"""CUDA training step for train_batched (FusedStep): M models, W never gets an autograd gradient.

* Solve: A = I - base has a positive-definite symmetric part while ||base||_2 < 1
  (the spectral penalty encourages this; it does not guarantee it), and then LU without pivoting
  is stable. Recursive LU (halve,
  recurse, one large GEMM update) with 64x64 leaves factored and inverted in registers by
  Triton; FP32. A model whose smallest pivot is below PIVOT_MIN (or NaN) keeps its no-pivot
  factor: its solves are iteratively refined against A (REFINE_SWEEPS) and accepted at normwise
  backward error <= REFINE_TOL; models that miss it are solved with pivoted LU (those models
  only). Every failing-pivot model is kept in FusedStep.fallbacks ("pivoted": needed pivoted
  LU); steps that needed pivoted LU are logged as warnings, fully refined steps at debug level.
* Spectral penalty: ONE warm-started power iteration (v carries over between steps), no backprop
  through it, so dsigma/d|W| = u v^T. z = W^T (W x) is one pass over W (_wtw).
* dL/dW is never written: it is offdiag*keep/kp * (Gb X^T + (base/|base|)(P Q^T + l1/n^2)
  + 2 l2/n^2 base) with Gb X^T rank b (solve backward) and P Q^T rank 1 (penalty). Two Triton
  passes build it tile by tile: per-model norm (for the clip), then clip + Adam in place.
"""
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

LEAF = 64
PIVOT_MIN = 0.05  # smallest no-pivot LU pivot accepted as is; below it the model's solves are refined
REFINE_TOL = 1e-7  # normwise backward error a refined solve must reach (pivoted LU's level), else pivoted LU
REFINE_SWEEPS = (2, 2)  # refinement sweeps before the first acceptance check, then before the second


# ---------------------------------------------------------------- no-pivot recursive LU
@triton.jit
def _leaf_lu_inv(A_ptr, Li_ptr, Ui_ptr, n, k0, end, NB: tl.constexpr):
    """In place: LU of A[m, k0:end, k0:end] (end-k0 <= NB, identity-padded). Writes L^-1, U^-1."""
    m = tl.program_id(0)
    i = tl.arange(0, NB)
    rows, cols = i[:, None], i[None, :]
    inb = (k0 + rows < end) & (k0 + cols < end)
    ptr = A_ptr + m.to(tl.int64) * n * n + (k0 + rows) * n + (k0 + cols)
    eye = tl.where(rows == cols, 1.0, 0.0)
    a = tl.where(inb, tl.load(ptr, mask=inb, other=0.0), eye)
    for k in range(NB):  # Doolittle elimination
        rowk = tl.sum(tl.where(rows == k, a, 0.0), axis=0)
        colk = tl.sum(tl.where(cols == k, a, 0.0), axis=1)
        piv = tl.sum(tl.where(i == k, rowk, 0.0), axis=0)
        l = tl.where(i > k, colk / piv, 0.0)
        a = a - l[:, None] * tl.where(i > k, rowk, 0.0)[None, :]
        a = tl.where((cols == k) & (rows > k), l[:, None], a)
    tl.store(ptr, a, mask=inb)
    x = eye  # L^-1 by forward substitution on I
    for k in range(NB):
        lk = tl.sum(tl.where((cols == k) & (rows > k), a, 0.0), axis=1)
        xk = tl.sum(tl.where(rows == k, x, 0.0), axis=0)
        x = x - lk[:, None] * xk[None, :]
    tl.store(Li_ptr + m * NB * NB + rows * NB + cols, x)
    y = eye  # U^-1 by back substitution on I
    for kk in range(NB):
        k = NB - 1 - kk
        uk = tl.sum(tl.where((cols == k) & (rows <= k), a, 0.0), axis=1)
        ukk = tl.sum(tl.where(i == k, uk, 0.0), axis=0)
        y = tl.where(rows == k, y / ukk, y)
        yk = tl.sum(tl.where(rows == k, y, 0.0), axis=0)
        y = y - tl.where(i < k, uk, 0.0)[:, None] * yk[None, :]
    tl.store(Ui_ptr + m * NB * NB + rows * NB + cols, y)


def _split(k0, k1):
    blocks = -(-(k1 - k0) // LEAF)
    return None if blocks == 1 else k0 + LEAF * (blocks // 2)


class _NoPivotLU:
    """Recursive no-pivot LU of a contiguous (M, n, n) batch, factored in place."""

    def __init__(self, A):
        self.A, self.Linv, self.Uinv = A, {}, {}
        self._lu(0, A.shape[-1])

    def _inv(self, k0, k1, lower, T):
        inv = (self.Linv if lower else self.Uinv)[k0][:, :k1 - k0, :k1 - k0]
        return inv.mT if T else inv

    def _lu(self, k0, k1):
        A, h = self.A, _split(k0, k1)
        if h is None:
            li = torch.empty(A.shape[0], LEAF, LEAF, device=A.device)
            ui = torch.empty_like(li)
            _leaf_lu_inv[(A.shape[0],)](A, li, ui, A.shape[-1], k0, k1, NB=LEAF, num_warps=4)
            self.Linv[k0], self.Uinv[k0] = li, ui
            return
        self._lu(k0, h)
        self._lower(k0, h, A[..., k0:h, h:k1], unit_L=True)            # U12 = L11^-1 A12
        self._upper_right(k0, h, A[..., h:k1, k0:h])                    # L21 = A21 U11^-1
        A[..., h:k1, h:k1].baddbmm_(A[..., h:k1, k0:h], A[..., k0:h, h:k1], alpha=-1)
        self._lu(h, k1)

    def _lower(self, k0, k1, B, unit_L):  # T X = B in place, T = L (unit) or U^T
        h = _split(k0, k1)
        if h is None:
            B.copy_(self._inv(k0, k1, lower=unit_L, T=not unit_L) @ B)
            return
        A = self.A
        self._lower(k0, h, B[..., :h - k0, :], unit_L)
        off = A[..., h:k1, k0:h] if unit_L else A[..., k0:h, h:k1].mT
        B[..., h - k0:, :].baddbmm_(off, B[..., :h - k0, :], alpha=-1)
        self._lower(h, k1, B[..., h - k0:, :], unit_L)

    def _upper(self, k0, k1, B, unit_LT):  # T X = B in place, T = U or L^T (unit)
        h = _split(k0, k1)
        if h is None:
            B.copy_(self._inv(k0, k1, lower=unit_LT, T=unit_LT) @ B)
            return
        A = self.A
        self._upper(h, k1, B[..., h - k0:, :], unit_LT)
        off = A[..., h:k1, k0:h].mT if unit_LT else A[..., k0:h, h:k1]
        B[..., :h - k0, :].baddbmm_(off, B[..., h - k0:, :], alpha=-1)
        self._upper(k0, h, B[..., :h - k0, :], unit_LT)

    def _upper_right(self, k0, k1, B):  # X U = B in place
        h = _split(k0, k1)
        if h is None:
            B.copy_(B @ self._inv(k0, k1, lower=False, T=False))
            return
        self._upper_right(k0, h, B[..., :, :h - k0])
        B[..., :, h - k0:].baddbmm_(B[..., :, :h - k0], self.A[..., k0:h, h:k1], alpha=-1)
        self._upper_right(h, k1, B[..., :, h - k0:])

    def solve(self, b, trans=False):
        x = b.clone(memory_format=torch.contiguous_format)
        n = self.A.shape[-1]
        if trans:  # A^T = U^T L^T
            self._lower(0, n, x, unit_L=False)
            self._upper(0, n, x, unit_LT=True)
        else:
            self._lower(0, n, x, unit_L=True)
            self._upper(0, n, x, unit_LT=False)
        return x

    def subset(self, idx):
        """The factor of models idx only (same solve code, smaller batch)."""
        s = object.__new__(_NoPivotLU)
        s.A = self.A[idx]
        s.Linv = {k: v[idx] for k, v in self.Linv.items()}
        s.Uinv = {k: v[idx] for k, v in self.Uinv.items()}
        return s


class _PivotedLU:
    """Pivoted LU factor (torch.linalg.lu_factor) of the models _RefinedLU could not refine; solved there."""

    def __init__(self, A):
        self.lu, self.piv = torch.linalg.lu_factor(A)


def _backward_error(A, R, x):
    """Normwise backward error ||R - A x|| / (||A|| ||x|| + ||R||) per model (Frobenius)."""
    res = (R - A @ x).norm(dim=(1, 2))
    return res / (A.norm(dim=(1, 2)) * x.norm(dim=(1, 2)) + R.norm(dim=(1, 2))).clamp_min(1e-30)


class _RefinedLU:
    """No-pivot factor lu of A (unfactored) whose models idx failed the pivot check: their solves are
    refined against A, and solved with pivoted LU (those models only) if they miss REFINE_TOL."""

    def __init__(self, lu, A, idx):
        self.lu, self.A, self.idx, self.sub = lu, A, idx, lu.subset(idx)
        self.pivoted, self.plu = [], None  # models in plu (sorted); factored lazily, reused by the second solve

    def _pivoted_rows(self, models):
        if not set(models) <= set(self.pivoted):  # union if the second solve needs more models
            self.pivoted = sorted(set(self.pivoted) | set(models))
            self.plu = _PivotedLU(self.A[self.pivoted])
        return [self.pivoted.index(m) for m in models]

    def solve(self, b, trans=False):
        x, idx = self.lu.solve(b, trans=trans), self.idx
        Ab, bb, xb = (self.A[idx].mT if trans else self.A[idx]), b[idx], x[idx]
        # All failing models keep refining until all pass or REFINE_SWEEPS runs out.
        for k in REFINE_SWEEPS:
            for _ in range(k):
                xb = xb + self.sub.solve(bb - Ab @ xb, trans=trans)
            ok = _backward_error(Ab, bb, xb) <= REFINE_TOL  # NaN fails
            if bool(ok.all()):
                break
        if not bool(ok.all()):
            fail = (~ok).nonzero().flatten()
            pos = self._pivoted_rows(idx[fail].tolist())
            xb[fail] = torch.linalg.lu_solve(self.plu.lu[pos], self.plu.piv[pos], bb[fail], adjoint=trans)
        x[idx] = xb
        return x


# ---------------------------------------------------------------- fused power iteration
@triton.jit
def _wtw_kernel(W_ptr, x_ptr, y_ptr, part_ptr, n, n_ch, RCH: tl.constexpr, BR: tl.constexpr, NP: tl.constexpr):
    """Rows [ch*RCH, ch*RCH+RCH) of model m: y = W x for these rows, partial z = W[rows]^T y."""
    ch, m = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, NP)
    cm = cols < n
    x = tl.load(x_ptr + m * n + cols, mask=cm, other=0.0)
    acc = tl.zeros([NP], dtype=tl.float32)
    base = W_ptr + m.to(tl.int64) * n * n
    for r0 in range(ch * RCH, ch * RCH + RCH, BR):
        rows = r0 + tl.arange(0, BR)
        rm = rows < n
        tile = tl.load(base + rows[:, None] * n + cols[None, :], mask=rm[:, None] & cm[None, :], other=0.0)
        y = tl.sum(tile * x[None, :], axis=1)
        tl.store(y_ptr + m * n + rows, y, mask=rm)
        acc += tl.sum(tile * y[:, None], axis=0)
    tl.store(part_ptr + (m * n_ch + ch) * NP + cols, acc, mask=cm)


def _wtw(W, x):
    """(W x, W^T W x) for W (M, n, n) contiguous fp32, x (M, n), in one read of W."""
    M, n = x.shape
    NP = triton.next_power_of_2(n)
    BR = max(1, 8192 // NP)  # whole rows in registers; n > ~16k needs a column-tiled kernel
    RCH = max(64, BR)
    n_ch = triton.cdiv(n, RCH)
    y = torch.empty_like(x)
    part = torch.empty(M, n_ch, NP, device=x.device)
    _wtw_kernel[(n_ch, M)](W, x.contiguous(), y, part, n, n_ch, RCH=RCH, BR=BR, NP=NP, num_warps=8)
    return y, part[..., :n].sum(1)


# ---------------------------------------------------------------- fused dL/dW -> clip -> Adam
@triton.jit
def _grad_tile(W_ptr, K_ptr, GB_ptr, X_ptr, P_ptr, Q_ptr, m, n, inv_keep, l1c, l2c,
               K1: tl.constexpr, K2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Tile of model m's dL/dW, rebuilt from W, the int8 dropout mask and the thin factors."""
    nc = tl.cdiv(n, BN)
    r = (tl.program_id(0) // nc) * BM + tl.arange(0, BM)
    c = (tl.program_id(0) % nc) * BN + tl.arange(0, BN)
    inb = (r[:, None] < n) & (c[None, :] < n)
    flat = m.to(tl.int64) * n * n + r[:, None] * n + c[None, :]
    w = tl.load(W_ptr + flat, mask=inb, other=0.0)
    keep = (tl.load(K_ptr + flat, mask=inb, other=0) != 0) & (r[:, None] != c[None, :]) & inb
    base = tl.where(keep, w * inv_keep, 0.0)
    t1 = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in tl.static_range(0, K1, 32 if K1 > 32 else K1):  # K slices: shared memory independent of batch
        k1 = k0 + tl.arange(0, 32 if K1 > 32 else K1)
        gb = tl.load(GB_ptr + (m * n + r[:, None]) * K1 + k1[None, :], mask=r[:, None] < n, other=0.0)
        xt = tl.load(X_ptr + (m * n + c[None, :]) * K1 + k1[:, None], mask=c[None, :] < n, other=0.0)
        t1 += tl.dot(gb, xt, input_precision="ieee")
    k2 = tl.arange(0, K2)
    p = tl.load(P_ptr + (m * n + r[:, None]) * K2 + k2[None, :], mask=r[:, None] < n, other=0.0)
    qt = tl.load(Q_ptr + (m * n + c[None, :]) * K2 + k2[:, None], mask=c[None, :] < n, other=0.0)
    dbase = (t1 + base / tl.sqrt(base * base + 1e-12) * (tl.dot(p, qt, input_precision="ieee") + l1c)
             + 2.0 * l2c * base)
    return tl.where(keep, dbase * inv_keep, 0.0), w, flat, inb


@triton.jit
def _norm_kernel(W_ptr, K_ptr, GB_ptr, X_ptr, P_ptr, Q_ptr, part_ptr, n, inv_keep, l1c, l2c, n_tiles,
                 K1: tl.constexpr, K2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(1)
    g, _, _, _ = _grad_tile(W_ptr, K_ptr, GB_ptr, X_ptr, P_ptr, Q_ptr, m, n, inv_keep, l1c, l2c, K1, K2, BM, BN)
    tl.store(part_ptr + m * n_tiles + tl.program_id(0), tl.sum(g * g))


@triton.jit
def _adam_kernel(W_ptr, K_ptr, GB_ptr, X_ptr, P_ptr, Q_ptr, EA_ptr, ES_ptr, coef_ptr, n, inv_keep, l1c, l2c,
                 step_size, b1, b2, eps, bc2_sqrt,
                 K1: tl.constexpr, K2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Same update as torch.optim.Adam (no weight decay) on the clipped gradient, in place."""
    m = tl.program_id(1)
    g, w, flat, inb = _grad_tile(W_ptr, K_ptr, GB_ptr, X_ptr, P_ptr, Q_ptr, m, n, inv_keep, l1c, l2c, K1, K2, BM, BN)
    g = g * tl.load(coef_ptr + m)
    ea = tl.load(EA_ptr + flat, mask=inb, other=0.0) * b1 + (1.0 - b1) * g
    es = tl.load(ES_ptr + flat, mask=inb, other=0.0) * b2 + (1.0 - b2) * g * g
    tl.store(EA_ptr + flat, ea, mask=inb)
    tl.store(ES_ptr + flat, es, mask=inb)
    tl.store(W_ptr + flat, w - step_size * ea / (tl.sqrt(es) / bc2_sqrt + eps), mask=inb)


BM, BN, BN_NORM = 32, 32, 64  # fastest tiles on an A40 (bigger ones spill registers)


def _pad_cols(x, k):
    """(M, n, j) -> contiguous (M, n, k), zero-padded (tl.dot needs k >= 16)."""
    out = x.new_zeros(*x.shape[:-1], k)
    out[..., :x.shape[-1]] = x
    return out


def _front(W, offdiag, eye, p: float):
    keep = torch.bernoulli(torch.full_like(W, 1.0 - p)) if p > 0 else torch.ones_like(W)
    base = W * offdiag * keep / (1.0 - p)
    w_mag = torch.sqrt(base.pow(2) + 1e-12)
    A = eye - base
    return keep.to(torch.int8), w_mag, A, w_mag.pow(2).mean(dim=(1, 2)), w_mag.mean(dim=(1, 2))


class FusedStep:
    """train_batched's CUDA step: W (M, n, n) is updated in place (clip + Adam fused), B's gradient
    is left on B.grad for the caller's optimizer. Loss as _batched_step_loss except the spectral
    penalty (module docstring)."""

    def __init__(self, W, cfg, compile_front=True):
        n = W.shape[-1]
        self.W, self.cfg, self.t = W, cfg, 0
        self.fallbacks = []  # one {"model", "epoch", "step", "min_pivot", "pivoted"} per model per failing-pivot step
        self.min_pivot = torch.full((W.shape[0],), float("inf"), device=W.device)  # per model, over all steps
        self.eye = torch.eye(n, device=W.device)
        self.offdiag = 1.0 - self.eye
        self.exp_avg, self.exp_avg_sq = torch.zeros_like(W), torch.zeros_like(W)
        self.front = torch.compile(_front) if compile_front else _front

    def step(self, B, v, u_b, d_b, se_b, group, train_w, epoch=-1, step=-1):
        """One step on batch (u_b, d_b, se_b); group = W's optimizer param group (lr, betas, eps).
        epoch / step only label fallback records and logs. Returns (per-model loss, next v)."""
        with torch.cuda.device(self.W.device):  # Triton launches on the current device, not W's
            return self._step(B, v, u_b, d_b, se_b, group, train_w, epoch, step)

    def _step(self, B, v, u_b, d_b, se_b, group, train_w, epoch, step):
        cfg, W = self.cfg, self.W
        n, p = W.shape[-1], cfg.w_dropout
        with torch.no_grad():
            keep, w_mag, A, l2m, l1m = self.front(W, self.offdiag, self.eye, p)
            # A stays unfactored for the residuals and pivoted LU, an extra n^2 per model. Recomputing it for
            # the failing models only would change the rounding.
            Af = A.clone()
            lu = _NoPivotLU(Af)  # factored in place, diag(Af) = the pivots
            piv = Af.diagonal(dim1=-2, dim2=-1).amin(dim=1)  # (M,) smallest pivot per model, NaN propagates
            torch.minimum(self.min_pivot, piv, out=self.min_pivot)
            failed = []
            # One host sync per step.
            if not bool((piv >= PIVOT_MIN).all()):
                bad = torch.nonzero(~(piv >= PIVOT_MIN)).flatten()
                failed = [{"model": m, "epoch": epoch, "step": step, "min_pivot": p_min}
                          for m, p_min in zip(bad.tolist(), piv[bad].tolist())]
                lu = _RefinedLU(lu, A, bad)
            x = lu.solve((u_b * B[:, None, :]).transpose(1, 2))                    # (M, n, b)
            # 1 power iteration, no backprop: sigma lags if the top singular gap is small or W moves fast.
            v = _wtw(w_mag, v / (v.norm(dim=1, keepdim=True) + 1e-8))[1]
            v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
            s = _wtw(w_mag, v)[0]                                                  # W v
            sigma = s.norm(dim=1)
        pred = x.transpose(1, 2).requires_grad_()
        w_2d = 1.0 / (se_b + 1e-8)
        w_2d = w_2d / (w_2d.mean(dim=(1, 2), keepdim=True) + 1e-8)
        contrast = (w_2d * (pred - d_b).abs()).mean(dim=(1, 2))
        (g_pred,) = torch.autograd.grad((cfg.delta_weight * contrast).sum(), pred)
        with torch.no_grad():
            excess = torch.relu(sigma - cfg.tau)
            loss = (cfg.delta_weight * contrast + cfg.stability_weight * excess ** 2 + cfg.l2_weight * l2m
                    + cfg.l1_weight * l1m + cfg.direct_l1_weight * B.abs().mean(dim=1))
            gb = lu.solve(g_pred.transpose(1, 2).contiguous(), trans=True)         # A^-T dL/dx
            if failed:
                for f in failed:
                    f["pivoted"] = f["model"] in lu.pivoted
                self.fallbacks += failed
                logger.log(logging.WARNING if lu.pivoted else logging.DEBUG,
                           "No-pivot LU pivot < %g at epoch %d step %d (model: min pivot %s); refined, "
                           "pivoted LU for models %s", PIVOT_MIN, epoch, step,
                           {f["model"]: f["min_pivot"] for f in failed}, lu.pivoted)
            B.grad = (gb.transpose(1, 2) * u_b).sum(1) + cfg.direct_l1_weight * torch.sign(B) / n
            sq = B.grad.pow(2).sum(1)
            if train_w:
                K1 = max(16, triton.next_power_of_2(gb.shape[-1]))
                g_s = (cfg.stability_weight * 2.0 * excess / sigma)[:, None, None] * s[..., None]
                args = (W, keep, _pad_cols(gb, K1), _pad_cols(x, K1), _pad_cols(g_s, 16), _pad_cols(v[..., None], 16))
                consts = (n, 1.0 / (1.0 - p), cfg.l1_weight / n ** 2, cfg.l2_weight / n ** 2)
                if cfg.grad_clip > 0:
                    nt = triton.cdiv(n, BM) * triton.cdiv(n, BN_NORM)
                    part = torch.empty(W.shape[0], nt, device=W.device)
                    _norm_kernel[(nt, W.shape[0])](*args, part, *consts, nt, K1, 16, BM, BN_NORM)
                    sq = sq + part.sum(1)
            coef = ((cfg.grad_clip / (sq.sqrt() + 1e-6)).clamp(max=1.0) if cfg.grad_clip > 0
                    else torch.ones_like(sq))
            B.grad.mul_(coef[:, None])
            if train_w:
                self.t += 1
                (b1, b2), lr = group["betas"], group["lr"]
                nt = triton.cdiv(n, BM) * triton.cdiv(n, BN)
                _adam_kernel[(nt, W.shape[0])](*args, self.exp_avg, self.exp_avg_sq, coef, *consts,
                                               lr / (1 - b1 ** self.t), b1, b2, group["eps"],
                                               (1 - b2 ** self.t) ** 0.5, K1, 16, BM, BN)
        return loss, v
