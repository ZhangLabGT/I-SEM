"""Interventional SEM: fits delta_c = (I - W)^-1 (B * u_c) to the mean shift of each knockdown.

W[i, j] is the effect of gene j on gene i, diag(W) = 0, u_c marks the knocked-down gene of condition c,
and B holds one learned knockdown strength per gene. Models train through train_batched.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def perturbation_to_vector(perturbations: List[str], genes: List[str]) -> np.ndarray:
    """Convert perturbation labels to binary vectors."""
    gene_map = {gene: i for i, gene in enumerate(genes)}
    vectors = np.zeros((len(perturbations), len(genes)), dtype=np.float32)
    for i, p_str in enumerate(perturbations):
        p_list = str(p_str).split(",")
        for p in p_list:
            p = p.strip()
            if p in gene_map:
                vectors[i, gene_map[p]] = 1.0
    return vectors


def infer_control_label(interventions: List[str]) -> str:
    """Heuristic to pick a control label from intervention strings."""
    candidates = ["non-targeting", "control", "ctrl", ""]
    available = set(interventions)
    available_lower = {s.lower(): s for s in available}
    for cand in candidates:
        if cand in available:
            return cand
        if cand.lower() in available_lower:
            matched = available_lower[cand.lower()]
            logger.warning(
                "Control label '%s' matched case-insensitively (exact candidate: '%s')",
                matched, cand,
            )
            return matched
    values, counts = np.unique(np.array(interventions), return_counts=True)
    fallback = values[int(np.argmax(counts))]
    logger.warning(
        "No standard control label found. Falling back to most frequent label: '%s' (%d cells). "
        "Available labels: %s",
        fallback, int(counts.max()), sorted(available)[:20],
    )
    return fallback


def build_intervention_targets(
    expression: np.ndarray,
    interventions: List[str],
    gene_names: List[str],
    control_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray, Optional[np.ndarray]]:
    """
    Build interventional training targets from condition means.

    Returns:
        u_mat: (n_conditions, n_genes) perturbation vectors
        delta_obs: (n_conditions, n_genes) mean shift vs control
        weights: (n_conditions,) number of cells per condition
        control_label: resolved control label
        control_mean: (n_genes,) control mean expression
        se_obs: (n_conditions, n_genes) standard error of delta_obs per condition-gene,
                computed as pooled SE = sqrt(var_cond/n_cond + var_ctrl/n_ctrl).

    Raises:
        ValueError: if the control or a used perturbation condition has fewer than 2 cells
            (its variance, hence the SE, is undefined).
    """
    interventions_arr = np.array(interventions)
    if control_label is None:
        control_label = infer_control_label(interventions)

    control_mask = interventions_arr == control_label
    if control_mask.sum() == 0:
        raise ValueError(f"No control cells found for label '{control_label}'.")

    control_mean = expression[control_mask].mean(axis=0)
    n_ctrl = float(control_mask.sum())
    if n_ctrl < 2:
        raise ValueError(f"Control '{control_label}' has {int(n_ctrl)} cell; the standard error needs >= 2 cells.")
    ctrl_var = expression[control_mask].var(axis=0, ddof=1)

    u_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []
    weight_list: List[float] = []
    se_list: List[np.ndarray] = []

    unique_conditions = np.unique(interventions_arr)
    dropped_unmatched = []
    too_few_cells = []
    for label in unique_conditions:
        if label == control_label:
            continue
        cond_mask = interventions_arr == label
        n_cond = float(cond_mask.sum())
        if n_cond == 0:
            continue
        u_vec = perturbation_to_vector([label], gene_names)[0]
        if u_vec.sum() == 0:
            dropped_unmatched.append(label)
            continue
        if n_cond < 2:
            too_few_cells.append(str(label))
            continue
        cond_expr = expression[cond_mask]
        cond_mean = cond_expr.mean(axis=0)
        delta_obs = cond_mean - control_mean
        cond_var = cond_expr.var(axis=0, ddof=1)
        se = np.sqrt(cond_var / n_cond + ctrl_var / n_ctrl + 1e-12)
        u_list.append(u_vec)
        delta_list.append(delta_obs)
        weight_list.append(n_cond)
        se_list.append(se)

    if too_few_cells:
        raise ValueError(
            f"{len(too_few_cells)} condition(s) have 1 cell; the standard error needs >= 2 cells: "
            f"{too_few_cells[:10]}"
        )
    if dropped_unmatched:
        logger.warning(
            "Dropped %d perturbation conditions with no matching gene name: %s",
            len(dropped_unmatched), dropped_unmatched[:10],
        )
    n_total = len(unique_conditions) - 1  # exclude control
    n_kept = len(u_list)
    if n_kept < n_total:
        logger.info(
            "Kept %d / %d perturbation conditions (dropped %d unmatched)",
            n_kept, n_total, n_total - n_kept,
        )

    if not u_list:
        raise ValueError("No valid perturbation conditions found for training.")

    return (
        np.stack(u_list, axis=0),
        np.stack(delta_list, axis=0),
        np.array(weight_list, dtype=np.float32),
        control_label,
        control_mean.astype(np.float32),
        np.stack(se_list, axis=0).astype(np.float32),
    )


class InterventionalSEM(nn.Module):
    """Start values of one model. train_batched stacks them, so the draw order fixes the results."""

    def __init__(self, n_genes: int, init_scale: float = 1e-3) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.weights = nn.Parameter(torch.randn(n_genes, n_genes) * init_scale)
        self.direct = nn.Parameter(torch.full((n_genes,), 1.0))
        self.register_buffer("_spectral_v", torch.randn(n_genes))


def initialize_direct_from_deltas(
    model: InterventionalSEM,
    u_mat: np.ndarray,
    delta_obs: np.ndarray,
) -> None:
    """Set B[j] to mean(|delta_obs[c, j]|) over conditions knocking down j; other genes keep B[j] = 1."""
    b_init = np.ones(model.n_genes, dtype=np.float32)
    for j in range(model.n_genes):
        mask = u_mat[:, j] > 0.5
        if mask.any():
            b_init[j] = np.mean(np.abs(delta_obs[mask, j]))
    b_init = np.clip(b_init, 0.01, 100.0)
    with torch.no_grad():
        model.direct.copy_(torch.from_numpy(b_init))


@dataclass
class InterventionalSEMConfig:
    """Training settings. src.pipeline._sem_config sets the shipped values."""
    epochs: int = 500
    batch_size: int = 32
    lr: float = 0.05
    lr_weights: float = 0.01
    stability_weight: float = 1.0
    delta_weight: float = 10.0
    l2_weight: float = 1e-4
    direct_l1_weight: float = 0.0
    tau: float = 0.9
    grad_clip: float = 1.0
    warmup_epochs: int = 0  # epochs that train B with W frozen
    l1_weight: float = 0.0
    w_dropout: float = 0.0


def _batched_step_loss(
    W: torch.Tensor,
    B: torch.Tensor,
    v: torch.Tensor,
    u_b: torch.Tensor,
    d_b: torch.Tensor,
    se_b: torch.Tensor,
    offdiag: torch.Tensor,
    eye: torch.Tensor,
    w_dropout: float,
    delta_weight: float,
    stability_weight: float,
    l2_weight: float,
    l1_weight: float,
    direct_l1_weight: float,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-model training loss (M,) on the CPU path; CUDA uses batched_linalg.FusedStep.
    Returns (loss, updated spectral vector v)."""
    base = W * offdiag
    if w_dropout > 0.0:
        base = base * torch.bernoulli(torch.full_like(base, 1.0 - w_dropout)) / (1.0 - w_dropout)
    w_mag = torch.sqrt(base.pow(2) + 1e-12)

    drive = u_b * B[:, None, :]
    pred = torch.linalg.solve_ex(eye - base, drive.transpose(1, 2))[0].transpose(1, 2)
    w_2d = 1.0 / (se_b + 1e-8)
    w_2d = w_2d / (w_2d.mean(dim=(1, 2), keepdim=True) + 1e-8)
    contrast = (w_2d * (pred - d_b).abs()).mean(dim=(1, 2))

    # Spectral penalty: the gradient flows through the power iterations.
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    for _ in range(5):
        v = torch.bmm(w_mag.transpose(1, 2), torch.bmm(w_mag, v[..., None]))[..., 0]
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    sigma = torch.bmm(w_mag, v[..., None])[..., 0].norm(dim=1)
    stability = F.relu(sigma - tau) ** 2

    loss = (
        delta_weight * contrast
        + stability_weight * stability
        + l2_weight * w_mag.pow(2).mean(dim=(1, 2))
        + l1_weight * w_mag.abs().mean(dim=(1, 2))
        + direct_l1_weight * B.abs().mean(dim=1)
    )
    return loss, v.detach()


def _clip_per_model_(grads: List[torch.Tensor], max_norm: float) -> None:
    """clip_grad_norm_ applied separately to each model's slice of the batched grads."""
    sq = sum(g.pow(2).flatten(1).sum(dim=1) for g in grads)
    coef = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(coef.view(-1, *([1] * (g.dim() - 1))))


def _model_norms(W: torch.Tensor) -> Dict[str, List[float]]:
    """Per-model norms of trained W (M, n, n), diagonal zero, in float64:
    ||abs(W)||_2 (what the stability penalty acts on), ||W||_2, rho(W) and sigma_min(I - W)."""
    out: Dict[str, List[float]] = {"sigma_abs": [], "norm2_W": [], "rho_W": [], "sigma_min_IminusW": []}
    eye = torch.eye(W.shape[-1], dtype=torch.float64, device=W.device)
    for w in W:  # one model at a time: a float64 copy of all M models would double peak memory
        w = w.double()
        out["sigma_abs"].append(torch.linalg.matrix_norm(w.abs(), ord=2).item())
        out["norm2_W"].append(torch.linalg.matrix_norm(w, ord=2).item())
        out["rho_W"].append(torch.linalg.eigvals(w).abs().max().item())
        out["sigma_min_IminusW"].append(torch.linalg.svdvals(eye - w)[-1].item())
    return out


def train_batched(
    u_mat: np.ndarray,
    delta_obs: np.ndarray,
    se_obs: np.ndarray,
    folds: List[np.ndarray],
    seeds: List[int],
    config: InterventionalSEMConfig,
    device: str = "cpu",
    return_params: bool = False,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Train len(folds) independent models at once on one device.

    Model m trains on conditions folds[m], with torch.manual_seed(seeds[m]) and B from its fold's shifts.
    W and B are stacked into (M, n, n) and (M, n) parameters. Adam is elementwise and the gradient clip is
    per model, so the models never interact. Single models are chaotic: compare ensembles, not models.

    On CUDA, batched_linalg.FusedStep replaces autograd: no-pivot recursive FP32 LU, with models whose pivot
    is below PIVOT_MIN refined and pivoted LU only for those that miss the accuracy target. The spectral
    penalty uses one warm-started power iteration without backprop (five with backprop on CPU).

    Returns:
        causal: (M, n, n) |(I-W)^{-1} diag(B)| per model, diagonal zeroed.
        final_loss: (M,) mean training loss over the last epoch.
        W, B: only if return_params. (M, n, n) trained W (diagonal zeroed) and (M, n) B.
        diagnostics: only if return_diagnostics (last element). Per-model lists, also logged: _model_norms
            of the trained W, "min_pivot" = smallest no-pivot LU pivot over all steps (CUDA; NaN on CPU),
            and "fallbacks" = FusedStep.fallbacks (one record per model per step with a pivot < PIVOT_MIN;
            "pivoted" = that model's solves needed pivoted LU after refinement).
    """
    cfg = config
    if len(folds) != len(seeds) or not folds:
        raise ValueError("folds and seeds must be non-empty and the same length")
    half = len(folds[0])
    if any(len(f) != half for f in folds):
        raise ValueError("train_batched needs equal-size folds")

    M, n = len(folds), u_mat.shape[1]
    W0, B0, V0 = [], [], []
    for fold, seed in zip(folds, seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = InterventionalSEM(n_genes=n)
        initialize_direct_from_deltas(model, u_mat[fold], delta_obs[fold])
        W0.append(model.weights.data)
        B0.append(model.direct.data)
        V0.append(model._spectral_v)
    W = nn.Parameter(torch.stack(W0).to(device))
    B = nn.Parameter(torch.stack(B0).to(device))
    v = torch.stack(V0).to(device)
    del W0, B0, V0

    idx = torch.as_tensor(np.stack(folds), device=device)                      # (M, half)
    U, D, S = (torch.as_tensor(a, dtype=torch.float32, device=device)[idx]    # (M, half, n)
               for a in (u_mat, delta_obs, se_obs))
    eye = torch.eye(n, device=device)
    offdiag = 1.0 - eye

    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        from src.models.batched_linalg import FusedStep
        fused = FusedStep(W.data, cfg)

    optimizer = torch.optim.Adam(
        [{"params": [W], "lr": cfg.lr_weights}, {"params": [B], "lr": cfg.lr}], fused=on_cuda,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    shuffle_gen = torch.Generator(device=device).manual_seed(int(seeds[0]))
    torch.manual_seed(int(seeds[0]))  # dropout masks

    epoch_loss = torch.zeros(M, device=device)
    n_batches = 1
    for epoch in range(cfg.epochs):
        # Warm-up: W frozen, B trains.
        train_w = not (cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs)
        W.requires_grad_(train_w)
        perm = torch.argsort(torch.rand(M, half, device=device, generator=shuffle_gen), dim=1)
        epoch_loss.zero_()
        n_batches = 0
        for start in range(0, half, cfg.batch_size):
            bi = perm[:, start:start + cfg.batch_size, None]
            u_b, d_b, se_b = (torch.take_along_dim(x, bi, dim=1) for x in (U, D, S))
            optimizer.zero_grad()
            if on_cuda:  # updates W in place, leaves B's clipped gradient on B.grad
                loss, v = fused.step(B, v, u_b, d_b, se_b, optimizer.param_groups[0], train_w,
                                     epoch=epoch, step=start // cfg.batch_size)
            else:
                loss, v = _batched_step_loss(
                    W, B, v, u_b, d_b, se_b, offdiag, eye, cfg.w_dropout, cfg.delta_weight,
                    cfg.stability_weight, cfg.l2_weight, cfg.l1_weight, cfg.direct_l1_weight, cfg.tau,
                )
                loss.sum().backward()  # parameters are disjoint per model -> per-model gradients
                if cfg.grad_clip > 0:
                    _clip_per_model_([g for g in (W.grad, B.grad) if g is not None], cfg.grad_clip)
            optimizer.step()
            epoch_loss += loss.detach()  # no host sync per step
            n_batches += 1
        scheduler.step()

    with torch.no_grad():
        inv = torch.linalg.inv(eye - W * offdiag)
        effect = (inv * B[:, None, :] * offdiag).abs()
    out = (effect.cpu().numpy(), (epoch_loss / n_batches).cpu().numpy())
    if return_params:
        out += ((W * offdiag).detach().cpu().numpy(), B.detach().cpu().numpy())
    if return_diagnostics:
        diagnostics = _model_norms((W * offdiag).detach())
        diagnostics["min_pivot"] = fused.min_pivot.tolist() if on_cuda else [float("nan")] * M
        diagnostics["fallbacks"] = list(fused.fallbacks) if on_cuda else []
        logger.info(
            "train_batched: %d models, min/median/max %s; %d failing-pivot records, %d needed pivoted LU",
            M, "; ".join(f"{k} {np.min(x):.4g}/{np.median(x):.4g}/{np.max(x):.4g}"  # NaN shows as nan
                         for k, x in diagnostics.items() if k != "fallbacks"),
            len(diagnostics["fallbacks"]), sum(f["pivoted"] for f in diagnostics["fallbacks"]),
        )
        out += (diagnostics,)
    return out
