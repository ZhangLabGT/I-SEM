"""The three I-SEM variants of scripts/config_tests (README.md). Pure functions: data in, (result, meta) out.

  fit_c  GRNInferencePipeline() defaults: the shipped method, a dense ranking S (W[i, j] = effect of j on i).
  fit_a  E7's held-out graph-size rule (ksel.py, copied unchanged): top-k* edges of the full-data default fit.
  fit_b  L1 weight by minimum 5-fold CV error of the held-out condition-mean shifts; refit on all cells.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "scripts/experiments/e7_stability")]
import ksel  # noqa: E402
from src.models.interventional_nb_sem import build_intervention_targets, infer_control_label, train_batched  # noqa: E402
from src.pipeline import GRNInferencePipeline, _complementary_pairs, _sem_config  # noqa: E402

L1_GRID = np.geomspace(5e-4, 5e-2, 9)      # index 4 = 0.005 = the pipeline default
L1_GRID[4] = GRNInferencePipeline().l1_weight   # geomspace gives 0.004999999999999999; pin the default exactly
N_FOLDS = 5

# Copied from retune.py line 42 (imported by E7's select(), so active when E7 chose k*): CausalBench's FOR cap is
# irrelevant outside CausalBench. Without it the rule is not the one E7 ran.
ksel.unconnected_pairs = lambda *_: ksel.OMISSION_SIZE


def default_device():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def pipeline_c(device):
    """The shipped method: every GRNInferencePipeline default; only the device is set."""
    return GRNInferencePipeline(device=device)


def topk_mask(S, k):
    """Top-k ordered pairs of |S|, diagonal excluded, ties by index (as e9.topk / e8._topk_mask)."""
    n = len(S)
    s = np.where(~np.eye(n, dtype=bool), np.abs(S), -np.inf).ravel()
    m = np.zeros(n * n, bool)
    m[np.argsort(-s, kind="mergesort")[:k]] = True
    return m.reshape(n, n)


# ------------------------------------------------------------------------------------------------ (c)
def fit_c(X, labels, genes, device=None):
    device = device or default_device()
    t0 = time.time()
    res = pipeline_c(device).run(X, list(labels), list(genes))
    return res.adjacency, {"variant": "c", "n_models": res.metadata["n_models"], "device": device,
                           "wall_s": time.time() - t0}


# ------------------------------------------------------------------------------------------------ (a)
# Validator and ksel_score: copied from the earlier hyperparameter retuning code
# retune.py lines 81-119 (sha256 in README), unchanged except that the module-level names they use are imported here.
class Validator:
    """Welch p-values for every (source, target) on validation cells: identical to ksel.EdgeTester, vectorized."""

    def __init__(self, Xv, ivv, genes):
        self.tester = ksel.EdgeTester(Xv, ivv, genes)
        n = len(genes)
        self.P = np.full((n, n), np.nan)  # P[source, target]
        self.ok = np.zeros(n, bool)
        for g, i in self.tester.g2i.items():
            if self.tester.testable(g):
                self.P[i] = ttest_ind(self.tester.kd[g], self.tester.ctrl, axis=0, equal_var=False).pvalue
                self.ok[i] = True
        self.SIG = np.where(self.ok[:, None], (np.nan_to_num(self.P, nan=1.0) < ksel.P_EDGE).astype(np.int8), -1)

    def check(self, genes, rng, n=300):
        """Assert the vectorized table equals EdgeTester.significant on random edges."""
        e = [(genes[a], genes[b]) for a, b in rng.randint(len(genes), size=(n, 2)) if a != b]
        g2i = self.tester.g2i
        mine = np.array([self.SIG[g2i[s], g2i[t]] for s, t in e])
        assert np.array_equal(mine, self.tester.significant(e)), "vectorized Welch table != EdgeTester"


def ksel_score(S, val, genes, seed):
    n = len(genes)
    M = S.copy(); np.fill_diagonal(M, -np.inf)
    order = np.argsort(-M.ravel(), kind="stable")[: n * (n - 1)]
    tgt, src = np.divmod(order, n)  # S[i, j] = effect of j on i  ->  edge j -> i
    ranked = list(zip(src.tolist(), tgt.tolist()))
    null_sig = val.tester.significant(val.tester.null_edges(ksel.N_NULL, seed))
    sig = lambda edges: np.array([val.SIG[s, t] for s, t in edges], np.int8)
    k, rec = ksel.select_k(ranked, sig, null_sig, [str(i) for i in range(n)], ksel.BLOCK)
    acc = [b for b in rec["curve"] if b["end"] <= k]
    rec["excess_validated"] = float(sum(b["validated"] - rec["random_rate"] * b["tested"] for b in acc))
    s_all = val.SIG[src, tgt]
    for K in (1000, 5000, 20000):
        s = s_all[:K]; s = s[s >= 0]
        rec[f"precision_at_{K}"] = float(s.mean()) if len(s) else float("nan")
    rec["curve"] = rec["curve"][: len(acc) + 1]
    return rec
# ------------------------------------------------------------------------------- end of retune.py copy


def fit_a(X, labels, genes, S_full, seed=0, device=None):
    """E7 select(): split_cells(labels, 0.2, seed) -> defaults on the 80% -> k* on the held-out 20% (ksel_score
    on the rule-a score = the pipeline adjacency) -> top-k* of S_full (this run's full-data default fit).
    ksel's tester finds controls by ksel.CONTROL ("non-targeting"), so a differently named control label (the
    simulators' "control") is renamed for the tester only; the split uses the labels as given."""
    device = device or default_device()
    t0 = time.time()
    iv = np.asarray(labels)
    val = ksel.split_cells(iv, ksel.VAL_FRACTION, seed)
    S80, m80 = fit_c(X[~val], iv[~val], genes, device)
    ctrl = infer_control_label(list(iv))
    ivk = np.where(iv == ctrl, ksel.CONTROL, iv)
    validator = Validator(X[val], ivk[val], list(genes))
    validator.check(list(genes), np.random.RandomState(seed))
    rec = ksel_score(S80, validator, list(genes), seed)
    k = int(rec["k"])
    return topk_mask(S_full, k), {"variant": "a", "k_star": k, "seed": seed, "val_fraction": ksel.VAL_FRACTION,
                                  "n_val_cells": int(val.sum()), "control_label": str(ctrl), "ksel": rec,
                                  "device": device, "fit80_wall_s": m80["wall_s"], "wall_s": time.time() - t0}


# ------------------------------------------------------------------------------------------------ (b)
def targets(X, iv, genes):
    """build_intervention_targets on the conditions (control included) with >= 2 of these cells; the rest are
    skipped (their SE is undefined). Returns (u, delta, se, skipped labels)."""
    lab, cnt = np.unique(iv, return_counts=True)
    keep = np.isin(iv, lab[cnt >= 2])
    u, delta, _, _, _, se = build_intervention_targets(X[keep], list(iv[keep]), list(genes))
    return u, delta, se, [str(x) for x in lab[cnt < 2]]


def heldout_error(P, u_te, d_te, se_te, seen):
    """Mean of |delta_test - pred| / se_test over the held-out conditions whose knockdown some model saw
    (P[:, j] = predicted shift of knockdown j). Returns (error, number of conditions used)."""
    tv = u_te.argmax(1)
    keep = seen[tv]
    err = np.abs(d_te[keep] - P[:, tv[keep]].T) / se_te[keep]
    err = err[np.isfinite(err)]
    return (float(err.mean()) if err.size else float("nan")), int(keep.sum())


def cv_folds(labels, seed, n_splits=N_FOLDS):
    """StratifiedKFold over cells, stratified by the cell's label (controls included)."""
    from sklearn.model_selection import StratifiedKFold
    iv = np.asarray(labels)
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(iv)), iv))


def train_ensemble(u, delta, se, l1, device):
    """The pipeline's 120-model ensemble (same config, jobs and single chunk as GRNInferencePipeline.run) at L1 l1.
    Returns W, B, membership[m, j] = knockdown j is in model m's half."""
    p = GRNInferencePipeline()
    cfg = dataclasses.replace(_sem_config(p.epochs, p.warmup_epochs, p.l1_weight, p.w_dropout), l1_weight=float(l1))
    jobs = _complementary_pairs(u.shape[0], p.n_resamples)
    assert p.models_per_chunk >= len(jobs), "the pipeline would train in several chunks"
    _, _, W, B, _ = train_batched(u, delta, se, folds=[j[0] for j in jobs], seeds=[j[1] for j in jobs],
                                  config=cfg, device=device, return_params=True, return_diagnostics=True)
    return W, B, np.stack([(u[j[0]] > .5).any(0) for j in jobs])


def fit_b(X, labels, genes, seed=0, device=None, grid=L1_GRID, n_splits=N_FOLDS, cache=None, shard=None):
    """cache: directory of per-(fold, lambda) CV results fold<f>_l<i>.json, each written as soon as its fit finishes
    and reused if present (all pairs must share one device). shard=(i, n): fit only the row-major (fold, lambda) pairs k with k % n == i,
    then return None. Without shard, missing pairs are fitted here, lambda is chosen and the refit runs on device."""
    from e7_real import heldout_pred
    device = device or default_device()
    t0 = time.time()
    iv = np.asarray(labels)
    pairs = [(f, l) for f in range(n_splits) for l in range(len(grid))]
    mine = set(pairs if shard is None else pairs[shard[0]::shard[1]])
    path = lambda f, l: Path(cache) / f"fold{f}_l{l}.json"
    cv = {}
    for f, (tr, te) in enumerate(cv_folds(iv, seed, n_splits)):
        for l in range(len(grid)):
            if cache and path(f, l).exists():
                cv[f, l] = json.loads(path(f, l).read_text())
                assert (cv[f, l]["lambda"], cv[f, l]["seed"]) == (float(grid[l]), seed), f"stale {path(f, l)}"
        todo = [l for l in range(len(grid)) if (f, l) in mine and (f, l) not in cv]
        if not todo:
            continue
        assert not np.intersect1d(tr, te).size, "a test cell is in the training fold"
        u, d, se, sk_tr = targets(X[tr], iv[tr], genes)
        u_te, d_te, se_te, sk_te = targets(X[te], iv[te], genes)
        for l in todo:
            t1 = time.time()
            W, B, mem = train_ensemble(u, d, se, grid[l], device)
            P = heldout_pred(W, B, mem, device=device)
            e, n = heldout_error(P, u_te, d_te, se_te, mem.any(0))
            cv[f, l] = {"fold": f, "lambda_index": l, "lambda": float(grid[l]), "error": e, "n_test_conditions": n,
                        "skipped_train": sk_tr, "skipped_test": sk_te, "seed": seed, "device": device,
                        "wall_s": time.time() - t1}
            if cache:
                Path(cache).mkdir(parents=True, exist_ok=True)
                tmp = path(f, l).with_suffix(".tmp")
                tmp.write_text(json.dumps(cv[f, l]))
                tmp.rename(path(f, l))                                  # atomic: readers never see a partial file
            print(f"b cv fold {f} lambda {l} ({grid[l]:.4g}) error {e:.6g} on {device} {time.time() - t1:.0f}s",
                  flush=True)
    if shard is not None:
        return None
    devices = {r["device"] for r in cv.values()}
    assert len(devices) == 1, f"CV fits ran on {sorted(devices)}: CPU and CUDA errors differ by more than the grid's spread"
    err = np.array([[cv[f, l]["error"] for l in range(len(grid))] for f in range(n_splits)])
    n_cond = np.array([[cv[f, l]["n_test_conditions"] for l in range(len(grid))] for f in range(n_splits)])
    skipped = [{"fold": f, "train": cv[f, 0]["skipped_train"], "test": cv[f, 0]["skipped_test"]} for f in range(n_splits)]
    mean = np.nanmean(err, 0)
    best = int(np.flatnonzero(mean == np.nanmin(mean)).max())        # ties -> larger lambda (grid ascending)
    lam = float(grid[best])
    res = GRNInferencePipeline(l1_weight=lam, device=device).run(X, list(iv), list(genes))
    return res.adjacency, {"variant": "b", "lambda": lam, "lambda_index": best, "grid": [float(g) for g in grid],
                           "cv_error": err.tolist(), "cv_error_mean": mean.tolist(), "n_test_conditions": n_cond.tolist(),
                           "cv_device": [[cv[f, l]["device"] for l in range(len(grid))] for f in range(n_splits)],
                           "skipped_conditions": skipped, "seed": seed, "device": device,
                           "n_folds": n_splits, "wall_s": time.time() - t0}
