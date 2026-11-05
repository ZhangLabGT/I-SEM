#!/usr/bin/env python3
"""Run I-SEM and the in-repo baselines on the real datasets and save their score matrices.

Methods: I-SEM (src/pipeline.py defaults), mean difference, and the observational baselines NOTEARS,
GOLEM-NV and ICA-LiNGAM, and the interventional IGSP and RICE (scripts/baselines/runners.py; authors' code, unmodified).
inspre and ADAPRE run from scripts/experiments (inspre_official.py, e8_adapre/) with their authors' settings.
Every matrix is saved as results/<method>_adj_<dataset>.npy with adj[i, j] = effect of gene j on gene i,
exactly as the method returns it (never thresholded or symmetrized here). Scoring is done by
scripts/experiments/paper_tables.py.

Usage:
    python -u scripts/run_all_baselines.py --methods notears --datasets hepg2 > results/notears_hepg2.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_all_metrics import load_dataset  # noqa: E402
from scripts.baselines import runners  # noqa: E402

RESULTS_ROOT = ROOT / "results"


def standardize_to_control(expression, control_mask):
    """Z-score each gene against control cells: (x - mu_ctrl) / sd_ctrl.

    This is the input regime inspre was built for (Brown et al. use Replogle's
    control-z-normalized release). Genes with zero control SD are only centered.
    """
    ctrl = expression[control_mask]
    mu = ctrl.mean(axis=0)
    sd = ctrl.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return ((expression - mu) / sd).astype(np.float32)


def run_isem(expression, interventions, gene_names, seed=0):
    """I-SEM with the pipeline defaults (directed |W.B| readout, 120 models). GPU: CUDA_VISIBLE_DEVICES."""
    import torch
    from src.pipeline import GRNInferencePipeline

    t0 = time.time()
    pipe = GRNInferencePipeline(device="cuda" if torch.cuda.is_available() else "cpu")
    adj = pipe.run(expression, interventions, gene_names).adjacency
    elapsed = time.time() - t0
    print(f"  I-SEM done in {elapsed:.1f}s", flush=True)
    return adj, elapsed


def run_mean_diff(expression, interventions, gene_names, seed=0, max_cells=None):
    """Mean difference: score[:, j] = |mean(x | KO_j) - mean(x | control)|, no model."""
    from scripts.eval_all_metrics import build_delta_baseline, gene_names_to_symbols
    _, idx_to_sym = gene_names_to_symbols(gene_names)
    t0 = time.time()
    print(f"  MeanDiff: {expression.shape[0]} cells x {len(gene_names)} genes", flush=True)
    adj = build_delta_baseline(expression, list(interventions), list(gene_names), idx_to_sym)
    elapsed = time.time() - t0
    print(f"  MeanDiff done in {elapsed:.1f}s, {int((adj > 0).sum())} non-zero entries", flush=True)
    return adj, elapsed


def observational_input(expression, interventions, seed=0, max_cells=10000):
    """Input of the observational baselines (LiNGAM, NOTEARS, GOLEM): every cell, all conditions pooled,
    z-scored against the control cells (as in the inspre authors' benchmark data), then a seeded subsample of
    max_cells cells (None = all). Returns (X, labels of the kept cells)."""
    from src.models.interventional_nb_sem import infer_control_label
    lab = np.asarray(interventions)
    X = standardize_to_control(np.asarray(expression), lab == infer_control_label(list(interventions)))
    if max_cells is not None and len(X) > max_cells:
        idx = np.sort(np.random.RandomState(seed).choice(len(X), max_cells, replace=False))
        X, lab = X[idx], lab[idx]
    return X, lab


def _observational(fit):
    def run(expression, interventions, gene_names, seed=0, max_cells=10000, mode="readme"):
        X, lab = observational_input(expression, interventions, seed, max_cells)
        print(f"  {fit.__name__}: {X.shape[0]} cells x {X.shape[1]} genes, mode {mode}", flush=True)
        t0 = time.time()
        kw = {"mode": mode, "targets": lab} if fit is not runners.lingam else {"seed": seed}
        if fit is runners.notears and mode == "cv":
            kw["n_jobs"] = int(os.environ.get("NOTEARS_CV_JOBS", "1"))
        adj, meta = fit(X, **kw)
        adj = np.asarray(adj, float)
        np.fill_diagonal(adj, 0)
        print(f"  {fit.__name__} done in {time.time() - t0:.1f}s, {int((adj != 0).sum())} edges", flush=True)
        return adj, time.time() - t0, meta
    run.__name__ = f"run_{fit.__name__}"
    return run


run_notears = _observational(runners.notears)
run_golem = _observational(runners.golem)
run_lingam = _observational(runners.lingam)


def run_igsp(expression, interventions, gene_names, seed=0):
    """IGSP on every cell, control-z-scored: controls = observational setting, one setting per knocked-down
    panel gene; cells whose target is off the panel ("excluded") are dropped."""
    from src.models.interventional_nb_sem import infer_control_label
    lab = np.asarray(interventions)
    ctrl = lab == infer_control_label(list(interventions))
    keep = ctrl | np.isin(lab, gene_names)
    X = standardize_to_control(np.asarray(expression), ctrl)[keep]
    labels = np.where(ctrl, "control", lab)[keep]
    print(f"  IGSP: {X.shape[0]} cells ({int(ctrl.sum())} control) x {X.shape[1]} genes, "
          f"{len(set(labels)) - 1} knockdowns; {int((~keep).sum())} off-panel cells dropped", flush=True)
    t0 = time.time()
    adj, meta = runners.igsp(X, list(labels), list(gene_names), seed=seed)
    print(f"  IGSP done in {time.time() - t0:.1f}s, {int((adj != 0).sum())} edges", flush=True)
    return adj, time.time() - t0, meta


def run_rice(expression, interventions, gene_names, ds):
    """RICE on the dataset's raw counts (README data table; the processed expression is not used): the
    authors' run_Replogle_2022.py with our panel as the gene set (README.md)."""
    from scripts.experiments.inspre_official import H5AD
    t0 = time.time()
    processed = RESULTS_ROOT / f"rice_processed_{ds}.npz"         # reused if present (resume after a failed fit)
    adj, meta = runners.rice(H5AD[ds], list(gene_names), processed=processed, reuse=processed.exists())
    adj = np.asarray(adj, float)                                     # diagonal already 0 (fit zeroes it)
    print(f"  RICE done in {time.time() - t0:.1f}s, {int((adj != 0).sum())} nonzero", flush=True)
    return adj, time.time() - t0, meta


METHOD_RUNNERS = {"isem": run_isem, "mean_diff": run_mean_diff, "notears": run_notears,
                  "golem": run_golem, "lingam": run_lingam, "igsp": run_igsp,
                  "rice": run_rice}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["k562", "rpe1", "hepg2", "jurkat"])
    ap.add_argument("--methods", nargs="+", choices=list(METHOD_RUNNERS), default=list(METHOD_RUNNERS))
    ap.add_argument("--mode", choices=["readme", "cv"], default="readme",
                    help="notears/golem: authors' defaults (readme) or the inspre CV protocol (cv, saved as <m>_cv)")
    args = ap.parse_args()
    RESULTS_ROOT.mkdir(exist_ok=True)
    for ds in args.datasets:
        expression, interventions, gene_names = load_dataset(ds)
        print(f"\n=== {ds}: {expression.shape[0]} cells x {len(gene_names)} genes", flush=True)
        for m in args.methods:
            kw = {"mode": args.mode} if m in ("notears", "golem") else {"ds": ds} if m == "rice" else {}
            adj, elapsed, *meta = METHOD_RUNNERS[m](expression, list(interventions), list(gene_names), **kw)
            name = f"{m}_cv" if kw.get("mode") == "cv" else m
            np.save(RESULTS_ROOT / f"{name}_adj_{ds}.npy", adj)
            (RESULTS_ROOT / f"{name}_run_{ds}.json").write_text(json.dumps(
                {"method": name, "dataset": ds, "elapsed_s": round(elapsed, 1), "n_nonzero": int((adj != 0).sum()),
                 "host": platform.node(), **(meta[0] if meta else {})}, indent=1, default=float))


if __name__ == "__main__":
    main()
