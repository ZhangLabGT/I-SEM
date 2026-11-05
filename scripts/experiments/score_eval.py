#!/usr/bin/env python3
"""E1/E3 scoring with the paper evaluator, scripts/experiments/paper_eval.py (imported via paper_eval).

Protocol (results_eval.py): every score is |A| with the diagonal zeroed, directed, not symmetrized;
all ordered pairs i != j; a pair is positive if its unordered pair is in the reference (STRING v11.5
score >= 400, full CORUM 5.3), built exactly as results_eval.run builds it; degree-preserving nulls
= paper_eval.rewire(G, N_REWIRE=5, seed=7); point metrics = paper_eval.metrics().

Uncertainty: paper_eval.boot()'s node bootstrap (rng.choice(n, n, replace=True), pairs of the same
resampled gene dropped), with n_boot >= 500 draws (§8), the same draws for every method (paired).
boot() returns only AUROC/AUPRC, so each draw here evaluates all six metrics; one sort per draw and
method serves the real labels and the nulls (auc_ap_multi == paper_eval.auc_ap, see check()).

Metrics: auroc, auprc_lift (= auprc / positive rate), p_at_1000, edge_specific_auroc
(auroc - mean null auroc), edge_specific_auprc_ratio (auprc / mean null auprc); auprc and pos_rate
are reported too. Paired differences: I-SEM minus the best baseline per metric (point estimate),
over the same draws. "Baselines" excludes controls (plug-in, control expression) and ablation rows.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import paper_eval as pe  # noqa: E402

ROOT = pe.ROOT
METRICS = ["auroc", "auprc", "auprc_lift", "p_at_1000", "edge_specific_auroc", "edge_specific_auprc_ratio"]
CONTROLS = {"Plug-in direct |Ŵ|", "Control expression e_i·e_j"}


def auc_ap_multi(Y, s):
    """paper_eval.auc_ap for several label rows Y (k, N) sharing one score vector s."""
    o = np.argsort(-s, kind="mergesort"); s = s[o]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(Y[:, o], axis=1)[:, last].astype(float); fps = last[None, :] + 1 - tps
    P, N = tps[:, -1:], fps[:, -1:]
    ap = np.sum(np.diff(np.c_[np.zeros(len(Y)), tps / P], axis=1) * tps / (tps + fps), axis=1)
    tpr, fpr = np.c_[np.zeros(len(Y)), tps / P], np.c_[np.zeros(len(Y)), fps / N]
    return np.trapz(tpr, fpr, axis=1), ap


def references(genes):
    """{name: symmetric 0/1 int8 adjacency} exactly as results_eval.run builds them."""
    from scripts.eval_all_metrics import edges_to_adj, gene_names_to_symbols, load_corum, load_string
    _, i2s = gene_names_to_symbols(list(genes)); ss = set(i2s.values())
    out = {}
    for name, edges in {"STRING": load_string(400), "CORUM": load_corum()}.items():
        e = {k: v for k, v in edges.items() if k[0] in ss and k[1] in ss}
        A = edges_to_adj(e, list(genes), i2s); G = ((A + A.T) > 0).astype(np.int8); np.fill_diagonal(G, 0)
        out[name] = G
    return out


def _all_metrics(Y, s, prevalence_row=0):
    auc, ap = auc_ap_multi(Y, s)
    y = Y[prevalence_row]
    top = np.argsort(-s, kind="mergesort")[:1000]
    return {"auroc": auc[0], "auprc": ap[0], "auprc_lift": ap[0] / y.mean(), "p_at_1000": y[top].mean(),
            "edge_specific_auroc": auc[0] - auc[1:].mean(), "edge_specific_auprc_ratio": ap[0] / ap[1:].mean()}


def score_reference(scores, G, n_boot=500, seed=0, baselines=None, isem="I-SEM"):
    """Point estimates, node-bootstrap CIs and paired differences for one reference."""
    if n_boot < 500:
        raise ValueError("the evaluation protocol uses >= 500 node-bootstrap draws")
    n = G.shape[0]
    S = {m: pe.absd(A) for m, A in scores.items()}
    iu = pe.offdiag(n)
    nulls = pe.rewire(G, pe.N_REWIRE, seed=7)
    Yfull = np.stack([G[iu]] + [H[iu] for H in nulls])
    out = {}
    for m, A in S.items():
        base = pe.metrics(A, G, iu)                       # paper_eval point metrics, verbatim
        null = auc_ap_multi(Yfull[1:], A[iu])
        out[m] = {**{k: float(v) for k, v in base.items()},
                  "null_auroc_mean": float(null[0].mean()), "null_auprc_mean": float(null[1].mean()),
                  "edge_specific_auroc": float(base["auroc"] - null[0].mean()),
                  "edge_specific_auprc_ratio": float(base["auprc"] / null[1].mean())}
    rng = np.random.default_rng(seed)
    draws = {m: {k: [] for k in METRICS} for m in S}
    p, q = iu
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        keep = idx[p] != idx[q]; a, b = idx[p[keep]], idx[q[keep]]
        Y = np.stack([G[a, b]] + [H[a, b] for H in nulls])
        for m, A in S.items():
            for k, v in _all_metrics(Y, A[a, b]).items():
                draws[m][k].append(float(v))
    for m in S:
        for k in METRICS:
            out[m][f"{k}_ci"] = pe.ci(np.asarray(draws[m][k]))
    if isem in S:
        for m in S:                       # as results_eval's isem_minus_this, for every other row
            if m != isem:
                out[m]["isem_minus_this"] = {}
                for k in METRICS:
                    d = np.asarray(draws[isem][k]) - np.asarray(draws[m][k])
                    out[m]["isem_minus_this"][k] = {"difference": out[isem][k] - out[m][k],
                                                    "difference_ci": pe.ci(d), "p_le0": float(np.mean(d <= 0))}
        cands = [m for m in (baselines if baselines is not None else S) if m in S and m != isem and m not in CONTROLS]
        if cands:
            out[isem]["paired_vs_best_baseline"] = {}
            for k in METRICS:
                best = max(cands, key=lambda m: out[m][k])
                d = np.asarray(draws[isem][k]) - np.asarray(draws[best][k])
                out[isem]["paired_vs_best_baseline"][k] = {
                    "baseline": best, "difference": out[isem][k] - out[best][k], "difference_ci": pe.ci(d),
                    "p_le0": float(np.mean(d <= 0))}
    return out


def meta(extra=None):
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    return {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "date": datetime.now(timezone.utc).isoformat(),
            "hardware": {"cpu": cpu, "logical_cpus": os.cpu_count(), "host": platform.node()},
            "evaluator": "scripts/experiments/paper_eval.py via scripts/experiments/score_eval.py",
            "n_rewire": pe.N_REWIRE, "rewire_seed": 7, **(extra or {})}


def check():
    r = np.random.default_rng(1)
    for ties in (False, True):
        s = r.integers(0, 30, 4000).astype(float) if ties else r.random(4000)
        Y = (r.random((4, 4000)) < .1).astype(np.int8)
        auc, ap = auc_ap_multi(Y, s)
        for k in range(4):
            assert np.allclose((auc[k], ap[k]), pe.auc_ap(Y[k], s)), "auc_ap_multi != paper_eval.auc_ap"
    G = (r.random((60, 60)) < .1).astype(np.int8); G = ((G + G.T) > 0).astype(np.int8); np.fill_diagonal(G, 0)
    res = score_reference({"I-SEM": G + r.random((60, 60)) * .1, "noise": r.random((60, 60))}, G, n_boot=500)
    assert res["I-SEM"]["auroc"] > .99 and res["I-SEM"]["auroc_ci"][0] > .95
    assert res["I-SEM"]["paired_vs_best_baseline"]["auroc"]["difference_ci"][0] > 0
    print("check ok")


if __name__ == "__main__":
    check()
