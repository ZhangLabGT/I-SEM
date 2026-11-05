#!/usr/bin/env python3
"""Directness test (simulations only): does a method score direct edges above indirect (path-mediated) pairs?

In A -> B -> C, knocking down A shifts C (total effect) with no direct A -> C edge. A total-effect score (mean
difference) ranks A -> C high; a direct-effect estimator should not. Truth: direct edges D = e9.edges(W) (the
simulator's own threshold; W = G^T, W[i, j] = effect of j on i). Indirect pairs: not in D but j reaches i through D.
Unconnected: no path in D. Indirect effect size: |(I - W)^-1 - I - W| (the total effect minus the direct one).
W's sub-threshold entries are floating-point round-off (max ~1e-16, at exactly the reachable pairs), so paths are
taken over D, never over W != 0 (scripts/experiments/e2_planted.py's evaluate uses W != 0 and would count them).

Metrics over the ordered off-diagonal pairs, |S| as the method outputs it:
- lift_indirect: AUPRC(indirect vs unconnected, among non-direct pairs) / share of indirect pairs (E2's
  auprc_total_only, normalized). 1 = indifferent to indirectness (ideal for a direct estimator); > 1 = scores like a
  total-effect method.
- auroc_direct_vs_indirect: AUROC of direct (1) vs indirect (0) pairs. 1 = every direct edge above every indirect pair.
- auroc_direct_vs_strong: the same against only the strongest indirect pairs (largest indirect effect, as many as there
  are direct edges): the pairs a total-effect score confuses most.
Per setting: mean over reps with a 95% t-CI, and the paired difference I-SEM (c) − method. Path methods (inspre,
ADAPRE): median over the lambda path, plus inspre's own lambda rule.

  bash scripts/experiments/run_env.sh python -u scripts/config_tests/directness.py
Outputs: scripts/results/config_tests/directness_sims.csv and directness.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as sc  # noqa: E402  (sets the repo paths)
from direction import SCRATCH, path_label  # noqa: E402
from run_sims import main_datasets  # noqa: E402

METRICS = ["lift_indirect", "auroc_direct_vs_indirect", "auroc_direct_vs_strong"]


def classes(W, D):
    """Direct, indirect, unconnected masks over off-diagonal pairs, and the indirect effect size."""
    n = len(W)
    off = ~np.eye(n, dtype=bool)
    reach = np.isfinite(shortest_path(D.T.astype(np.int8), unweighted=True)).T & off   # reach[i, j]: j ~> i
    eff = np.abs(np.linalg.inv(np.eye(n) - W) - np.eye(n) - W)
    return D & off, reach & ~D, off & ~reach, eff


def metrics(S, direct, indirect, unconn, eff):
    A = np.abs(np.asarray(S, float))
    out = dict.fromkeys(METRICS, np.nan)
    rest = indirect | unconn
    y = indirect[rest]
    if 0 < y.sum() < y.size:
        out["lift_indirect"] = sc.auc_ap(y.astype(np.int8), A[rest])[1] / y.mean()
    if direct.any() and indirect.any():
        lab = np.r_[np.ones(direct.sum()), np.zeros(indirect.sum())].astype(np.int8)
        out["auroc_direct_vs_indirect"] = sc.auc_ap(lab, np.r_[A[direct], A[indirect]])[0]
        k = min(int(direct.sum()), int(indirect.sum()))
        strong = np.argsort(-eff[indirect], kind="mergesort")[:k]
        lab = np.r_[np.ones(direct.sum()), np.zeros(k)].astype(np.int8)
        out["auroc_direct_vs_strong"] = sc.auc_ap(lab, np.r_[A[direct], A[indirect][strong]])[0]
    return out


def sim_rows(d):
    m, W, E, _, _ = sc.sim_entries(d)
    direct, indirect, unconn, eff = classes(W, sc.e9.edges(W, m))
    k = min(int(direct.sum()), int(indirect.sum()))
    base = {"setting": m["setting"], "rep": m["rep"], "n_direct": int(direct.sum()), "n_indirect": int(indirect.sum()),
            "n_unconnected": int(unconn.sum()),
            "strong_min_effect": float(np.sort(eff[indirect])[::-1][k - 1]) if k else np.nan}
    return [{**base, "method": e["method"], "lambda_index": e["l"], "c2": path_label(e),
             **metrics(e["S"], direct, indirect, unconn, eff)}
            for e in E if e["S"] is not None and e["method"] not in SCRATCH]


def per_rep(df):
    """One value per (setting, rep, method, statistic); path methods: path median and inspre's own rule."""
    out = []
    for (s, r, meth), g in df.groupby(["setting", "rep", "method"], sort=False):
        base = {"setting": s, "rep": r, "method": meth}
        if g.lambda_index.isna().all():
            out.append({**base, "stat": "", **g.iloc[0][METRICS]})
            continue
        out.append({**base, "stat": "path median", **g[METRICS].median()})
        for _, c in g[g.c2.notna()].iterrows():
            out.append({**base, "stat": c.c2, **c[METRICS]})
    return pd.DataFrame(out)


def table(df, raw):
    ci = lambda x: "" if x.notna().sum() < 2 else (lambda lo, hi: f"{x.mean():.3f} [{lo:.3f}, {hi:.3f}]")(*sc.e9.tci(x.dropna()))
    dci = lambda x: "" if x.notna().sum() < 2 else (lambda lo, hi: f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")(*sc.e9.tci(x.dropna()))
    L = ["### Simulations", "",
         "Mean over the 10 reps per setting, 95% t-CI; Δ = I-SEM (c) − method, paired by rep. lift_indirect: 1 = ideal "
         "for a direct estimator, > 1 = total-effect-like (lower is better, so Δ < 0 favours I-SEM). AUROCs: higher is "
         "better (Δ > 0 favours I-SEM).", "",
         "| setting | direct / indirect / unconnected pairs (mean) | strongest-indirect min effect (mean) |", "|---|---|---|"]
    for s, g in raw.drop_duplicates(["setting", "rep"]).groupby("setting", sort=False):
        L.append(f"| {s} | {g.n_direct.mean():.0f} / {g.n_indirect.mean():.0f} / {g.n_unconnected.mean():.0f} | "
                 f"{g.strong_min_effect.mean():.3f} |")
    L += ["", "| setting | method | statistic | lift_indirect | Δ | AUROC direct vs indirect | Δ | AUROC direct vs strongest indirect | Δ |",
          "|---|---|---|---|---|---|---|---|---|"]
    isem = df[df.method == sc.ISEM["c"]].set_index(["setting", "rep"])[METRICS]
    for (s, meth, st), g in df.groupby(["setting", "method", "stat"], sort=False):
        g = g.set_index(["setting", "rep"])
        row = [s, meth, st]
        for mt in METRICS:
            row += [ci(g[mt]), "" if meth == sc.ISEM["c"] else dci(isem[mt].reindex(g.index) - g[mt])]
        L.append("| " + " | ".join(row) + " |")
    return L + [""]


def main():
    raw = pd.DataFrame([r for d in main_datasets() for r in sim_rows(d)])
    raw.to_csv(sc.RES / "directness_sims.csv", index=False)
    L = ["# Directness test (generated by directness.py; do not edit)", "", __doc__.split("\n\n")[1], ""]
    (sc.RES / "directness.md").write_text("\n".join(L + table(per_rep(raw), raw)) + "\n")
    print("wrote", sc.RES / "directness.md", flush=True)


if __name__ == "__main__":
    main()
