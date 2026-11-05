#!/usr/bin/env python3
"""Direction test: does a method's score put the arrow the right way round?

For every one-way edge j -> i of a directed truth T (T[i, j] = 1, T[j, i] = 0; W orientation, W[i, j] = effect of j
on i), a method that scores the pair in either direction is right if |S[i, j]| > |S[j, i]| (tie = 0.5). Chance = 0.5.
Two-way (cyclic) pairs have no single right direction and are skipped.

- Simulations (the 30 main datasets, truth = e9.edges(W)): orientation accuracy per dataset, mean over reps with a
  95% t-CI; plus AUROC of S, of S^T and of max(S, S^T) against the directed truth (S - S^T and S - sym gaps show how
  much of the score is directional).

Simulations only.

Methods are scored exactly as score.py loads them (sim_entries); I-SEM (a)/(b) are not reported. Path methods
(inspre, ADAPRE): the median over their lambda path, plus inspre's own lambda rule.

  bash scripts/experiments/run_env.sh python -u scripts/config_tests/direction.py
Outputs: scripts/results/config_tests/direction_sims.csv and direction.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as sc  # noqa: E402  (sets the repo paths)
from run_sims import main_datasets  # noqa: E402

SCRATCH = (sc.ISEM["a"], sc.ISEM["b"])


def orientation(S, T):
    """Per one-way truth pair scored by the method: 1 right, 0 wrong, 0.5 tie; the pair's regulator index; the
    number of one-way pairs."""
    A = np.abs(np.asarray(S, float))
    off = ~np.eye(len(A), dtype=bool)
    i, j = np.nonzero((T > 0) & (T.T == 0) & off)
    f, r = A[i, j], A[j, i]
    cov = (f > 0) | (r > 0)
    return np.where(f > r, 1.0, np.where(f < r, 0.0, 0.5))[cov], j[cov], len(i)


def path_label(e):
    """inspre's own lambda-selection rule(s) (its C2 label); None for every other lambda and method."""
    return e["sel"]["C2"] if "C2" in e["sel"] and e["method"].startswith("inspre") else None


# ------------------------------------------------------------------------------------------------- sims
def sim_rows(d):
    m, W, E, _, _ = sc.sim_entries(d)
    tru = sc.e9.edges(W, m).astype(np.int8)
    off = ~np.eye(len(W), dtype=bool)
    y = tru[off]
    rows = []
    for e in E:
        if e["S"] is None or e["method"] in SCRATCH:
            continue
        A = np.abs(e["S"])
        out, _, n_one = orientation(A, tru)
        rows.append({"setting": m["setting"], "rep": m["rep"], "method": e["method"], "lambda_index": e["l"],
                     "c2": path_label(e), "n_one_way": n_one, "n_scored": len(out),
                     "orient_acc": out.mean() if len(out) else np.nan,
                     "auroc": sc.auc_ap(y, A[off])[0], "auroc_T": sc.auc_ap(y, A.T[off])[0],
                     "auroc_sym": sc.auc_ap(y, np.maximum(A, A.T)[off])[0]})
    return rows


def per_rep(df):
    """One value per (setting, rep, method, statistic): path methods give the path median and inspre's own rule."""
    out = []
    for (s, r, meth), g in df.groupby(["setting", "rep", "method"], sort=False):
        base = {"setting": s, "rep": r, "method": meth}
        if g.lambda_index.isna().all():
            out.append({**base, "stat": "", **g.iloc[0][["orient_acc", "auroc", "auroc_T", "auroc_sym", "n_scored", "n_one_way"]]})
            continue
        h = g.dropna(subset=["orient_acc"])
        num = ["orient_acc", "auroc", "auroc_T", "auroc_sym", "n_scored", "n_one_way"]
        out.append({**base, "stat": "path median", **h[num].median()})
        c2 = g[g.c2.notna()]
        if len(c2):
            out.append({**base, "stat": c2.c2.iloc[0], **c2.iloc[0][num]})
    return pd.DataFrame(out)


def sim_table(df):
    L = ["### Simulations", "",
         "Mean over 10 reps per setting, 95% t-CI. Orientation = share of one-way true edges (scored in either direction) "
         "put the right way round; chance 0.5. ΔAUROC(S − S^T) > 0: the directed score beats its own transpose; "
         "ΔAUROC(S − sym) ≥ 0: symmetrizing loses information.", "",
         "| setting | method | statistic | scored / one-way edges | orientation [CI] | AUROC S | ΔAUROC S − S^T [CI] | ΔAUROC S − sym [CI] |",
         "|---|---|---|---|---|---|---|---|"]
    ci = lambda x: (lambda lo, hi: f"{x.mean():.3f} [{lo:.3f}, {hi:.3f}]")(*sc.e9.tci(x)) if x.notna().sum() > 1 else ""
    dci = lambda x: (lambda lo, hi: f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")(*sc.e9.tci(x))
    for (s, meth, st), g in df.groupby(["setting", "method", "stat"], sort=False):
        L.append(f"| {s} | {meth} | {st} | {g.n_scored.mean():.0f} / {g.n_one_way.mean():.0f} | {ci(g.orient_acc.dropna())} | "
                 f"{g.auroc.mean():.3f} | {dci(g.auroc - g.auroc_T)} | {dci(g.auroc - g.auroc_sym)} |")
    return L + [""]


def main():
    sims = pd.DataFrame([r for d in main_datasets() for r in sim_rows(d)])
    sims.to_csv(sc.RES / "direction_sims.csv", index=False)
    L = ["# Direction test (generated by direction.py; do not edit)", "", __doc__.split("\n\n")[1], ""]
    (sc.RES / "direction.md").write_text("\n".join(L + sim_table(per_rep(sims))) + "\n")
    print("wrote", sc.RES / "direction.md", flush=True)

if __name__ == "__main__":
    main()
