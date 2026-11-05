#!/usr/bin/env python3
"""The first-replicate directness macros recomputed on seeds 12-21.

    python3 seeds_directness.py            prints the macros (\\renewcommand) and a check table
    python3 seeds_directness.py --test     self-check of the formatting helpers

Same statistics as the number ledger (SIM.DIR.rcc.bestlambda / repbest / lambdawins / isem.ind and
SIM.DIR.demo2.*lift*), read from scripts/fresh_seeds/results/directness.csv.gz (read only). That file has
I-SEM, inspre, ADAPRE and mean difference only: the four general causal-discovery rows stay on the first replicates.
inspre's lowest-CV-error row is `selected == True` (the old file's c2 == "which.min eps_hat_G").
"""
import math, sys
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
from scipy.stats import t as tdist

SRC = Path(__file__).resolve().parents[2] / "scripts/fresh_seeds/results/directness.csv.gz"
ISEM, INS, ADA, MD, RCC = "I-SEM (c) defaults", "inspre (official, DAG)", "ADAPRE", "Mean difference", "rand_cycles_confound_7"
SV, LIFT, MINUS = "auroc_direct_vs_strong", "lift_indirect", r"\ensuremath{-}"


def tci(v):
    v = np.asarray([x for x in v if x == x], float)
    h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def rnd(x, d):
    s = f"{Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP):f}"
    return s.lstrip("-") if float(s) == 0 else s


def num(x, d=3, sign=False):
    s = rnd(x, d)
    neg, s = (x < 0 if sign else s.startswith("-")), s.lstrip("-")
    i, _, f = s.partition(".")
    s = f"{int(i):,}".replace(",", "{,}") + ("." + f if f else "")
    return (MINUS if neg else "+" if sign and x > 0 else "") + s


ci3s = lambda v: f"{num(v[0], 3, True)} [{num(v[1], 3, True)}, {num(v[2], 3, True)}]"  # noqa: E731


def per_rep(d, setting, method, col, selected=None):
    """One value per rep: I-SEM / mean difference their row; path methods the median over the path, or the selected row."""
    x = d[(d.setting == setting) & (d.method == method)]
    if selected is not None:
        x = x[x.selected == selected] if selected else x
    return x.groupby("rep")[col].median().to_dict()


def paired(a, b):
    return tci([a[k] - b[k] for k in sorted(set(a) & set(b))])


def path(d, setting, method):
    return d[(d.setting == setting) & (d.method == method) & d.lambda_index.notna()]


def main():
    d = pd.read_csv(SRC)
    d = d[d.sim.isin(["inspre", "adapre"])]
    assert sorted(d.rep.unique()) == list(range(12, 22)), d.rep.unique()
    isem = lambda s=RCC, col=SV: per_rep(d, s, ISEM, col)  # noqa: E731
    out, check = {}, []
    for nm, m in [("Inspre", INS), ("Adapre", ADA)]:
        p = path(d, RCC, m)
        pm = p.groupby("lambda_index")[SV].mean()
        li = pm.idxmax()
        best = paired(isem(), p[p.lambda_index == li].set_index("rep")[SV].to_dict())
        out[f"numRccDelta{nm}BestDirStrong"] = ci3s(best)
        rep_best = p.groupby("rep")[SV].max().to_dict()
        out[f"numRccDelta{nm}OracleRepDirStrong"] = ci3s(paired(isem(), rep_best))
        cis = [paired(isem(), p[p.lambda_index == l].set_index("rep")[SV].to_dict()) for l in sorted(pm.index)]
        out[f"numRccLambdaWins{nm}"] = str(sum(c[1] > 0 for c in cis))
        check.append(f"{nm}: best position {li:.0f} (mean {pm[li]:.3f}); positions {len(cis)}; "
                     f"wins {sum(c[1] > 0 for c in cis)}, losses {sum(c[2] < 0 for c in cis)}")
    out["numRccLambdaPositions"] = str(path(d, RCC, INS).lambda_index.nunique())
    out["numRccIsemDirInd"] = num(tci(isem(RCC, "auroc_direct_vs_indirect").values())[0])
    out["numDemoTwoIsemLift"] = num(tci(isem("demo_set2", LIFT).values())[0])
    for nm, m in [("Inspre", INS), ("Adapre", ADA)]:
        out[f"numDemoTwo{nm}Lift"] = num(tci(per_rep(d, "demo_set2", m, LIFT, False).values())[0])
        out[f"numDemoTwoDelta{nm}Lift"] = ci3s(paired(isem("demo_set2", LIFT), per_rep(d, "demo_set2", m, LIFT, False)))
    rows = [("inspre path median", INS, False), ("inspre lowest-CV-error", INS, True), ("ADAPRE path median", ADA, False),
            ("mean difference", MD, None)]
    worse = []
    for lab, m, s in rows:
        c = paired(isem("demo_set2", LIFT), per_rep(d, "demo_set2", m, LIFT, s))
        check.append(f"demo_set2 lift, I-SEM minus {lab}: {ci3s(c)}")
        worse.append(c[1] > 0)
    out["numDemoTwoLiftWorseSeeds"] = str(sum(worse))  # new: of the four rows these seeds have
    out["numDemoTwoLiftRowsSeeds"] = str(len(rows))
    return out, check


def test():
    assert num(-0.0004, 3, True) == MINUS + "0.000" and num(0.0105, 3) == "0.011" and num(1234.5, 0) == "1{,}235"
    assert tci([1, 1, 1])[0] == 1 and paired({1: 2, 2: 3}, {1: 1, 2: 2})[0] == 1
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        test()
    else:
        out, check = main()
        for k, v in out.items():
            print(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}")
        print("% " + "\n% ".join(check))
