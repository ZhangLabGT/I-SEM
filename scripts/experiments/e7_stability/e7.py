#!/usr/bin/env python3
"""E7: I-SEM root-cause sweep (stability cap, dropout x half-splits, realistic noise). Diagnostic only.

Every write stays under scripts/results/paper/e7_stability/. src/ is used unchanged: hyperparameters are
overridden with dataclasses.replace on src.pipeline._sem_config, and models are trained with the same
train_batched call GRNInferencePipeline makes (120 models in one chunk), so the control config
(tau 0.9, dropout 0.2, half-split, rule a) is the paper model.

Configs  tau in {0.9, 2.5, off (stability_weight 0)} x w_dropout in {0.2, 0.0} x design in
         {half: 60 complementary pairs = 120 models (paper), full: 10 models on ALL knockdowns, seeds 42+2k}.
Scoring  rule a (paper): mean_m |W_ij B_j|, B_j = 1 for knockdowns model m did not see;
         rule b (seen-only): mean of |W_ij B_j| over the models whose half contains j's knockdown.
         For design full every model sees every knockdown, so a == b.

  bash scripts/experiments/run_env.sh python -u scripts/experiments/e7_stability/e7.py stage1
  bash scripts/experiments/run_env.sh python -u scripts/experiments/e7_stability/e7.py summary1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments")]

OUT = ROOT / "scripts/results/paper/e7_stability"
E2_SCORES = OUT / "e2_scores"
TAUS, DROPOUTS, DESIGNS = [0.9, 2.5, "off"], [0.2, 0.0], ["half", "full"]
CONTROL = "tau0.9_do0.2_half"
N_FULL = 10


def cfg_name(tau, dropout, design):
    return f"tau{tau}_do{dropout}_{design}"


def parse_cfg(name):
    t, d, design = name.split("_")
    tau = t[3:]
    return (tau if tau == "off" else float(tau)), float(d[2:]), design


def all_configs():
    return [cfg_name(t, d, g) for t in TAUS for d in DROPOUTS for g in DESIGNS]


def fit(expr, interv, genes, name, device="cuda:0"):
    """Train one ensemble for config `name`. Returns (S_a, S_b, info)."""
    from src.models.interventional_nb_sem import build_intervention_targets
    u, delta, _, _, _, se = build_intervention_targets(expr, list(interv), list(genes))
    W, B, membership, info = train(u, delta, se, name, device)
    S_a, S_b = scores(W, B, membership)
    assert np.isfinite(S_a).all() and np.isfinite(S_b).all(), name
    return S_a, S_b, info


def train(u, delta, se, name, device="cuda:0"):
    """train_batched for config `name` on prepared targets. Returns (W, B, membership, info)."""
    import torch
    from src.models.interventional_nb_sem import train_batched
    from src.pipeline import _complementary_pairs, _sem_config
    tau, dropout, design = parse_cfg(name)
    cfg = _sem_config(250, 50, 0.005, dropout)      # GRNInferencePipeline defaults
    cfg = dataclasses.replace(cfg, stability_weight=0.0) if tau == "off" else dataclasses.replace(cfg, tau=tau)
    if design == "half":
        jobs = _complementary_pairs(len(u), 60)
    else:
        jobs = [(np.arange(len(u)), 42 + 2 * k) for k in range(N_FULL)]
    torch.cuda.synchronize()
    t0 = time.monotonic()
    _, loss, W, B, diag = train_batched(u, delta, se, folds=[j[0] for j in jobs], seeds=[j[1] for j in jobs],
                                        config=cfg, device=device, return_params=True, return_diagnostics=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    membership = np.stack([(u[j[0]] > .5).any(0) for j in jobs])
    info = {"config": name, "tau": tau, "w_dropout": dropout, "design": design, "n_models": len(jobs),
            "stability_weight": cfg.stability_weight, "cfg_tau": cfg.tau, "wall_s": elapsed,
            "final_loss_median": float(np.median(loss)),
            **{f"{k}_{s}": float(f(diag[k])) for k in ("sigma_abs", "norm2_W", "rho_W", "sigma_min_IminusW")
               for s, f in (("median", np.median), ("min", np.min), ("max", np.max))},
            # failing-pivot records (no-pivot LU pivot < PIVOT_MIN; key names kept for the stored JSONs); records
            # without "pivoted" are pre-option-A runs, where every failing step was re-solved with pivoted LU
            "n_fallbacks": len(diag["fallbacks"]),
            "n_models_with_fallback": len({f["model"] for f in diag["fallbacks"]}),
            "n_models_pivoted_lu": len({f["model"] for f in diag["fallbacks"] if f.get("pivoted", True)}),
            "min_pivot": float(np.min(diag["min_pivot"])),
            "n_models_nonpositive_pivot": int(np.sum(np.asarray(diag["min_pivot"]) <= 0)),
            "per_model": {k: v for k, v in diag.items() if k != "fallbacks"}}
    return W, B, membership, info


def scores(W, B, membership):
    """Rule a: mean_m |W B| with out-of-half B = 1 (asserted); rule b: average over in-half models only."""
    from src.pipeline import _mean_ensemble
    assert np.all(B[~membership] == 1.0), "out-of-half B drifted"
    counts = membership.sum(0)
    assert np.all(counts > 0), "a knockdown is in no model's half"
    S_a = _mean_ensemble(np.abs(W * B[:, None, :]))
    S_b = np.zeros(W.shape[1:])
    for m in range(len(W)):
        S_b += np.abs(W[m].astype(np.float64) * B[m][None, :]) * membership[m][None, :]
    S_b /= counts[None, :]
    np.fill_diagonal(S_b, 0)
    return S_a, S_b


def save(path, S_a, S_b, info):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, S_a=S_a.astype(np.float32), S_b=S_b.astype(np.float32))
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(info, default=float))


# ---------------------------------------------------------------- Stage 1: E2 default setting
def stage1(configs, reps):
    import e2_planted as E
    p, key = E.DEFAULT, E.key(E.DEFAULT)
    for rep in reps:
        expr, interv, genes, W = E.simulate(p, rep)
        assert np.array_equal(W, np.load(E.CACHE / key / f"rep{rep}" / "W.npy"))
        for name in configs:
            f = E2_SCORES / name / f"rep{rep}.npz"
            if f.exists():
                continue
            S_a, S_b, info = fit(expr, interv, genes, name)
            save(f, S_a, S_b, info)
            ev = E.evaluate(S_a, W)
            print(f"rep{rep} {name}: a {ev['auprc_direct']:.3f}/{ev['auprc_total_only']:.3f} "
                  f"b {E.evaluate(S_b, W)['auprc_direct']:.3f} sig|W| {info['sigma_abs_median']:.2f} "
                  f"rho {info['rho_W_median']:.2f} fb {info['n_fallbacks']} {info['wall_s']:.0f}s", flush=True)


# ---------------------------------------------------------------- Stage 2: every E2 sweep setting
# The control on these settings is the paper's cached isem.npy (Stage 1 showed the re-run is bit-identical).
def stage2(configs, reps, keys=None):
    import e2_planted as E
    for rep in reps:
        for sweep, level, p in E.settings():
            k = E.key(p)
            if p == E.DEFAULT or (keys and k not in keys):
                continue
            todo = [c for c in configs if not (OUT / "e2_sweep" / k / c / f"rep{rep}.npz").exists()]
            if not todo:
                continue
            expr, interv, genes, W = E.simulate(p, rep)
            assert np.array_equal(W, np.load(E.CACHE / k / f"rep{rep}" / "W.npy"))
            for c in todo:
                S_a, S_b, info = fit(expr, interv, genes, c)
                save(OUT / "e2_sweep" / k / c / f"rep{rep}.npz", S_a, S_b, info)
                print(f"{k} rep{rep} {c}: a {E.evaluate(S_a, W)} sig|W| {info['sigma_abs_median']:.2f} "
                      f"rho {info['rho_W_median']:.2f} fb {info['n_fallbacks']} piv {info['min_pivot']:.3g} "
                      f"{info['wall_s']:.0f}s", flush=True)


def summary2(configs, reps):
    import pandas as pd
    import e2_planted as E
    ref = pd.read_csv(ROOT / "scripts/results/paper/e2_planted.csv")
    rows = []
    for sweep, level, p in E.settings():
        k = E.key(p)
        for c in configs:
            for rep in reps:
                f = (E2_SCORES / c if p == E.DEFAULT else OUT / "e2_sweep" / k / c) / f"rep{rep}.npz"
                if not f.exists():
                    continue
                info = json.loads(f.with_suffix(".json").read_text())
                W = np.load(E.CACHE / k / f"rep{rep}" / "W.npy")
                for rule in (["a", "b"] if info["design"] == "half" else ["a"]):
                    rows.append({"sweep": sweep, "level": level, "rep": rep, "method": f"{c}:{rule}",
                                 **E.evaluate(np.load(f)[f"S_{rule}"], W),
                                 **{x: info[x] for x in ("sigma_abs_median", "rho_W_median", "rho_W_max", "norm2_W_median",
                                                         "n_fallbacks", "n_models_with_fallback", "min_pivot",
                                                         "n_models_nonpositive_pivot", "wall_s")}})
    mine = pd.DataFrame(rows)
    ref = ref[ref.method.isin(["isem", "plugin", "inspre", "mean_diff"]) & ref.rep.isin(reps)]
    df = pd.concat([ref, mine], ignore_index=True)
    df.to_csv(OUT / "stage2_e2_sweeps.csv", index=False)
    methods = ["isem"] + [m for m in mine.method.unique()] + ["plugin", "inspre", "mean_diff"]
    lines = ["# Stage 2: every E2 sweep setting (one factor at a time around the default)", "",
             "isem = paper model (cached e2_cache scores = control config). Mean [95% t-CI over reps].", ""]
    for metric, lab in [("auprc_direct", "direct AUPRC"), ("auprc_total_only", "leakage (auprc_total_only)"),
                        ("auroc_direct", "direct AUROC")]:
        lines += [f"## {lab}", "", "| sweep | level | " + " | ".join(methods) + " |",
                  "|---|---|" + "---|" * len(methods)]
        for (sweep, level), x in df.groupby(["sweep", "level"], sort=False):
            cells = [fmt(x[x.method == m][metric]) if (x.method == m).any() else "–" for m in methods]
            lines.append(f"| {sweep} | {level} | " + " | ".join(cells) + " |")
        lines.append("")
    if not mine.empty:
        dg = mine.groupby(["sweep", "level", "method"], sort=False).agg(
            sigma_abs=("sigma_abs_median", "median"), rho_med=("rho_W_median", "median"), rho_max=("rho_W_max", "max"),
            fallbacks=("n_fallbacks", "sum"), models_fb=("n_models_with_fallback", "sum"),
            min_pivot=("min_pivot", "min"), nonpos_pivot_models=("n_models_nonpositive_pivot", "sum"))
        lines += ["## Numerical diagnostics (per setting x config, over reps)", "", dg.round(4).to_markdown(), ""]
    (OUT / "stage2_e2_sweeps.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def tci(x):
    """Mean and 95% t-interval over replicates."""
    from scipy.stats import t
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = x.mean()
    h = t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return m, m - h, m + h


def fmt(x):
    m, lo, hi = tci(x)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def summary1(reps):
    import pandas as pd
    import e2_planted as E
    key = E.key(E.DEFAULT)
    rows = []
    for name in all_configs():
        for rep in reps:
            f = E2_SCORES / name / f"rep{rep}.npz"
            if not f.exists():
                continue
            W = np.load(E.CACHE / key / f"rep{rep}" / "W.npy")
            z, info = np.load(f), json.loads(f.with_suffix(".json").read_text())
            base = {k: v for k, v in info.items() if k != "per_model"}
            for rule in (["a", "b"] if info["design"] == "half" else ["a"]):
                rows.append({**base, "rep": rep, "rule": rule, **E.evaluate(z[f"S_{rule}"], W)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stage1_e2.csv", index=False)

    ref = pd.read_csv(ROOT / "scripts/results/paper/e2_planted.csv")
    ref = ref[(ref.sweep == "rho") & (ref.level == 0.8)]
    lines = ["# Stage 1: E2 default setting (rho 0.8, noise 0.3, cycle_frac 0.5, 100 cells)", "",
             f"Replicates per config: {sorted(df.rep.unique().tolist())}. Mean [95% t-CI over reps].", "",
             "## Reference methods (scripts/results/paper/e2_planted.csv, same reps)", "",
             "| method | direct AUPRC | direct AUROC | leakage (auprc_total_only) |", "|---|---|---|---|"]
    for m in ["isem", "plugin", "inspre", "mean_diff", "notears"]:
        r = ref[(ref.method == m) & ref.rep.isin(reps)]
        lines.append(f"| {m} | {fmt(r.auprc_direct)} | {fmt(r.auroc_direct)} | {fmt(r.auprc_total_only)} |")

    # control check vs the paper's cached isem scores
    c = df[(df.config == CONTROL) & (df.rule == "a")].set_index("rep")
    r = ref[ref.method == "isem"].set_index("rep").loc[c.index]
    d_auprc, d_leak = c.auprc_direct.mean() - 0.584, c.auprc_total_only.mean() - 0.508
    lines += ["", "## Control check", "",
              f"Control ({CONTROL}, rule a): direct AUPRC {c.auprc_direct.mean():.3f} (target 0.584, diff {d_auprc:+.3f}), "
              f"leakage {c.auprc_total_only.mean():.3f} (target 0.508, diff {d_leak:+.3f}); "
              f"per-rep max |diff| vs e2_planted.csv isem: AUPRC {np.abs(c.auprc_direct - r.auprc_direct).max():.4f}, "
              f"leakage {np.abs(c.auprc_total_only - r.auprc_total_only).max():.4f}. "
              f"PASS: {abs(d_auprc) <= .02 and abs(d_leak) <= .02}"]

    lines += ["", "## All configs", "",
              "| config | rule | direct AUPRC | direct AUROC | leakage | med ‖abs W‖₂ | med ρ(W) | med ‖W‖₂ | "
              "failing-pivot records (models) | min pivot | wall s/ensemble |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    g = df.groupby(["config", "rule"], sort=False)
    for (name, rule), x in g:
        lines.append(f"| {name} | {rule} | {fmt(x.auprc_direct)} | {fmt(x.auroc_direct)} | {fmt(x.auprc_total_only)} | "
                     f"{x.sigma_abs_median.median():.2f} | {x.rho_W_median.median():.2f} | {x.norm2_W_median.median():.2f} | "
                     f"{x.n_fallbacks.sum()} ({x.n_models_with_fallback.sum()}) | {x.min_pivot.min():.3g} | "
                     f"{x.wall_s.mean():.1f} |")

    # root cause 2: 2x2 per tau, rule a, with paired effects
    lines += ["", "## Root cause 2: {half-split, full} x {dropout 0.2, 0}, direct AUPRC, rule a", "",
              "Stand-in prediction (linear, noisy): half only 0.815, dropout only 0.773, both 0.448.", ""]
    piv = df[df.rule == "a"].pivot_table(index="rep", columns="config", values="auprc_direct")
    for tau in TAUS:
        k = {(g_, d): cfg_name(tau, d, g_) for g_ in DESIGNS for d in DROPOUTS}
        if not all(v in piv for v in k.values()):
            continue
        both, half_only, drop_only, neither = (piv[k[("half", .2)]], piv[k[("half", 0.)]],
                                               piv[k[("full", .2)]], piv[k[("full", 0.)]])
        lines += [f"### tau {tau}", "", "| design | dropout 0.2 | dropout 0 |", "|---|---|---|",
                  f"| half-split | {fmt(both)} (both) | {fmt(half_only)} (half only) |",
                  f"| full | {fmt(drop_only)} (dropout only) | {fmt(neither)} (neither) |", "",
                  f"- dropout effect within half-split (0.2 − 0): {fmt(both - half_only)}",
                  f"- dropout effect within full: {fmt(drop_only - neither)}",
                  f"- half-split effect at dropout 0 (half − full): {fmt(half_only - neither)}",
                  f"- half-split effect at dropout 0.2: {fmt(both - drop_only)}",
                  f"- interaction (both − half only − dropout only + neither): "
                  f"{fmt(both - half_only - drop_only + neither)}", ""]

    # gate 1
    agg = g.agg(auprc=("auprc_direct", "mean"), leak=("auprc_total_only", "mean")).reset_index()
    best = agg.sort_values("auprc", ascending=False).iloc[0]
    ok = agg[(agg.auprc >= .75) & (agg.leak <= .35)]
    bx = df[(df.config == best.config) & (df.rule == best.rule)]
    lines += ["## GATE 1", "",
              f"Best by direct AUPRC: {best.config} rule {best.rule}: AUPRC {fmt(bx.auprc_direct)}, "
              f"leakage {fmt(bx.auprc_total_only)} (inspre 0.590, plug-in 0.778).",
              f"Configs meeting AUPRC ≥ 0.75 and leakage ≤ 0.35: "
              f"{', '.join(f'{a} ({b})' for a, b in zip(ok.config, ok.rule)) or 'none'}.",
              f"GATE 1: {'GO' if len(ok) else 'NO-GO'}"]
    (OUT / "stage1_e2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stage1", "summary1", "stage2", "summary2"])
    ap.add_argument("--keys", nargs="+", help="stage2: only these E2 setting keys")
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--reps", type=int, nargs="+", default=None)
    a = ap.parse_args()
    if a.cmd == "stage1":
        stage1(a.configs or [CONTROL] + [c for c in all_configs() if c != CONTROL], a.reps or range(10))
    elif a.cmd == "summary1":
        summary1(a.reps or list(range(10)))
    elif a.cmd == "stage2":
        stage2(a.configs, a.reps or range(10), a.keys)
    else:
        summary2(a.configs, a.reps or list(range(10)))


if __name__ == "__main__":
    main()
