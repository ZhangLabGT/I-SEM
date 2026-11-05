#!/usr/bin/env python3
"""E7 Stage 3: the Stage-1 fix on real datasets. Every write stays under scripts/results/paper/e7_stability/.

SELECTION RULE (fixed before any real-data run; never reads STRING/CORUM):
  Candidates: CANDIDATES below (control = paper model, the two Stage-1 winners, tau 1.5 in both designs).
  Per (dataset, split seed): hold out 20% of the cells of every condition (control included; split_cells
  below), build training
  targets from the other 80% and train each candidate. Primary criterion = held-out-cell fit: the ensemble's
  predicted mean shift for each knockdown j, mean over the models that saw j of (I - W_m)^-1[:, j] B_m[j],
  against the held-out cells' observed shift (build_intervention_targets on the held-out cells), mean squared
  error over all (condition, gene) pairs. Lower is better.
  Shared pick (the one config taken to full fits): lowest mean over datasets of MSE(config) / MSE(control).
  Per-dataset picks are reported too.
Only after the pick: full fits on all cells, and Table-1 metrics with the node bootstrap (500 draws) vs current
I-SEM and mean difference, for both scoring rules (a = paper readout, b = seen-only). vs_sparse: matched-k
precision against official inspre (E1b) and ADAPRE (E8).

  bash scripts/experiments/run_env.sh python -u scripts/experiments/e7_stability/e7_real.py select --ds hepg2 --seed 0
  bash scripts/experiments/run_env.sh python -u scripts/experiments/e7_stability/e7_real.py pick
  bash scripts/experiments/run_env.sh python -u scripts/experiments/e7_stability/e7_real.py full --ds hepg2 --configs C
  bash scripts/experiments/run_env.sh python -u scripts/experiments/e7_stability/e7_real.py eval --ds hepg2 --ref STRING --config C
  bash scripts/experiments/run_env.sh python -u scripts/experiments/e7_stability/e7_real.py summary --config C
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import e7  # noqa: E402  (also puts ROOT and scripts/experiments on sys.path)

ROOT, OUT = e7.ROOT, e7.OUT
VAL_FRACTION = 0.2
SEL, FULL, EVAL = OUT / "real_select", OUT / "real_full", OUT / "real_eval"
DATASETS = ["hepg2", "jurkat", "rpe1", "k562"]
CANDIDATES = [e7.CONTROL, "tau2.5_do0.0_half", "tau2.5_do0.0_full", "tau1.5_do0.0_half", "tau1.5_do0.0_full"]


def load(ds):
    from scripts.eval_all_metrics import load_dataset
    X, iv, genes = load_dataset(ds)
    return X, np.asarray(iv), genes


def split_cells(interventions, frac, seed):
    """Boolean validation mask: round(frac * n_c) random cells of every condition c (fewer than 1 -> none)."""
    iv = np.asarray(interventions)
    rng = np.random.RandomState(seed)
    val = np.zeros(len(iv), bool)
    for c in np.unique(iv):
        idx = np.flatnonzero(iv == c)
        rng.shuffle(idx)
        val[idx[:int(round(frac * len(idx)))]] = True
    return val


def heldout_pred(W, B, membership, device="cuda:0"):
    """P[:, j] = mean over models m that saw knockdown j of (I - W_m)^-1[:, j] B_m[j]."""
    import torch
    n = W.shape[1]
    P = torch.zeros(n, n, dtype=torch.float64, device=device)
    eye = torch.eye(n, dtype=torch.float64, device=device)
    for Wm, Bm, mm in zip(W, B, membership):
        T = torch.linalg.inv(eye - torch.as_tensor(Wm, dtype=torch.float64, device=device))
        P += T * torch.as_tensor(Bm * mm, dtype=torch.float64, device=device)[None, :]
    return (P / torch.as_tensor(membership.sum(0), dtype=torch.float64, device=device)[None, :]).cpu().numpy()


def select(ds, seed, configs):
    from src.models.interventional_nb_sem import build_intervention_targets
    out = SEL / f"{ds}_seed{seed}"
    todo = [c for c in configs if not (out / f"{c}.json").exists()]
    if not todo:
        return
    out.mkdir(parents=True, exist_ok=True)
    X, iv, genes = load(ds)
    val = split_cells(iv, VAL_FRACTION, seed)
    u, delta, _, _, _, se = build_intervention_targets(X[~val], list(iv[~val]), genes)
    uv, dv, *_ = build_intervention_targets(X[val], list(iv[val]), genes)
    tf, tv = (u > .5).argmax(1), (uv > .5).argmax(1)
    assert np.all((u > .5).sum(1) == 1) and np.all((uv > .5).sum(1) == 1)
    D_fit = np.full((len(genes), len(genes)), np.nan); D_fit[:, tf] = delta.T     # observed 80% shift, column j
    keep = np.isin(tv, tf)
    del X
    for c in todo:
        W, B, membership, info = e7.train(u, delta, se, c)
        P = heldout_pred(W, B, membership)
        err = dv[keep] - P[:, tv[keep]].T
        rec = {"dataset": ds, "seed": seed, "config": c, "n_val_conditions": int(keep.sum()),
               "heldout_mse": float(np.mean(err ** 2)), "heldout_mae": float(np.mean(np.abs(err))),
               "observed80_mse": float(np.mean((dv[keep] - D_fit[:, tv[keep]].T) ** 2)),   # no-model reference
               **{k: v for k, v in info.items() if k != "per_model"}}
        (out / f"{c}.json").write_text(json.dumps(rec, indent=1, default=float))
        print(f"{ds} seed{seed} {c}: mse {rec['heldout_mse']:.6g} (80%-obs {rec['observed80_mse']:.6g}) "
              f"sig|W| {info['sigma_abs_median']:.2f} rho {info['rho_W_median']:.2f} fb {info['n_fallbacks']} "
              f"{info['wall_s']:.0f}s", flush=True)


def pick():
    import pandas as pd
    rows = [json.loads(f.read_text()) for f in sorted(SEL.glob("*/*.json"))]
    df = pd.DataFrame([{k: r[k] for k in ("dataset", "seed", "config", "heldout_mse", "heldout_mae", "observed80_mse",
                                          "sigma_abs_median", "rho_W_median", "rho_W_max", "n_fallbacks",
                                          "min_pivot", "wall_s")} for r in rows])
    df.to_csv(OUT / "stage3_selection.csv", index=False)
    m = df.groupby(["dataset", "config"]).mean(numeric_only=True)
    rel = (m.heldout_mse / m.heldout_mse.xs(e7.CONTROL, level="config")).rename("mse_rel_control")
    per_ds = rel.groupby(level="dataset").idxmin().map(lambda x: x[1])
    complete = [c for c in CANDIDATES if all((d, c) in rel.index for d in df.dataset.unique())]
    shared = rel.groupby(level="config").mean().loc[complete].sort_values()
    lines = ["# Stage 3 selection (label-free; held-out 20% of cells per condition)", "",
             f"Datasets/seeds: {sorted(set(zip(df.dataset, df.seed)))}", "",
             (m.assign(mse_rel_control=rel)[["mse_rel_control", "heldout_mse", "observed80_mse", "heldout_mae",
                                               "sigma_abs_median", "rho_W_median",
                                               "rho_W_max", "n_fallbacks", "min_pivot", "wall_s"]]
              .round(5).to_markdown()), "",
             "Per-dataset pick (lowest held-out MSE): " + ", ".join(f"{d}: {c}" for d, c in per_ds.items()), "",
             "Shared ranking (mean over datasets of MSE / MSE(control)): " +
             ", ".join(f"{c} {v:.4f}" for c, v in shared.items()), "",
             f"SHARED PICK: {shared.index[0]}"]
    (OUT / "stage3_selection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return shared.index[0]


def full(ds, configs):
    from src.models.interventional_nb_sem import build_intervention_targets
    X, iv, genes = load(ds)
    u, delta, _, _, _, se = build_intervention_targets(X, list(iv), genes)
    del X
    for c in configs:
        f = FULL / ds / f"{c}.npz"
        if f.exists():
            continue
        W, B, membership, info = e7.train(u, delta, se, c)
        S_a, S_b = e7.scores(W, B, membership)
        e7.save(f, S_a, S_b, info)
        print(f"{ds} {c}: sig|W| {info['sigma_abs_median']:.2f} rho med/max {info['rho_W_median']:.2f}/"
              f"{info['rho_W_max']:.2f} fb {info['n_fallbacks']} piv {info['min_pivot']:.3g} {info['wall_s']:.0f}s",
              flush=True)


def evaluate(ds, ref, config, n_boot):
    import score_eval as se
    from paper_tables import load as load_cache
    z = load_cache(ds)
    genes = list(z["gene_names"])
    G = se.references(genes)[ref]
    new = np.load(FULL / ds / f"{config}.npz")
    base = {"I-SEM current": z["S_current"],
            "Mean difference": np.load(ROOT / f"results/mean_diff_adj_{ds}.npy")}
    out = {}
    for rule in ("a", "b"):
        out[rule] = se.score_reference({"I-SEM": new[f"S_{rule}"], **base}, G, n_boot=n_boot,
                                       baselines=["Mean difference"])
    f = EVAL / config / f"{ds}_{ref}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"ds": ds, "ref": ref, "config": config, "rules": out,
                             "meta": se.meta({"n_boot": n_boot})}, indent=1, allow_nan=True))
    print("wrote", f, flush=True)


def summary(config):
    lines = [f"# Stage 3: {config} on real datasets vs current I-SEM (node bootstrap, paired)", "",
             "Δ = new − other, over the same bootstrap draws; CI = 95% percentile interval; * = CI excludes 0.", ""]
    wins = {"auroc": 0, "edge_specific_auroc": 0}
    n_pairs = 0
    for rule in ("a", "b"):
        lines += [f"## rule {rule} ({'paper readout, unseen B = 1' if rule == 'a' else 'seen-only'})", "",
                  "| dataset | ref | metric | new | I-SEM current | Δ vs current [CI] | "
                  "Mean diff | Δ vs Mean diff |", "|---|---|---|---|---|---|---|---|"]
        for ds in DATASETS:
            for ref in ("STRING", "CORUM"):
                f = EVAL / config / f"{ds}_{ref}.json"
                if not f.exists():
                    lines.append(f"| {ds} | {ref} | missing | | | | | |"); continue
                r = json.loads(f.read_text())["rules"][rule]
                if rule == "a":
                    n_pairs += 1
                for k in se_metrics():
                    def d(m):
                        x = r[m]["isem_minus_this"][k]
                        lo, hi = x["difference_ci"]
                        return f"{x['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}]{'*' if lo > 0 or hi < 0 else ''}"
                    if rule == "a" and k in wins and r["I-SEM current"]["isem_minus_this"][k]["difference_ci"][0] > 0:
                        wins[k] += 1
                    lines.append(f"| {ds} | {ref} | {k} | {r['I-SEM'][k]:.4f} | {r['I-SEM current'][k]:.4f} | "
                                 f"{d('I-SEM current')} | "
                                 f"{r['Mean difference'][k]:.4f} | {d('Mean difference')} |")
        lines.append("")
    lines += ["## GATE 2 (rule a)", "",
              f"Pairs with paired CI > 0 vs current I-SEM: AUROC {wins['auroc']}/{n_pairs}, "
              f"edge-specific AUROC {wins['edge_specific_auroc']}/{n_pairs}."]
    for ds in DATASETS:
        f = FULL / ds / f"{config}.json"
        if f.exists():
            i = json.loads(f.read_text())
            lines.append(f"- {ds}: median ‖abs W‖₂ {i['sigma_abs_median']:.3f} (max {i['sigma_abs_max']:.3f}), "
                         f"ρ(W) median {i['rho_W_median']:.3f} (max {i['rho_W_max']:.3f}), ‖W‖₂ median "
                         f"{i['norm2_W_median']:.3f}, σ_min(I−W) min {i['sigma_min_IminusW_min']:.3g}, "
                         f"failing-pivot records {i['n_fallbacks']} ({i['n_models_with_fallback']} models), min pivot "
                         f"{i['min_pivot']:.3g}, wall {i['wall_s']:.0f}s")
    (OUT / f"stage3_{config}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def tausweep():
    """One row per (dataset, config): label-free selection fit (seed 0) next to the full-fit STRING/CORUM
    scores (rule a). Descriptive only: the STRING/CORUM columns never feed the selection rule."""
    lines = ["# Stage 3 tau sweep on real datasets (exploratory; STRING/CORUM columns are descriptive, not used "
             "for selection)", "",
             "Selection fit = 80% of cells, seed 0; MSE rel = held-out MSE / control's. Full fit = all cells. "
             "ES = edge-specific AUROC. Δ = vs I-SEM current, * = paired bootstrap CI excludes 0.", ""]
    for ds in DATASETS:
        cfgs = sorted({f.stem for f in (SEL / f"{ds}_seed0").glob("*.json")} | {f.stem for f in (FULL / ds).glob("*.json")},
                      key=lambda c: (str(e7.parse_cfg(c)[0]), -e7.parse_cfg(c)[1], c))
        if not cfgs:
            continue
        ctl = SEL / f"{ds}_seed0" / f"{e7.CONTROL}.json"
        ctl = json.loads(ctl.read_text())["heldout_mse"] if ctl.exists() else np.nan
        lines += [f"## {ds}", "",
                  "| config | MSE rel | sel ρ med/max | sel fb (models) | full ρ med/max | full fb | "
                  "STRING AUROC Δ | STRING ES Δ | CORUM AUROC Δ | CORUM ES Δ |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c in cfgs:
            row = [c]
            f = SEL / f"{ds}_seed0" / f"{c}.json"
            if f.exists():
                s = json.loads(f.read_text())
                row += [f"{s['heldout_mse'] / ctl:.3f}",
                        f"{s['rho_W_median']:.2f}/{s['rho_W_max']:.2f}", f"{s['n_fallbacks']} ({s['n_models_with_fallback']})"]
            else:
                row += [""] * 3
            f = FULL / ds / f"{c}.json"
            if f.exists():
                i = json.loads(f.read_text())
                row += [f"{i['rho_W_median']:.2f}/{i['rho_W_max']:.2f}", f"{i['n_fallbacks']} ({i['n_models_with_fallback']})"]
            else:
                row += [""] * 2
            for ref in ("STRING", "CORUM"):
                f = EVAL / c / f"{ds}_{ref}.json"
                if not f.exists():
                    row += [""] * 2; continue
                r = json.loads(f.read_text())["rules"]["a"]
                for k in ("auroc", "edge_specific_auroc"):
                    x = r["I-SEM current"]["isem_minus_this"][k]
                    lo, hi = x["difference_ci"]
                    row.append(f"{r['I-SEM'][k]:.4f} ({x['difference']:+.4f}{'*' if lo > 0 or hi < 0 else ''})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    (OUT / "stage3_tausweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def vs_sparse(configs, n_boot):
    """Matched-k precision vs the sparse baselines (official inspre, E1b; ADAPRE, E8): each I-SEM variant's top-k
    pairs, k = the baseline's edge count, full panel, paired node bootstrap. Same call as e8_adapre.score_real
    (paper_tables.matched_k), so the 'I-SEM current' rows reproduce E8's numbers. Rule a scores."""
    import paper_tables as pt
    import score_eval as se
    base = {"inspre (official)": lambda ds: np.load(OUT.parent / f"e1b_inspre_official/{ds}.npy"),
            "ADAPRE": lambda ds: np.load(OUT.parent / f"e8_adapre/{ds}_adapre.npy")}
    rows = []
    for ds in DATASETS:
        z = pt.load(ds)
        G_all = se.references([str(g) for g in z["gene_names"]])
        S = {"I-SEM current": z["S_current"], **{c: np.load(FULL / ds / f"{c}.npz")["S_a"] for c in configs
                                                  if (FULL / ds / f"{c}.npz").exists()}}
        for ref, G in G_all.items():
            for bname, bload in base.items():
                B = bload(ds)
                for m, A in S.items():
                    res = {}
                    pt.matched_k(res, A, B, G, n_boot)
                    i, b = res[pt.MATCHED_K], res[pt.INSPRE]
                    d = i["isem_minus_inspre_official"]["precision_at_k"]
                    rows.append({"dataset": ds, "ref": ref, "baseline": bname, "isem": m, "k": i["k"],
                                 "isem_precision": i["precision_at_k"], "baseline_precision": b["precision_at_k"],
                                 "diff": d["difference"], "diff_lo": d["difference_ci"][0], "diff_hi": d["difference_ci"][1]})
                    print(f"{ds} {ref} vs {bname} k={i['k']} {m}: {i['precision_at_k']:.3f} vs {b['precision_at_k']:.3f} "
                          f"({d['difference']:+.3f} [{d['difference_ci'][0]:+.3f}, {d['difference_ci'][1]:+.3f}])", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stage3_vs_sparse.csv", index=False)
    lines = ["# I-SEM variants vs sparse baselines at matched edge count (exploratory)", "",
             "Precision of each I-SEM variant's top-k pairs vs the baseline's k edges, full panel, rule a; "
             f"Δ = I-SEM − baseline, paired node bootstrap ({n_boot} draws), * = CI excludes 0. "
             "AUROC is not used here: the baselines score most pairs exactly 0.", ""]
    for bname in base:
        lines += [f"## vs {bname}", "", "| dataset | ref | k | baseline | " + " | ".join(S) + " |",
                  "|---|---|---|---|" + "---|" * len(S)]
        for (ds, ref), g in df[df.baseline == bname].groupby(["dataset", "ref"], sort=False):
            g = g.set_index("isem")
            cells = [f"{g.loc[m, 'isem_precision']:.3f} ({g.loc[m, 'diff']:+.3f}"
                     f"{'*' if g.loc[m, 'diff_lo'] > 0 or g.loc[m, 'diff_hi'] < 0 else ''})" if m in g.index else ""
                     for m in S]
            lines.append(f"| {ds} | {ref} | {g.k.iloc[0]} | {g.baseline_precision.iloc[0]:.3f} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUT / "stage3_vs_sparse.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def se_metrics():
    return ["auroc", "auprc_lift", "p_at_1000", "edge_specific_auroc", "edge_specific_auprc_ratio"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "pick", "full", "eval", "summary", "tausweep", "vs_sparse"])
    ap.add_argument("--ds", nargs="+", default=DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", nargs="+", default=CANDIDATES)
    ap.add_argument("--config")
    ap.add_argument("--ref", nargs="+", default=["STRING", "CORUM"])
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    if a.cmd == "select":
        for ds in a.ds:
            select(ds, a.seed, a.configs)
    elif a.cmd == "pick":
        pick()
    elif a.cmd == "full":
        for ds in a.ds:
            full(ds, a.configs)
    elif a.cmd == "eval":
        for ds in a.ds:
            for ref in a.ref:
                evaluate(ds, ref, a.config, a.n_boot)
    elif a.cmd == "tausweep":
        tausweep()
    elif a.cmd == "vs_sparse":
        vs_sparse(a.configs, a.n_boot)
    else:
        summary(a.config)


if __name__ == "__main__":
    main()
