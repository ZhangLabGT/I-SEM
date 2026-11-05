#!/usr/bin/env python3
"""Scores results/ and writes results/REPORT.md (generated; do not edit) and the CSVs behind it."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist, wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
I = "I-SEM (c) defaults"
N, A, MD = "inspre (official, DAG)", "ADAPRE", "Mean difference"
METRICS = ("auroc", "auprc")
DIR = ("auroc_direct_vs_strong", "auroc_direct_vs_indirect", "lift_indirect")
LOCKED = "rand_cycles_confound_7"
PRIMARY = {"H1": ("ins_sel", "auroc"), "H2": ("ins_sel", "auprc"), "H3": ("ada_med", "auroc"), "H4": ("ada_med", "auprc")}
COMP = {"ins_sel": "inspre, selected penalty", "ins_med": "inspre, median penalty", "ins_best": "inspre, best penalty (upper limit)",
        "ada_med": "ADAPRE, median penalty", "ada_best": "ADAPRE, best penalty (upper limit)"}


def holm(p):
    """Holm-adjusted p-values, same order as p."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    out = np.empty(len(p))
    out[o] = adj
    return out


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    h = tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def wil(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(wilcoxon(d).pvalue) if len(d) > 5 and (d != 0).any() else np.nan


def primary_verdicts(diffs):
    """diffs: {id: per-setting differences}. All must have Holm p < 0.05 and mean > 0 for the claim to pass."""
    ids = list(diffs)
    adj = holm([wil(diffs[k]) for k in ids])
    rows = [dict(id=k, mean_diff=float(np.nanmean(diffs[k])), wins=int(np.nansum(np.asarray(diffs[k]) > 0)), n=int(np.isfinite(diffs[k]).sum()),
                 p_holm=float(adj[i]), passed=bool(adj[i] < 0.05 and np.nanmean(diffs[k]) > 0)) for i, k in enumerate(ids)]
    return rows, all(r["passed"] for r in rows)


def path_summary(rows):
    """One row per dataset: I-SEM, inspre selected, median/best penalty for both path methods."""
    c1, c2 = rows[rows.config == "C1"], rows[rows.config == "C2"]
    out = []
    for (sim, st, rep), g in c1.groupby(["sim", "setting", "rep"]):
        r = dict(sim=sim, setting=st, rep=rep)
        i = g[g.method == I]
        if len(i):
            r.update(isem_auroc=i.auroc.iloc[0], isem_auprc=i.auprc.iloc[0])
        for nm, m in (("ins", N), ("ada", A)):
            p = g[g.method == m]
            if len(p):
                for mt in METRICS:
                    r[f"{nm}_med_{mt}"] = p[mt].median()
                    r[f"{nm}_best_{mt}"] = p[mt].max()
        q = c2[(c2.setting == st) & (c2.rep == rep) & (c2.sim == sim) & (c2.method == N)]
        if len(q):
            r.update(ins_sel_auroc=q.auroc.iloc[0], ins_sel_auprc=q.auprc.iloc[0])
        out.append(r)
    return pd.DataFrame(out)


def direct_summary(d):
    out = []
    for (sim, st, rep), g in d.groupby(["sim", "setting", "rep"]):
        r = dict(sim=sim, setting=st, rep=rep)
        for mt in DIR:
            i = g[g.method == I]
            if len(i):
                r[f"isem_{mt}"] = i[mt].iloc[0]
            mm = g[g.method == MD]
            if len(mm):
                r[f"md_{mt}"] = mm[mt].iloc[0]
            for nm, m in (("ins", N), ("ada", A)):
                p = g[g.method == m]
                if len(p):
                    r[f"{nm}_med_{mt}"] = p[mt].median()
            s = g[(g.method == N) & g.selected]
            if len(s):
                r[f"ins_sel_{mt}"] = s[mt].iloc[0]
        out.append(r)
    return pd.DataFrame(out)


def f1_table(rows, mask, label):
    out = []
    c1 = rows[(rows.config == "C1") & mask]
    for m in (N, A):
        p = c1[(c1.method == m) & (c1.k > 0)]
        for scope, q in (("all penalties", p), ("k <= 2x true edges", p[p.k <= 2 * p.n_true_edges])):
            dd = q.c_topk_f1 - q.f1
            out.append(dict(block=label, method=m, scope=scope, penalties=len(q), win=(dd > 1e-9).mean(), tie=(dd.abs() <= 1e-9).mean(),
                            lose=(dd < -1e-9).mean(), isem_f1=q.c_topk_f1.mean(), method_f1=q.f1.mean()))
    return out


def md(df, nd=3):
    d = df.copy()
    for c in d:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    L = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    return L + ["| " + " | ".join(map(str, r)) + " |" for r in d.values]


def fit_loglog(h, y, cols):
    h = h[(h[y] > 0)]
    Y = np.log(h[y])
    X = np.column_stack([np.ones(len(Y))] + [np.log(h[c]) for c in cols])
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    return b[1:], 1 - ((Y - X @ b) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()


def main():
    rows = pd.read_csv(RES / "rows.csv.gz", low_memory=False)
    meta = pd.read_csv(RES / "datasets.csv")
    dirn = pd.read_csv(RES / "directness.csv.gz")
    st = pd.read_csv(RES / "strength.csv")
    fits = pd.read_csv(RES / "fits.csv")
    miss = pd.read_csv(RES / "missing_fits.csv")
    T = path_summary(rows)
    T.to_csv(RES / "per_dataset.csv", index=False)
    fac = meta[meta.sim == "inspre"].drop_duplicates("setting").set_index("setting")[["graph", "cycles", "confounders", "density", "effects", "instruments", "int_beta"]]
    fac["confounded"] = fac.confounders > 0
    G = T[T.sim == "inspre"]
    S = G.groupby("setting").mean(numeric_only=True).drop(columns="rep").join(fac)
    S.to_csv(RES / "per_setting.csv")
    assert (G.groupby("setting").size() == 10).all() and len(S) == 64
    B = S[(S.effects == "small") & (S.instruments == "weak")]
    assert len(B) == 16
    CB = S[(S.effects == "large") & (S.instruments == "strong")]
    assert len(CB) == 16
    L = ["# Simulated-data claim: confirmatory results (generated by analyze.py; do not edit)", "",
         "Seeds 12-21. Unit: setting mean over seeds unless stated.", ""]
    # primary
    diffs = {k: (B[f"isem_{mt}"] - B[f"{c}_{mt}"]).values for k, (c, mt) in PRIMARY.items()}
    prows, ok = primary_verdicts(diffs)
    P = pd.DataFrame(prows)
    P.insert(1, "comparison", [COMP[PRIMARY[k][0]] for k in P.id])
    P.insert(2, "metric", [PRIMARY[k][1] for k in P.id])
    P["isem"] = [B[f"isem_{PRIMARY[k][1]}"].mean() for k in P.id]
    P["baseline"] = [B[f"{PRIMARY[k][0]}_{PRIMARY[k][1]}"].mean() for k in P.id]
    L += [f"## Primary tests: {'ALL PASS' if ok else 'NOT ALL PASS'}", ""] + md(P) + [""]
    # S1
    s1 = []
    for f in ("graph", "cycles", "confounded"):
        for lv, g in B.groupby(f):
            r = {"factor": f, "level": lv}
            good = True
            for k, (c, mt) in PRIMARY.items():
                d = g[f"isem_{mt}"] - g[f"{c}_{mt}"]
                r[k] = f"{d.mean():+.3f} ({int((d > 0).sum())}/{len(d)})"
                good &= d.mean() > 0 and (d > 0).sum() >= 6
            r["passed"] = good
            s1.append(r)
    S1 = pd.DataFrame(s1)
    L += [f"## S1 'across': {'PASS' if S1.passed.all() else 'FAIL'}", "", "Cell = mean difference (settings I-SEM higher).", ""] + md(S1) + [""]
    # best penalty, upper limit, in the block
    u = []
    for c in ("ins_med", "ins_best", "ada_best"):
        for mt in METRICS:
            d = B[f"isem_{mt}"] - B[f"{c}_{mt}"]
            u.append(dict(comparison=COMP[c], metric=mt, isem=B[f"isem_{mt}"].mean(), baseline=B[f"{c}_{mt}"].mean(), mean_diff=d.mean(), wins=f"{int((d > 0).sum())}/16", p=wil(d)))
    L += ["## Other comparisons in the block (no pass criterion)", ""] + md(pd.DataFrame(u)) + [""]
    u = []
    for f in ("graph",):
        for lv, g in B.groupby(f):
            for mt in METRICS:
                d = g[f"isem_{mt}"] - g[f"ada_best_{mt}"]
                u.append(dict(graph=lv, metric=mt, mean_diff_vs_ada_best=d.mean(), wins=f"{int((d > 0).sum())}/{len(d)}"))
    L += ["ADAPRE at its best penalty, by graph type:", ""] + md(pd.DataFrame(u)) + [""]
    # S2
    AD = T[T.sim == "adapre"].groupby("setting").mean(numeric_only=True)
    s2 = []
    allok = True
    for s, r in AD.iterrows():
        near = min(r.isem_auroc, r.ins_sel_auroc, r.ada_med_auroc) >= 0.98
        comp = abs(r.isem_auroc - r.ada_med_auroc) <= 0.02 and abs(r.isem_auprc - r.ada_med_auprc) <= 0.02
        allok &= near and comp
        s2.append(dict(set=s, isem_auroc=r.isem_auroc, ins_sel_auroc=r.ins_sel_auroc, ada_med_auroc=r.ada_med_auroc, ada_best_auroc=r.ada_best_auroc,
                       isem_auprc=r.isem_auprc, ins_sel_auprc=r.ins_sel_auprc, ada_med_auprc=r.ada_med_auprc, ada_best_auprc=r.ada_best_auprc, near_ceiling=near, comparable=comp))
    L += [f"## S2 ADAPRE simulator: {'PASS' if allok else 'FAIL'}", "", "Means over seeds 12-21; each seed's two sets share a network.", ""] + md(pd.DataFrame(s2)) + [""]
    # S3
    dc = {"inspre selected": (CB.isem_auprc - CB.ins_sel_auprc).values, "ADAPRE median": (CB.isem_auprc - CB.ada_med_auprc).values}
    adj = holm([wil(v) for v in dc.values()])
    s3 = [dict(comparison=k, isem=CB.isem_auprc.mean(), baseline=(CB.isem_auprc - v).mean(), mean_diff=v.mean(), wins=f"{int((v > 0).sum())}/16", p_holm=adj[i], passed=bool(adj[i] < .05 and v.mean() < 0)) for i, (k, v) in enumerate(dc.items())]
    L += [f"## S3 contrast block (large effects, strong knockdowns), AUPRC: {'PASS' if all(x['passed'] for x in s3) else 'FAIL'}", ""] + md(pd.DataFrame(s3)) + [""]
    # S4
    lk = G[G.setting == LOCKED]
    s4 = []
    for c in ("ins_best", "ada_best"):
        for mt in METRICS:
            d = (lk[f"isem_{mt}"] - lk[f"{c}_{mt}"]).values
            lo, hi = ci95(d)
            s4.append(dict(comparison=COMP[c], metric=mt, isem=lk[f"isem_{mt}"].mean(), baseline=lk[f"{c}_{mt}"].mean(), mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, seeds_won=f"{int((d > 0).sum())}/10", passed=bool(lo > 0)))
    L += [f"## S4 locked setting vs best penalties: {'PASS' if all(x['passed'] for x in s4) else 'FAIL'}", ""] + md(pd.DataFrame(s4)) + [""]
    # S5 and blocks
    b5 = []
    for (e, k), g in S.groupby(["effects", "instruments"]):
        r = dict(effects=e, knockdown=k, n=len(g))
        for c in ("ins_sel", "ins_med", "ada_med", "ins_best", "ada_best"):
            for mt in METRICS:
                r[f"{c}_{mt}"] = f"{(g[f'isem_{mt}'] - g[f'{c}_{mt}']).mean():+.3f} ({int((g[f'isem_{mt}'] > g[f'{c}_{mt}']).sum())})"
        b5.append(r)
    all5 = []
    for c in ("ins_sel", "ins_med", "ada_med", "ins_best", "ada_best"):
        for mt in METRICS:
            d = S[f"isem_{mt}"] - S[f"{c}_{mt}"]
            all5.append(dict(comparison=COMP[c], metric=mt, isem=S[f"isem_{mt}"].mean(), baseline=S[f"{c}_{mt}"].mean(), mean_diff=d.mean(), wins=f"{int((d > 0).sum())}/64"))
    L += ["## S5 whole grid (64 settings)", ""] + md(pd.DataFrame(all5)) + ["", "By block, cell = mean difference (settings I-SEM higher):", ""] + md(pd.DataFrame(b5)) + [""]
    # F1
    fac_r = rows.merge(meta[["sim", "setting", "rep", "effects", "instruments"]], on=["sim", "setting", "rep"], how="left")
    inm = fac_r.sim == "inspre"
    f = f1_table(fac_r, inm, "all 64") + f1_table(fac_r, inm & (fac_r.effects == "small") & (fac_r.instruments == "weak"), "small, weak") + \
        f1_table(fac_r, inm & (fac_r.effects == "large") & (fac_r.instruments == "strong"), "large, strong") + f1_table(fac_r, fac_r.sim == "adapre", "ADAPRE simulator")
    L += ["## F1 at matched edge count (no pass criterion)", ""] + md(pd.DataFrame(f)) + [""]
    # directness
    D = direct_summary(dirn)
    D.to_csv(RES / "directness_per_dataset.csv", index=False)
    DS = D[D.sim == "inspre"].groupby("setting").mean(numeric_only=True).drop(columns="rep").join(fac)
    DS.to_csv(RES / "directness_per_setting.csv")
    dr = []
    for lab, g in (("all 64", DS), ("small, weak", DS[(DS.effects == "small") & (DS.instruments == "weak")]), ("large, strong", DS[(DS.effects == "large") & (DS.instruments == "strong")])):
        for mt in DIR:
            for c, nm in (("ins_sel", "inspre selected"), ("ins_med", "inspre median"), ("ada_med", "ADAPRE median"), ("md", "mean difference")):
                d = g[f"isem_{mt}"] - g[f"{c}_{mt}"]
                better = (d < 0).sum() if mt == "lift_indirect" else (d > 0).sum()
                dr.append(dict(block=lab, metric=mt, vs=nm, isem=g[f"isem_{mt}"].mean(), baseline=g[f"{c}_{mt}"].mean(), mean_diff=d.mean(), isem_better=f"{int(better)}/{len(d)}", p=wil(d)))
    L += ["## Direct vs strong indirect (no pass criterion; lift_indirect: lower is better)", ""] + md(pd.DataFrame(dr)) + [""]
    lockd = D[D.setting == LOCKED]
    L += [f"Locked setting, I-SEM auroc_direct_vs_strong {lockd.isem_auroc_direct_vs_strong.mean():.3f}; inspre selected {lockd.ins_sel_auroc_direct_vs_strong.mean():.3f}; "
          f"ADAPRE median {lockd.ada_med_auroc_direct_vs_strong.mean():.3f}; mean difference {lockd.md_auroc_direct_vs_strong.mean():.3f}.", ""]
    w = DS[DS.isem_auroc_direct_vs_strong <= DS.md_auroc_direct_vs_strong].index.tolist()
    L += [f"Settings where I-SEM does not beat mean difference on auroc_direct_vs_strong: {len(w)}/64 {w[:12]}", ""]
    # strength
    SG = st[st.sim == "inspre"].merge(fac.reset_index()[["setting", "effects", "instruments", "int_beta"]], on="setting").groupby("int_beta")[["d", "t", "d_sqrt_n", "min_t", "frac_t_lt3"]].median().reset_index()
    L += ["## Knockdown strength (no pass criterion)", "", "inspre grid, median over settings and seeds, by int_beta:", ""] + md(SG, 2) + [""]
    L += ["Lowest |t| of any knockdown, inspre grid: %.2f. ADAPRE simulator, by set:" % st[st.sim == "inspre"].min_t.min(), ""] + md(st[st.sim == "adapre"].groupby("setting")[["d", "t", "d_sqrt_n", "min_t"]].median().reset_index(), 2) + [""]
    # sweep
    SW = T[T.sim == "sweep"]
    cells = ["d2_n25", "d1_n100", "d0.5_n400", "d0.25_n1600", "d0.25_n400", "d0.3_n100", "d0.25_n100", "d0.25_n25", "mixed"]
    M = SW.groupby("setting").mean(numeric_only=True).reindex(cells)
    tt = st[st.sim == "sweep"].groupby("setting")[["t", "d_sqrt_n", "frac_t_lt3"]].median().reindex(cells)
    ok_ins = fits[(fits.sim == "sweep") & (fits.method == "inspre_dag")].groupby("setting").size().reindex(cells).fillna(0).astype(int)
    tab = pd.DataFrame({"cell": cells, "t": tt.t.values, "d_sqrt_n": tt.d_sqrt_n.values, "inspre_fits": ok_ins.values})
    for mt in METRICS:
        for c, nm in (("isem", "isem"), ("ins_sel", "ins_sel"), ("ada_med", "ada_med"), ("ada_best", "ada_best")):
            tab[f"{nm}_{mt}"] = M[f"{c}_{mt}"].values if f"{c}_{mt}" in M else np.nan
    L += ["## Knockdown sweep, our extension, 10 reps per cell (no pass criterion)", ""] + md(tab) + [""]
    gaps = []
    for cell in cells:
        g = SW[SW.setting == cell]
        for c in ("ins_sel", "ada_med", "ada_best"):
            for mt in METRICS:
                if f"{c}_{mt}" not in g or g[f"{c}_{mt}"].notna().sum() < 2:
                    continue
                d = (g[f"isem_{mt}"] - g[f"{c}_{mt}"]).dropna()
                lo, hi = ci95(d)
                gaps.append(dict(cell=cell, vs=COMP[c], metric=mt, reps=len(d), mean_gap=d.mean(), ci_lo=lo, ci_hi=hi))
    pd.DataFrame(gaps).to_csv(RES / "sweep_gaps.csv", index=False)
    L += ["Gaps are in `sweep_gaps.csv`.", ""]
    fc = pd.read_csv(RES / "filter_check.csv")
    fr = []
    for (st_, rp), g in fc.groupby(["setting", "rep"]):
        a = g[g.method == "adapre"]
        r = dict(setting=st_, rep=rp, K=g.K.iloc[0], isem_auroc=g[g.method == "isem"].auroc.iloc[0], isem_auprc=g[g.method == "isem"].auprc.iloc[0], ada_med_auroc=a.auroc.median(), ada_best_auroc=a.auroc.max(), ada_med_auprc=a.auprc.median(), ada_best_auprc=a.auprc.max())
        fr.append(r)
    FR = pd.DataFrame(fr).groupby("setting").mean(numeric_only=True).drop(columns="rep").reindex(cells).reset_index()
    L += ["Filter check: all methods rescored on only the genes ADAPRE kept (gaps here cannot come from gene loss):", ""] + md(FR) + [""]
    L += ["Mixed cell, out-edges by source knockdown strength (pairs pooled over reps):", ""] + md(pd.read_csv(RES / "mixed_weak_sources.csv")) + [""]
    # ratio
    Rd = pd.read_csv(RES / "ratio_datasets.csv")
    Rk = pd.read_csv(RES / "ratio_knockdowns.csv")
    Q = Rk[Rk.cell != "mixed"].copy()
    Q["at"], Q["ad"] = Q.t.abs(), Q.shift_sd.abs()
    Q = Q[(Q["at"] > 0) & (Q.ad > 0) & (Q.err_med > 0)]
    Q["x"] = Q.ad * np.sqrt(Q.n)
    a, ra = fit_loglog(Q, "err_med", ["at"])
    b, rb = fit_loglog(Q, "err_med", ["ad"])
    bin_ = pd.cut(Q["at"], [0, 1, 2, 3, 5, 10, 20, 40, 1e5])
    BQ = Q.groupby(bin_, observed=True).agg(knockdowns=("t", "size"), err_med=("err_med", "median"), err_p99=("err_p99", "median"), err_max=("err_max", "median"), abs_gt2=("frac_abs_gt2", "mean"), se_gt1=("frac_se_gt1", "mean"), flip=("flip", "mean")).reset_index()
    BQ["at"] = BQ["at"].astype(str)
    L += ["## Ratio test, simulator, our extension (no pass criterion)", "", f"t / (d sqrt n) median {(Q['at'] / Q.x).median():.3f}: the simulator cannot separate t from d sqrt n.",
          f"log error vs log t: slope {a[0]:.3f}, R2 {ra:.3f}; vs log d alone: slope {b[0]:.3f}, R2 {rb:.3f}.", ""] + md(BQ) + [""]
    cz = Rd[~Rd.mixed].groupby(["d", "n_kd"])[["err_med", "err_p99", "err_max", "flip_frac", "K_after_filter", "cor_R"]].mean().reset_index()
    L += md(cz) + [""]
    # real
    R = pd.read_csv(RES / "real_knockdowns.csv")
    Z = R[(R.t > 0) & (R.d > 0) & (R.err_med > 0)].copy()
    Z["x"] = Z.d * np.sqrt(Z.n)
    Z["inflate"] = Z.t / Z.x
    L += ["## Real datasets, split-half (no truth; no pass criterion)", ""]
    L += md(Z.groupby("screen").agg(knockdowns=("t", "size"), n=("n", "median"), d=("d", "median"), t=("t", "median"), d_sqrt_n=("x", "median"), t_over_dsqrtn=("inflate", "median"), pct_t_lt3=("t", lambda s: 100 * (s < 3).mean()), pct_x_lt3=("x", lambda s: 100 * (s < 3).mean()), err=("err_med", "median")).reset_index(), 2) + [""]
    reg = []
    for y in ("err_med", "log_div_diff"):
        a, ra = fit_loglog(Z, y, ["d", "n"])
        b, rb = fit_loglog(Z, y, ["x"])
        c, rc = fit_loglog(Z, y, ["t"])
        e, re = fit_loglog(Z, y, ["x", "t"])
        reg.append(dict(outcome=y, slope_d=a[0], slope_n=a[1], r2_d_n=ra, slope_dsqrtn=b[0], r2_dsqrtn=rb, slope_t=c[0], r2_t=rc, both_dsqrtn=e[0], both_t=e[1], r2_both=re))
    pr = 2 * 0.6745 / Z.x
    L += md(pd.DataFrame(reg)) + ["", f"Observed / predicted split-half error (0.6745 x 2 / (d sqrt n)): median {(Z.err_med / pr).median():.3f}; Spearman {Z[['err_med']].assign(p=pr).corr('spearman').iloc[0, 1]:.3f}.", ""]
    bq = Z.groupby(pd.cut(Z.t, [0, 1, 2, 3, 5, 10, 20, 40, 1e5]), observed=True).agg(knockdowns=("t", "size"), flip=("flip_rate", "mean"), err=("err_med", "median"), p99_95=("err_p99", lambda s: s.quantile(.95)), div=("log_div_diff", "median")).reset_index()
    bq["t"] = bq["t"].astype(str)
    L += md(bq) + [""]
    # run facts
    fx = fits.groupby(["sim", "method"]).agg(fits=("wall", "size"), wall_median=("wall", "median"), wall_max=("wall", "max"), dropped=("dropped", "sum"), oom=("oom_retries", "sum"), models_min=("n_models", "min"), models_max=("n_models", "max")).reset_index()
    L += ["## Run facts", "", f"Missing or failed fits: {len(miss)}.", ""] + md(fx, 1) + [""]
    if len(miss):
        L += md(miss.groupby(["sim", "setting", "method"]).size().rename("missing").reset_index()) + [""]
    L += ["Timings come from parallel runs on a shared GPU and host, so they are not speed measurements.", ""]
    (RES / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
