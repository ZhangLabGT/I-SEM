#!/usr/bin/env python3
"""Scores the ablation fits; descriptive, no pass criterion. Comparator and shipped rows are read from the stored reports.

  bash run.sh analyze.py sims
  bash run.sh analyze.py real --n-boot 500 --workers 6 [--ds hepg2 ...]
  bash run.sh analyze.py real-report
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import harness as h
import directness as dn
score, e9 = h.score, h.e9

HERE = h.HERE
RES = HERE / "results"
REMOVALS = ("no_push_signed", "no_push_abs", "no_se", "no_stability", "no_half", "wald", "plugin")   # 7 arms of the Holm family
ROWS = ("shipped", *REMOVALS)
CACHE_NAME = {"no_push_abs": "no_push"}
REAL_REPORT = h.ROOT / "scripts/real_data/res_A"
COMPS = (("ins_sel", "inspre_selected"), ("ins_med", "inspre_median"), ("ins_best", "inspre_best"), ("ada_med", "adapre_median"), ("ada_best", "adapre_best"))
BLOCK_NUMBERS = {"weak": {"isem_auroc": .780, "ins_sel_auroc": .696, "ada_med_auroc": .751, "isem_auprc": .344, "ins_sel_auprc": .309, "ada_med_auprc": .278},
                 "contrast": {"isem_auprc": .879, "ins_sel_auprc": .986, "ada_med_auprc": .984}}     # sim_claim REPORT.md


def sim_analyze():
    """fresh_seeds/analyze.py's pure summaries; its module-level RES is only a path."""
    spec = importlib.util.spec_from_file_location("sim_claim_analyze", h.SIM / "analyze.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    out = np.empty(len(p)); out[o] = adj
    return out


def label(g, delta, sig):
    """g = shipped - comparator; delta = arm - shipped; sig = Holm significant."""
    if not np.isfinite(g):
        return "NA"
    if g == 0:
        return "NO GAP"
    if not sig:
        return "NONE"
    if g > 0:
        if delta < 0:
            return "CARRIES" if -delta >= .5 * g else "PARTIAL"
        return "HELPS"
    if delta > 0:
        return "EXPLAINS" if delta >= .5 * abs(g) else "PARTIAL"
    return "WORSENS"


def q1(labels):
    return "PASS" if all(x == "CARRIES" for x in labels) else "FAIL"


def q2(labels):
    return "PASS" if all(x == "EXPLAINS" for x in labels) else "FAIL"


def real_verdict(g, delta):
    """Per-dataset g and delta for one (arm, reference, metric, comparator). Returns (verdict, leads, carried, deficits, explained)."""
    g, delta = np.asarray(g, float), np.asarray(delta, float)
    lead, deficit = g > 0, g < 0
    carried = int((delta[lead] <= -.5 * g[lead]).sum()); explained = int((delta[deficit] >= .5 * np.abs(g[deficit])).sum())
    nl, nd = int(lead.sum()), int(deficit.sum())
    if nl >= 3 and carried >= 3:
        v = "CARRIES ON REAL"
    elif nd >= 3 and explained >= 3:
        v = "EXPLAINS ON REAL"
    elif nl < 3 and nd < 3:
        v = "NOT ASSESSABLE"
    else:
        v = "NOT MET"
    return v, nl, carried, nd, explained


def score_dataset(d):
    d = Path(d)
    m, _, _, _, W = e9.load(d)
    tru = e9.edges(W, m); off = ~np.eye(len(W), dtype=bool)
    y = tru[off].astype(np.int8)
    rel = d.relative_to(e9.DATA / "inspre")
    out = []
    with np.load(h.SIM / "cache/inspre" / rel / "isem.npz") as z:      # G1: the cached shipped fit
        cached = score.auc_ap(y, np.abs(np.asarray(z["S"], float))[off])
    for arm in ROWS:
        f = HERE / "cache/sims" / rel / f"{CACHE_NAME.get(arm, arm)}.npz"
        r = dict(setting=m["setting"], rep=m["rep"], arm=arm, failed=True, err=f.with_suffix(".err").exists(), missing=not f.exists())
        if f.exists():
            with np.load(f) as z:
                S = np.asarray(z["S"], float); meta = json.loads(str(z["meta"]))
            r.update({k: meta.get(k) for k in ("wall_seconds", "rho_gt1", "rho_nan", "rho_max", "fallback_models", "pivoted_models", "models_nonfinite_W", "models_nonfinite_B",
                                               "S_nonfinite", "B_drift_max", "B_drift_rel_max", "B_sign_mismatch_in_half", "control_max_abs_diff", "cond_T", "oom_retries")})
            if np.isfinite(S).all():                                    # G4: any non-finite score = failed, not imputed
                r["failed"] = False
                r["auroc"], r["auprc"] = score.auc_ap(y, np.abs(S)[off])
        if arm == "shipped":
            r["cached_auroc"], r["cached_auprc"] = cached
        out.append(r)
    return out


def comparator_check(comp, ds):
    """rows.csv.gz vs a rescore of fresh_seeds/cache, one rep per setting (comparators are defined by the cache)."""
    for d in [Path(x) for x in ds if x.endswith("rep12")]:
        m, _, _, genes, W = e9.load(d)
        tru = e9.edges(W, m); off = ~np.eye(len(W), dtype=bool); y = tru[off].astype(np.int8)
        c = h.SIM / "cache/inspre" / d.relative_to(e9.DATA / "inspre")
        row = comp[(comp.setting == m["setting"]) & (comp.rep == 12)].iloc[0]
        path, meta = h.e8.load_fit(c / "inspre_dag.npz")
        A = np.abs(h.e8.embed(path[:, :, meta["best_index"] - 1], meta["kept_genes"], genes))
        a, ap = score.auc_ap(y, A[off])
        assert abs(a - row.ins_sel_auroc) < 1e-9 and abs(ap - row.ins_sel_auprc) < 1e-9, (m["setting"], a, row.ins_sel_auroc)
        path, meta = h.e8.load_fit(c / "adapre.npz")
        v = [score.auc_ap(y, np.abs(h.e8.embed(path[:, :, l], meta["kept_genes"], genes))[off]) for l in range(path.shape[2])]
        assert abs(np.median([x[0] for x in v]) - row.ada_med_auroc) < 1e-9 and abs(np.median([x[1] for x in v]) - row.ada_med_auprc) < 1e-9, m["setting"]


def sims():
    RES.mkdir(exist_ok=True)
    an = sim_analyze()
    stored = pd.read_csv(h.SIM / "results/rows.csv.gz", low_memory=False)
    stored = stored[stored.sim == "inspre"]
    all_rows, all_ds, verdicts, labels, g1, diag_rows = [], [], [], [], [], []
    for blk in ("weak", "contrast"):
        ds = [str(d) for d in h.block(blk)]
        names = sorted({Path(d).parent.name for d in ds})
        with ProcessPoolExecutor(16, mp_context=mp.get_context("fork")) as ex:
            df = pd.DataFrame([r for part in ex.map(score_dataset, ds, chunksize=4) for r in part])
        df.insert(0, "block", blk)
        all_ds.append(df)
        comp = an.path_summary(stored[stored.setting.isin(names)])
        comparator_check(comp, ds)
        cs = comp.groupby("setting").mean(numeric_only=True)
        for k, v in BLOCK_NUMBERS[blk].items():
            assert abs(cs[k].mean() - v) < 6e-4, (blk, k, cs[k].mean())
        ship = df[df.arm == "shipped"].set_index(["setting", "rep"])
        # G1
        for mt in ("auroc", "auprc"):
            e = (ship[mt] - ship[f"cached_{mt}"]).abs().groupby("setting").mean()
            g1.append(dict(block=blk, metric=mt, max_setting_mean_abs_diff=e.max(), worst_setting=e.idxmax(), passed=bool(e.max() <= .002)))
        g1_ok = all(x["passed"] for x in g1 if x["block"] == blk)
        # G2 / G4 / diagnostics
        fail_pct, g2 = {}, {}
        for arm in ROWS:
            q = df[df.arm == arm]
            fail_pct[arm] = 100 * q.failed.mean()
            if arm in ("no_push_abs", "no_push_signed"):
                drift_ok = q.B_drift_rel_max.max() <= .01
                g2[arm] = bool(drift_ok and (arm == "no_push_abs" or q.B_sign_mismatch_in_half.sum() == 0))
            diag_rows.append(dict(block=blk, arm=arm, fits=len(q), failed=int(q.failed.sum()), missing=int(q.missing.sum()), errored=int(q.err.sum()),
                                  S_nonfinite=q.get("S_nonfinite", pd.Series(dtype=float)).sum(), models_nonfinite_W=q.get("models_nonfinite_W", pd.Series(dtype=float)).sum(),
                                  rho_gt1=q.get("rho_gt1", pd.Series(dtype=float)).sum(), rho_max=q.get("rho_max", pd.Series(dtype=float)).max(),
                                  pivoted_models=q.get("pivoted_models", pd.Series(dtype=float)).sum(), B_drift_max=q.get("B_drift_max", pd.Series(dtype=float)).max(),
                                  B_drift_rel_max=q.get("B_drift_rel_max", pd.Series(dtype=float)).max(), B_sign_mismatch=q.get("B_sign_mismatch_in_half", pd.Series(dtype=float)).sum(),
                                  control_max_abs_diff=q.get("control_max_abs_diff", pd.Series(dtype=float)).max(), wall_mean=q.get("wall_seconds", pd.Series(dtype=float)).mean(),
                                  cond_median=q.get("cond_T", pd.Series(dtype=float)).median(), cond_max=q.get("cond_T", pd.Series(dtype=float)).max(), g2_passed=g2.get(arm)))
        # paired per setting: an arm's failed datasets leave its pairs (arm and shipped alike); comparators on the same reps
        rows = {}
        comp_rep = comp.set_index(["setting", "rep"])
        for arm in ROWS:
            ok = df[(df.arm == arm) & ~df.failed].set_index(["setting", "rep"])
            ok = ok.loc[np.array([c in ship.index and not ship.loc[c, "failed"] for c in ok.index], dtype=bool)]
            for mt in ("auroc", "auprc"):
                arm_s = ok[mt].groupby("setting").mean()
                ship_s = ship.loc[ok.index, mt].groupby("setting").mean()
                rows[arm, mt] = dict(arm_s=arm_s, ship_s=ship_s, idx=ok.index)
        ps = {(arm, mt): (float(wilcoxon((rows[arm, mt]["arm_s"] - rows[arm, mt]["ship_s"]).values).pvalue)
                          if (rows[arm, mt]["arm_s"] - rows[arm, mt]["ship_s"]).abs().sum() > 0 else 1.0) for arm in REMOVALS for mt in ("auroc", "auprc")}
        keys = list(ps); adj = dict(zip(keys, holm([ps[k] for k in keys])))
        for arm in ROWS:
            lab = {}
            for mt in ("auroc", "auprc"):
                R = rows[arm, mt]
                delta_s = R["arm_s"] - R["ship_s"]
                r = dict(block=blk, arm=arm, metric=mt, mean=R["arm_s"].mean(), delta=delta_s.mean(), p=ps.get((arm, mt)), p_holm=adj.get((arm, mt)),
                         wins_arm_higher=int((delta_s > 0).sum()), n=len(delta_s), failed_fits=int(df[df.arm == arm].failed.sum()),
                         g1_flag=not g1_ok, g2_flag=(arm in g2 and not g2[arm]))
                sig = arm != "shipped" and adj[arm, mt] < .05
                for c, nm in COMPS:
                    cc = comp_rep.loc[R["idx"], f"{c}_{mt}"]
                    cm = cc.groupby("setting").mean()
                    g = (R["ship_s"] - cm).mean()
                    r.update({nm: cm.mean(), f"gap_shipped_{nm}": g, f"gap_arm_{nm}": (R["arm_s"] - cm).mean()})
                    if c in ("ins_sel", "ada_med"):
                        lb = label(g, r["delta"], sig) if arm != "shipped" else "-"
                        r[f"label_{nm}"] = lb; lab[mt, nm] = lb
                labels.append(r)
            if arm != "shipped":
                adq = {nm: comp_rep.loc[rows[arm, "auprc"]["idx"], f"{c}_auprc"].groupby("setting").mean().mean() for c, nm in (("ins_sel", "inspre_selected"), ("ada_med", "adapre_median"))}
                gate = "FAILED" if fail_pct[arm] > 10 else None
                flag = "" if g1_ok and g2.get(arm, True) else " (UNRELIABLE: G1/G2 flag)"
                v = dict(block=blk, arm=arm, fail_pct=fail_pct[arm])
                if blk == "weak":
                    v["verdict"] = "W1" if arm == "wald" else "Q1"
                    v["result"] = (gate or q1([lab[m, n] for m in ("auroc", "auprc") for n in ("inspre_selected", "adapre_median")])) + flag
                else:
                    v["verdict"] = "W2" if arm == "wald" else "Q2"
                    v["result"] = (gate or q2([lab["auprc", n] for n in ("inspre_selected", "adapre_median")])) + flag
                    v["auroc_labels"] = "/".join(lab["auroc", n] for n in ("inspre_selected", "adapre_median"))
                    v["reaches_inspre_selected_auprc"] = bool(rows[arm, "auprc"]["arm_s"].mean() >= adq["inspre_selected"])
                    v["reaches_adapre_median_auprc"] = bool(rows[arm, "auprc"]["arm_s"].mean() >= adq["adapre_median"])
                verdicts.append(v)
    pd.concat(all_ds).to_csv(RES / "sims_per_dataset.csv", index=False)
    pd.DataFrame(labels).to_csv(RES / "sims_rows.csv", index=False)
    pd.DataFrame(verdicts).to_csv(RES / "sims_verdicts.csv", index=False)
    pd.DataFrame(g1).to_csv(RES / "sims_g1.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(RES / "sims_diagnostics.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 60, "display.max_rows", 200)
    print(pd.DataFrame(g1).to_string()); print(pd.DataFrame(verdicts).to_string())


# ------------------------------------------------------------------------------------------------- real
_CTX = {}
METRICS = ("auroc", "auprc_lift", "p_at_1000", "edge_specific_auroc")
BASELINES = ("inspre-GWPS which.min", "inspre-GWPS D-hat", "inspre-default which.min", "ADAPRE median path", "ADAPRE best path")


def chunk(arm):
    import score_eval as se
    return arm, se.score_reference({arm: _CTX["S"][arm], "shipped": _CTX["S"]["shipped"]}, _CTX["G"], n_boot=_CTX["n_boot"], isem="shipped")[arm]


def real_scores(ds):
    """Arm -> S ("shipped" = the harness control); the stored fit the report scored; genes."""
    genes = h.real_targets(ds)[3]
    with np.load(h.SHIPPED_REAL / ds / "isem_c.npz") as z:
        stored = np.asarray(z["S"], float)
    S = {}
    for arm in ROWS:
        f = HERE / "cache/real" / ds / f"{CACHE_NAME.get(arm, arm)}.npz"
        if f.exists():
            with np.load(f) as z:
                S[arm] = np.asarray(z["S"], float)
    return S, stored, genes


def real_ds(ds, ref, n_boot, workers):
    import score_eval as se
    out_f = RES / f"real_parts/{ds}_{ref}.json"
    S, _, genes = real_scores(ds)
    bad = [a for a, A in S.items() if not np.isfinite(A).all()]                     # G4
    good = {a: A for a, A in S.items() if a not in bad}
    assert "shipped" in good, f"{ds}: control missing or failed"
    _CTX.update(S=good, G=se.references(genes)[ref], n_boot=n_boot)
    arms = list(good)
    with mp.get_context("fork").Pool(min(workers, len(arms))) as p:
        res = dict(p.map(chunk, arms, chunksize=1))
    rows = []
    for a in arms:
        t = res[a]
        r = {"ds": ds, "ref": ref, "arm": a}
        for k in METRICS:
            r[k] = t[k]; r[f"{k}_lo"], r[f"{k}_hi"] = t[f"{k}_ci"]
            if "isem_minus_this" in t:                    # shipped - arm in the scorer; Delta = arm - shipped here
                d = t["isem_minus_this"][k]
                r[f"delta_{k}"] = -d["difference"]; r[f"delta_{k}_lo"], r[f"delta_{k}_hi"] = -d["difference_ci"][1], -d["difference_ci"][0]
            else:
                r[f"delta_{k}"] = 0.
        rows.append(r)
    for a in bad:
        rows.append({"ds": ds, "ref": ref, "arm": a, "nonfinite": int((~np.isfinite(S[a])).sum())})
    out_f.parent.mkdir(exist_ok=True, parents=True)
    out_f.write_text(json.dumps(rows, indent=1, default=float))
    print(ds, ref, "scored", arms, "nonfinite:", bad, flush=True)


def topk_precision(S, G, ks):
    """Precision of the top-k ordered pairs, with paper_tables.matched_k's tie rule (mergesort on -|S|)."""
    import score_eval as se
    n = len(G); off = ~np.eye(n, dtype=bool)
    s = np.where(off, se.pe.absd(S), -np.inf).ravel()
    order = np.argsort(-s, kind="mergesort")
    hits = np.cumsum((G.astype(bool) & off).ravel()[order])
    return {k: hits[k - 1] / k for k in ks}


def baseline_rows(c1, c2, ds, ref):
    a = c1[(c1.ds == ds) & (c1.ref == ref)]; b = c2[(c2.ds == ds) & (c2.ref == ref)]
    q = b[b.method == "inspre (official GWPS settings)"]
    picks = {"inspre-GWPS D-hat": q[q.selection.str.contains("D_bar")].iloc[0], "inspre-GWPS which.min": q[q.selection.str.contains("which.min")].iloc[0]}
    q = b[b.method == "inspre (package defaults)"]
    if len(q):
        picks["inspre-default which.min"] = q[q.selection.str.contains("which.min")].iloc[0]
    return picks, a


def real_report():
    import score_eval as se
    c1 = pd.read_csv(REAL_REPORT / "real_C1.csv", low_memory=False); c2 = pd.read_csv(REAL_REPORT / "real_C2.csv", low_memory=False)
    long, tops, diag, ctrl = [], [], [], []
    for ds in h.SCREENS:
        S, stored, genes = real_scores(ds)
        refs = se.references(genes)
        for arm in ROWS:
            f = HERE / "cache/real" / ds / f"{CACHE_NAME.get(arm, arm)}.json"
            m = json.loads(f.read_text()) if f.exists() else {"missing": True}
            g2 = None
            if arm in ("no_push_abs", "no_push_signed") and f.exists():
                g2 = bool(m["B_drift_rel_max"] <= .01 and (arm == "no_push_abs" or m.get("B_sign_mismatch_in_half") == 0))
            diag.append({"ds": ds, "row": arm, "g2_passed": g2, "error": (f.with_suffix(".err").read_text()[:300] if f.with_suffix(".err").exists() else None), **m})
        same = "shipped" in S and np.array_equal(np.asarray(S["shipped"], np.float32), np.asarray(stored, np.float32))
        ctrl.append(dict(ds=ds, control_max_abs_diff=float(np.abs(S["shipped"] - stored).max()), identical=same))
        for ref in ("STRING", "CORUM"):
            f = RES / f"real_parts/{ds}_{ref}.json"
            if not f.exists():
                continue
            by = {r["arm"]: r for r in json.loads(f.read_text())}
            picks, a = baseline_rows(c1, c2, ds, ref)
            rep = a[a.method == score.ISEM["c"]].iloc[0]
            if same:      # the control equals the fit the report scored, so its point metrics must too
                assert abs(rep.auroc - by["shipped"]["auroc"]) < 1e-9 and abs(rep.p_at_1000 - by["shipped"]["p_at_1000"]) < 1e-9, (ds, ref)
            ada = a[a.method == "ADAPRE"]
            for metric in METRICS:
                base = {k: v[metric] for k, v in picks.items()}
                base["ADAPRE median path"] = ada[metric].median(); base["ADAPRE best path"] = ada[metric].max()
                for arm, r in by.items():
                    row = dict(ds=ds, ref=ref, arm=arm, metric=metric, nonfinite=r.get("nonfinite", 0), value=r.get(metric), lo=r.get(f"{metric}_lo"), hi=r.get(f"{metric}_hi"),
                               delta=r.get(f"delta_{metric}"), delta_lo=r.get(f"delta_{metric}_lo"), delta_hi=r.get(f"delta_{metric}_hi"),
                               g2_flag=any(x["ds"] == ds and x["row"] == arm and x["g2_passed"] is False for x in diag))
                    for nm in BASELINES:
                        v = base.get(nm, np.nan)
                        row[f"baseline_{nm}"] = v
                        row[f"lead_{nm}"] = r[metric] - v if metric in r else np.nan
                        row[f"gap_shipped_{nm}"] = by["shipped"][metric] - v
                    long.append(row)
            nets = {nm: v for nm, v in picks.items() if v.k > 0}
            G = refs[ref]
            ada_k = ada[ada.k > 0]
            for arm, A in S.items():
                if not np.isfinite(A).all():
                    continue
                ks = sorted({int(v.k) for v in nets.values()} | set(ada_k.k.astype(int)))
                P = topk_precision(A, G, ks)
                for nm, v in nets.items():
                    if arm == "shipped" and same:
                        assert abs(P[int(v.k)] - v.c_topk_precision) < 1e-9, (ds, ref, nm)
                    tops.append(dict(ds=ds, ref=ref, arm=arm, network=nm, k=int(v.k), arm_topk_P=P[int(v.k)], network_P=v.precision, delta=P[int(v.k)] - v.precision))
                d = np.array([P[int(x.k)] - x.precision for x in ada_k.itertuples()])
                tops.append(dict(ds=ds, ref=ref, arm=arm, network="ADAPRE path", k=len(d), delta=float(np.median(d)), path_lambdas_won=int((d > 0).sum())))
    L = pd.DataFrame(long)
    L.to_csv(RES / "real_rows.csv", index=False)
    pd.DataFrame(tops).to_csv(RES / "real_topk.csv", index=False)
    pd.DataFrame(diag).to_csv(RES / "real_diagnostics.csv", index=False)
    pd.DataFrame(ctrl).to_csv(RES / "real_control.csv", index=False)
    ver = []
    for arm in ROWS:
        q = L[L.arm == arm]
        for ref in ("STRING", "CORUM"):
            for metric in ("auroc", "p_at_1000"):
                for nm in ("inspre-GWPS which.min", "ADAPRE median path"):
                    z = q[(q.ref == ref) & (q.metric == metric)].set_index("ds").reindex(list(h.SCREENS))
                    g = z[f"gap_shipped_{nm}"].astype(float).where(z.delta.notna())
                    v, nl, c, nd, e = real_verdict(g.values, z.delta.astype(float).values)
                    flag = bool(z.g2_flag.fillna(False).any())
                    ver.append(dict(arm=arm, ref=ref, metric=metric, comparator=nm, verdict=v + (" (UNRELIABLE: G2 flag)" if flag else ""), leads=nl, carried=c, deficits=nd, explained=e))
    pd.DataFrame(ver).to_csv(RES / "real_verdicts.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 500)
    print(pd.DataFrame(ctrl).to_string()); print(pd.DataFrame(ver).to_string())


def counts():
    """Coordinator's full count, before scoring: outputs + A1 FAILED must fill every slot."""
    crash = pd.read_csv(RES / "crashes.csv") if (RES / "crashes.csv").exists() else pd.DataFrame(columns=["arm", "dataset"])
    n_crash = crash.groupby(["arm", "dataset"]).size().to_dict()
    arms = [CACHE_NAME.get(a, a) for a in ROWS if a != "plugin"]
    slots = {"weak": [("sims", str(Path(d).relative_to(e9.DATA / "inspre")), a) for d in h.block("weak") for a in arms],
             "contrast": [("sims", str(Path(d).relative_to(e9.DATA / "inspre")), a) for d in h.block("contrast") for a in arms],
             "real": [("real", ds, a) for ds in h.SCREENS for a in arms],
             "plugin": [("sims", str(Path(d).relative_to(e9.DATA / "inspre")), "plugin") for b in ("weak", "contrast") for d in h.block(b)] + [("real", ds, "plugin") for ds in h.SCREENS]}
    want = {"weak": 1120, "contrast": 1120, "real": 28, "plugin": 324}
    out, failed, reruns = [], [], []
    for name, fits in slots.items():
        assert len(fits) == want[name], (name, len(fits))
        present = 0
        for kind, key, a in fits:
            f = HERE / "cache" / kind / key / f"{a}.npz"
            c = n_crash.get((a, key), 0)
            if f.exists():
                present += 1
                if c:
                    reruns.append(dict(group=name, arm=a, dataset=key, crashes=c))
            else:
                failed.append(dict(group=name, arm=a, dataset=key, crashes=c, err=f.with_suffix(".err").exists()))
        out.append(dict(group=name, expected=want[name], present=present))
    print(pd.DataFrame(out).to_string()); print("missing or FAILED:", failed); print("reruns in the final tables:", reruns)
    pd.DataFrame(reruns).to_csv(RES / "reruns.csv", index=False)
    pd.DataFrame(failed).to_csv(RES / "failed.csv", index=False)
    assert all(x["crashes"] >= 3 or x["err"] for x in failed), "an unfinished fit is neither FAILED nor errored"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sims", "real", "real-report", "counts"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ds", nargs="+", default=list(h.SCREENS))
    a = ap.parse_args()
    if a.cmd == "counts":
        counts()
    elif a.cmd == "sims":
        sims()
    elif a.cmd == "real":
        for ds in a.ds:
            for ref in ("STRING", "CORUM"):
                real_ds(ds, ref, a.n_boot, a.workers)
    else:
        real_report()
