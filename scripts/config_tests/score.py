#!/usr/bin/env python3
"""C1 / C2 tables for the E9 simulations and the real datasets, from this folder's I-SEM caches and the baselines'
stored outputs (read only). Metrics are the repo's own functions: paper_eval.auc_ap, e8_adapre._prf, e9.topk (sims);
score_eval.score_reference, paper_tables.matched_k (real, node bootstrap). Nothing here selects anything by truth;
ORACLE rows are labelled ceilings.

  bash scripts/experiments/run_env.sh python -u scripts/config_tests/score.py [--real hepg2 rpe1 jurkat k562] [--n-boot 500] [--workers 32]

Real (ds, ref) parts are cached in scripts/results/config_tests/real_parts/ with their inputs' mtime and size; a part
is recomputed when any input appears, disappears or changes (delete one to force it).
Outputs: scripts/results/config_tests/{sims,real}_{C1,C2}.csv and summary.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/experiments/e9_author_sims")]
import e9_author_sims as e9  # noqa: E402
from paper_eval import auc_ap  # noqa: E402
from run_sims import main_datasets  # noqa: E402

e8 = e9.e8
RES = ROOT / "scripts/results/config_tests"
PAPER = ROOT / "scripts/results/paper"
PARTS = RES / "real_parts"
RANKERS = {"C1": ("c",), "C2": ("b", "c")}          # I-SEM rankings compared at matched k, per config
ISEM = {"c": "I-SEM (c) defaults", "b": "I-SEM (b) L1 by 5-fold CV", "a": "I-SEM (a) top-k* (ksel)"}
FIXED = {"notears": "NOTEARS (README default)", "golem": "GOLEM-NV (README default)",
         "lingam": "ICA-LiNGAM (Bpruned)", "igsp": "IGSP (alpha 1e-3)"}
INSPRE_THR = 0.015                                  # the authors' GWPS |G| zeroing (inspre_official.py)
SCREENS = ("hepg2", "rpe1", "jurkat", "k562")
BASELINES = {**FIXED, "mean_diff": "Mean difference"}
REAL_ONLY = {"rice": "RICE (authors' Replogle settings)"}   # count model: not run on the (continuous) simulations
SIMS_NA = "N/A (count model; simulations are continuous)"
REAL_BASELINES = {**BASELINES, **REAL_ONLY}
E8_REAL = ("adapre", "inspre_default")             # E8's stored real-data lambda paths scored here (labels: e8.LABEL)


def entry(method, S, net=None, sel=None, l=None, lam=None, rule=None):
    """sel: {config: selection label}. A path lambda belongs to C1 always and to C2 when a C2 rule picks it.
    net = the method's own network (its nonzero edges); rule = sims only, the same estimate under the authors' edge
    rule (e9.edges, as E9 scores it), reported in secondary rule_* columns."""
    return {"method": method, "S": S, "net": net, "sel": sel, "l": l, "lam": lam, "rule": rule}


def path_entries(method, mats, nets, lams, c2, rules=None):
    """mats[l], nets[l] along the path; c2 = {lambda index: C2 selection label}."""
    rules = rules or [None] * len(mats)
    return [entry(method, S, net, {"C1": "whole path", **({"C2": c2[l]} if l in c2 else {})}, l, float(lams[l]), rule)
            for l, (S, net, rule) in enumerate(zip(mats, nets, rules))]


def nonzero(S):
    """The method's own network as it outputs it: nonzero ordered off-diagonal pairs (never thresholded)."""
    return (np.abs(S) > 0) & ~np.eye(len(S), dtype=bool)


def which_min(eps):
    """R's which.min (NaN skipped) as a 0-based index. The JSON metas store NaN as the string "NaN"."""
    return int(np.nanargmin(np.asarray(eps, float)))


def both(label):
    return {"C1": label, "C2": label}


# ------------------------------------------------------------------------------------------------- sims
def sim_entries(d):
    m, _, _, genes, W = e9.load(d)
    rel = d.relative_to(e9.DATA)
    mine, base = RES / "sims" / rel, e9.CACHE / rel
    E, missing, rank = [], [], {}
    for v in ("c", "b", "a"):
        f = mine / f"isem_{v}.npz"
        if not f.exists():
            missing.append(f"I-SEM ({v})"); continue
        with np.load(f) as z:
            meta = json.loads(str(z["meta"]))
            if v == "a":
                E.append(entry(ISEM[v], None, z["edges"], {"C2": "held-out k* (E7 ksel)"}))
            else:
                rank[v] = z["S"]
                E.append(entry(ISEM[v], z["S"], None, both("defaults") if v == "c" else
                               {"C2": "min 5-fold CV error"}, lam=meta.get("lambda")))
    edges = lambda S: e9.edges(S, m)
    f = mine / "inspre_dag_path.npz"
    if f.exists():
        with np.load(f) as z:
            path, meta = z["W_path"], json.loads(str(z["meta"]))
        diff = meta.get("max_abs_diff_vs_e9_cache")
        best = meta["best_index"] - 1
        if diff != 0.0:                       # the C2 row must be E9's official fit; anything else is not scored
            missing.append(f"inspre path does not reproduce E9's inspre_dag.npz (max |diff| {diff}); path not scored")
        else:
            mats = [np.abs(e8.embed(path[:, :, l], meta["kept_genes"], genes)) for l in range(path.shape[2])]
            E += path_entries("inspre (official, DAG)", mats, [nonzero(S) for S in mats], meta["lambda"],
                              {best: "which.min eps_hat_G"}, [edges(S) for S in mats])
    else:
        missing.append("inspre path (run_sims.py --variants inspre_path)")
    f = base / "adapre.npz"
    if f.exists():
        path, meta = e8.load_fit(f)
        mats = [e8.embed(path[:, :, l], meta["kept_genes"], genes) for l in range(path.shape[2])]
        wm = which_min(meta["eps_hat_G"])
        assert wm == e8.rules(meta)["min_eps_g"]
        E += path_entries("ADAPRE", mats, [nonzero(S) for S in mats], meta["lambda"],
                          {wm: "which.min eps_hat_G"}, [edges(S) for S in mats])
    else:
        missing.append("ADAPRE")
    for meth, label in BASELINES.items():
        f = base / f"{meth}.npz"
        if not f.exists():
            missing.append(label); continue
        S = np.load(f)["S"]
        dense = meth == "mean_diff"
        E.append(entry(label, S, None if dense else nonzero(S), both("no selection step"), rule=None if dense else edges(S)))
    return m, W, E, rank, missing


def best_k_f1(S, tru):
    """ORACLE: the best F1 over every top-k of S against the truth (a ceiling, never a method's choice)."""
    n = len(S); off = ~np.eye(n, dtype=bool)
    o = np.argsort(-np.abs(S)[off], kind="mergesort")
    tp = np.cumsum(tru[off][o]); k = np.arange(1, len(o) + 1)
    f1 = 2 * tp / (k + tru[off].sum())
    return float(f1.max()), int(f1.argmax() + 1)


def sim_rows(d):
    m, W, E, rank, missing = sim_entries(d)
    tru = e9.edges(W, m); off = ~np.eye(len(W), dtype=bool)
    base = {"sim": m["sim"], "setting": m["setting"], "rep": m["rep"], "n_true_edges": int(tru.sum())}
    rows = []
    for e in E:
        r0 = {"method": e["method"], "lambda_index": e["l"], "lambda": e["lam"]}
        if e["S"] is not None:
            r0["auroc"], r0["auprc"] = auc_ap(tru[off].astype(np.int8), np.abs(e["S"])[off])
        if e["net"] is not None:
            r0.update(e8._prf(e["net"] & off, tru))
        if e["rule"] is not None:
            r0.update({f"rule_{k}": x for k, x in e8._prf(e["rule"], tru).items()})
        for cfg, sel in e["sel"].items():
            r = {"config": cfg, **base, "selection": sel, **r0}
            if e["net"] is not None and r["k"] > 0:
                for v in RANKERS[cfg]:
                    if v in rank and not (e["method"] == ISEM["a"] and v == "c"):   # (a) = c's top-k*: Δ 0 by construction
                        r.update({f"{v}_topk_{k}": x for k, x in e9.topk(rank[v], tru, r["k"]).items() if k != "k"})
            rows.append(r)
    for v, S in rank.items():
        f1, k = best_k_f1(S, tru)
        for cfg in ("C1", "C2") if v == "c" else ("C2",):
            rows.append({"config": cfg, **base, "method": f"{ISEM[v]} ORACLE best k (ceiling)",
                         "selection": "ORACLE (truth)", "k": k, "f1": f1})
    return rows, [(m["sim"], m["setting"], m["rep"], x) for x in missing]


# ------------------------------------------------------------------------------------------------- real
def real_inputs(ds):
    """Every file a real (ds, ref) part reads, present or not."""
    return [RES / "real" / ds / f"isem_{v}.npz" for v in ("c", "b", "a")] + \
           [PAPER / f"e1b_inspre_official/{ds}{x}" for x in ("_Gpath.npz", ".json", ".npy")] + \
           [PAPER / f"e8_adapre/{ds}_{m}{x}" for m in E8_REAL for x in ("_Gpath.npz", ".json", ".npy")] + \
           [ROOT / f"results/{meth}_adj_{ds}.npy" for meth in REAL_BASELINES]


def fingerprint(ds, n_boot):
    st = lambda f: [f.stat().st_mtime_ns, f.stat().st_size] if f.exists() else None
    return {"n_boot": n_boot, **{str(f.relative_to(ROOT)): st(f) for f in real_inputs(ds)}}


def real_status(ds):
    """Missing / pending inputs of a dataset (listed for every dataset, scored or not)."""
    miss = [f"I-SEM ({v}) not fitted (run_real.py --ds {ds} --variants {v})"
            for v in ("c", "b", "a") if not (RES / "real" / ds / f"isem_{v}.npz").exists()]
    miss += [f"{label} pending: results/{meth}_adj_{ds}.npy" for meth, label in REAL_BASELINES.items()
             if not (ROOT / f"results/{meth}_adj_{ds}.npy").exists()]
    miss += [f"{e8.LABEL[m]} pending: scripts/results/paper/e8_adapre/{ds}_{m}{x}" for m in E8_REAL
             for x in ("_Gpath.npz", ".json", ".npy") if not (PAPER / f"e8_adapre/{ds}_{m}{x}").exists()]
    return miss


def e8_path(ds, method, genes, d=PAPER / "e8_adapre"):
    """E8's stored real-data lambda path, embedded exactly as e8.embed (the R wrapper already wrote W[i, j] = effect
    of j on i = t(G); kept genes zero-padded to the panel, non-finite -> 0, diagonal 0). Returns (mats, meta, rules)
    with rules["min_eps_g"] = which.min(eps_hat_G); asserts the stored D_bar .npy is reproduced."""
    path, _ = e8.load_fit(d / f"{ds}_{method}_Gpath.npz")
    meta = json.loads((d / f"{ds}_{method}.json").read_text())
    assert meta["panel_genes"] == genes, f"{ds}_{method}: panel differs"
    mats = [e8.embed(path[:, :, l], meta["kept_genes"], genes) for l in range(path.shape[2])]
    r = e8.rules(meta)
    r["min_eps_g"] = which_min(meta["eps_hat_G"])     # e8.rules' argmin sees "NaN" strings; must agree with nanargmin
    assert r["min_eps_g"] == meta["selection"]["min_eps_g"]
    assert np.array_equal(mats[r["dbar"]], np.load(d / f"{ds}_{method}.npy")), f"E8 {ds}_{method}.npy not reproduced"
    return mats, meta, r


def real_entries(ds, genes):
    E, rank = [], {}
    for v in ("c", "b", "a"):
        f = RES / "real" / ds / f"isem_{v}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            meta = json.loads(str(z["meta"]))
            assert meta["genes"] == genes, f"{f}: gene order differs from the paper panel"
            if v == "a":
                E.append(entry(ISEM[v], None, z["edges"], {"C2": "held-out k* (E7 ksel)"}))
            else:
                rank[v] = z["S"]
                E.append(entry(ISEM[v], z["S"], None, both("defaults") if v == "c" else
                               {"C2": "min 5-fold CV error"}, lam=meta.get("lambda")))
    # inspre official (E1b): the stored path, the authors' |G| < 0.015 zeroing, zero-padded to the panel
    with np.load(PAPER / f"e1b_inspre_official/{ds}_Gpath.npz") as z:
        path, eps, lams, kept = z["W_path"], z["eps_hat_G"], z["lambda_"], [str(g) for g in z["kept_genes"]]
    sel = json.loads((PAPER / f"e1b_inspre_official/{ds}.json").read_text())["selection"]
    assert np.isfinite(path).all()
    mats = []
    for l in range(path.shape[2]):
        Wl = path[:, :, l].copy(); Wl[np.abs(Wl) < INSPRE_THR] = 0.0
        mats.append(e8.embed(Wl, kept, genes))
    official = sel["chosen_index"] - 1
    assert np.array_equal(mats[official], np.load(PAPER / f"e1b_inspre_official/{ds}.npy")), "E1b .npy not reproduced"
    c2 = {which_min(eps): "which.min eps_hat_G"}
    c2[official] = " + ".join(filter(None, [c2.get(official), "D_bar (E1b official GWPS rule; extra row)"]))
    E += path_entries("inspre (official GWPS settings)", mats, [S != 0 for S in mats], lams, c2)
    # ADAPRE and inspre at its package defaults (E8): stored paths, e8.embed. C1 = whole path, C2 = which.min eps_hat_G
    # (+ ADAPRE's E8 primary D_bar rule as a labelled extra row). A missing fit is listed by real_status.
    for m in E8_REAL:
        if not all((PAPER / f"e8_adapre/{ds}_{m}{x}").exists() for x in ("_Gpath.npz", ".json", ".npy")):
            continue
        mats, meta, r = e8_path(ds, m, genes)
        c2 = {r["min_eps_g"]: "which.min eps_hat_G"}
        if m == "adapre":
            c2[r["dbar"]] = " + ".join(filter(None, [c2.get(r["dbar"]), "D_bar (E8 primary rule; extra row)"]))
        E += path_entries(e8.LABEL[m], mats, [S != 0 for S in mats], meta["lambda"], c2)
    for meth, label in REAL_BASELINES.items():
        f = ROOT / f"results/{meth}_adj_{ds}.npy"
        if not f.exists():
            continue                                           # listed by real_status
        S = np.load(f)
        assert S.shape == (len(genes),) * 2, f"{f}: shape {S.shape}"
        E.append(entry(label, S, None if meth == "mean_diff" else np.abs(S) > 0, both("no selection step")))
    return E, rank


_CTX = {}      # real_part's inputs, inherited by forked workers


def _score_chunk(ms):
    """score_reference on methods ms (+ I-SEM (c) for the paired differences). Each bootstrap draw depends only on
    the seed, never on which methods are scored, so a chunk's rows equal the all-methods call's."""
    import score_eval as se
    S, isem = _CTX["S"], _CTX["isem"]
    sub = {m: S[m] for m in ms} | ({isem: S[isem]} if isem else {})
    out = se.score_reference(sub, _CTX["G"], n_boot=_CTX["n_boot"], **({"isem": isem} if isem else {}))
    return {m: out[m] for m in ms}


def _mk_job(job):
    import paper_tables as pt
    res = {}
    pt.matched_k(res, _CTX["rank"][job[1]], _CTX["nets"][job[0]], _CTX["G"], _CTX["n_boot"])
    return res


def _map(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as p:
        return p.map(fn, jobs, chunksize=1)


def real_part(ds, ref, n_boot, workers=1):
    import paper_tables as pt
    import score_eval as se
    genes = [str(g) for g in pt.load(ds)["gene_names"]]
    G = se.references(genes)[ref]
    fp = fingerprint(ds, n_boot)
    E, rank = real_entries(ds, genes)
    key = lambda e: e["method"] + ("" if e["l"] is None else f" @lambda[{e['l']}]")
    S = {key(e): e["S"] for e in E if e["S"] is not None}
    isem = ISEM["c"] if ISEM["c"] in S else None
    off = ~np.eye(len(G), dtype=bool)
    nets, jobs = {}, []                                  # every (network, I-SEM ranking) matched-k comparison
    for e in E:
        net = None if e["net"] is None else e["net"] & off
        if net is None or not net.any():
            continue
        nets[key(e)] = net
        for cfg in e["sel"]:
            for v in (RANKERS[cfg] if e["method"] != ISEM["a"] else ("c",)):
                if v in rank and (key(e), v) not in jobs:
                    jobs.append((key(e), v))
    _CTX.update(S=S, G=G, n_boot=n_boot, isem=isem, rank=rank, nets=nets)
    keys = list(S)
    tf = {}
    for part in _map(_score_chunk, [c for c in (keys[i::workers] for i in range(workers)) if c], workers):
        tf.update(part)
    mk = dict(zip(jobs, _map(_mk_job, jobs, workers)))
    rows = []
    for e in E:
        r0 = {"method": e["method"], "lambda_index": e["l"], "lambda": e["lam"]}
        if e["S"] is not None:
            t = tf[key(e)]
            for k in ("auroc", "auprc", "auprc_lift", "p_at_1000", "edge_specific_auroc"):
                r0[k] = t[k]; r0[f"{k}_lo"], r0[f"{k}_hi"] = t[f"{k}_ci"]
            if "isem_minus_this" in t:
                d = t["isem_minus_this"]["auroc"]
                r0["c_minus_auroc"] = d["difference"]; r0["c_minus_auroc_lo"], r0["c_minus_auroc_hi"] = d["difference_ci"]
        own = None
        for cfg, sel in e["sel"].items():
            r = {"config": cfg, "ds": ds, "ref": ref, "selection": sel, **r0}
            if e["net"] is not None:
                net = e["net"] & off
                for v in (RANKERS[cfg] if e["method"] != ISEM["a"] else ("c",)) if net.any() else ():
                    if v not in rank:
                        continue
                    res = mk[key(e), v]
                    own = res[pt.INSPRE]
                    if e["method"] == ISEM["a"]:
                        break                                # its own network only (= c's top-k*)
                    i, dd = res[pt.MATCHED_K], res[pt.MATCHED_K]["isem_minus_inspre_official"]["precision_at_k"]
                    r.update({f"{v}_topk_precision": i["precision_at_k"], f"{v}_topk_recall": i["recall_at_k"],
                              f"{v}_topk_f1": f1(i["precision_at_k"], i["recall_at_k"]),
                              f"{v}_minus_precision": dd["difference"], f"{v}_minus_precision_lo": dd["difference_ci"][0],
                              f"{v}_minus_precision_hi": dd["difference_ci"][1]})
                if own is not None:
                    r.update({"k": own["k"], "precision": own["precision_at_k"], "precision_lo": own["precision_at_k_ci"][0],
                              "precision_hi": own["precision_at_k_ci"][1], "recall": own["recall_at_k"],
                              "f1": f1(own["precision_at_k"], own["recall_at_k"])})
                else:
                    r["k"] = int(net.sum())                   # empty network: precision undefined
            rows.append(r)
    return {"rows": rows, "inputs": fp, "meta": se.meta({"n_boot": n_boot})}


def f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


# ---------------------------------------------------------------------------------------------- summary
def reported(df):
    """The reporting rule (README "What is reported"): I-SEM (c) is the method; (a) and (b) are scratch tests. Every
    method is reported in C1; C2 only for methods whose authors prescribe a lambda-selection step (inspre), with
    I-SEM (c) as the reference row. Other rows go to *_scratch.csv."""
    base = df.method.str.replace(" ORACLE best k (ceiling)", "", regex=False)
    c2 = base.str.startswith("inspre") | (base == ISEM["c"])
    return ~base.isin([ISEM["a"], ISEM["b"]]) & ((df.config == "C1") | c2)


def write_csvs(df, kind):
    """{kind}_{C1,C2}.csv = reported rows (I-SEM (b) columns dropped); {kind}_scratch.csv = the rest."""
    keep = reported(df)
    for cfg in ("C1", "C2"):
        d = df[keep & (df.config == cfg)].dropna(axis=1, how="all")
        d.drop(columns=[c for c in d if c.startswith("b_")]).to_csv(RES / f"{kind}_{cfg}.csv", index=False)
    df[~keep].dropna(axis=1, how="all").to_csv(RES / f"{kind}_scratch.csv", index=False)
    return df[keep]


def fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def sim_summary(df):
    L = []
    for cfg in ("C1", "C2"):
        d = df[df.config == cfg]
        fitted = set(map(tuple, d.loc[d.method == ISEM["c"], ["sim", "setting", "rep"]].values))
        d = d[[t in fitted for t in map(tuple, d[["sim", "setting", "rep"]].values)]]    # reps with this run's I-SEM
        d = d.assign(l=np.where(d.selection == "whole path", d.lambda_index, -1))
        L += [f"### Simulations, {cfg}", "",
              "Means over the reps where I-SEM (c) is fitted (n). Network = the method's own nonzero edges, as it outputs them "
              "(the CSV's rule_* columns score the same estimate under the authors' edge rule, e9.edges, as E9 does). "
              "Path rows: lambda position l (the lambda value differs by rep). Δ = I-SEM top-k F1 − method F1 at the "
              "method's own k, paired over reps, 95% t-CI. F1 of networks of different sizes is size-dependent.", ""]
        rk = ("c",)
        head = ["setting", "method", "selection", "l", "k", "P", "R", "F1"] + \
               sum([[f"({v}) top-k F1", f"Δ({v}) [CI]"] for v in rk], []) + ["AUROC", "AUPRC", "n"]
        L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for (sim, setting, meth, sel, l), g in d.groupby(["sim", "setting", "method", "selection", "l"],
                                                         dropna=False, sort=False):
            row = [f"{sim}/{setting}", meth, sel, "" if l < 0 else str(int(l)), fmt(g.k.mean(), 0) if "k" in g else "",
                   fmt(g.get("precision", pd.Series(dtype=float)).mean()), fmt(g.get("recall", pd.Series(dtype=float)).mean()),
                   fmt(g.get("f1", pd.Series(dtype=float)).mean())]
            for v in rk:
                c = f"{v}_topk_f1"
                if c in g and g[c].notna().any():
                    dl = (g[c] - g.f1).dropna(); lo, hi = e9.tci(dl)
                    row += [fmt(g[c].mean()), f"{dl.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"]
                else:
                    row += ["", ""]
            row += [fmt(g.get("auroc", pd.Series(dtype=float)).mean()), fmt(g.get("auprc", pd.Series(dtype=float)).mean()), str(len(g))]
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    return L


def real_summary(df):
    L = []
    for cfg in ("C1", "C2"):
        d = df[df.config == cfg]
        rk = ("c",)
        L += [f"### Real datasets, {cfg}", "",
              "Network = the method's own nonzero edges (inspre (official GWPS settings) after the authors' |G| < 0.015 "
              "zeroing; inspre (package defaults) and ADAPRE unthresholded). P = precision vs "
              "the reference (undirected, either order), 95% node-bootstrap CI. Δ = I-SEM top-k precision − method "
              "precision at the method's k, paired node bootstrap. F1 at different network sizes is size-dependent.", ""]
        head = ["ds", "ref", "method", "selection", "l", "k", "P [CI]", "R", "F1"] + \
               sum([[f"({v}) top-k P", f"ΔP({v}) [CI]"] for v in rk], []) + ["AUROC [CI]", "AUPRC lift"]
        L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for _, r in d.iterrows():
            g = lambda c: r[c] if c in r and pd.notna(r[c]) else None
            row = [r.ds, r.ref, r.method, r.selection, "" if g("lambda_index") is None else str(int(r.lambda_index)),
                   "" if g("k") is None else str(int(r.k)),
                   "" if g("precision") is None else f"{r.precision:.3f} [{r.precision_lo:.3f}, {r.precision_hi:.3f}]",
                   fmt(g("recall")), fmt(g("f1"))]
            for v in rk:
                if g(f"{v}_topk_precision") is None:
                    row += ["", ""]
                else:
                    row += [f"{r[f'{v}_topk_precision']:.3f}", f"{r[f'{v}_minus_precision']:+.3f} "
                            f"[{r[f'{v}_minus_precision_lo']:+.3f}, {r[f'{v}_minus_precision_hi']:+.3f}]"]
            row += ["" if g("auroc") is None else f"{r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]", fmt(g("auprc_lift"), 2)]
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="*", default=list(SCREENS))
    ap.add_argument("--refs", nargs="+", default=["STRING", "CORUM"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--skip-sims", action="store_true")
    ap.add_argument("--workers", type=int, default=32, help="processes per real part (results identical to 1)")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    L = ["# config_tests summary (generated by score.py; do not edit)", "",
         "Configs, scoring and caveats: scripts/config_tests/README.md. ORACLE rows are ceilings chosen against the "
         "truth, never a method's choice.", "",
         "Reported: I-SEM (c), the shipped pipeline, against every baseline in C1 (each method as its authors ship it); "
         "C2 only for inspre, whose authors prescribe a lambda-selection step. I-SEM (a)/(b) and ADAPRE's CV-chosen "
         "lambda (no rule from its authors) are scratch tests, kept in {sims,real}_scratch.csv.", ""]
    missing = []
    if not a.skip_sims:
        res = [sim_rows(d) for d in main_datasets()]
        sims = pd.DataFrame([r for rows, _ in res for r in rows])
        missing += [f"sims {s}/{t}/rep{r}: {x}" for _, miss in res for s, t, r, x in miss]
        L += sim_summary(write_csvs(sims, "sims"))
        L += [f"- {label}: {SIMS_NA}" for label in REAL_ONLY.values()] + [""]
    real = []
    PARTS.mkdir(parents=True, exist_ok=True)
    for ds in SCREENS:
        missing += [f"real {ds}: {x}" for x in real_status(ds)]
        if ds not in a.real:
            missing.append(f"real {ds}: not scored in this run (not in --real)"); continue
        if not (RES / "real" / ds / "isem_c.npz").exists():
            missing.append(f"real {ds}: real tables skipped until I-SEM (c) is fitted"); continue
        for ref in a.refs:
            f = PARTS / f"{ds}_{ref}.json"
            if not f.exists() or json.loads(f.read_text()).get("inputs") != fingerprint(ds, a.n_boot):
                f.write_text(json.dumps(real_part(ds, ref, a.n_boot, a.workers), indent=1, default=float, allow_nan=True))
                print("wrote", f, flush=True)
            real += json.loads(f.read_text())["rows"]
    if real:
        L += real_summary(write_csvs(pd.DataFrame(real), "real"))
    from collections import Counter
    cnt = Counter(x.split(": ", 1)[1] if x.startswith("sims") else x for x in missing)
    L += ["## Missing / pending", ""] + ([f"- {x}" + (f" ({n} datasets)" if n > 1 else "") for x, n in sorted(cnt.items())]
                                        or ["- nothing"])
    (RES / "summary.md").write_text("\n".join(L) + "\n")
    print(f"wrote {RES / 'summary.md'}; {len(missing)} missing/pending items", flush=True)


if __name__ == "__main__":
    main()
