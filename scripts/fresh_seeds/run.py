#!/usr/bin/env python3
"""Confirmatory run for the simulated-data claim. Seeds 12-21; the fits are skipped when their output exists.

  bash run.sh check                     # generator bytes and method reproduction against stored fits
  bash run.sh fit --methods inspre_dag adapre mean_diff --workers 12
  bash run.sh fit --methods isem --workers 12
  bash run.sh score
  bash run.sh real
"""
import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import common as c

e9, e8, score = c.e9, c.e8, c.score
RES = c.HERE / "results"
KINDS = ("inspre", "adapre", "sweep")


def check():
    """Seed 1 regenerates byte-identical data; each method reproduces its stored fit on one stored dataset."""
    bad = 0
    for kind in ("inspre", "adapre"):
        n = 0
        for s in sorted((e9.DATA / kind).iterdir()):
            for f in ("Y.bin", "G.bin", "targets.txt"):
                bad += (s / "rep1" / f).read_bytes() != (c.STORED / "data" / kind / s.name / "rep1" / f).read_bytes()
            n += 1
        print(f"{kind}: {n} settings regenerated at seed 1, {bad} files differ so far", flush=True)
    assert bad == 0
    for s in (e9.DATA / "inspre").iterdir():      # the confirmatory seeds really are new data
        a = (s / "rep12" / "Y.bin").read_bytes()
        assert a != (c.STORED / "data/inspre" / s.name / "rep1" / "Y.bin").read_bytes()
    for rel in (Path("inspre/rand_cycles_confound_7/rep1"), Path("adapre/demo_set2/rep1")):
        d, cache = c.STORED / "data" / rel, c.HERE / "check" / rel
        m, _, _, genes, W = e9.load(d)
        off = ~np.eye(len(W), dtype=bool)
        fits = {x: c.fit(d, x, cache / f"{x}.npz") if not (cache / f"{x}.npz").exists() else None for x in ("adapre", "inspre_dag", "isem", "mean_diff")}
        a, _ = e8.load_fit(cache / "adapre.npz")
        b, _ = e8.load_fit(c.STORED / "cache" / rel / "adapre.npz")
        da = float(np.nanmax(np.abs(a - b)))
        p, mn = e8.load_fit(cache / "inspre_dag.npz")
        Sp = np.abs(e8.embed(p[:, :, mn["best_index"] - 1], mn["kept_genes"], genes))
        dn = float(np.abs(Sp - np.load(c.STORED / "cache" / rel / "inspre_dag.npz")["S"]).max())
        md = float(np.abs(np.load(cache / "mean_diff.npz")["S"] - np.load(c.STORED / "cache" / rel / "mean_diff.npz")["S"]).max())
        Si = np.load(cache / "isem.npz")["S"]
        Sc = np.load(c.ROOT / "scripts/results/config_tests/sims" / rel / "isem_c.npz")["S"]
        rho = float(np.corrcoef(np.abs(Si)[off], np.abs(Sc)[off])[0, 1])
        print(f"{rel}: ADAPRE cv0 vs stored max|diff| {da}; inspre {dn}; mean diff {md}; I-SEM GPU vs stored CPU corr {rho:.6f}", flush=True)
        assert da == 0 and dn == 0 and md < 1e-5 and rho > 0.999
    print("check OK", flush=True)


def fit(methods, workers):
    ds = [d for k in KINDS for d in c.datasets(k)]
    ctx = "spawn" if "isem" in methods else "fork"
    jobs = [(d, m) for d in ds for m in methods]
    fails = 0
    with ProcessPoolExecutor(workers, mp_context=mp.get_context(ctx)) as ex:
        for f in as_completed([ex.submit(c.job, d, m) for d, m in jobs]):
            r = f.result()
            fails += r.startswith("FAIL")
            print(r, flush=True)
    print(f"done: {len(jobs)} jobs, {fails} failed", flush=True)


def strength(d):
    m, Y, labels, genes, _ = e9.load(d)
    labels = np.asarray(labels)
    ctl = labels == "control"
    sd = Y[ctl].std(0, ddof=1)
    rows = []
    for j, g in enumerate(genes):
        k = labels == g
        if k.sum() < 2:
            continue
        sh = Y[k, j].mean() - Y[ctl, j].mean()
        se = np.sqrt(Y[k, j].var(ddof=1) / k.sum() + sd[j] ** 2 / ctl.sum())
        rows.append((abs(sh) / sd[j], abs(sh) / se, abs(sh) / sd[j] * np.sqrt(k.sum())))
    a = np.array(rows)
    return dict(sim=m["sim"], setting=m["setting"], rep=m["rep"], d=np.median(a[:, 0]), t=np.median(a[:, 1]), d_sqrt_n=np.median(a[:, 2]),
                min_t=a[:, 1].min(), frac_t_lt3=(a[:, 1] < 3).mean(), frac_dsqrtn_lt3=(a[:, 2] < 3).mean())


def filter_check(d):
    """Sweep only: every method rescored on the genes ADAPRE kept, separating gene loss from ratio noise."""
    m, _, _, genes, W = e9.load(d)
    cdir = e9.CACHE / d.relative_to(e9.DATA)
    if not (cdir / "adapre.npz").exists() or not (cdir / "isem.npz").exists():
        return []
    path, meta = e8.load_fit(cdir / "adapre.npz")
    keep = np.isin(genes, meta["kept_genes"])
    if keep.sum() < 3:
        return []
    tru = e9.edges(W, m)[np.ix_(keep, keep)]
    off = ~np.eye(keep.sum(), dtype=bool)
    if not tru[off].any():
        return []
    au = lambda S: score.auc_ap(tru[off].astype(np.int8), np.abs(S)[np.ix_(keep, keep)][off])
    rows = [dict(method="isem", l=-1, auroc=au(np.load(cdir / "isem.npz")["S"])[0], auprc=au(np.load(cdir / "isem.npz")["S"])[1])]
    for l in range(path.shape[2]):
        x = au(e8.embed(path[:, :, l], meta["kept_genes"], genes))
        rows.append(dict(method="adapre", l=l, auroc=x[0], auprc=x[1]))
    if (cdir / "inspre_dag.npz").exists():
        p, mn = e8.load_fit(cdir / "inspre_dag.npz")
        x = au(np.abs(e8.embed(p[:, :, mn["best_index"] - 1], mn["kept_genes"], genes)))
        rows.append(dict(method="inspre_selected", l=-1, auroc=x[0], auprc=x[1]))
    return [{**r, "setting": m["setting"], "rep": m["rep"], "K": int(keep.sum())} for r in rows]


def weak_sources(d):
    """Sweep mixed cell: out-edges of weak vs strong knockdown sources, one set of pairs per dataset."""
    m, _, _, genes, W = e9.load(d)
    b = np.asarray(m["params"]["int_beta"])[: len(genes)]
    tru = e9.edges(W, m)
    cdir = e9.CACHE / d.relative_to(e9.DATA)
    S = {"isem": np.abs(np.load(cdir / "isem.npz")["S"])}
    if (cdir / "inspre_dag.npz").exists():        # a crashed inspre fit is left out of inspre's pool, never imputed
        p, mn = e8.load_fit(cdir / "inspre_dag.npz")
        S["inspre_selected"] = np.abs(e8.embed(p[:, :, mn["best_index"] - 1], mn["kept_genes"], genes))
    pa, ma = e8.load_fit(cdir / "adapre.npz")
    ada = [np.abs(e8.embed(pa[:, :, l], ma["kept_genes"], genes)) for l in range(pa.shape[2])]
    rows = []
    for grp, src in (("weak", np.flatnonzero(b == -0.25)), ("strong", np.flatnonzero(b == -2))):
        keep = [(i, j) for j in src for i in range(len(genes)) if i != j]
        ii, jj = np.array(keep).T
        y = tru[ii, jj].astype(np.int8)
        row = dict(rep=m["rep"], group=grp, n_pairs=len(y), n_true=int(y.sum()))
        for k, A in S.items():
            row[k] = A[ii, jj]
        row["adapre_path"] = [A[ii, jj] for A in ada]
        row["y"] = y
        rows.append(row)
    return rows


def score_one(d):
    score.sim_entries = c.entries
    rows, missing = score.sim_rows(d)
    import directness as dn
    m, W, E, _, _ = c.entries(d)
    direct, indirect, unconn, eff = dn.classes(W, e9.edges(W, m))
    drows = [{"sim": m["sim"], "setting": m["setting"], "rep": m["rep"], "method": e["method"], "lambda_index": e["l"],
              "selected": bool(e["sel"] and "C2" in e["sel"] and e["l"] is not None), **dn.metrics(e["S"], direct, indirect, unconn, eff)}
             for e in E if e["S"] is not None]
    meta = dict(sim=m["sim"], setting=m["setting"], rep=m["rep"], n_true=int(e9.edges(W, m).sum()),
                **{k: v for k, v in e9.describe(json.loads((d / "meta.json").read_text())).items() if k != "main"} if m["sim"] != "sweep" else {})
    return rows, missing, drows, meta, strength(d), (filter_check(d) if m["sim"] == "sweep" else [])


def score_all():
    RES.mkdir(exist_ok=True)
    ds = [d for k in KINDS for d in c.datasets(k)]
    rows, missing, drows, metas, st, fc = [], [], [], [], [], []
    with ProcessPoolExecutor(48, mp_context=mp.get_context("fork")) as ex:
        for r, mi, dr, me, s, f in ex.map(score_one, ds, chunksize=4):
            rows += r; missing += mi; drows += dr; metas.append(me); st.append(s); fc += f
    pd.DataFrame(rows).to_csv(RES / "rows.csv.gz", index=False)
    pd.DataFrame(drows).to_csv(RES / "directness.csv.gz", index=False)
    pd.DataFrame(metas).to_csv(RES / "datasets.csv", index=False)
    pd.DataFrame(st).to_csv(RES / "strength.csv", index=False)
    pd.DataFrame(fc).to_csv(RES / "filter_check.csv", index=False)
    pd.DataFrame(missing, columns=["sim", "setting", "rep", "method"]).to_csv(RES / "missing_fits.csv", index=False)
    wall = []
    for d in ds:
        for f in (e9.CACHE / d.relative_to(e9.DATA)).glob("*.npz"):
            with np.load(f) as z:
                mt = json.loads(str(z["meta"]))
            wall.append(dict(sim=d.parts[-3], setting=d.parts[-2], rep=d.name, method=f.stem, wall=mt["wall_seconds"], K=mt.get("K"),
                             dropped=mt.get("n_prefilter_dropped"), oom_retries=mt.get("oom_retries"), n_models=mt.get("n_models")))
    pd.DataFrame(wall).to_csv(RES / "fits.csv", index=False)
    mix = [r for d in e9.DATA.glob("sweep/mixed/rep*/meta.json") for r in weak_sources(d.parent)]
    out = []
    for grp in ("weak", "strong"):
        g = [r for r in mix if r["group"] == grp]
        y = np.concatenate([r["y"] for r in g])
        o = dict(group=grp, reps=len(g), n_pairs=len(y), n_true=int(y.sum()))
        o["isem_auroc"], o["isem_auprc"] = score.auc_ap(y, np.concatenate([r["isem"] for r in g]))
        ok = [r for r in g if "inspre_selected" in r]
        o["inspre_reps"] = len(ok)
        o["inspre_selected_auroc"], o["inspre_selected_auprc"] = score.auc_ap(np.concatenate([r["y"] for r in ok]), np.concatenate([r["inspre_selected"] for r in ok]))
        ok_y = np.concatenate([r["y"] for r in ok])          # I-SEM on the same reps, for a like-for-like comparison
        o["isem_same_reps_auroc"], o["isem_same_reps_auprc"] = score.auc_ap(ok_y, np.concatenate([r["isem"] for r in ok]))
        L = len(g[0]["adapre_path"])
        path = [score.auc_ap(y, np.concatenate([r["adapre_path"][l] for r in g])) for l in range(L)]
        o["adapre_med_auroc"], o["adapre_med_auprc"] = np.median([p[0] for p in path]), np.median([p[1] for p in path])
        o["adapre_best_auroc"], o["adapre_best_auprc"] = max(p[0] for p in path), max(p[1] for p in path)
        out.append(o)
    pd.DataFrame(out).to_csv(RES / "mixed_weak_sources.csv", index=False)
    print(f"{len(rows)} rows, {len(missing)} missing fits", flush=True)


def real():
    """Real datasets, read-only. Ratio per half of each knockdown's cells; stability = absolute split-half difference."""
    from scripts.eval_all_metrics import load_dataset
    from src.models.interventional_nb_sem import infer_control_label
    rng = np.random.default_rng(20261102)
    out = []
    for ds in ("hepg2", "rpe1", "jurkat", "k562"):
        X, iv, genes = load_dataset(ds)
        iv = np.asarray(iv).astype(str); X = np.asarray(X, np.float64); genes = [str(g) for g in genes]
        ctl = infer_control_label(list(iv))
        Xc = X[iv == ctl]; mu, sd = Xc.mean(0), Xc.std(0, ddof=1); nc = len(Xc)
        use = sd > 0
        for g, j in {g: i for i, g in enumerate(genes)}.items():
            idx = np.flatnonzero(iv == g)
            if len(idx) < 8 or not use[j]:
                continue
            Xk = X[idx]
            sh = Xk[:, j].mean() - mu[j]
            se = np.sqrt(Xk[:, j].var(ddof=1) / len(idx) + sd[j] ** 2 / nc)
            oth = use.copy(); oth[j] = False
            med, p99, flip, ldiv = [], [], [], []
            for _ in range(5):
                p = rng.permutation(len(idx)); a, b = p[: len(p) // 2], p[len(p) // 2:]
                da, db = (Xk[a].mean(0) - mu) / sd, (Xk[b].mean(0) - mu) / sd
                flip.append(np.sign(da[j]) != np.sign(db[j]))
                if da[j] == 0 or db[j] == 0:
                    continue
                e = np.abs(da[oth] / da[j] - db[oth] / db[j])
                med.append(np.median(e)); p99.append(np.quantile(e, .99)); ldiv.append(abs(np.log(abs(da[j] / db[j]))))
            if med:
                out.append(dict(screen=ds, gene=g, n=len(idx), d=abs(sh) / sd[j], t=abs(sh) / se, signed_shift_sd=sh / sd[j],
                                err_med=np.mean(med), err_p99=np.mean(p99), flip_rate=np.mean(flip), log_div_diff=np.mean(ldiv), splits=len(med)))
        print(ds, "done", flush=True)
    RES.mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(RES / "real_knockdowns.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "fit", "score", "real"])
    ap.add_argument("--methods", nargs="+", default=["inspre_dag", "adapre", "mean_diff"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    {"check": check, "fit": lambda: fit(a.methods, a.workers), "score": score_all, "real": real}[a.cmd]()
