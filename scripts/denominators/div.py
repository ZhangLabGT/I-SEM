#!/usr/bin/env python3
"""Divisor swap: inspre and ADAPRE given the true own-gene shift, 160 weak-block sims, seeds 12-21.
  bash run.sh check | fit --methods inspre_dag adapre --workers 12 | score | analyze
Repo modules read-only; data and shipped-fit caches read from fresh_seeds, all writes here."""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SC = ROOT / "scripts/fresh_seeds"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/config_tests"), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/experiments/e9_author_sims")]
import run_sims  # noqa: E402
import score  # noqa: E402

e9, e8 = score.e9, score.e9.e8
e9.OUT, e9.DATA, e9.CACHE, e9.LOGS, e9.TMP = HERE, SC / "data", HERE / "cache", HERE / "logs", HERE / "tmp"
e8.TMP = run_sims.TMP = HERE / "tmp/r"   # rcpp_cache lives here: warm it with one serial adapre fit before any pool
e9.RSCRIPT = e8.RSCRIPT = HERE / "shim.sh"
SEEDS = range(12, 22)
ARMS = ("observed", "true_divisor", "true_R")
METHODS = ("inspre_dag", "adapre")
RES = HERE / "results"


def datasets():
    out = []
    for s in sorted((e9.DATA / "inspre").iterdir()):
        d0 = json.loads((s / "rep12/meta.json").read_text())
        de = e9.describe(d0)
        if de["effects"] == "small" and de["instruments"] == "weak":
            assert s.name.endswith(("_7", "_8")), s.name
            out += [s / f"rep{r}" for r in SEEDS]
    assert len(out) == 160, len(out)
    return out


def truth(d):
    """True own-gene shift and ratio matrix from the saved (normalised, exposure x outcome) G. Generator:
    Y_int = (XB int_beta sd + eps) (I - G_raw)^-1, z-scored by the control sd, so mean(Y_k | KD j) = int_beta T[j, k],
    T = (I - G_saved)^-1 (the sds cancel into G_saved); ratio R[j, k] = T[j, k] / T[j, j], which is the generator's R."""
    m = json.loads((d / "meta.json").read_text())
    D = m["D"]
    ib = m["params"]["int_beta"]
    ib = ib[0] if isinstance(ib, list) else ib
    assert np.isscalar(ib), f"non-scalar int_beta in {d}"
    G = np.fromfile(d / "G.bin", np.float64).reshape(D, D, order="F")
    T = np.linalg.inv(np.eye(D) - G)
    return ib * np.diag(T), T / np.diag(T)[:, None], ib * T, list(m["genes"])


def check_one(d):
    """z = (observed - true) / empirical SE, own gene and all pairs; plus divisor noise."""
    m, Y, labels, genes, _ = e9.load(d)
    div, R, M, g = truth(d)
    assert g == genes
    lab = np.asarray(labels)
    n = (lab == genes[0]).sum()
    obs = np.stack([Y[lab == t].mean(0) for t in genes]).astype(float)
    sd = np.stack([Y[lab == t].std(0, ddof=1) for t in genes]).astype(float)
    se = sd / np.sqrt(n)
    own = np.arange(len(genes))
    oth = ~np.eye(len(genes), dtype=bool)
    z = (obs - M) / se
    return dict(setting=m["setting"], rep=m["rep"], z_own_mean=z[own, own].mean(), z_own_sd=z[own, own].std(ddof=1),
                z_pair_mean=z[oth].mean(), z_pair_sd=z[oth].std(ddof=1), true_abs=np.abs(div).mean(),
                rel_err=np.mean(np.abs(obs[own, own] - div) / np.abs(div)), cv=np.mean(se[own, own] / np.abs(div)),
                t_min=np.min(np.abs(div) / se[own, own]), t_med=np.median(np.abs(div) / se[own, own]),
                flips=int((np.sign(obs[own, own]) != np.sign(div)).sum()), diag_T_min=np.abs(div).min(), diag_T_max=np.abs(div).max(),
                ratio_err_med=np.median(np.abs(obs[oth] - (obs[own, own][:, None] * R)[oth])), n_kd=len(genes))


def check():
    env = {**e8.r_env(1), "DIVISOR_HERE": str(HERE), "PATH": f"{ROOT}/.envs/inspre/bin:{os.environ['PATH']}"}
    r = subprocess.run([str(ROOT / ".envs/inspre/bin/Rscript"), str(HERE / "check_patched.R"), str(ROOT / "scripts/baselines/adapre/upstream/R/adapre_inspre_core.R")],
                       capture_output=True, text=True, env=env)
    print(r.stdout[-300:], r.stderr[-800:] if r.returncode else "", flush=True)
    assert r.returncode == 0, "override identity check failed"
    with ProcessPoolExecutor(16, mp_context=mp.get_context("fork")) as ex:
        df = pd.DataFrame(ex.map(check_one, datasets()))
    df.to_csv(RES / "truth_check.csv", index=False)
    print(df[["z_own_mean", "z_own_sd", "z_pair_mean", "z_pair_sd", "cv", "rel_err", "t_min", "t_med", "flips"]].describe().loc[["mean", "min", "max"]].to_string())
    assert abs(df.z_own_mean.mean()) < 0.1 and 0.9 < df.z_own_sd.mean() < 1.1, "own-gene formula off"
    assert abs(df.z_pair_mean.mean()) < 0.05 and 0.9 < df.z_pair_sd.mean() < 1.1, "pair formula off"
    print("truth check OK", flush=True)


def fit_job(job):
    d, method, arm = job
    d = Path(d)
    rel = d.relative_to(e9.DATA)
    out = HERE / "cache" / rel / f"{method}__{arm}.npz"
    tag = f"{rel} {method} {arm}"
    if out.exists():
        return f"{tag} cached"
    t0 = time.time()
    try:
        m, Y, labels, genes, _ = e9.load(d)
        div, R, _, g = truth(d)
        assert g == genes
        with tempfile.TemporaryDirectory(dir=HERE / "tmp") as td:
            td = Path(td)
            div.astype(np.float64).tofile(td / "div.bin")
            np.asarray(R, np.float64).T.tofile(td / "R.bin")      # column-major for R
            (td / "genes.txt").write_text("\n".join(genes) + "\n")
            os.environ.update(DIVISOR_ARM=arm, DIVISOR_TRUTH=str(td), DIVISOR_HERE=str(HERE))
            if method == "inspre_dag":
                path, meta = run_sims.inspre_path(Y, labels, genes, m["rep"], None)
            else:
                path, meta = e8.fit(Y, labels, genes, "adapre", seed=m["rep"], ncores=2, its=100, solve_its=10, cv=0)
        meta = {**meta, "wall_seconds": time.time() - t0, "arm": arm}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, W_path=path, meta=np.asarray(json.dumps(meta, default=float)))
        tmp.replace(out)
    except Exception as e:                         # recorded, reported; never retried with other settings
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".err").write_text(repr(e))
        return f"FAIL {tag}: {repr(e)[-300:]}"
    out.with_suffix(".err").unlink(missing_ok=True)
    return f"{tag} {time.time() - t0:.0f}s"


def fit(methods, arms, workers, limit):
    jobs = [(str(d), me, a) for d in datasets()[:limit] for me in methods for a in arms]
    print(f"{len(jobs)} fits, {workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        for i, r in enumerate(ex.map(fit_job, jobs), 1):
            if r.startswith("FAIL") or i % 40 == 0:
                print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {r}", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


def control(limit=None):
    """Arm observed against the cached as-released fits."""
    rows = []
    for d in datasets()[:limit]:
        rel = d.relative_to(e9.DATA)
        for me in METHODS:
            f = HERE / "cache" / rel / f"{me}__observed.npz"
            if not f.exists():
                continue
            a, ma = e8.load_fit(f)
            b, mb = e8.load_fit(SC / "cache" / rel / f"{me}.npz")
            same_shape = a.shape == b.shape
            dif = float(np.nanmax(np.abs(a - b))) if same_shape else float("inf")
            rows.append(dict(setting=d.parent.name, rep=d.name, method=me, same_shape=same_shape, max_abs_diff=dif,
                             nan_same=bool(same_shape and (np.isnan(a) == np.isnan(b)).all()), kept_same=ma["kept_genes"] == mb["kept_genes"],
                             lambda_same=ma["lambda"] == mb["lambda"], best_same=ma.get("best_index") == mb.get("best_index")))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "control.csv", index=False)
    print(df.groupby("method").agg(n=("rep", "size"), max_abs_diff=("max_abs_diff", "max"), shape=("same_shape", "all"), nan=("nan_same", "all"),
                                   kept=("kept_same", "all"), lam=("lambda_same", "all"), best=("best_same", "all")).to_string(), flush=True)
    return df


def score_one(d):
    m, _, _, genes, W = e9.load(d)
    tru = e9.edges(W, m)
    off = ~np.eye(len(W), dtype=bool)
    y = tru[off].astype(np.int8)
    rel = d.relative_to(e9.DATA)
    base = dict(setting=m["setting"], rep=m["rep"], n_true=int(tru.sum()))
    rows, kept = [], []
    with np.load(SC / "cache" / rel / "isem.npz") as z:
        au, ap = score.auc_ap(y, np.abs(z["S"])[off])
    rows.append({**base, "method": "isem", "arm": "shipped", "rule": "shipped", "lambda_index": -1, "auroc": au, "auprc": ap})
    for me in METHODS:
        for arm in ARMS:
            f = HERE / "cache" / rel / f"{me}__{arm}.npz"
            if not f.exists():
                continue
            path, meta = e8.load_fit(f)
            L = path.shape[2]
            per = [score.auc_ap(y, np.abs(e8.embed(path[:, :, l], meta["kept_genes"], genes))[off]) for l in range(L)]
            au = np.array([p[0] for p in per]); ap = np.array([p[1] for p in per])
            kept.append({**base, "method": me, "arm": arm, "K": meta["K"], "L": L, "wall": meta["wall_seconds"],
                         "selected_index": meta.get("best_index", np.nan), "nonfinite": int((~np.isfinite(path)).sum())})
            rows.append({**base, "method": me, "arm": arm, "rule": "median", "lambda_index": -1, "auroc": np.median(au), "auprc": np.median(ap)})
            rows.append({**base, "method": me, "arm": arm, "rule": "best (upper limit)", "lambda_index": -1, "auroc": au.max(), "auprc": ap.max()})
            if me == "inspre_dag":
                b = meta["best_index"] - 1
                rows.append({**base, "method": me, "arm": arm, "rule": "selected", "lambda_index": b, "auroc": au[b], "auprc": ap[b]})
    return rows, kept


def score_all():
    ds = datasets()
    rows, kept = [], []
    with ProcessPoolExecutor(16, mp_context=mp.get_context("fork")) as ex:
        for r, k in ex.map(score_one, ds, chunksize=4):
            rows += r; kept += k
    pd.DataFrame(rows).to_csv(RES / "rows.csv", index=False)
    pd.DataFrame(kept).to_csv(RES / "kept.csv", index=False)
    print(len(rows), "rows", flush=True)


def analyze():
    df = pd.read_csv(RES / "rows.csv")
    kept = pd.read_csv(RES / "kept.csv")
    sm = df.groupby(["method", "arm", "rule", "setting"])[["auroc", "auprc"]].mean().reset_index()
    sm.to_csv(RES / "per_setting.csv", index=False)
    iso = sm[sm.method == "isem"].set_index("setting")
    out = []
    for me, rule in (("inspre_dag", "selected"), ("inspre_dag", "median"), ("inspre_dag", "best (upper limit)"), ("adapre", "median"), ("adapre", "best (upper limit)")):
        g = sm[(sm.method == me) & (sm.rule == rule)]
        for metric in ("auroc", "auprc"):
            p = g.pivot(index="setting", columns="arm", values=metric)
            i = iso.loc[p.index, metric]
            gap0 = (i - p.observed).mean()
            for arm in ("observed", "true_divisor", "true_R"):
                if arm not in p:
                    continue
                gap = i - p[arm]
                r = dict(method=me, rule=rule, metric=metric, arm=arm, n=int(p[arm].notna().sum()), isem=i.mean(), mean=p[arm].mean(),
                         isem_minus_arm=gap.mean(), isem_wins=int((gap > 0).sum()), p_isem_vs_arm=wilcoxon(gap).pvalue if (gap != 0).any() else 1.0)
                if arm != "observed":
                    dl = p[arm] - p.observed
                    r.update(arm_minus_observed=dl.mean(), arm_wins=int((dl > 0).sum()), p_arm_vs_observed=wilcoxon(dl).pvalue if (dl != 0).any() else 1.0,
                             gap_closed_frac=(gap0 - gap.mean()) / gap0 if gap0 else np.nan)
                out.append(r)
    t = pd.DataFrame(out)
    t.to_csv(RES / "summary.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(t.round(4).to_string(index=False))
    print(kept.groupby(["method", "arm"]).agg(n=("K", "size"), K_mean=("K", "mean"), K_min=("K", "min"), nonfinite=("nonfinite", "sum"), wall=("wall", "mean")).round(2).to_string())
    print("selected index:", kept[kept.method == "inspre_dag"].groupby("arm").selected_index.describe()[["mean", "min", "max"]].round(2).to_string(), sep="\n")
    err = list((HERE / "cache").rglob("*.err"))
    print(f"{len(err)} crashes", [str(e.relative_to(HERE)) for e in err][:10])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "fit", "control", "score", "analyze"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    {"check": check, "fit": lambda: fit(a.methods, a.arms, a.workers, a.limit), "control": lambda: control(a.limit), "score": score_all, "analyze": analyze}[a.cmd]()
