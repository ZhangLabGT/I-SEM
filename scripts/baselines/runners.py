"""Entry points for the observational baselines NOTEARS, GOLEM-NV, ICA-LiNGAM, and the interventional IGSP and RICE.

Each returns (W, meta) with W[i, j] = effect of gene j on gene i (the repo orientation), the method's output after
its own default post-processing only; nothing is symmetrized. X is cells x genes, all conditions pooled.

mode "readme": one fit at the authors' published settings (real datasets).
mode "cv":     the inspre authors' benchmark protocol (Rmd/python_methods_comparison/run_python_methods.py):
               10-value lambda grid, 5-fold StratifiedKFold by target, lowest held-out loss, refit on all data.
               NOTEARS: their script scored the full-data fits on the held-out folds (G_all instead of G_cv);
               here the held-out loss uses the training-fold fits, i.e. the bug is fixed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENVS = ROOT / ".envs"
LAMBDA_GRID = np.exp(np.arange(np.log(1), np.log(1e-6), (np.log(1e-6) - np.log(1)) / 10 + 1e-8))[1:]  # inspre's
sys.path.insert(0, str(ROOT / "scripts" / "baselines" / "notears" / "upstream"))


def _notears_l2_loss(W, X):
    """The inspre authors' loss_notears (l2): 0.5 / n * ||X - X W||_F^2."""
    R = X - X @ W
    return 0.5 / X.shape[0] * float((R ** 2).sum())


def _notears_fit(X, lam, w_threshold):
    from notears.linear import notears_linear
    return notears_linear(X, lam, "l2", w_threshold=w_threshold)


def notears(X, mode="readme", targets=None, n_jobs=1):
    """n_jobs: parallel processes for the CV fits (independent and deterministic, so results do not change)."""
    from notears.linear import notears_linear
    X = np.asarray(X, np.float64)
    t0 = time.time()
    if mode == "readme":
        W_est = notears_linear(X, lambda1=0.1, loss_type="l2")           # README/CLI defaults, w_threshold 0.3
        meta = {"lambda1": 0.1, "w_threshold": 0.3}
    else:
        from sklearn.model_selection import StratifiedKFold
        from concurrent.futures import ProcessPoolExecutor
        folds = list(StratifiedKFold(n_splits=5).split(X, np.asarray(targets)))
        jobs = [(None, lam) for lam in LAMBDA_GRID] + [(f, lam) for f in range(5) for lam in LAMBDA_GRID]
        args = [(X if f is None else X[folds[f][0]], lam, 0.0) for f, lam in jobs]
        if n_jobs > 1:
            with ProcessPoolExecutor(n_jobs) as ex:
                out = list(ex.map(_notears_fit, *zip(*args)))
        else:
            out = [_notears_fit(*a) for a in args]
        L = len(LAMBDA_GRID)
        fits = out[:L]
        eps = [[_notears_l2_loss(out[L + f * L + i], X[folds[f][1]]) for i in range(L)] for f in range(5)]
        best = int(np.argmin(np.mean(eps, 0)))
        W_est = fits[best]
        meta = {"lambda_grid": LAMBDA_GRID.tolist(), "heldout_loss": np.mean(eps, 0).tolist(), "best_index": best,
                "lambda1": float(LAMBDA_GRID[best]), "w_threshold": 0.0, "cv_note": "held-out loss of training-fold fits"}
    meta.update(method="NOTEARS-linear", mode=mode, upstream="github.com/xunzheng/notears @ 4a9ab19",
                n=int(X.shape[0]), d=int(X.shape[1]), wall_seconds=time.time() - t0)
    return W_est.T.copy(), meta                                               # W_est[i, j] = i -> j


def golem(X, mode="readme", targets=None, seed=1, gpu=True, threads=0):
    env = {**os.environ, "TF_DETERMINISTIC_OPS": "1", "TF_CPP_MIN_LOG_LEVEL": "2",
           "TF_FORCE_GPU_ALLOW_GROWTH": "true"}   # shared GPU: TF otherwise maps all memory up front
    if threads:  # TF reads these when the session config leaves the pools at 0 (GOLEM's does)
        env.update(TF_NUM_INTRAOP_THREADS=str(threads), TF_NUM_INTEROP_THREADS="1",
                   OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        d = Path(d)
        np.save(d / "X.npy", np.asarray(X, np.float32))
        cmd = [str(ENVS / "golem/bin/python"), str(ROOT / "scripts/baselines/golem/fit.py"), str(d / "X.npy"),
               str(d / "out.npz"), "--mode", mode, "--seed", str(seed)]
        if mode == "cv":
            (d / "t.txt").write_text("\n".join(map(str, targets)) + "\n")
            cmd += ["--targets", str(d / "t.txt")]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode:
            raise RuntimeError(f"GOLEM failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
        with np.load(d / "out.npz") as z:
            B, meta = z["B"], json.loads(str(z["meta"]))
    return B.T.copy(), meta                                                   # B[i, j] = i -> j


def lingam(X, seed=0, threads=1):
    X = np.ascontiguousarray(X, np.float32)
    n, p = X.shape
    env = {k: v for k, v in os.environ.items() if k not in ("R_HOME", "R_LIBS", "R_LIBS_USER", "R_LIBS_SITE")}
    env.update(R_LIBS_USER="", R_LIBS="", R_LIBS_SITE="", R_PROFILE_USER="/dev/null", R_ENVIRON_USER="/dev/null",
               OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        d = Path(d)
        X.tofile(d / "X.bin")
        cmd = [str(ENVS / "lingam/bin/Rscript"), str(ROOT / "scripts/baselines/lingam/fit.R"), str(d / "X.bin"),
               str(n), str(p), str(d / "B.bin"), str(seed)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode:
            raise RuntimeError(f"LiNGAM failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
        B = np.fromfile(d / "B.bin", np.float64).reshape((p, p), order="F")
        meta = json.loads((d / "B.bin.json").read_text())
    return B, meta                                                            # Bpruned[i, j] = j -> i already


def igsp(X, labels, genes, seed=0, threads=1):
    """IGSP (causaldag 0.1a163, scripts/baselines/igsp/fit.py). labels[c] = "control" or the gene knocked down in cell c
    (a name in genes); controls are the observational setting, each knocked-down gene one intervention setting."""
    g2i = {g: i for i, g in enumerate(genes)}
    bad = sorted({l for l in labels if l != "control" and l not in g2i})
    if bad:
        raise ValueError(f"IGSP labels neither 'control' nor a panel gene: {bad[:5]}")
    t = np.array([-1 if l == "control" else g2i[l] for l in labels], np.int64)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(PYTHONNOUSERSITE="1", OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        d = Path(d)
        np.save(d / "X.npy", np.asarray(X, np.float32))
        np.save(d / "t.npy", t)
        cmd = [str(ENVS / "igsp/bin/python"), "-s", str(ROOT / "scripts/baselines/igsp/fit.py"), str(d / "X.npy"),
               str(d / "t.npy"), str(d / "out.npz"), "--seed", str(seed)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=d)
        if r.returncode:
            raise RuntimeError(f"IGSP failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
        with np.load(d / "out.npz") as z:
            A, meta = z["A"], json.loads(str(z["meta"]))
    return A.T.astype(np.float64), meta                                       # A[i, j] = i -> j


def rice(raw_h5ad, genes, jobs=16, processed=None, reuse=False):
    """RICE (scripts/baselines/rice/fit.py in .envs/rice, GPU): the authors' run_Replogle_2022.py on the raw dataset
    raw_h5ad with our panel `genes` (Ensembl IDs, in order). The fit's output streams to this process's stdout.
    processed: optional .npz path for the script's processed arrays (its line-164 np.savez); reuse=True resumes
    from it (fit.py --reuse-processed: only the expensive regressions/PCA are read back)."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUPY_CACHE_DIR=str(ENVS / ".cache/cupy"))
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        d = Path(d)
        (d / "panel.txt").write_text("\n".join(genes) + "\n")
        cmd = [str(ENVS / "rice/bin/python"), "-s", "-u", str(ROOT / "scripts/baselines/rice/fit.py"), str(raw_h5ad),
               str(d / "panel.txt"), str(d / "out.npz"), "--jobs", str(jobs)] + \
              (["--processed", str(processed)] if processed else []) + (["--reuse-processed"] if reuse else [])
        r = subprocess.run(cmd, env=env, cwd=d)
        if r.returncode:
            raise RuntimeError(f"RICE failed (exit {r.returncode}); see the output above")
        with np.load(d / "out.npz") as z:
            W, genes_out, meta = z["W"], [str(g) for g in z["genes"]], json.loads(str(z["meta"]))
    assert genes_out == list(genes)
    return W.T.copy(), meta                                                   # W[i, j] = i -> j
