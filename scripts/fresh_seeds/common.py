"""Shared setup. Repo modules are imported read-only; every write path is redirected into this folder at import, so
spawned GPU workers inherit the redirect too."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/config_tests"), str(ROOT / "scripts/experiments"),
                str(ROOT / "scripts/experiments/e9_author_sims")]
import run_sims  # noqa: E402
import score  # noqa: E402

e9, e8 = score.e9, score.e9.e8
e2 = e9.e2
STORED = ROOT / "scripts/results/paper/e9_author_sims"
e9.OUT, e9.DATA, e9.CACHE, e9.LOGS, e9.TMP = HERE, HERE / "data", HERE / "cache", HERE / "logs", HERE / "tmp"
e8.TMP = run_sims.TMP = HERE / "tmp"
SEEDS = tuple(range(12, 22))
OOM_WAIT, OOM_MAX = 60, 240    # I-SEM waits and retries on the GPU; never falls back to CPU or splits the model chunk
GPU_FULL = ("out of memory", "alloc_failed", "cusolver")   # how a full shared GPU surfaces (allocator, CUBLAS, CUSOLVER)
CPU_METHODS = ("inspre_dag", "adapre", "mean_diff")
LABELS = {"isem": score.ISEM["c"], "inspre_dag": "inspre (official, DAG)", "adapre": "ADAPRE", "mean_diff": "Mean difference"}


def fit(d, method, out):
    m, Y, labels, genes, _ = e9.load(d)
    t0 = time.time()
    if method == "isem":
        import torch
        for attempt in range(OOM_MAX):
            try:
                A, meta = e2.score("isem", Y, labels, genes, m["rep"])
                break
            except RuntimeError as e:     # torch.cuda.OutOfMemoryError is a RuntimeError
                if not any(x in str(e).lower() for x in GPU_FULL) or attempt == OOM_MAX - 1:
                    raise
                torch.cuda.empty_cache()
                time.sleep(OOM_WAIT)
        assert torch.cuda.max_memory_allocated() > 0, "I-SEM did not use the GPU"
        meta = {**meta, "device": "cuda:0", "oom_retries": attempt}
        arrays = {"S": np.asarray(A, np.float32)}
    elif method == "inspre_dag":
        path, meta = run_sims.inspre_path(Y, labels, genes, m["rep"], None)
        arrays = {"W_path": path}
    elif method == "adapre":          # the demo's call at code defaults
        path, meta = e8.fit(Y, labels, genes, "adapre", seed=m["rep"], ncores=2, its=100, solve_its=10, cv=0)
        arrays = {"W_path": path}
    elif method == "mean_diff":
        A, meta = e2.score("mean_diff", Y, labels, genes, m["rep"])
        arrays = {"S": np.asarray(A, np.float32)}
    else:
        raise ValueError(method)
    meta = {**meta, "wall_seconds": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays, meta=np.asarray(json.dumps(meta, default=float)))
    tmp.replace(out)
    return meta


def job(d, method):
    out = e9.CACHE / d.relative_to(e9.DATA) / f"{method}.npz"
    tag = str(d.relative_to(e9.DATA))
    if out.exists():
        return f"{tag} {method} cached"
    try:
        meta = fit(d, method, out)
    except Exception as e:            # recorded and reported, never silently dropped
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / f"{method}.err").write_text(repr(e))
        return f"FAIL {tag} {method}: {repr(e)[-200:]}"
    (out.parent / f"{method}.err").unlink(missing_ok=True)   # an earlier failed attempt
    return f"{tag} {method} {meta['wall_seconds']:.0f}s"


def datasets(kind):
    """Confirmatory datasets only: seed 1 is a reproduction check, seed 11 is exploratory and is never generated here."""
    if kind == "sweep":
        return sorted(p.parent for p in e9.DATA.glob("sweep/*/rep*/meta.json"))
    out = [e9.DATA / kind / s.name / f"rep{r}" for s in sorted((e9.DATA / kind).iterdir()) for r in SEEDS]
    missing = [d for d in out if not (d / "meta.json").exists()]
    assert not missing, f"missing datasets: {missing[:3]}"
    return out


def entries(d):
    """score.sim_entries for this folder's fits. A missing fit is recorded, never imputed. Same helpers, orientation
    and embedding; drops the stored-E9 comparison and ADAPRE's which.min row (its CV is off, its authors prescribe none)."""
    m, _, _, genes, W = e9.load(d)
    c = e9.CACHE / d.relative_to(e9.DATA)
    E, rank, missing = [], {}, []
    edges = lambda A: e9.edges(A, m)
    if (c / "isem.npz").exists():
        with np.load(c / "isem.npz") as z:
            rank["c"] = z["S"]
        E.append(score.entry(score.ISEM["c"], rank["c"], None, score.both("defaults")))
    else:
        missing.append("isem")
    for meth, ab in (("inspre_dag", True), ("adapre", False)):
        if not (c / f"{meth}.npz").exists():
            missing.append(meth)
            continue
        path, meta = e8.load_fit(c / f"{meth}.npz")
        mats = [e8.embed(path[:, :, l], meta["kept_genes"], genes) for l in range(path.shape[2])]
        mats = [np.abs(A) for A in mats] if ab else mats
        c2 = {meta["best_index"] - 1: "which.min eps_hat_G"} if ab else {}
        E += score.path_entries(LABELS[meth], mats, [score.nonzero(A) for A in mats], meta["lambda"], c2, [edges(A) for A in mats])
    if (c / "mean_diff.npz").exists():
        with np.load(c / "mean_diff.npz") as z:
            E.append(score.entry(LABELS["mean_diff"], z["S"], None, {"C1": "no selection step"}))
    return m, W, E, rank, missing
