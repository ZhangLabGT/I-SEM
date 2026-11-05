#!/usr/bin/env python3
"""Fit the I-SEM variants (and inspre's whole lambda path) on the 40 main E9 datasets. Resumable: cached fits are
skipped. Writes only under scripts/results/config_tests/sims/<sim>/<setting>/rep<r>/.

  bash scripts/experiments/run_env.sh python -u scripts/config_tests/run_sims.py --variants a b c [--shard i/n] [--limit N]
  bash scripts/experiments/run_env.sh python -u scripts/config_tests/run_sims.py --variants inspre_path

Variant a needs this run's c (fitted first when both are asked for; otherwise read from this cache).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "scripts/experiments/e9_author_sims")]
import e9_author_sims as e9  # noqa: E402

OUT = ROOT / "scripts/results/config_tests/sims"
TMP = ROOT / "scripts/results/config_tests/tmp"
e9.e8.TMP = TMP                                   # r_env()'s TMPDIR / Rcpp cache -> our own folder
VARIANTS = ("c", "a", "b", "inspre_path")        # fit order (a reuses c)


def main_datasets():
    return [d for d in e9.datasets() if e9.describe(json.loads((d / "meta.json").read_text()))["main"]]


def save(f, meta, **arrays):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays, meta=np.asarray(json.dumps(meta, default=float)))
    tmp.replace(f)
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))


def inspre_path(Y, labels, genes, seed, cached):
    """inspre_path.R (E9's inspre_DAG call, whole path). Path oriented as W (G^T); checks the which.min slice
    against E9's cached S (|G|^T at which.min, zero for dropped genes)."""
    TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TMP) as d:
        d = Path(d)
        np.ascontiguousarray(Y, np.float32).tofile(d / "x.bin")
        (d / "t.txt").write_text("\n".join(labels) + "\n")
        (d / "g.txt").write_text("\n".join(genes) + "\n")
        cmd = [str(e9.RSCRIPT), str(HERE / "inspre_path.R"), str(d / "x.bin"), str(len(Y)), str(d / "t.txt"),
               str(d / "g.txt"), str(d / "out"), str(seed), str(ROOT / "scripts/experiments/e2_inspre.R")]
        r = subprocess.run(cmd, capture_output=True, text=True, env=e9.e8.r_env(blas_threads=1))
        if r.returncode:
            raise RuntimeError(f"inspre_path failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        meta = json.loads((d / "out.json").read_text())
        K, L = meta["K"], meta["L"]
        G = np.fromfile(d / "out.Gpath.bin", np.float32).reshape((K, K, L), order="F")
    path = np.ascontiguousarray(G.transpose(1, 0, 2))           # W[i, j] = effect of j on i
    best = meta["best_index"] - 1
    S = np.abs(e9.e8.embed(path[:, :, best], meta["kept_genes"], genes))
    meta["max_abs_diff_vs_e9_cache"] = float(np.abs(S - cached).max()) if cached is not None else None
    return path, meta


def fit_dataset(d, variants, device):
    out = OUT / d.relative_to(e9.DATA)
    m, Y, labels, genes, _ = e9.load(d)
    from isem_variants import fit_a, fit_b, fit_c
    msgs = []
    for v in [v for v in VARIANTS if v in variants]:
        f = out / ("inspre_dag_path.npz" if v == "inspre_path" else f"isem_{v}.npz")
        if f.exists():
            msgs.append(f"{v} cached"); continue
        t0 = time.time()
        if v == "c":
            S, meta = fit_c(Y, labels, genes, device)
            save(f, meta, S=np.asarray(S, np.float32))
        elif v == "a":
            fc = out / "isem_c.npz"
            if not fc.exists():
                msgs.append("a skipped: fit c first (same run)"); continue
            with np.load(fc) as z:
                S_c = z["S"]
            E, meta = fit_a(Y, labels, genes, S_c, seed=m["rep"], device=device)
            save(f, meta, edges=E)
        elif v == "b":
            S, meta = fit_b(Y, labels, genes, seed=m["rep"], device=device)
            save(f, meta, S=np.asarray(S, np.float32))
        else:
            cf = e9.CACHE / d.relative_to(e9.DATA) / "inspre_dag.npz"
            cached = np.load(cf)["S"] if cf.exists() else None
            path, meta = inspre_path(Y, labels, genes, m["rep"], cached)
            save(f, meta, W_path=path)
        msgs.append(f"{v} {time.time() - t0:.0f}s")
    return ", ".join(msgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=["c", "a", "b"])
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th dataset starting at i")
    ap.add_argument("--limit", type=int, help="first N datasets of this shard (smoke tests)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    ds = main_datasets()[i::n][:a.limit]
    print(f"config_tests sims: {len(ds)} datasets, variants {a.variants}", flush=True)
    for d in ds:
        print(f"{d.relative_to(e9.DATA)}: {fit_dataset(d, a.variants, a.device)}", flush=True)


if __name__ == "__main__":
    main()
