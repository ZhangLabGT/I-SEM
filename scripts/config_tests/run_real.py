#!/usr/bin/env python3
"""Fit the I-SEM variants on a real dataset. Resumable; writes only under scripts/results/config_tests/real/<ds>/.

  bash scripts/experiments/run_env.sh python -u scripts/config_tests/run_real.py --ds hepg2 --variants a b c
  bash scripts/experiments/run_env.sh python -u scripts/config_tests/run_real.py --ds hepg2 --variants c --max-genes 40 --out-tag smoke

Variant b's 45 CV fits are cached one per file in real/<ds>/b_cv/ and can run as shards in separate processes
(--b-shard i/n fits pairs i, i+n, ...; all of a dataset's shards on one device); a final run without --b-shard picks
lambda from the cache (fitting any missing pair itself) and refits on its device:
  bash scripts/experiments/run_env.sh python -u scripts/config_tests/run_real.py --ds k562 --variants b --b-shard 0/5 --device cuda:0
  bash scripts/experiments/run_env.sh python -u scripts/config_tests/run_real.py --ds k562 --variants b --device cuda:0

--max-genes keeps the first N panel genes and the cells of their knockdowns plus controls (smoke tests only; the
result goes to real/<ds>_<out-tag>/ and is never read by score.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
from run_sims import save  # noqa: E402

OUT = ROOT / "scripts/results/config_tests/real"
SEED = 0                                           # E7's split seed


def load(ds, max_genes=None):
    from scripts.eval_all_metrics import load_dataset
    from src.models.interventional_nb_sem import infer_control_label
    X, iv, genes = load_dataset(ds)
    iv = np.asarray([str(x).strip() for x in iv])
    if max_genes:
        genes = genes[:max_genes]
        keep = np.isin(iv, genes + [infer_control_label(list(iv))])
        X, iv = np.asarray(X[keep][:, :max_genes]), iv[keep]
    return X, iv, genes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", required=True, choices=["hepg2", "rpe1", "jurkat", "k562"])
    ap.add_argument("--variants", nargs="+", choices=["c", "a", "b"], default=["c", "a", "b"])
    ap.add_argument("--max-genes", type=int)
    ap.add_argument("--out-tag")
    ap.add_argument("--device", default=None)
    ap.add_argument("--b-shard", help="i/n: fit only variant b's CV pairs i, i+n, ... and exit")
    a = ap.parse_args()
    if bool(a.max_genes) != bool(a.out_tag):
        ap.error("--max-genes and --out-tag go together")
    shard = tuple(map(int, a.b_shard.split("/"))) if a.b_shard else None
    if shard and (a.variants != ["b"] or not 0 <= shard[0] < shard[1]):
        ap.error("--b-shard i/n needs --variants b and 0 <= i < n")
    from isem_variants import fit_a, fit_b, fit_c
    out = OUT / (f"{a.ds}_{a.out_tag}" if a.out_tag else a.ds)
    X, iv, genes = load(a.ds, a.max_genes)
    print(f"{a.ds}: {X.shape[0]} cells, {len(genes)} genes -> {out}", flush=True)
    for v in [v for v in ("c", "a", "b") if v in a.variants]:
        f = out / f"isem_{v}.npz"
        if f.exists():
            print(f"{v} cached", flush=True); continue
        t0 = time.time()
        if v == "c":
            S, meta = fit_c(X, iv, genes, a.device)
            save(f, {**meta, "genes": genes}, S=np.asarray(S, np.float32))
        elif v == "a":
            if not (out / "isem_c.npz").exists():
                print("a skipped: fit c first (same run)", flush=True); continue
            E, meta = fit_a(X, iv, genes, np.load(out / "isem_c.npz")["S"], seed=SEED, device=a.device)
            save(f, {**meta, "genes": genes}, edges=E)
        elif shard:
            fit_b(X, iv, genes, seed=SEED, device=a.device, cache=out / "b_cv", shard=shard)
        else:
            S, meta = fit_b(X, iv, genes, seed=SEED, device=a.device, cache=out / "b_cv")
            save(f, {**meta, "genes": genes}, S=np.asarray(S, np.float32))
        print(f"{v} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
