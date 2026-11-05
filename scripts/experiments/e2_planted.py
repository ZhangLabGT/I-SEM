#!/usr/bin/env python3
"""E2: planted-truth cyclic simulation. Every write stays under scripts/.

Ground truth. N_GENES genes. A fraction `cycle_frac` of them is split into disjoint directed simple
cycles (lengths 2-5); the rest are free. Cycles and free genes are put in a random order and DAG edges
only run forward in that order, so the strongly connected components are exactly the planted cycles.
The eigenvalues of W are then the union of the cycles' eigenvalues (|prod w|^(1/L) in modulus), and
scaling the cycle edges alone sets rho(W) exactly; DAG edge weights do not depend on rho.
W[i, j] is the effect of j on i, as in the I-SEM code.

Cells. Log-rate z = m + T b_j e_j + N eps, T = (I - W)^-1, eps ~ N(0, I); counts ~ Poisson(exp(z));
expression = log1p(counts), the regime of the real datasets. Every gene is knocked down; N_CONTROL
control cells. Condition means follow the paper's model exactly: delta_c = T (B * u_c) (§5).
  * kd_j ~ U(0.5, 0.9) is the realised knockdown of the target itself, as a dataset measures it:
    b_j = log(1 - kd_j) / T_jj, so z_j drops by log(1 - kd_j) whatever rho is.
  * noise is the steady-state s.d. of every gene's control log-rate: N = diag(noise / ||T_i.||) T,
    i.e. noise propagated through the network (keeping its gene-gene correlation, the coexpression
    signal) and rescaled per gene. Without both, T_ii ~ 1/(1 - rho^L) ties the rho sweep to effect
    size and SNR (at rho = 0.95 |T| reaches ~21 and Poisson rates overflow).
The cell-level noise is not iid structural noise; methods that use only condition means see the
model above exactly. Near-unstable positive feedback amplifies knockdowns by ~1/(1 - rho^L): at
rho = 0.95 about 0.2% of (knockdown, gene) pairs shift by more than 4 log units (none at rho <= 0.8).
This is not clipped (it is the regime the rho sweep probes); collect() records it per setting.

Pairing. Topology, raw weights, m and kd depend only on (rep, cycle_frac), so the rho, noise and cells
sweeps change one factor on the same networks. Sweeps are one-at-a-time around DEFAULT.

Metrics (directed, ordered pairs i != j, scores |S| with the diagonal zeroed, exact tie-aware AUC/AP
from scripts/experiments/paper_eval.py):
  auprc_direct, auroc_direct  positives W_ij != 0
  auprc_total_only            pairs with W_ij = 0 only; positive if j reaches i through a path
                              (a total effect without a direct edge), negative if j cannot reach i

  python -u scripts/experiments/e2_planted.py check
  python -u scripts/experiments/e2_planted.py run --methods isem mean_diff plugin
  python -u scripts/experiments/e2_planted.py run --methods inspre_dag notears
  python -u scripts/experiments/e2_planted.py collect      # -> scripts/results/paper/e2_planted.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from paper_eval import absd, auc_ap  # noqa: E402

PAPER = ROOT / "scripts/results/paper"
CACHE = PAPER / "e2_cache"
N_GENES, N_CONTROL, REPS = 150, 1000, 10
DEFAULT = {"rho": 0.8, "noise": 0.3, "cycle_frac": 0.5, "cells": 100}
SWEEPS = {"rho": [0.5, 0.8, 0.95], "noise": [0.15, 0.3, 0.6],
          "cycle_frac": [0.0, 0.25, 0.5, 0.75], "cells": [25, 100, 400]}
# Fig 2 methods. Baselines get every cell (max_cells=None), so the cells sweep means what it says.
# inspre appears only with the authors' official settings: their inspre_DAG variant,
# cached as inspre_dag and written to the CSV as "inspre". The DAG=FALSE run is not reported.
MAIN = ["isem", "inspre_dag", "notears", "mean_diff", "plugin"]
CSV_NAME = {"inspre_dag": "inspre"}
CONTROL = "non-targeting"


def key(p):
    return f"rho{p['rho']}_noise{p['noise']}_cyc{p['cycle_frac']}_cells{p['cells']}"


def settings():
    """(sweep, level, params) for every figure point; the default point recurs in every sweep."""
    return [(s, lv, {**DEFAULT, s: lv}) for s, levels in SWEEPS.items() for lv in levels]


def planted(n, cycle_frac, rho, rep):
    """Planted W, gene log-means m and knockdown fractions kd. Only rho is applied after drawing."""
    rng = np.random.default_rng([rep, round(100 * cycle_frac), 2027])
    perm = rng.permutation(n)
    k = int(round(cycle_frac * n))
    cycles, used = [], 0
    while k - used >= 2:
        L = int(min(rng.integers(2, 6), k - used))
        if k - used - L == 1:            # never leave a single gene that cannot form a cycle
            L = L + 1 if L < 5 else L - 1
        cycles.append(perm[used:used + L]); used += L
    blocks = cycles + [np.array([g]) for g in perm[used:]]
    order = rng.permutation(len(blocks))
    rank = np.empty(n, int)
    for r, b in enumerate(order):
        rank[blocks[b]] = r
    W = np.zeros((n, n))
    for i in range(n):                   # DAG edges j -> i from strictly earlier blocks
        earlier = np.flatnonzero(rank < rank[i])
        p = min(rng.poisson(1.2), len(earlier))
        if p:
            js = rng.choice(earlier, p, replace=False)
            W[i, js] = rng.uniform(0.25, 0.75, p) * rng.choice([-1, 1], p)
    C = np.zeros((n, n))
    for c in cycles:                     # c[0] -> c[1] -> ... -> c[0]
        C[np.roll(c, -1), c] = rng.uniform(0.5, 1.0, len(c)) * rng.choice([-1, 1], len(c))
    if cycles:
        rho_c = max(np.prod(np.abs(C[np.roll(c, -1), c])) ** (1 / len(c)) for c in cycles)
        W += C * (rho / rho_c)
    m = np.log(rng.uniform(3, 30, n))
    kd = rng.uniform(0.5, 0.9, n)
    return W, m, kd, cycles


def simulate(p, rep, n=N_GENES):
    W, m, kd, _ = planted(n, p["cycle_frac"], p["rho"], rep)
    T = np.linalg.inv(np.eye(n) - W)
    rng = np.random.default_rng([rep, round(100 * p["cycle_frac"]), round(100 * p["rho"]),
                                 round(1000 * p["noise"]), p["cells"], 7])
    genes = [f"G{g:03d}" for g in range(n)]
    blocks, labels = [], []
    N = (p["noise"] / np.linalg.norm(T, axis=1))[:, None] * T
    b = np.log1p(-kd) / np.diag(T)
    for j in [None] + list(range(n)):
        cells = N_CONTROL if j is None else p["cells"]
        z = m + rng.standard_normal((cells, n)) @ N.T
        if j is not None:
            z += T[:, j] * b[j]
        blocks.append(np.log1p(rng.poisson(np.exp(z))).astype(np.float32))
        labels += [CONTROL if j is None else genes[j]] * cells
    return np.concatenate(blocks), labels, genes, W


def _inspre(expr, interv, genes, rep, dag):
    from scripts.run_all_baselines import standardize_to_control
    rscript = ROOT / ".envs/inspre/bin/Rscript"   # built by scripts/envs/build.sh; CV needs caret
    lab = np.array(interv)
    z = standardize_to_control(expr, lab == CONTROL)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        d = Path(d)
        np.ascontiguousarray(z, np.float32).tofile(d / "x.bin")
        (d / "t.txt").write_text("\n".join(np.where(lab == CONTROL, "control", lab)) + "\n")
        (d / "g.txt").write_text("\n".join(genes) + "\n")
        cmd = [str(rscript), str(Path(__file__).with_name("e2_inspre.R")), str(d / "x.bin"),
               str(len(z)), str(d / "t.txt"), str(d / "g.txt"), str(d / "S.bin"), str(dag).upper(), str(rep)]
        env = {**os.environ, "R_LIBS_USER": "", "R_LIBS": "", "R_LIBS_SITE": "",   # env's own library only;
               "R_PROFILE_USER": "/dev/null", "R_ENVIRON_USER": "/dev/null"}  # ~/.Rprofile prepends ~/R/...
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode:
            raise RuntimeError(f"inspre failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        n = len(genes)
        S = np.fromfile(d / "S.bin", np.float32).reshape((n, n), order="F")
        return S, json.loads((d / "S.bin.json").read_text())


def score(method, expr, interv, genes, rep):
    """Score matrix S[i, j] (effect of j on i) and method metadata."""
    if method == "isem":
        from src.pipeline import GRNInferencePipeline
        pipe = GRNInferencePipeline(device="cuda:0")
        assert (pipe.n_resamples, pipe.readout, pipe.symmetrize) == (60, "direct", False), "not the paper model"
        res = pipe.run(expr, interv, genes)
        return res.adjacency, {"n_models": res.metadata.get("n_models")}
    if method == "plugin":                # I - diag(b) T^-1, b_j = 1 / [T^-1]_jj
        from src.models.interventional_nb_sem import build_intervention_targets
        u, delta = build_intervention_targets(expr, interv, genes)[:2]
        n = len(genes)
        T = np.zeros((n, n)); T[:, u.argmax(1)] = delta.T
        Ti = np.linalg.inv(T)
        return np.eye(n) - (1 / np.diag(Ti))[:, None] * Ti, {"cond_T": float(np.linalg.cond(T))}
    if method == "inspre_dag":
        return _inspre(expr, interv, genes, rep, dag=True)
    import scripts.run_all_baselines as rab
    runner = {"mean_diff": rab.run_mean_diff, "notears": rab.run_notears}[method]
    kw = {} if method == "mean_diff" else {"max_cells": len(expr)}   # every cell, no subsampling
    return runner(expr, interv, genes, seed=rep, **kw)[0], {"n_cells_used": int(len(expr))}


def evaluate(S, W):
    n = len(W)
    off = ~np.eye(n, dtype=bool)
    S = absd(S)
    direct = (W != 0) & off
    reach = np.isfinite(shortest_path((W != 0).T.astype(np.int8), unweighted=True)).T  # reach[i, j]: j ~> i
    auroc, auprc = auc_ap(direct[off].astype(np.int8), S[off])
    rest = off & ~direct
    y = reach[rest].astype(np.int8)
    auprc_total = auc_ap(y, S[rest])[1] if 0 < y.sum() < y.size else float("nan")
    return {"auprc_direct": auprc, "auroc_direct": auroc, "auprc_total_only": auprc_total}


def provenance():
    files = ["src/pipeline.py", "scripts/eval_all_metrics.py", "src/models/interventional_nb_sem.py", "src/models/batched_linalg.py",
             "scripts/run_all_baselines.py", "scripts/baselines/notears/upstream/notears/linear.py",
             "scripts/experiments/paper_eval.py", "scripts/experiments/e2_planted.py",
             "scripts/experiments/e2_inspre.R"]
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        gpu = "none"
    return {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "date": datetime.now(timezone.utc).isoformat(),
            "hardware": {"cpu": cpu, "logical_cpus": os.cpu_count(), "gpu": gpu, "host": platform.node()},
            "source_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}}


def run(methods, reps):
    uniq = {key(p): p for _, _, p in settings()}
    for rep in reps:
        for k, p in uniq.items():
            d = CACHE / k / f"rep{rep}"
            todo = [m for m in methods if not (d / f"{m}.npy").exists()]
            if not todo:
                continue
            d.mkdir(parents=True, exist_ok=True)
            try:
                expr, interv, genes, W = simulate(p, rep)
            except Exception as e:      # recorded per (setting, rep); never aborts the other settings
                (d / "simulate.err").write_text(repr(e))
                print(f"FAIL simulate {k} rep{rep}: {e!r}", flush=True)
                continue
            tmp = d / f"W.{os.getpid()}.tmp.npy"   # several method groups run concurrently
            np.save(tmp, W); tmp.replace(d / "W.npy")
            for m in todo:
                t0 = time.time()
                try:
                    S, info = score(m, expr, interv, genes, rep)
                except Exception as e:  # recorded, never silently dropped: collect lists it as missing
                    (d / f"{m}.err").write_text(repr(e))
                    print(f"FAIL {k} rep{rep} {m}: {e!r}", flush=True)
                    continue
                S = np.asarray(S, dtype=np.float32)
                assert S.shape == W.shape and np.isfinite(S).all(), f"{m}: bad score matrix"
                np.save(d / f"{m}.npy", S)
                (d / f"{m}.json").write_text(json.dumps({"elapsed_s": time.time() - t0, **info,
                                                         **provenance()}, default=str))
                print(f"{k} rep{rep} {m}: {evaluate(S, W)} ({time.time() - t0:.0f}s)", flush=True)


def collect(reps):
    import pandas as pd
    rows, missing = [], []
    for sweep, level, p in settings():
        for rep in range(reps):
            d = CACHE / key(p) / f"rep{rep}"
            for m in MAIN:
                f = d / f"{m}.npy"
                if not f.exists():
                    missing.append(f"{key(p)}/rep{rep}/{m}"); continue
                rows.append({"sweep": sweep, "level": level, "rep": rep, "method": CSV_NAME.get(m, m),
                             **evaluate(np.load(f), np.load(d / "W.npy"))})
    PAPER.mkdir(parents=True, exist_ok=True)
    cols = ["sweep", "level", "rep", "method", "auprc_direct", "auroc_direct", "auprc_total_only"]
    pd.DataFrame(rows, columns=cols).to_csv(PAPER / "e2_planted.csv", index=False)
    shifts = {}
    for _, _, p in settings():
        mx, f4 = [], []
        for rep in range(reps):
            W, m, kd, _ = planted(N_GENES, p["cycle_frac"], p["rho"], rep)
            T = np.linalg.inv(np.eye(N_GENES) - W)
            S = np.abs(T * (np.log1p(-kd) / np.diag(T))[None, :]); np.fill_diagonal(S, 0)
            mx.append(float(S.max())); f4.append(float((S > 4).mean()))
        shifts[key(p)] = {"max_abs_mean_shift_per_rep": mx, "frac_pairs_abs_shift_gt4_mean": float(np.mean(f4))}
    meta = {"n_genes": N_GENES, "knockdown_shift_stats": shifts, "n_control": N_CONTROL, "reps": reps, "default": DEFAULT, "sweeps": SWEEPS,
            "methods": [CSV_NAME.get(m, m) for m in MAIN], "csv_method_names": CSV_NAME, "missing": sorted(set(missing)),
            "doc": __doc__, **provenance()}
    (PAPER / "e2_planted_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(rows)} rows; "
          f"{len(set(missing))} (setting, rep, method) cells missing", flush=True)


def check():
    """Self-check of the generator and the metric definitions."""
    for cf in SWEEPS["cycle_frac"]:
        for rho in SWEEPS["rho"]:
            W, _, _, cyc = planted(N_GENES, cf, rho, rep=3)
            on = sum(len(c) for c in cyc)
            assert on == round(cf * N_GENES), (cf, on)
            r = np.abs(np.linalg.eigvals(W)).max()
            assert (np.isclose(r, rho) if cf else r < 1e-6), (cf, rho, r)
            assert np.all(np.diag(W) == 0)
            a = W != 0                   # SCCs are exactly the planted cycles
            from scipy.sparse.csgraph import connected_components
            ncomp, lab = connected_components(a.T, directed=True, connection="strong")
            assert ncomp == N_GENES - on + len(cyc)
            assert all(2 <= len(c) <= 5 for c in cyc)
    for _, _, p in settings():          # every figure point, every rep: exact knockdown, bounded rates
        for rep in range(REPS):
            W, m, kd, _ = planted(N_GENES, p["cycle_frac"], p["rho"], rep)
            T = np.linalg.inv(np.eye(N_GENES) - W)
            shift = T * (np.log1p(-kd) / np.diag(T))[None, :]   # column j: mean shift when j is knocked down
            assert np.allclose(np.diag(shift), np.log1p(-kd))
            assert m.max() + np.abs(shift).max() + 6 * p["noise"] < 34, (p, rep)   # rate < 6e14 << numpy's 9e18
    W, *_ = planted(N_GENES, 0.5, 0.8, rep=0)
    e = evaluate(np.abs(W), W)
    assert np.isclose(e["auprc_direct"], 1) and np.isclose(e["auroc_direct"], 1)
    T = np.linalg.inv(np.eye(N_GENES) - W)
    e = evaluate(np.abs(T) * (W == 0), W)   # pure total-effect scores rank indirect pairs first
    assert e["auprc_total_only"] > 0.99, e
    # orientation: mean_diff on noiseless-ish data puts the knockdown target in the column
    expr, interv, genes, W = simulate({**DEFAULT, "noise": 0.05, "cells": 400}, rep=0, n=30)
    S, _ = score("plugin", expr, interv, genes, 0)
    assert evaluate(S, W)["auroc_direct"] > 0.9, evaluate(S, W)
    print("check ok", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "run", "collect"])
    ap.add_argument("--methods", nargs="+", choices=MAIN, default=MAIN)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--rep-list", type=int, nargs="+", help="run only these replicates (shard slow methods)")
    a = ap.parse_args()
    {"check": check, "run": lambda: run(a.methods, a.rep_list or range(a.reps)), "collect": lambda: collect(a.reps)}[a.cmd]()


if __name__ == "__main__":
    main()
