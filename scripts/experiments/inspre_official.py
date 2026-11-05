#!/usr/bin/env python3
"""Run the authors' official GWPS-style inspre fit on each paper panel.

The NPZ expression matrix supplies the paper's cell and gene panel ordering.
Raw h5ad observation metadata supplies per-cell guide identifiers and targets,
which are required for the authors' guide-efficacy filter and guide-level IVs.
All generated inputs, logs, and outputs stay under scripts/.
Input normalisation: genes z-scored against non-targeting controls (as run_all_baselines.py).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "scripts"
OUTPUT = STUDY / "results" / "paper" / "e1b_inspre_official"
SMOKE = STUDY / "tmp" / "e1b_smoke"
TMP = STUDY / "tmp"
RUNNER = STUDY / "experiments" / "run_inspre_official.R"
DATASETS = ("k562", "rpe1", "hepg2", "jurkat")
H5AD = {
    "k562": ROOT / "data" / "k562.h5ad",
    "rpe1": ROOT / "data" / "rpe1.h5ad",
    "hepg2": ROOT / "data" / "nadig2025" / "GSE264667_hepg2_raw_singlecell_01.h5ad",
    "jurkat": ROOT / "data" / "nadig2025" / "GSE264667_jurkat_raw_singlecell_01.h5ad",
}
SEEDS = {"k562": 26001, "rpe1": 26002, "hepg2": 26003, "jurkat": 26004}


def _decode_obs_column(obs: h5py.Group, key: str) -> np.ndarray:
    """Decode an AnnData categorical column as strings."""
    ds = obs[key]
    if ds.dtype.kind in "iu" and "__categories" in obs and key in obs["__categories"]:
        categories = obs["__categories"][key].asstr()[:]
        codes = ds[:]
        out = np.full(codes.shape, "", dtype=object)
        valid = (codes >= 0) & (codes < len(categories))
        out[valid] = categories[codes[valid]]
        return out.astype(str)
    if ds.dtype.kind in "OSU":
        return np.asarray(ds.asstr()[:], dtype=str)
    raise ValueError(f"Cannot decode obs/{key}: dtype={ds.dtype}")


def prepare_inputs(dataset: str, out: Path, max_genes: int | None):
    """Write the control-z-scored panel (genes x cells h5) and per-cell guide metadata."""
    source_npz = ROOT / "data" / f"dataset_{dataset}.npz"
    source_h5ad = H5AD[dataset]
    if not source_npz.is_file() or not source_h5ad.is_file():
        raise FileNotFoundError(f"Missing inputs: {source_npz} or {source_h5ad}")
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    stem = cache / f"{dataset}_inspre_official"

    with np.load(source_npz, allow_pickle=True) as data:
        expression = np.asarray(data["expression_matrix"], dtype=np.float32)
        genes = [str(x) for x in data["var_names"]]
        conditions = np.asarray([str(x).strip() for x in data["interventions"]], dtype=str)
    if expression.shape != (len(conditions), len(genes)) or len(set(genes)) != len(genes):
        raise ValueError(f"Malformed panel arrays for {dataset}: {expression.shape}")
    if max_genes:
        genes, expression = genes[:max_genes], expression[:, :max_genes]

    with h5py.File(source_h5ad, "r") as h5:
        obs = h5["obs"]
        guide_ids = _decode_obs_column(obs, "gene_transcript")
        obs_targets = _decode_obs_column(obs, "gene_id")
    if len(guide_ids) != expression.shape[0]:
        raise ValueError(f"Cell-row mismatch for {dataset}: npz {expression.shape[0]}, h5ad {len(guide_ids)}")
    bad = (conditions != "excluded") & (conditions != obs_targets)
    if bad.any():
        r = int(np.flatnonzero(bad)[0])
        raise ValueError(f"npz/h5ad target mismatch at row {r}: {conditions[r]!r} vs {obs_targets[r]!r}")
    # Controls: obs target is non-targeting. 'excluded' (off-panel targets) cells are NOT controls.
    control = obs_targets == "non-targeting"
    if control.sum() < 2:
        raise ValueError(f"No non-targeting controls in {dataset}")
    guide_ids = guide_ids.copy()
    guide_ids[control] = "control"

    # Same convention as scripts/run_all_baselines.py::standardize_to_control (sd 0 -> 1).
    mu = expression[control].mean(axis=0)
    sd = expression[control].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = ((expression - mu) / sd).astype(np.float32)

    h5_path = stem.with_suffix(".h5")
    if h5_path.exists():
        h5_path.unlink()
    # Stored cells x genes in C order like an h5ad X; hdf5r reverses dims, so R sees genes x cells,
    # which is what fit_inspre_from_h5X expects (the runner asserts dim(X) == genes x cells).
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=z, dtype="float32", chunks=(64, len(genes)))  # no lzf: R HDF5 lacks the filter
    paths = [stem.with_suffix(e) for e in (".guides.txt", ".targets.txt", ".genes.txt")]
    for p, v in zip(paths, (guide_ids, obs_targets, np.asarray(genes))):
        p.write_text("\n".join(v) + "\n", encoding="utf-8")
    return h5_path, *paths, genes


def _resolve_rscript(configured: str | None) -> str:
    local = ROOT / ".envs" / "inspre" / "bin" / "Rscript"
    cand = Path(configured) if configured else local
    if not cand.is_file():
        raise FileNotFoundError(f"Rscript not found: {cand}")
    return str(cand)


def _hardware() -> dict:
    cpu = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count() or 0,
            "gpu": "not used (CPU-only E1b)"}


def _commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def run_one(dataset: str, rscript: str, overwrite: bool, out: Path, max_genes: int | None, ncores: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    score_path, detail_path = out / f"{dataset}.npy", out / f"{dataset}.json"
    gpath_npz, log_path = out / f"{dataset}_Gpath.npz", out / "logs" / f"{dataset}.log"
    if score_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {score_path}")

    h5_path, guides, targets, genes_path, panel_genes = prepare_inputs(dataset, out, max_genes)
    raw = out / f".{dataset}.run"
    env = {k: v for k, v in os.environ.items() if k not in ("R_HOME", "R_LIBS_USER", "R_LIBS", "R_LIBS_SITE")}
    env.update(R_LIBS_USER="", R_LIBS="", R_LIBS_SITE="", R_PROFILE_USER="/dev/null",
               R_ENVIRON_USER="/dev/null", PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(TMP))
    TMP.mkdir(parents=True, exist_ok=True)
    cmd = [rscript, str(RUNNER), str(h5_path), str(guides), str(targets), str(genes_path),
           str(raw), str(SEEDS[dataset]), str(ncores)]
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"dataset={dataset} commit={_commit()} max_genes={max_genes}\ncommand={json.dumps(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        raise RuntimeError(f"inspre failed for {dataset} (exit {proc.returncode}); see {log_path}")

    meta = json.loads(Path(str(raw) + ".meta.json").read_text(encoding="utf-8"))
    kept = meta["kept_genes"]
    if isinstance(kept, str):
        kept = [kept]
    K, L = len(kept), meta["L"]
    path = np.fromfile(str(raw) + ".Gpath.bin", dtype=np.float32).reshape((K, K, L), order="F")
    sel = meta["selection"]
    np.savez(gpath_npz, W_path=path, lambda_=np.asarray(sel["lambda"]), D_hat=np.asarray(sel["D_hat"]),
             eps_hat_G=np.asarray(sel["eps_hat_G"]), eps_hat_I=np.asarray(sel["eps_hat_I"]),
             kept_genes=np.asarray(kept))
    W = path[:, :, sel["chosen_index"] - 1].copy()  # R index is 1-based
    W[np.abs(W) < 0.015] = 0.0
    idx = {g: i for i, g in enumerate(panel_genes)}
    pos = np.asarray([idx[g] for g in kept], dtype=int)
    full = np.zeros((len(panel_genes),) * 2, dtype=np.float32)
    full[np.ix_(pos, pos)] = W
    np.fill_diagonal(full, 0.0)
    if not np.isfinite(full).all():
        raise ValueError("non-finite scores")
    np.save(score_path, full, allow_pickle=False)
    meta.update(
        dataset=dataset, panel_genes=panel_genes, max_genes=max_genes, panel_shape=list(full.shape),
        orientation="W[i,j]=effect of j on i = t(inspre G); signed; |W| used by the evaluator",
        input_normalisation="genes z-scored vs non-targeting control cells (mean/sd of controls, sd 0 -> 1); "
                            "as scripts/run_all_baselines.py::standardize_to_control",
        n_nonzero=int((full != 0).sum()), gpath_file=str(gpath_npz.relative_to(ROOT)),
        elapsed_seconds=elapsed, commit=_commit(), hardware=_hardware(),
        run_date_utc=dt.datetime.now(dt.timezone.utc).isoformat(), log=str(log_path.relative_to(ROOT)))
    detail_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for f in out.glob(f".{dataset}.run*"):
        f.unlink()
    print(f"{dataset}: wrote {score_path} shape={full.shape} kept={K} nnz={meta['n_nonzero']} "
          f"lambda_idx={sel['chosen_index']} elapsed={elapsed:.1f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS + ("all",), default="all")
    p.add_argument("--rscript", help="Rscript path (default .envs/inspre/bin/Rscript)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-genes", type=int, help="smoke test: first K panel genes; writes to scripts/tmp/e1b_smoke")
    p.add_argument("--ncores", type=int, default=16)
    a = p.parse_args()
    rscript = _resolve_rscript(a.rscript)
    out = SMOKE if a.max_genes else OUTPUT
    for d in DATASETS if a.dataset == "all" else (a.dataset,):
        run_one(d, rscript, a.overwrite or bool(a.max_genes), out, a.max_genes, a.ncores)


if __name__ == "__main__":
    main()
