"""Idle-GPU timing of the shipped I-SEM (src/ at git HEAD, option-A solver), for the paper's runtime table.

Fit: (train_batched over the 60 complementary pairs = 120 models, shipped config),
plus the GPU model, nvidia-smi before and after (no other compute process),
timer scope = train_batched only (data loading and the shift/SE build excluded). The scores are checked against
scripts/results/paper/reference_option_a/{ds}.npz (the pinned option-A reference), so the timed fit is the shipped one.
Writes only timing/{ds}.json and timing/{ds}.npz. Aborts without saving if src/ differs from HEAD or changes mid-fit.

Usage (repo root): bash scripts/experiments/run_env.sh python -u scripts/timing/fit_idle.py DATASET
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
for k, sub in [("XDG_CACHE_HOME", ""), ("TMPDIR", "tmp"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
               ("TRITON_CACHE_DIR", "triton"), ("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba")]:
    os.environ[k] = str(CACHE / sub)
    os.makedirs(os.environ[k], exist_ok=True)

import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_all_metrics import load_dataset  # noqa: E402
from src.models.interventional_nb_sem import build_intervention_targets, train_batched  # noqa: E402
from src.pipeline import GRNInferencePipeline, _complementary_pairs, _mean_ensemble, _sem_config  # noqa: E402

SOURCES = sorted((ROOT / "src/models").glob("*.py")) + [ROOT / "src/pipeline.py"]


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}


def gpu_state():
    return {"time": datetime.now(timezone.utc).isoformat(),
            "compute_apps": sh("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"),
            "gpu": sh("nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu",
                      "--format=csv,noheader")}


if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src"], cwd=ROOT).returncode:
    raise SystemExit("src/ differs from git HEAD: not the shipped model")
h0 = hashes()
ds = sys.argv[1]
expression, interventions, genes = load_dataset(ds)
u, delta, _, _, _, se = build_intervention_targets(expression, interventions, genes)
del expression
pipe = GRNInferencePipeline(device="cuda:0")
cfg = _sem_config(pipe.epochs, pipe.warmup_epochs, pipe.l1_weight, pipe.w_dropout)
pairs = _complementary_pairs(len(u), pipe.n_resamples)
print(f"{ds}: n={len(genes)} conditions={len(u)} models={len(pairs)}", flush=True)

torch.cuda.synchronize()
before = gpu_state()  # queried after CUDA init, so this process is listed; any other pid = co-tenant
t0 = time.monotonic()
total, loss, W, B = train_batched(u, delta, se, folds=[p[0] for p in pairs], seeds=[p[1] for p in pairs],
                                  config=cfg, device="cuda:0", return_params=True)
torch.cuda.synchronize()
elapsed = time.monotonic() - t0
after = gpu_state()
if hashes() != h0:
    raise SystemExit("model source files changed during the fit: results not saved")

scores = _mean_ensemble(np.abs(W * B[:, None, :]))
ref = np.load(ROOT / f"scripts/results/paper/reference_option_a/{ds}.npz")["S_a"]
off = ~np.eye(len(genes), dtype=bool)
maxrel = float(np.max(np.abs(scores - ref)) / np.max(np.abs(ref)))
rho = float(spearmanr(scores[off], ref[off]).statistic)
me = str(os.getpid())
others = lambda s: [ln for ln in s["compute_apps"].splitlines() if ln.strip() and ln.split(",")[0].strip() != me]  # noqa: E731
meta = {"dataset": ds, "seconds": elapsed, "n_genes": len(genes), "n_conditions": int(len(u)), "n_models": len(pairs),
        "timer_scope": "train_batched only (120 models trained together), torch.cuda.synchronize() on both sides; "
                       "excludes data loading, the shift/SE build and the readout (fit_idle.py)",
        "device": "cuda:0", "torch": torch.__version__, "cuda": torch.version.cuda,
        "code_commit": sh("git", "rev-parse", "HEAD"), "model_source_sha256": h0, "pid": os.getpid(),
        "nvidia_smi_before": before, "nvidia_smi_after": after,
        "other_processes_before": others(before), "other_processes_after": others(after),
        "idle": not others(before) and not others(after),
        "reference_check": {"max_relative_difference": maxrel, "spearman_offdiag": rho}}
np.savez_compressed(HERE / f"{ds}.npz", scores=scores)
(HERE / f"{ds}.json").write_text(json.dumps(meta, indent=1))
print(f"DONE {ds}: {elapsed:.1f}s idle={meta['idle']} maxrel={maxrel:.2e} rho={rho:.12f}", flush=True)
