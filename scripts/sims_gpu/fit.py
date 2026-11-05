"""Shipped I-SEM (c) (isem_variants.fit_c, unchanged; src/ at git HEAD = option-A solver) on cuda:0 for the three
locked simulation settings (run_sims.main_datasets(): inspre rand_cycles_confound_7 + ADAPRE demo_set1/demo_set2, 10 reps
each). Writes only to res/.
Guard: src/ must equal git HEAD; model sources hashed before/after every fit.
Usage (repo root): flock scripts/gpu.lock bash scripts/experiments/run_env.sh python -u <this>
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/config_tests")]
import run_sims  # noqa: E402
from isem_variants import fit_c  # noqa: E402

run_sims.OUT = HERE / "res/sims"
run_sims.TMP = run_sims.e9.e8.TMP = HERE / "tmp"
SOURCES = sorted((ROOT / "src/models").glob("*.py")) + [ROOT / "src/pipeline.py"]
hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}  # noqa: E731

if subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "src"]).returncode:
    raise SystemExit("src/ differs from git HEAD: refusing to fit")
commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
e9 = run_sims.e9
for d in run_sims.main_datasets():
    f = run_sims.OUT / d.relative_to(e9.DATA) / "isem_c.npz"
    if f.exists():
        print(f"{f.relative_to(HERE)} cached", flush=True)
        continue
    _, Y, labels, genes, _ = e9.load(d)
    h0 = hashes()
    torch.cuda.synchronize()
    S, meta = fit_c(Y, labels, genes, "cuda:0")
    torch.cuda.synchronize()
    if hashes() != h0:
        raise SystemExit("model source files changed during the fit: not saved")
    meta.update(device="cuda:0", code_commit=commit, model_source_sha256=h0)
    run_sims.save(f, meta, S=np.asarray(S, np.float32))
    print(f"{f.relative_to(HERE)}: {meta['wall_s']:.1f}s", flush=True)
