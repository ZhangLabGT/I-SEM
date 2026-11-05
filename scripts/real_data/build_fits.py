"""I-SEM readouts for the real_data scripts (keys S, W_mean, W_abs, B, membership, genes, delta, se, control_mean,
delta_jj, cond_target_gene_idx, n_c). No refit: every array is taken from (or recomputed from the per-model W in)
scripts/results/paper/cache/, the train_paper.py --write-reference fits.
  S = _mean_ensemble(|W * B|), W_mean = W.mean(0), W_abs = _mean_ensemble(|W|).
Checks (abort on failure): S bitwise == config_tests/run_real.py S == reference S_a; S, W_mean, W_abs recomputed
from the per-model W/B equal the cached readouts.

Run (repo root): bash scripts/experiments/run_env.sh python -u scripts/real_data/build_fits.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import _mean_ensemble  # noqa: E402

CACHE = ROOT / "scripts/results/paper/cache"
TREE = ROOT / "scripts/results/config_tests/real"
REF = ROOT / "scripts/results/paper/reference_option_a"
OUT = HERE / "fits"

OUT.mkdir(exist_ok=True)
for ds in ["hepg2", "rpe1", "jurkat", "k562"]:
    c = dict(np.load(CACHE / f"{ds}.npz", allow_pickle=True))
    meta = json.loads((CACHE / f"{ds}.json").read_text())
    W = np.load(CACHE / f"{ds}_W_models.npy")
    B = c["B_models"]
    S = _mean_ensemble(np.abs(W * B[:, None, :]))
    W_mean, W_abs = W.mean(0), _mean_ensemble(np.abs(W))
    assert np.array_equal(S, c["S_current"]) and np.array_equal(S, c["scores"])
    assert np.array_equal(W_mean, c["ensemble_W"]) and np.array_equal(W_abs, c["W_only"])
    assert np.array_equal(S, np.load(TREE / ds / "isem_c.npz")["S"]), f"{ds}: S != option-A tree"
    assert np.array_equal(S, np.load(REF / f"{ds}.npz")["S_a"]), f"{ds}: S != reference_option_a"
    new = dict(S=S, W_mean=W_mean, W_abs=W_abs, B=B, membership=c["membership"], genes=c["gene_names"],
               delta=c["delta"], se=c["se"], control_mean=c["control_mean"], delta_jj=c["delta_jj"],
               cond_target_gene_idx=c["cond_target_gene_idx"], n_c=c["n_c"])
    np.savez_compressed(OUT / f"{ds}.npz", **new)
    (OUT / f"{ds}.json").write_text(json.dumps(dict(
        dataset=ds, source=str((CACHE / f"{ds}.npz").relative_to(ROOT)), code_commit=meta["code_commit"],
        source_sha256=meta["source_sha256"]), indent=1))
    print(ds, "ok", flush=True)
