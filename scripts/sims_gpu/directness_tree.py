"""Run scripts/config_tests/directness.py unchanged against sims_gpu/res (score.RES and run_sims.OUT redirected)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "scripts/config_tests")]
import run_sims  # noqa: E402
import score  # noqa: E402
import directness  # noqa: E402

score.RES, score.PARTS = HERE / "res", HERE / "res/real_parts"
run_sims.OUT, run_sims.TMP = HERE / "res/sims", HERE / "tmp"
run_sims.e9.e8.TMP = HERE / "tmp"
assert directness.sc is score
directness.main()
