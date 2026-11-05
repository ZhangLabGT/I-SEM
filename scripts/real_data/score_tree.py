"""Run scripts/config_tests score.py and direction.py unchanged against a scratch RES tree.
Every output global is redirected (score.RES, score.PARTS; direction writes via score.RES); run_sims' TMP too.

  python -u score_tree.py TREE [score.py args...]      e.g.  res_gpu_A --skip-sims --real hepg2
  python -u score_tree.py TREE --direction             direction.py only (sims)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "scripts/config_tests")]
import run_sims  # noqa: E402
import score  # noqa: E402
import direction  # noqa: E402

tree = (HERE / sys.argv[1]).resolve()
assert tree.parent == HERE, tree
tree.mkdir(exist_ok=True)
score.RES, score.PARTS = tree, tree / "real_parts"
run_sims.OUT, run_sims.TMP = tree / "sims", HERE / "tmp"
run_sims.e9.e8.TMP = HERE / "tmp"
assert direction.sc is score
args = sys.argv[2:]
if args == ["--direction"]:
    direction.main()
else:
    sys.argv = ["score.py", *args]
    score.main()
