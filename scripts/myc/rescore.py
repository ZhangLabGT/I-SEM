"""MYC -> Muhar 2018 targets (Jurkat), every row scored by size |A|.

Same statistic, gene sets, baseline selection rules, seed, bootstrap draws and Holm family size as
scripts/real_data/myc_sig.py; that script and its stored output are only read. The one change: each
row's MYC column is |A| instead of check_claims.activation. Holm pairs are I-SEM S and I-SEM |W| vs each network row,
because I-SEM's signed W has no separate row once scored by size. ADAPRE's oracle penalty is re-picked by size.

Gate: the unmodified signed analysis must reproduce the stored myc_sig.json before anything is rescored.

Run from the repo root (CPU only):
  CUDA_VISIBLE_DEVICES="" bash scripts/experiments/run_env.sh python -B -u scripts/myc/rescore.py \
      > rescore.log 2>&1
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECK = HERE.parent / "real_data"
sys.path.insert(0, str(CHECK))
import check_claims as cc  # noqa: E402
import myc_sig as ms  # noqa: E402

SIGNED_ISEM, SIGNED_ACTIVATION = list(ms.ISEM), cc.activation


def run(L, activation, isem):
    """myc_sig.analyse with one scoring rule; a fresh seed 0 gives every run the same bootstrap draws."""
    cc.activation, ms.ISEM = activation, isem
    return ms.analyse(L, np.random.default_rng(0))


def close(a, b, path=""):
    """Recursive equality of two JSON-like results, within float rounding; returns the first mismatching path."""
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: keys differ"
        return next((m for k in a if (m := close(a[k], b[k], f"{path}/{k}"))), None)
    if isinstance(a, (list, tuple)):
        return next((m for i, (x, y) in enumerate(zip(a, b)) if (m := close(x, y, f"{path}/{i}"))), None)
    if isinstance(a, (bool, str)) or a is None:
        return None if a == b else f"{path}: {a} != {b}"
    return None if (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9 else f"{path}: {a} != {b}"


def selfcheck():
    """Size scoring ignores sign, and the AUROC code ranks as expected."""
    y = np.array([True, True, False, False])
    assert cc.auroc(np.array([3.0, 2.0, 1.0, 0.0]), y) == 1.0
    assert cc.auroc(np.array([0.0, 1.0, 2.0, 3.0]), y) == 0.0
    a = np.array([-3.0, 2.0, 1.0, -0.5])
    assert cc.auroc(np.abs(a), y) == cc.auroc(np.abs(-a), y) == 1.0


def main():
    selfcheck()
    L = {"jurkat": cc.rt.load("jurkat", cc.rt._symbols())}

    signed = json.loads(json.dumps(run(L, SIGNED_ACTIVATION, SIGNED_ISEM), default=float))
    stored = json.loads((CHECK / "out/myc_sig.json").read_text())
    bad = close(signed, stored)
    if bad:
        sys.exit(f"GATE FAILED: signed rerun does not reproduce myc_sig.json at {bad}")
    print("gate passed: signed rerun reproduces myc_sig.json", flush=True)

    size = run(L, lambda name, A: np.abs(A), ["I-SEM S", "I-SEM |W|"])
    (HERE / "myc_size.json").write_text(json.dumps(size, indent=1, default=float))

    lines = ["| row | signed AUROC | size AUROC [95% CI] | I-SEM S minus row [95% CI] | Holm p vs S | Holm p vs abs W |",
             "|---|---|---|---|---|---|"]
    for m, r in size["rows"].items():
        vs = r.get("vs I-SEM S"), r.get("vs I-SEM |W|")
        cells = [f"{m}", f"{stored['rows'][m]['auroc']:.3f}", f"{r['auroc']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"]
        cells += [f"{vs[0]['diff']:+.3f} [{vs[0]['ci95'][0]:+.3f}, {vs[0]['ci95'][1]:+.3f}]" if vs[0] else "",
                  f"{vs[0].get('p_holm', float('nan')):.3g}" if vs[0] else "",
                  f"{vs[1].get('p_holm', float('nan')):.3g}" if vs[1] else ""]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (HERE / "RESULT.md").write_text(
        f"# MYC test scored by size\n\n{size['n_targets']} targets, {size['n_genes']} genes, "
        f"{size['n_holm']} Holm tests. Reference rows (raw z, mean difference) are outside the Holm family, "
        f"so their Holm columns are blank or nan.\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    main()
