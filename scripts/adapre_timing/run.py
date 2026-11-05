"""ADAPRE at its shipped cv_folds = 0 (released default), timed the same way as the stored E8 runs.

Same inputs, seed, its, solve_its, ncores and timer as scripts/experiments/e8_adapre (imported, not edited); only cv goes
from 5 to 0. All files stay in this folder: e8's TMP is redirected here. adapre_fit.R starts its timer before sourceCpp, and
the compile counts as part of the method, so each dataset's reported time is a cold run with an empty build cache.
"hepg2:cached" repeats HepG2 with its build kept, only to show the size of the compile; it is not a reported time.
The penalty path is compared with the stored 5-fold run (scores must not depend on the added CV).

  CUDA_VISIBLE_DEVICES="" bash scripts/experiments/run_env.sh python -B -u scripts/adapre_timing/run.py jurkat:cold k562:cold \
      > scripts/adapre_timing/run.log 2>&1
"""
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments/e8_adapre"))
import e8_adapre as em  # noqa: E402

STORED = em.OUT                                  # read only
os.environ["PATH"] = f"{ROOT / '.envs/inspre/bin'}:{os.environ['PATH']}"   # R env's compilers


def one(ds, k):
    em.TMP = HERE / "tmp" / ds                   # em.r_env and em.fit write here, not under scripts/results; one cache per dataset
    assert (k == "cold") != (em.TMP / "rcpp_cache").exists(), f"{ds}:{k} does not match the build cache state"
    old = json.loads((STORED / f"{ds}_adapre.json").read_text())
    s = old["settings"]
    z, labels, keep, panel, table = em.prepare_real(ds)
    path, meta = em.fit(z, labels, keep, "adapre", seed=26000 + em.DATASETS.index(ds), ncores=s["ncores"], its=s["its"],
                        solve_its=s["solve_its"], cv=0, log=HERE / f"{ds}_{k}.fit.log", blas_threads=s["ncores"])
    assert meta["settings"]["cv_folds"] == 0
    old_path, old_meta = em.load_fit(STORED / f"{ds}_adapre_Gpath.npz")
    same_genes = list(meta["kept_genes"]) == list(old_meta["kept_genes"])
    same_shape = path.shape == old_path.shape
    diff = float(np.nanmax(np.abs(path - old_path))) if same_genes and same_shape else None
    r = {"dataset": ds, "run": k, "K": meta["K"], "L": meta["L"], "n_cells": meta["n_cells"], "fit_seconds_cv0": meta["fit_seconds"],
         "fit_seconds_cv5_stored": old["fit_seconds"], "ratio_cv0_over_cv5": meta["fit_seconds"] / old["fit_seconds"],
         "same_kept_genes": same_genes, "path_max_abs_diff_vs_cv5": diff, "ncores": s["ncores"], "its": s["its"], "solve_its": s["solve_its"],
         "load_at_end": list(os.getloadavg())}
    print(json.dumps(r), flush=True)
    return r


if __name__ == "__main__":
    f = HERE / "times.json"
    out = json.loads(f.read_text()) if f.exists() else []   # add to earlier runs, never overwrite them
    for a in sys.argv[1:]:
        out.append(one(*a.split(":")))
        (HERE / "times.json").write_text(json.dumps(out, indent=1))
