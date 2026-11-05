"""IGSP (Wang, Solus, Yang & Uhler, NeurIPS 2017) with the authors' package, unmodified: causaldag 0.1a163
(`from causaldag import igsp`; the code is graphical_model_learning's). Runs inside .envs/igsp.

  python fit.py X.npy targets.npy out.npz [--seed 0]

X        cells x genes; targets[c] = index of the gene knocked down in cell c, -1 for a control cell.
Call     as the inspre authors' run_igsp (inspre Rmd/python_methods_comparison/helper_functions.py) and the
         causaldag docs: obs = control cells, one setting dict(interventions=[j]) per knocked-down gene,
         partial_correlation_suffstat(obs), gauss_invariance_suffstat(obs, [cells of each setting]),
         MemoizedCI_Tester(partial_correlation_test, alpha=1e-3),
         MemoizedInvarianceTester(gauss_invariance_test, alpha=1e-3), igsp(settings, nodes, ci, inv) with
         igsp()'s own defaults (depth 4, nruns 5, initial_undirected 'threshold'). alpha: see SOURCE.md.
Seed     igsp draws from Python's `random` (min-degree tie breaks, neighbour shuffles); seeded here.
Output   A (binary, A[i, j] = 1 means i -> j, est_dag.to_amat()[0]) and a JSON meta string.
"""
import argparse
import inspect
import json
import random
import time

import numpy as np

import conditional_independence as ci
from causaldag import igsp
from importlib.metadata import version

ALPHA = 1e-3      # CI test (partial correlation); causaldag/graphical_model_learning docs, see SOURCE.md
ALPHA_INV = 1e-3  # invariance test; causaldag UT-IGSP docs, see SOURCE.md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("X")
    ap.add_argument("targets")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    X = np.load(a.X).astype(np.float64)
    t = np.load(a.targets)
    p = X.shape[1]
    ko = sorted(set(t[t >= 0].tolist()))
    t0 = time.time()
    random.seed(a.seed)
    np.random.seed(a.seed)
    obs = X[t == -1]
    settings = [dict(interventions=[j]) for j in ko]
    ci_tester = ci.MemoizedCI_Tester(ci.partial_correlation_test, ci.partial_correlation_suffstat(obs), alpha=ALPHA)
    inv_tester = ci.MemoizedInvarianceTester(ci.gauss_invariance_test,
                                             ci.gauss_invariance_suffstat(obs, [X[t == j] for j in ko]),
                                             alpha=ALPHA_INV)
    est_dag = igsp(settings, list(range(p)), ci_tester, inv_tester)
    A = est_dag.to_amat(list(range(p)))[0]
    meta = {"method": "IGSP", "seed": a.seed, "alpha": ALPHA, "alpha_inv": ALPHA_INV,
            "igsp_defaults": {k: v.default for k, v in inspect.signature(igsp).parameters.items()
                              if v.default is not inspect.Parameter.empty},
            "n_obs": int(len(obs)), "n_settings": len(ko), "n": int(len(X)), "d": p,
            "n_ci_tests": len(ci_tester.ci_dict), "n_invariance_tests": len(inv_tester.invariance_dict),
            "upstream": {k: version(k) for k in ("causaldag", "graphical-model-learning", "graphical-models",
                                                 "conditional-independence", "numpy", "scipy")},
            "wall_seconds": time.time() - t0}
    np.savez(a.out, A=A.astype(np.int8), meta=json.dumps(meta))


if __name__ == "__main__":
    main()
