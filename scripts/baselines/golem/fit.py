"""GOLEM-NV (Ng, Ghassami & Zhang, NeurIPS 2020) with the authors' code in upstream/, unmodified.
Runs inside .envs/golem (TensorFlow 2.15, GOLEM's own tf.compat.v1 graph code).

  python fit.py X.npy out.npz --mode readme [--seed 1]
  python fit.py X.npy out.npz --mode cv --targets targets.txt [--seed 1]

readme  The README's GOLEM-NV: GOLEM-EV (lambda_1 2e-2, lambda_2 5) gives B_init for GOLEM-NV (lambda_1 2e-3,
        lambda_2 5); num_iter 1e5, learning rate 1e-3; then upstream main.py's postprocess(B, graph_thres=0.3)
        (threshold, then drop the weakest edges until a DAG).
cv      The inspre authors' benchmark (inspre Rmd/python_methods_comparison/run_python_methods.py):
        fit_golem(X, lam) = GOLEM-EV then GOLEM-NV with the same lambda_1, lambda_2 5; 10-value lambda grid;
        5-fold StratifiedKFold by target; held-out GOLEM-NV likelihood; refit at the best lambda on all data,
        returned unthresholded as they did. Their seeds were random; here they are fixed.

Output: B (final) and B_raw (before postprocess), B[i, j] = edge i -> j (X = X B), plus a JSON meta string.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

UP = Path(__file__).resolve().parent / "upstream" / "src"
sys.path[:0] = [str(UP), str(UP / "models"), str(UP / "trainers")]

import tensorflow as tf  # noqa: E402

tf.compat.v1.disable_eager_execution()
import golem  # noqa: E402
import golem_model  # noqa: E402
import golem_trainer  # noqa: E402
from utils.train import postprocess  # noqa: E402

LAMBDA_GRID = np.exp(np.arange(np.log(1), np.log(1e-6), (np.log(1e-6) - np.log(1)) / 10 + 1e-8))[1:]  # inspre's


def golem_nv(X, lam_ev, lam_nv, seed):
    B_ev = golem.golem(X, lambda_1=lam_ev, lambda_2=5.0, equal_variances=True, seed=seed)
    return golem.golem(X, lambda_1=lam_nv, lambda_2=5.0, equal_variances=False, seed=seed, B_init=B_ev)


def heldout_nv_likelihood(B_train, X_test, lam):
    model = golem_model.GolemModel(X_test.shape[0], X_test.shape[1], lam, lambda_2=5.0,
                                   equal_variances=False, B_init=B_train)
    model.sess.run(tf.compat.v1.global_variables_initializer())
    _, likelihood, _, _ = golem_trainer.GolemTrainer().eval_iter(model, X_test)
    return float(likelihood)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("X")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["readme", "cv"], required=True)
    ap.add_argument("--targets")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    X = np.load(a.X).astype(np.float32)
    t0 = time.time()
    meta = {"method": "GOLEM-NV", "mode": a.mode, "seed": a.seed, "n": int(X.shape[0]), "d": int(X.shape[1]),
            "upstream": "github.com/ignavierng/golem @ 9e99e61", "tf": tf.__version__,
            "gpu": bool(tf.config.list_physical_devices("GPU"))}
    if a.mode == "readme":
        B_raw = golem_nv(X, 2e-2, 2e-3, a.seed)
        B = postprocess(B_raw, graph_thres=0.3)
        meta.update(lambda_1_ev=2e-2, lambda_1_nv=2e-3, lambda_2=5.0, graph_thres=0.3)
    else:
        from sklearn.model_selection import StratifiedKFold
        targets = np.array(Path(a.targets).read_text().split())
        fits = [golem_nv(X, lam, lam, a.seed) for lam in LAMBDA_GRID]
        eps = []
        for tr, te in StratifiedKFold(n_splits=5).split(X, targets):
            eps.append([heldout_nv_likelihood(golem_nv(X[tr], lam, lam, a.seed), X[te], lam) for lam in LAMBDA_GRID])
        best = int(np.argmin(np.mean(eps, 0)))
        B_raw = B = fits[best]
        meta.update(lambda_grid=LAMBDA_GRID.tolist(), heldout_likelihood=np.mean(eps, 0).tolist(), best_index=best,
                    lambda_1=float(LAMBDA_GRID[best]), lambda_2=5.0)
    meta["wall_seconds"] = time.time() - t0
    np.savez(a.out, B=np.asarray(B, np.float64), B_raw=np.asarray(B_raw, np.float64), meta=json.dumps(meta))


if __name__ == "__main__":
    main()
