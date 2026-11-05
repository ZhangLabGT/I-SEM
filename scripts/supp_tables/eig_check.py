"""Read-only check: is each stored option-A model's steady state an attracting one?

Linear dynamics dx/dt = -x + W x + input settle to x* = (I - W)^{-1} input iff every eigenvalue of W has real
part < 1. The path-sum (I - W)^{-1} = sum_k W^k needs the stronger rho(W) < 1. Inputs: per-model W saved by the
option-A reference fits (scripts/results/paper/cache/<ds>_W_models.npy).
"""
import json, sys
import numpy as np

CACHE = "scripts/results/paper/cache"
out = {}
for ds in ["hepg2", "jurkat", "rpe1", "k562"]:
    Ws = np.load(f"{CACHE}/{ds}_W_models.npy", mmap_mode="r")
    rho, re_max, n_real_gt1 = [], [], 0
    for W in Ws:
        ev = np.linalg.eigvals(np.asarray(W, dtype=np.float64))
        rho.append(float(np.abs(ev).max()))
        re_max.append(float(ev.real.max()))
    rho, re_max = np.array(rho), np.array(re_max)
    out[ds] = dict(models=len(Ws), n_genes=int(Ws.shape[1]),
                   rho_gt1=int((rho > 1).sum()), rho_max=float(rho.max()),
                   re_ge1=int((re_max >= 1).sum()), re_max=float(re_max.max()),
                   re_max_median=float(np.median(re_max)))
    print(ds, out[ds], flush=True)
json.dump(out, open("scripts/supp_tables/eig_check.json", "w"), indent=1)
