#!/usr/bin/env python3
"""E1 (Table 1) and E3 (Table 2) from fresh post-Part-A fits; every write stays under scripts/results/paper.

  python -u scripts/experiments/paper_tables.py e3 --ds hepg2 --ref STRING   # one (dataset, reference) part
  python -u scripts/experiments/paper_tables.py e1 --ds k562  --ref CORUM
  python -u scripts/experiments/paper_tables.py merge e3                      # parts -> e3_ablation.json
  python -u scripts/experiments/paper_tables.py merge e1                      # parts -> e1_benchmark.json, e1_norms.json

Inputs: scripts/results/paper/cache/<ds>[_<variant>].npz from train_paper.py (each validated against the
Part A acceptance rule), baselines as their runs saved them in results/<method>_adj_<ds>.npy (scored as
provided: |A|, diagonal zeroed, never thresholded or symmetrized), inspre's official E1b output, and the
plug-in / control-expression controls (no trained model).
Schema: {ds: {ref: {method: {metric: value, metric_ci: [lo, hi], ...}}}, meta: {...}}; the E1 inspre
kept-gene evaluation uses ref names "<ref>@inspre_kept_genes". inspre appears only as the official E1b
run (baselines run as their authors provide them). E1 also reports precision/recall of official inspre's own edges and of I-SEM's top k
pairs, k = inspre's edge count on that panel (row MATCHED_K; only I-SEM is thresholded).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import score_eval as se  # noqa: E402

ROOT = se.ROOT
PAPER = ROOT / "scripts/results/paper"
CACHE = PAPER / "cache"
PARTS = PAPER / "parts"
DATASETS = ["k562", "rpe1", "hepg2", "jurkat"]
MATCHED_K = "I-SEM (top k = official inspre edge count)"
INSPRE = "inspre (official, E1b)"
BASELINES = {  # results/<file>_<ds>.npy from scripts/run_all_baselines.py; inspre = E1b official (below)
    "Mean difference": "mean_diff_adj", "NOTEARS": "notears_adj"}


def load(ds, variant="full"):
    f = CACHE / (f"{ds}.npz" if variant == "full" else f"{ds}_{variant}.npz")
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/experiments/train_paper.py --variants {variant}")
    with np.load(f, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def plugin_direct(z):
    """T[:, pert] = delta.T; W_hat = I - diag(1 / diag(T^-1)) T^-1."""
    n = len(z["gene_names"]); T = np.zeros((n, n)); T[:, z["cond_target_gene_idx"]] = z["delta"].T
    Ti = np.linalg.inv(T)
    return np.eye(n) - (1 / np.diag(Ti))[:, None] * Ti


def e3_scores(ds):
    z = load(ds)
    return {"Full I-SEM": z["S_current"],
            "Uniform weights (no 1/SE)": load(ds, "uniform")["scores"],
            "Stability penalty off": load(ds, "no_stability")["scores"],
            "Total readout |(I-W)^-1 B|": z["total_readout"],
            "Symmetrized score": z["symmetrized"],
            "In-half B only": z["S_fixed"],
            "M = 2 models": load(ds, "two_models")["scores"],
            "Plug-in Ŵ (no learning)": plugin_direct(z)}


def e1_scores(ds):
    z = load(ds)
    n = len(z["gene_names"])
    S = {"I-SEM": z["S_current"]}
    skipped = {}
    for m, f in BASELINES.items():
        p = ROOT / f"results/{f}_{ds}.npy"
        if not p.exists():
            skipped[m] = "not available for this dataset"; continue
        A = np.load(p)
        if A.shape != (n, n):
            skipped[m] = f"shape {A.shape} != panel"; continue
        S[m] = A
    off = PAPER / f"e1b_inspre_official/{ds}.npy"
    if off.exists():
        S[INSPRE] = np.load(off)
    else:
        skipped[INSPRE] = "E1b output missing"
    S["Plug-in direct |Ŵ|"] = plugin_direct(z)
    S["Control expression e_i·e_j"] = np.outer(z["control_mean"], z["control_mean"])
    return z, S, skipped


def matched_k(res, isem, inspre, G, n_boot, seed=0):
    """Precision/recall of official inspre's nonzero edges and of I-SEM's top-k pairs, k = inspre's edge
    count (directed, off-diagonal). Node-bootstrap CIs and the paired I-SEM - inspre difference use
    score_eval's draw scheme; the two edge sets are fixed on the full data, only the labels resample."""
    n = len(G); off = ~np.eye(n, dtype=bool)
    ins = (se.pe.absd(inspre) > 0) & off
    k = int(ins.sum())
    s = np.where(off, se.pe.absd(isem), -np.inf).ravel()
    top = np.zeros(n * n, bool); top[np.argsort(-s, kind="mergesort")[:k]] = True
    sel = {MATCHED_K: top.reshape(n, n), INSPRE: ins}
    pos = G.astype(bool) & off

    def pr(A, Y, P):
        tp = Y[A].sum()
        return {"precision_at_k": tp / max(A.sum(), 1), "recall_at_k": tp / P.sum()}
    point = {m: pr(A, pos, pos) for m, A in sel.items()}
    rng = np.random.default_rng(seed); p, q = np.nonzero(off)
    draws = {m: {key: [] for key in ("precision_at_k", "recall_at_k")} for m in sel}
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        keep = idx[p] != idx[q]; a, b = idx[p[keep]], idx[q[keep]]
        Y = pos[a, b]
        for m, A in sel.items():
            for key, v in pr(A[a, b], Y, Y).items():
                draws[m][key].append(float(v))
    for m in sel:
        row = res.setdefault(m, {})
        row.update({"k": k, **{key: float(v) for key, v in point[m].items()}})
        row.update({f"{key}_ci": se.pe.ci(np.asarray(v)) for key, v in draws[m].items()})
    res[MATCHED_K]["isem_minus_inspre_official"] = {
        key: {"difference": point[MATCHED_K][key] - point[INSPRE][key],
              "difference_ci": se.pe.ci(np.subtract(draws[MATCHED_K][key], draws[INSPRE][key])),
              "p_le0": float(np.mean(np.subtract(draws[MATCHED_K][key], draws[INSPRE][key]) <= 0))}
        for key in ("precision_at_k", "recall_at_k")}


def run_part(exp, ds, ref, n_boot):
    PARTS.mkdir(parents=True, exist_ok=True)
    out = {}
    if exp == "e3":
        S = e3_scores(ds); genes = load(ds)["gene_names"]
        G = se.references(genes)[ref]
        out[ref] = se.score_reference({("I-SEM" if k == "Full I-SEM" else k): v for k, v in S.items()},
                                      G, n_boot=n_boot, baselines=[k for k in S if k != "Full I-SEM"])
        out[ref]["Full I-SEM"] = out[ref].pop("I-SEM")
    else:
        z, S, skipped = e1_scores(ds)
        genes = list(z["gene_names"])
        G = se.references(genes)[ref]
        out[ref] = se.score_reference(S, G, n_boot=n_boot)
        if INSPRE in S:
            matched_k(out[ref], S["I-SEM"], S[INSPRE], G, n_boot)
        out["_skipped"] = skipped
        kept_json = PAPER / f"e1b_inspre_official/{ds}.json"
        if kept_json.exists():          # every method on inspre's kept genes (§4.3)
            kept = json.loads(kept_json.read_text())["kept_genes"]
            idx = np.array([genes.index(g) for g in kept])
            sub = {m: A[np.ix_(idx, idx)] for m, A in S.items()}
            out[f"{ref}@inspre_kept_genes"] = r = se.score_reference(sub, G[np.ix_(idx, idx)], n_boot=n_boot)
            matched_k(r, sub["I-SEM"], sub[INSPRE], G[np.ix_(idx, idx)], n_boot)
    f = PARTS / f"{exp}_{ds}_{ref}.json"
    f.write_text(json.dumps({"ds": ds, "parts": out, "meta": se.meta({"n_boot": n_boot})}, indent=1, allow_nan=True))
    print("wrote", f, flush=True)


def merge(exp):
    res, missing, metas = {}, [], {}
    for ds in DATASETS:
        for ref in ("STRING", "CORUM"):
            f = PARTS / f"{exp}_{ds}_{ref}.json"
            if not f.exists():
                missing.append(f"{ds}/{ref}"); continue
            d = json.loads(f.read_text())
            res.setdefault(ds, {}).update({k: v for k, v in d["parts"].items() if not k.startswith("_")})
            if "_skipped" in d["parts"]:
                res[ds]["_skipped_methods"] = d["parts"]["_skipped"]
            metas[f"{ds}/{ref}"] = d["meta"]
    for ds in res:                      # provenance of every fit that fed this dataset
        res[ds]["_fit_provenance"] = {f.stem: json.loads(f.read_text()) for f in sorted(CACHE.glob(f"{ds}*.json"))
                                      if not f.stem.endswith("_norms")}
    name = {"e1": "e1_benchmark.json", "e3": "e3_ablation.json"}[exp]
    res["meta"] = {**se.meta(), "parts": metas, "missing_parts": missing}
    (PAPER / name).write_text(json.dumps(res, indent=1))
    print(f"wrote {PAPER / name}; missing parts: {missing}", flush=True)
    if exp == "e1":
        write_norms()


def write_norms():
    """e1_norms.json from the full fits' per-model diagnostics (A2/A3), for P2 and Fig 5b."""
    if True:
        norms = {}
        for ds in DATASETS:
            f = CACHE / f"{ds}_norms.json"
            if f.exists():
                d = json.loads(f.read_text())
                norms[ds] = {k: d[k] for k in ("sigma_abs", "norm2_W", "rho_W", "min_pivot", "fallbacks")}
                norms[ds]["sigma_min_IminusW"] = d.get("sigma_min_IminusW")
        (PAPER / "e1_norms.json").write_text(json.dumps(norms, indent=1))
        print(f"wrote e1_norms.json for {sorted(norms)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["e1", "e3", "merge", "norms"])
    ap.add_argument("exp", nargs="?", choices=["e1", "e3"])
    ap.add_argument("--ds", choices=DATASETS)
    ap.add_argument("--ref", choices=["STRING", "CORUM"])
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    if a.cmd == "merge":
        merge(a.exp)
    elif a.cmd == "norms":
        write_norms()
    else:
        run_part(a.cmd, a.ds, a.ref, a.n_boot)


if __name__ == "__main__":
    main()
