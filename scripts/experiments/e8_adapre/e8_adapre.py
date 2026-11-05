#!/usr/bin/env python3
"""E8: ADAPRE (Sun, Kang & Keles, RECOMB 2026) as a baseline against I-SEM. Every write stays under
scripts/results/paper/e8_adapre/ (logs in logs/); nothing in other result folders is touched.

Fits (adapre_fit.R): ADAPRE as released, the same code with adaptive_lambda = FALSE, and inspre's
fit_inspre_from_X at package defaults. Each fit saves the whole lambda path; lambda rules are applied here:
  dbar        PRIMARY. Rule of our official inspre run (run_inspre_official.R header): smallest lambda with
              D_bar <= 0.01, D_bar = cummax of D_hat from the largest to the smallest lambda; else the
              largest lambda (flagged). The ADAPRE paper states no selection rule (ADAPTATIONS.md).
  min_eps_g   which.min(eps_hat_G), the 5-fold CV prediction error (the rule e2_inspre.R uses).
  min_train   which.min(train_error), the demo's non-oracle rule.
  oracle_f1   E2 only, ORACLE: max F1 against the true graph (the demo's select_best_by_f1).
  matched     E2 only, uses the TRUE EDGE COUNT: lambda whose edge count is closest to |E(W)|.
Scores are |W| at the chosen lambda in our orientation (W[i, j] = effect of j on i), diagonal zero, not
symmetrized, not thresholded. Edge sets (precision/recall/F1/SHD, k) use |W| > EDGE_THR = 1e-3, the demo's
edge_thr; SHD = FP + FN over ordered pairs (a reversed edge counts 2; the paper does not define SHD).

  python -u scripts/experiments/e8_adapre/e8_adapre.py check
  python -u scripts/experiments/e8_adapre/e8_adapre.py e2 --default-only --workers 10
  python -u scripts/experiments/e8_adapre/e8_adapre.py e2 --workers 10            # full sweep
  python -u scripts/experiments/e8_adapre/e8_adapre.py e2collect
  python -u scripts/experiments/e8_adapre/e8_adapre.py real --ds hepg2 --ncores 8
  python -u scripts/experiments/e8_adapre/e8_adapre.py score --ds hepg2
  python -u scripts/experiments/e8_adapre/e8_adapre.py bias
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, ttest_ind

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STUDY / "experiments"))
import e2_planted as e2  # noqa: E402

PAPER = STUDY / "results/paper"
OUT = PAPER / "e8_adapre"
TMP = OUT / "tmp"
RSCRIPT = ROOT / ".envs/inspre/bin/Rscript"
RUNNER = HERE / "adapre_fit.R"
EDGE_THR = 1e-3
DBAR_MAX = 0.01
FIT_METHODS = ("adapre", "adapre_noadapt", "inspre_default")
LABEL = {"adapre": "ADAPRE", "adapre_noadapt": "ADAPRE code, adaptive off", "inspre_default": "inspre (package defaults)"}
DATASETS = ("hepg2", "jurkat", "rpe1", "k562")      # run order (runtime is probed on HepG2 first)
E6_ORDER = ("k562", "rpe1", "hepg2", "jurkat")      # analyze.py's dataset order (its rng is shared)


# ----------------------------------------------------------------------------------------- fitting
def r_env(blas_threads):
    env = {k: v for k, v in os.environ.items() if k not in ("R_HOME", "R_LIBS_USER", "R_LIBS", "R_LIBS_SITE")}
    TMP.mkdir(parents=True, exist_ok=True)
    env.update(PATH=f"{ROOT / '.envs/inspre/bin'}:{os.environ['PATH']}",   # the R env's compilers, for ADAPRE's sourceCpp
               R_LIBS_USER="", R_LIBS="", R_LIBS_SITE="", R_PROFILE_USER="/dev/null", R_ENVIRON_USER="/dev/null",
               TMPDIR=str(TMP), E8_RCPP_CACHE=str(TMP / "rcpp_cache"), OPENBLAS_NUM_THREADS=str(blas_threads),
               OMP_NUM_THREADS=str(blas_threads), MKL_NUM_THREADS=str(blas_threads))
    return env


def fit(z, labels, genes, method, seed, ncores, its, solve_its, cv=5, log=None, blas_threads=1):
    """Run adapre_fit.R; return (W_path [K, K, L] float32, meta). labels: 'control' or a column of z."""
    TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TMP) as d:
        d = Path(d)
        np.ascontiguousarray(z, np.float32).tofile(d / "x.bin")
        (d / "t.txt").write_text("\n".join(labels) + "\n")
        (d / "g.txt").write_text("\n".join(genes) + "\n")
        cmd = [str(RSCRIPT), str(RUNNER), str(d / "x.bin"), str(len(z)), str(d / "t.txt"), str(d / "g.txt"),
               str(d / "out"), method, str(seed), str(ncores), str(its), str(solve_its), str(cv)]
        t0 = time.time()
        if log is None:
            r = subprocess.run(cmd, capture_output=True, text=True, env=r_env(blas_threads))
            if r.returncode:
                raise RuntimeError(f"{method} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        else:
            with open(log, "a") as fh:
                fh.write(f"command={json.dumps(cmd)}\n"); fh.flush()
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=r_env(blas_threads))
            if r.returncode:
                raise RuntimeError(f"{method} failed (exit {r.returncode}); see {log}")
        meta = json.loads((d / "out.meta.json").read_text())
        K, L = meta["K"], meta["L"]
        path = np.fromfile(d / "out.Gpath.bin", np.float32).reshape((K, K, L), order="F")
        meta["wall_seconds"] = time.time() - t0
        if isinstance(meta["kept_genes"], str):
            meta["kept_genes"] = [meta["kept_genes"]]
        return path, meta


def rule_dbar(D_hat, lam):
    """run_inspre_official.R: smallest lambda with cummax(D_hat, largest -> smallest lambda) <= 0.01."""
    ord_ = np.argsort(-np.asarray(lam), kind="stable")
    dbar = np.maximum.accumulate(np.asarray(D_hat, float)[ord_])
    ok = np.flatnonzero(dbar <= DBAR_MAX)
    return (int(ord_[0]), True) if len(ok) == 0 else (int(ord_[ok.max()]), False)


def rules(meta):
    """{rule: lambda index} for the non-oracle rules, plus the D_bar fallback flag."""
    out = {"min_train": int(np.argmin(meta["train_error"]))}
    if meta.get("D_hat") is not None:
        out["dbar"], fb = rule_dbar(meta["D_hat"], meta["lambda"])
        out["dbar_fallback"] = fb
    if meta.get("eps_hat_G") is not None:
        out["min_eps_g"] = int(np.argmin(meta["eps_hat_G"]))
    return out


def embed(Wk, kept, genes):
    """K x K score on the kept genes -> full panel (zeros for dropped genes), diagonal zero. Non-finite entries
    (row p of G = -V[p, ] / V_pp with V_pp shrunk to 0 by the solver's lasso; see adapre_fit.R) are scored 0:
    the gene's out-effects are undefined at that lambda. Counts are recorded per lambda in the fit meta."""
    Wk = np.where(np.isfinite(Wk), Wk, 0)
    idx = {g: i for i, g in enumerate(genes)}
    pos = np.asarray([idx[g] for g in kept], int)
    S = np.zeros((len(genes),) * 2, np.float32)
    S[np.ix_(pos, pos)] = Wk
    np.fill_diagonal(S, 0)
    return S


def save_fit(f, path, meta):
    np.savez_compressed(f, W_path=path, meta=np.asarray(json.dumps(meta)))


def load_fit(f):
    with np.load(f) as z:
        return z["W_path"], json.loads(str(z["meta"]))


# ----------------------------------------------------------------------------------------- metrics
def edge_metrics(S, W):
    """precision/recall/F1/SHD of the edge set |S| > EDGE_THR against W != 0 (ordered pairs, i != j)."""
    off = ~np.eye(len(W), dtype=bool)
    est = (np.abs(S) > EDGE_THR) & off
    tru = (W != 0) & off
    return _prf(est, tru)


def _prf(est, tru):
    tp = int((est & tru).sum()); fp = int((est & ~tru).sum()); fn = int((~est & tru).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if tp else 0.0
    return {"k": tp + fp, "precision": p, "recall": r, "f1": f, "shd": fp + fn}


def topk_metrics(S, W, k):
    """I-SEM-style dense score at matched sparsity: its top-k ordered pairs (ties broken by mergesort)."""
    n = len(W); off = ~np.eye(n, dtype=bool)
    s = np.where(off, np.abs(S), -np.inf).ravel()
    est = np.zeros(n * n, bool); est[np.argsort(-s, kind="mergesort")[:k]] = True
    return _prf(est.reshape(n, n), (W != 0) & off)


def rho(a, b):
    return float(spearmanr(a, b)[0])


def partial(x, y, z):
    """analyze.py's rank partial correlation of x and y given z."""
    rx, ry, rz = map(rankdata, (x, y, z))

    def res(a, b):
        A = np.c_[np.ones_like(b), b]
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(np.corrcoef(res(rx, rz), res(ry, rz))[0, 1])


def outstrength(A):
    """analyze.py: column sums of |A| with the diagonal zeroed (column j = regulator j)."""
    A = np.abs(np.asarray(A, float)).copy(); np.fill_diagonal(A, 0); return A.sum(0)


# ----------------------------------------------------------------------------------------------- E2
def e2_settings(default_only):
    uniq = {e2.key(p): p for _, _, p in e2.settings()}
    return {e2.key(e2.DEFAULT): e2.DEFAULT} if default_only else uniq


def _e2_job(k, p, rep, methods, ncores):
    from scripts.run_all_baselines import standardize_to_control
    d = OUT / "e2_cache" / k / f"rep{rep}"
    d.mkdir(parents=True, exist_ok=True)
    todo = [m for m in methods if not (d / f"{m}.npz").exists()]
    if not todo:
        return k, rep, "cached"
    expr, interv, genes, W = e2.simulate(p, rep)
    W_cached = e2.CACHE / k / f"rep{rep}" / "W.npy"          # read-only check: same network as e2_planted.csv
    if W_cached.exists():
        assert np.array_equal(np.load(W_cached), W), f"E2 generator drifted from e2_cache for {k} rep{rep}"
    lab = np.array(interv)
    z = standardize_to_control(expr, lab == e2.CONTROL)
    labels = np.where(lab == e2.CONTROL, "control", lab)
    msgs = []
    for m in todo:
        try:
            path, meta = fit(z, labels, genes, m, seed=rep, ncores=ncores, its=100, solve_its=10, cv=5)
        except Exception as e:                                # recorded, never silently dropped
            (d / f"{m}.err").write_text(repr(e)); msgs.append(f"FAIL {m}: {e!r}"); continue
        save_fit(d / f"{m}.npz", path, meta)
        msgs.append(f"{m} {meta['fit_seconds']:.0f}s K={meta['K']}")
    return k, rep, "; ".join(msgs)


def run_e2(default_only, workers, ncores, methods, reps):
    jobs = [(k, p, r) for r in reps for k, p in e2_settings(default_only).items()]
    print(f"E2: {len(jobs)} (setting, rep) jobs, methods={methods}, workers={workers}, ncores/fit={ncores}", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_e2_job, k, p, r, methods, ncores) for k, p, r in jobs]
        for f in as_completed(futs):
            k, rep, msg = f.result()
            print(f"{k} rep{rep}: {msg}", flush=True)


def _choose_e2(path, meta, W, genes):
    """{rule: (lambda index, full score)} incl. the oracle and true-edge-count-matched rules."""
    full = [embed(path[:, :, l], meta["kept_genes"], genes) for l in range(path.shape[2])]
    ch = {r: i for r, i in rules(meta).items() if r != "dbar_fallback"}
    f1 = [edge_metrics(S, W)["f1"] for S in full]
    ch["oracle_f1"] = int(np.nanargmax(f1))
    ne = int(((W != 0) & ~np.eye(len(W), dtype=bool)).sum())
    ks = [edge_metrics(S, W)["k"] for S in full]
    ch["matched"] = int(np.argmin(np.abs(np.asarray(ks) - ne)))
    return {r: (i, full[i]) for r, i in ch.items()}, rules(meta).get("dbar_fallback")


def e2_collect():
    import pandas as pd
    genes = [f"G{g:03d}" for g in range(e2.N_GENES)]
    rows, bias, missing = [], [], []
    for sweep, level, p in e2.settings():
        k = e2.key(p)
        for rep in range(e2.REPS):
            d = OUT / "e2_cache" / k / f"rep{rep}"
            if not any((d / f"{m}.npz").exists() for m in FIT_METHODS):
                continue                                          # setting not run (default-only mode)
            W = np.load(e2.CACHE / k / f"rep{rep}" / "W.npy")
            isem_f = e2.CACHE / k / f"rep{rep}" / "isem.npy"
            isem = np.load(isem_f) if isem_f.exists() else None
            kd = np.abs(np.log1p(-e2.planted(e2.N_GENES, p["cycle_frac"], p["rho"], rep)[2]))
            outdeg = ((W != 0) & ~np.eye(len(W), dtype=bool)).sum(0)
            base = {"sweep": sweep, "level": level, "rep": rep}
            scores = {}
            for m in FIT_METHODS:
                f = d / f"{m}.npz"
                if not f.exists():
                    missing.append(f"{k}/rep{rep}/{m}"); continue
                path, meta = load_fit(f)
                ch, fb = _choose_e2(path, meta, W, genes)
                for rule, (li, S) in ch.items():
                    em = edge_metrics(S, W)
                    row = {**base, "method": m, "rule": rule, "lambda_index": li, "lambda": meta["lambda"][li],
                           "dbar_fallback": fb, "n_kept": meta["K"], "fit_seconds": meta["fit_seconds"],
                           **e2.evaluate(S, W), **em}
                    if isem is not None and em["k"] > 0:
                        row.update({f"isem_topk_{x}": v for x, v in topk_metrics(isem, W, em["k"]).items() if x != "k"})
                    rows.append(row)
                    scores[f"{m}:{rule}"] = S                   # bias for every rule: dbar can be near-empty
            for m in ("isem", "inspre_dag", "plugin", "mean_diff"):    # read-only from e2_planted's cache
                f = e2.CACHE / k / f"rep{rep}" / f"{m}.npy"
                if f.exists():
                    scores[m] = np.load(f)
            if "inspre_dag" in scores:                               # official-settings inspre at its own sparsity
                S = scores["inspre_dag"]; em = edge_metrics(S, W)
                row = {**base, "method": "inspre_dag", "rule": "official (min eps_hat_G)", **e2.evaluate(S, W), **em}
                if isem is not None and em["k"] > 0:
                    row.update({f"isem_topk_{x}": v for x, v in topk_metrics(isem, W, em["k"]).items() if x != "k"})
                rows.append(row)
            scores["truth |W|"] = np.abs(W)
            for m, S in scores.items():                              # E6 on E2: out-strength vs kd and true out-degree
                o = outstrength(S)
                bias.append({**base, "method": m, "spearman_out_vs_kd": rho(o, kd),
                             "spearman_out_vs_true_outdeg": rho(o, outdeg),
                             "partial_out_kd_given_true_outdeg": partial(o, kd, outdeg),
                             "n_nonzero_out": int((o > 0).sum())})
    pd.DataFrame(rows).to_csv(OUT / "e2_adapre.csv", index=False)
    pd.DataFrame(bias).to_csv(OUT / "e2_bias.csv", index=False)
    (OUT / "e2_missing.json").write_text(json.dumps(sorted(set(missing)), indent=1))
    print(f"wrote e2_adapre.csv ({len(rows)} rows), e2_bias.csv ({len(bias)} rows); {len(set(missing))} missing", flush=True)


# ------------------------------------------------------------------------------------------ real data
def prepare_real(ds, min_cells=100, p_max=0.05):
    """Control-z-scored panel restricted to targets with significant knockdown and >= min_cells cells
    (the ADAPRE paper's teloHAEC rule: 'significant knockdown (p < 0.05) and at least 100 perturbed cells')."""
    from scripts.run_all_baselines import standardize_to_control
    with np.load(ROOT / "data" / f"dataset_{ds}.npz", allow_pickle=True) as zf:
        X = np.asarray(zf["expression_matrix"], np.float32)
        genes = [str(g) for g in zf["var_names"]]
        cond = np.asarray([str(c).strip() for c in zf["interventions"]])
    ctrl = cond == "non-targeting"
    z = standardize_to_control(X, ctrl)
    del X
    table = []
    for j, g in enumerate(genes):
        m = cond == g
        n = int(m.sum())
        p = float(ttest_ind(z[m, j], z[ctrl, j], equal_var=False, alternative="less").pvalue) if n > 1 else float("nan")
        table.append({"gene": g, "n_cells": n, "mean_z_target": float(z[m, j].mean()) if n else float("nan"),
                      "p_knockdown_welch_less": p, "kept": bool(n >= min_cells and p < p_max)})
    keep = [t["gene"] for t in table if t["kept"]]
    rows = ctrl | np.isin(cond, keep)
    gi = np.asarray([genes.index(g) for g in keep])
    labels = np.where(ctrl[rows], "control", cond[rows])
    return z[np.ix_(rows, gi)], labels, keep, genes, table


def run_real(ds, method, ncores, its, solve_its, cv, overwrite):
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "logs").mkdir(exist_ok=True)
    tag = f"{ds}_{method}"
    if (OUT / f"{tag}.npy").exists() and not overwrite:
        raise FileExistsError(f"{OUT / tag}.npy exists (use --overwrite)")
    t0 = time.time()
    z, labels, keep, panel, table = prepare_real(ds)
    print(f"{ds}: {len(keep)}/{len(panel)} targets pass the knockdown filter; {len(z)} cells", flush=True)
    log = OUT / "logs" / f"{tag}.fit.log"
    log.write_text(f"dataset={ds} method={method} its={its} solve_its={solve_its} cv={cv} ncores={ncores}\n")
    peak0 = _children_maxrss()
    path, meta = fit(z, labels, keep, method, seed=26000 + DATASETS.index(ds), ncores=ncores, its=its,
                     solve_its=solve_its, cv=cv, log=log, blas_threads=ncores)
    meta["peak_rss_gb_children"] = max(_children_maxrss(), peak0) / 2 ** 20
    save_fit(OUT / f"{tag}_Gpath.npz", path, meta)
    r = rules(meta)
    S = embed(path[:, :, r["dbar"]], meta["kept_genes"], panel)
    assert np.isfinite(S).all()
    np.save(OUT / f"{tag}.npy", S, allow_pickle=False)
    meta.update(dataset=ds, panel_genes=panel, target_filter=table, selection=r,
                chosen_lambda=meta["lambda"][r["dbar"]], n_nonzero=int((S != 0).sum()),
                nonfinite_at_chosen_lambda=int(meta["nonfinite_per_lambda"][r["dbar"]]),
                zero_diag_V_genes_at_chosen_lambda=meta["zero_diag_V_genes_per_lambda"][r["dbar"]],
                n_cells_total_input=int(len(z)), total_seconds=time.time() - t0,
                orientation="W[i,j] = effect of j on i = t(G); signed; |W| scored",
                input_normalisation="genes z-scored vs non-targeting control cells (standardize_to_control)")
    (OUT / f"{tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"{tag}: K={meta['K']} nnz={meta['n_nonzero']} lambda_idx={r['dbar']} fallback={r['dbar_fallback']} "
          f"fit={meta['fit_seconds']:.0f}s rss={meta['peak_rss_gb_children']:.1f}GB", flush=True)


def _children_maxrss():
    import resource
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss      # KiB on Linux


def score_real(ds, n_boot, method="adapre"):
    import paper_tables as pt
    import score_eval as se
    z = pt.load(ds)
    genes = [str(g) for g in z["gene_names"]]
    ada = np.load(OUT / f"{ds}_{method}.npy")
    meta = json.loads((OUT / f"{ds}_{method}.json").read_text())
    assert meta["panel_genes"] == genes, "panel order differs from the I-SEM cache"
    ins = np.load(PAPER / f"e1b_inspre_official/{ds}.npy")
    S = {"I-SEM": z["S_current"], pt.INSPRE: ins, "ADAPRE": ada}
    refs = se.references(genes)
    out = {}
    subsets = {"full_panel": np.arange(len(genes)),
               "adapre_kept_genes": np.asarray([genes.index(g) for g in meta["kept_genes"]]),
               "inspre_kept_genes": np.asarray([genes.index(g) for g in
                                                json.loads((PAPER / f"e1b_inspre_official/{ds}.json").read_text())["kept_genes"]])}
    for ref, G in refs.items():
        for sub, idx in subsets.items():
            Ss = {m: A[np.ix_(idx, idx)] for m, A in S.items()}
            Gs = G[np.ix_(idx, idx)]
            r = se.score_reference(Ss, Gs, n_boot=n_boot)
            for other, name in (("ADAPRE", "I-SEM (top k = ADAPRE edge count)"),
                                (pt.INSPRE, "I-SEM (top k = official inspre edge count)")):
                tmp = {}
                pt.matched_k(tmp, Ss["I-SEM"], Ss[other], Gs, n_boot)
                r[name] = tmp[pt.MATCHED_K]
                r[other].update({key: v for key, v in tmp[pt.INSPRE].items()})
            out[f"{ref}@{sub}"] = r
            print(f"{ds} {ref}@{sub}: " + ", ".join(f"{m} AUROC {r[m]['auroc']:.3f} P@k {r[m].get('precision_at_k', float('nan')):.3f}"
                                                   for m in ("I-SEM", "ADAPRE", pt.INSPRE)), flush=True)
    out["meta"] = se.meta({"n_boot": n_boot, "adapre_meta": {k: meta[k] for k in ("K", "chosen_lambda", "n_nonzero", "selection", "fit_seconds")}})
    (OUT / f"score_{ds}.json").write_text(json.dumps(out, indent=1, allow_nan=True))


def bias_real():
    """E6 for ADAPRE: analyze.py's out-strength Spearmans and its 500 gene resamples (same rng stream:
    default_rng(0), 500 draws per dataset in k562, rpe1, hepg2, jurkat order), paired vs I-SEM/inspre."""
    rng = np.random.default_rng(0)
    e6 = json.loads((PAPER / "e6_bias.json").read_text())
    res = {}
    for ds in E6_ORDER:
        z = np.load(PAPER / f"cache/{ds}.npz", allow_pickle=True)
        n = len(z["gene_names"]); genes = [str(g) for g in z["gene_names"]]
        idx = [rng.integers(0, n, n) for _ in range(500)]         # drawn even if ADAPRE is missing: keeps the stream
        f = OUT / f"{ds}_adapre.npy"
        if not f.exists():
            res[ds] = "ADAPRE not run"; continue
        kd = np.abs(z["delta_jj"])
        L = np.load(PAPER / f"cache/libsize_{ds}.npz")
        inter = np.load(ROOT / f"data/dataset_{ds}.npz", allow_pickle=True)["interventions"]
        lg = np.log(L["umi_obs"]); ctl = inter == "non-targeting"
        dl = np.array([lg[inter == g].mean() - lg[ctl].mean() for g in genes])
        meth = {"isem_S_current": z["S_current"], "inspre": np.load(PAPER / f"e1b_inspre_official/{ds}.npy"),
                "adapre": np.load(f)}
        if (OUT / f"{ds}_adapre_noadapt.npy").exists():
            meth["adapre_noadapt"] = np.load(OUT / f"{ds}_adapre_noadapt.npy")
        outs = {m: outstrength(A) for m, A in meth.items()}
        kept = np.isin(genes, json.loads((OUT / f"{ds}_adapre.json").read_text())["kept_genes"])
        d = {"n_genes": n, "n_adapre_kept": int(kept.sum()), "methods": {}, "paired": {}}
        for m, o in outs.items():
            d["methods"][m] = {
                "spearman_out_vs_absdeltajj": rho(o, kd), "spearman_out_vs_absdlogUMI": rho(o, np.abs(dl)),
                "spearman_out_vs_absdeltajj_ci": [float(x) for x in np.percentile([rho(o[i], kd[i]) for i in idx], [2.5, 97.5])],
                "spearman_out_vs_absdlogUMI_ci": [float(x) for x in np.percentile([rho(o[i], np.abs(dl)[i]) for i in idx], [2.5, 97.5])],
                "partial_out_absdeltajj_given_absdlogUMI": partial(o, kd, np.abs(dl)),
                "n_nonzero_out_strength_genes": int((o > 0).sum()),
                "on_adapre_kept_genes": {"spearman_out_vs_absdeltajj": rho(o[kept], kd[kept]),
                                         "spearman_out_vs_absdlogUMI": rho(o[kept], np.abs(dl)[kept])}}
            if m in e6[ds]["methods"]:                             # reproduction check against e6_bias.json
                ref = e6[ds]["methods"][m]
                d["methods"][m]["matches_e6_bias_json"] = bool(
                    np.isclose(ref["spearman_out_vs_absdeltajj"], d["methods"][m]["spearman_out_vs_absdeltajj"]) and
                    np.allclose(ref["spearman_out_vs_absdeltajj_ci"], d["methods"][m]["spearman_out_vs_absdeltajj_ci"]))
        for a in ("isem_S_current", "inspre"):
            for b in [m for m in outs if m.startswith("adapre")]:
                diffs = [rho(outs[a][i], kd[i]) - rho(outs[b][i], kd[i]) for i in idx]
                d["paired"][f"{a}-{b}"] = {"diff": d["methods"][a]["spearman_out_vs_absdeltajj"] - d["methods"][b]["spearman_out_vs_absdeltajj"],
                                           "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
        if "isem_S_current-inspre" in e6[ds]["bootstrap_rho_absdeltajj_diff_isem_minus_baseline"]:
            diffs = [rho(outs["isem_S_current"][i], kd[i]) - rho(outs["inspre"][i], kd[i]) for i in idx]
            d["isem_minus_inspre_ci_matches_e6"] = bool(np.allclose(
                [np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)],
                e6[ds]["bootstrap_rho_absdeltajj_diff_isem_minus_baseline"]["isem_S_current-inspre"]["ci95"]))
        res[ds] = d
        print(ds, {m: round(v["spearman_out_vs_absdeltajj"], 3) for m, v in d["methods"].items()}, flush=True)
    (OUT / "e6_adapre_bias.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------- knockdown-strength analyses (strong/weak, slopes, T_ii)
DENSITIES = (0.001, 0.005, 0.01, 0.03)   # ADAPRE Fig. 1: "very sparse (~0.1%) to moderately sparse (~3%)"


def isem_b_e2(default_only):
    """Refit the paper I-SEM on E2 with train_paper.py's exact train_batched call to recover B (only S was
    cached). Accepted only if the refit score reproduces e2_cache/<key>/rep<r>/isem.npy within train_paper's
    Part-A tolerance. Writes e8_adapre/e2_isemB/<key>/rep<r>.npz."""
    import torch
    from src.models.interventional_nb_sem import build_intervention_targets, train_batched
    from src.pipeline import GRNInferencePipeline, _complementary_pairs, _mean_ensemble, _sem_config
    pipe = GRNInferencePipeline(device="cuda:0")
    assert (pipe.n_resamples, pipe.epochs, pipe.warmup_epochs, pipe.l1_weight, pipe.w_dropout, pipe.models_per_chunk,
            pipe.readout, pipe.symmetrize) == (60, 250, 50, .005, 0.0, 120, "direct", False), "not the paper model"
    cfg = _sem_config(pipe.epochs, pipe.warmup_epochs, pipe.l1_weight, pipe.w_dropout)
    for k, p in e2_settings(default_only).items():
        for rep in range(e2.REPS):
            f = OUT / "e2_isemB" / k / f"rep{rep}.npz"
            if f.exists():
                continue
            expr, interv, genes, W = e2.simulate(p, rep)
            u, delta, n_c, control, cmean, se = build_intervention_targets(expr, interv, genes)
            pairs = _complementary_pairs(len(u), pipe.n_resamples)
            t0 = time.time()
            out = train_batched(u, delta, se, folds=[q[0] for q in pairs], seeds=[q[1] for q in pairs], config=cfg,
                                device="cuda:0", return_params=True, return_diagnostics=False)
            torch.cuda.synchronize()
            Wm, B = out[2], out[3]
            direct = _mean_ensemble(np.abs(Wm * B[:, None, :]))
            ref = np.load(e2.CACHE / k / f"rep{rep}" / "isem.npy")
            off = ~np.eye(len(ref), dtype=bool)
            maxrel = float(np.max(np.abs(direct - ref)) / np.max(np.abs(ref)))
            sp = float(spearmanr(direct[off], ref[off])[0])
            accepted = maxrel <= 1e-5 and sp >= 1 - 1e-10
            membership = np.stack([(u[q[0]] > .5).any(0) for q in pairs])
            counts = membership.sum(0)
            target = (u > .5).argmax(1)
            delta_jj = np.zeros(len(genes)); delta_jj[target] = delta[np.arange(len(delta)), target]
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, W_only=_mean_ensemble(np.abs(Wm)), inhalf_mean_B=(B * membership).sum(0) / counts, inhalf_mean_abs_B=(np.abs(B) * membership).sum(0) / counts,
                     delta_jj=delta_jj, maxrel=maxrel, spearman=sp, accepted=accepted, seconds=time.time() - t0)
            print(f"{k} rep{rep}: refit maxrel={maxrel:.2e} spearman={sp:.12f} accepted={accepted} ({time.time() - t0:.0f}s)", flush=True)


def _own_zshift(z, labels, genes):
    """ADAPRE's gamma-hat: mean control-z expression of each gene in its own knockdown cells (beta_obs)."""
    return np.array([z[labels == g, j].mean() if (labels == g).any() else np.nan for j, g in enumerate(genes)])


def tii_e2():
    """T_ii test: own-gene shift gamma_i T_ii vs gamma_i, and which estimate recovers gamma_i."""
    import pandas as pd
    from scripts.run_all_baselines import standardize_to_control
    rows = []
    for sweep, level, p in e2.settings():
        k = e2.key(p)
        for rep in range(e2.REPS):
            fb = OUT / "e2_isemB" / k / f"rep{rep}.npz"
            fa = OUT / "e2_cache" / k / f"rep{rep}" / "adapre.npz"
            if not (fb.exists() and fa.exists()):
                continue
            W, m, kd, cycles = e2.planted(e2.N_GENES, p["cycle_frac"], p["rho"], rep)
            T = np.linalg.inv(np.eye(e2.N_GENES) - W)
            Tii = np.diag(T)
            gT = np.log1p(-kd)                                  # realised own log-rate shift = gamma_i T_ii (simulator)
            g = gT / Tii                                        # gamma_i, the input strength
            on = np.zeros(e2.N_GENES, bool)
            for c in cycles:
                on[c] = True
            zb = np.load(fb)
            if not bool(zb["accepted"]):
                continue
            Bh, djj = zb["inhalf_mean_B"], zb["delta_jj"]
            _, ma = load_fit(fa)
            ah = np.full(e2.N_GENES, np.nan)
            ah[[int(x[1:]) for x in ma["kept_genes"]]] = ma["beta_obs"]
            for subset, msk in (("all", np.ones_like(on)), ("cycle", on), ("acyclic", ~on)):
                if msk.sum() < 5:
                    continue
                r = {"sweep": sweep, "level": level, "rep": rep, "genes": subset, "n": int(msk.sum()),
                     "Tii_median": float(np.median(Tii[msk])), "Tii_min": float(Tii[msk].min()), "Tii_max": float(Tii[msk].max()),
                     "frac_abs_Tii_minus1_gt_0.1": float(np.mean(np.abs(Tii[msk] - 1) > .1)),
                     "median_abs_log_gT_over_g": float(np.median(np.abs(np.log(Tii[msk]))))}
                for nm, est in (("isem_B", Bh), ("adapre_gamma_hat", ah), ("observed_delta_jj", djj)):
                    ok = msk & np.isfinite(est)
                    r[f"{nm}_spearman_gamma"] = rho(est[ok], g[ok])
                    r[f"{nm}_spearman_gammaT"] = rho(est[ok], gT[ok])
                    r[f"{nm}_partial_gamma_given_gammaT"] = partial(est[ok], g[ok], gT[ok])
                # scale-free T_ii recovery: estimate / observed own shift ~ 1 / T_ii ?
                for nm, est in (("isem_B", Bh), ("adapre_gamma_hat", ah)):
                    ok = msk & np.isfinite(est) & (np.abs(djj) > 1e-8)
                    r[f"{nm}_over_delta_jj_spearman_invTii"] = rho(est[ok] / djj[ok], 1 / Tii[ok])
                rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "e2_tii.csv", index=False)
    print(f"wrote e2_tii.csv ({len(rows)} rows)", flush=True)


def _topk_mask(S, k):
    n = len(S); off = ~np.eye(n, dtype=bool)
    s = np.where(off, np.abs(S), -np.inf).ravel()
    m = np.zeros(n * n, bool); m[np.argsort(-s, kind="mergesort")[:k]] = True
    return m.reshape(n, n)


def _path_at_k(path, kept, genes, k):
    """Path method at k edges: the densest-enough lambda (fewest nonzeros >= k), then its top-k |G|.
    Returns (edge mask, actual k, reached)."""
    nnz = [int((np.isfinite(path[:, :, l]) & (np.abs(path[:, :, l]) > 0)).sum()) for l in range(path.shape[2])]
    ok = [l for l in range(len(nnz)) if nnz[l] >= k]
    l = min(ok, key=lambda x: nnz[x]) if ok else int(np.argmax(nnz))
    S = embed(path[:, :, l], kept, genes)
    m = _topk_mask(S, min(k, nnz[l])) & (np.abs(S) > 0)
    return m, int(m.sum()), bool(ok)


def _slope(outdeg, x):
    from scipy.stats import linregress
    ok = np.isfinite(x)
    r = linregress(x[ok], outdeg[ok])
    return {"slope": float(r.slope), "slope_se": float(r.stderr), "p": float(r.pvalue), "spearman": rho(outdeg[ok], x[ok])}


def slopes_real():
    """ADAPRE Fig. 1a on every method: OLS of out-degree (edges out of regulator j) on gamma-hat_j (own-target
    knockdown effect, control-z units; negative = knockdown) at matched sparsity."""
    import pandas as pd
    rows = []
    for ds in DATASETS:
        fa = OUT / f"{ds}_adapre_Gpath.npz"
        if not fa.exists():
            continue
        gh_f = OUT / f"{ds}_gammahat.json"
        if not gh_f.exists():
            z, labels, keep, panel, table = prepare_real(ds)
            gh_f.write_text(json.dumps({t["gene"]: t["mean_z_target"] for t in table}))
        c = np.load(PAPER / f"cache/{ds}.npz", allow_pickle=True)
        genes = [str(x) for x in c["gene_names"]]
        gh = np.array([json.loads(gh_f.read_text())[g] for g in genes])
        absd = np.abs(c["delta_jj"])
        n = len(genes)
        paths = {"adapre": load_fit(fa)}
        if (OUT / f"{ds}_adapre_noadapt_Gpath.npz").exists():
            paths["adapre_noadapt"] = load_fit(OUT / f"{ds}_adapre_noadapt_Gpath.npz")
        with np.load(PAPER / f"e1b_inspre_official/{ds}_Gpath.npz") as zi:
            paths["inspre"] = (zi["W_path"], {"kept_genes": [str(x) for x in zi["kept_genes"]]})
        dense = {"isem_S_current": c["S_current"], "isem_W_only": c["W_only"]}
        for dens in DENSITIES:
            k = int(round(dens * n * (n - 1)))
            masks = {m: (_topk_mask(S, k), k, True) for m, S in dense.items()}
            masks.update({m: _path_at_k(pth, meta["kept_genes"], genes, k) for m, (pth, meta) in paths.items()})
            for m, (mask, kk, reached) in masks.items():
                od = mask.sum(0).astype(float)                     # column j = regulator j
                rows.append({"dataset": ds, "density": dens, "k_target": k, "k": kk, "reached": reached, "method": m,
                             **_slope(od, gh), "spearman_outdeg_absdeltajj": rho(od, absd),
                             "n_regulators_with_edges": int((od > 0).sum())})
    pd.DataFrame(rows).to_csv(OUT / "slopes_real.csv", index=False)
    print(f"wrote slopes_real.csv ({len(rows)} rows)", flush=True)


def slopes_e2():
    """ADAPRE-style slopes on E2: out-degree ~ gamma-hat (own-gene control-z shift, ADAPRE's beta_obs) and
    ~ true out-degree, at the density grid and at k = true edge count."""
    import pandas as pd
    rows = []
    for sweep, level, p in e2.settings():
        k0 = e2.key(p)
        for rep in range(e2.REPS):
            d = OUT / "e2_cache" / k0 / f"rep{rep}"
            fb = OUT / "e2_isemB" / k0 / f"rep{rep}.npz"
            if not (d / "adapre.npz").exists():
                continue
            W = np.load(e2.CACHE / k0 / f"rep{rep}" / "W.npy"); n = len(W)
            genes = [f"G{g:03d}" for g in range(n)]
            off = ~np.eye(n, dtype=bool); tdeg = ((W != 0) & off).sum(0)
            kd = np.abs(np.log1p(-e2.planted(n, p["cycle_frac"], p["rho"], rep)[2]))
            _, ma = load_fit(d / "adapre.npz")
            gh = np.full(n, np.nan); gh[[int(x[1:]) for x in ma["kept_genes"]]] = ma["beta_obs"]
            paths = {m: load_fit(d / f"{m}.npz") for m in FIT_METHODS if (d / f"{m}.npz").exists()}
            dense = {"isem_S": np.load(e2.CACHE / k0 / f"rep{rep}" / "isem.npy")}
            if fb.exists() and bool(np.load(fb)["accepted"]):
                dense["isem_W_only"] = np.load(fb)["W_only"]
            fi = e2.CACHE / k0 / f"rep{rep}" / "inspre_dag.npy"     # missing where official inspre failed in E2
            ins = np.load(fi) if fi.exists() else None
            ks = {f"{x:g}": int(round(x * n * (n - 1))) for x in DENSITIES}
            ks["true_E"] = int(tdeg.sum())
            for lab, k in ks.items():
                masks = {m: (_topk_mask(S, k), k, True) for m, S in dense.items()}
                masks.update({m: _path_at_k(pth, meta["kept_genes"], genes, k) for m, (pth, meta) in paths.items()})
                if ins is not None:
                    mi = _topk_mask(ins, k) & (np.abs(ins) > 0)
                    masks["inspre_dag (official, one lambda)"] = (mi, int(mi.sum()), bool(mi.sum() >= k))
                for m, (mask, kk, reached) in masks.items():
                    od = mask.sum(0).astype(float)
                    rows.append({"sweep": sweep, "level": level, "rep": rep, "density": lab, "k": kk, "reached": reached,
                                 "method": m, **_slope(od, gh), "slope_on_true_outdeg": _slope(od, tdeg.astype(float))["slope"],
                                 "spearman_outdeg_kd": rho(od, kd), "spearman_outdeg_true_outdeg": rho(od, tdeg)})
    pd.DataFrame(rows).to_csv(OUT / "slopes_e2.csv", index=False)
    print(f"wrote slopes_e2.csv ({len(rows)} rows)", flush=True)


def _half_metrics_pairs(S, pos, cols):
    """AUROC/AUPRC on ordered pairs (i, j), i != j, j in cols."""
    import score_eval as se
    n = len(S); off = ~np.eye(n, dtype=bool); colm = np.zeros(n, bool); colm[cols] = True
    sel = off & colm[None, :]
    auc, ap = se.auc_ap_multi(pos[sel][None, :].astype(np.int8), np.abs(S)[sel])
    return float(auc[0]), float(ap[0]), float(pos[sel].mean())


def split_e2():
    """ADAPRE Fig. 2b-style split: every metric separately for regulators in the strong / weak knockdown half."""
    import pandas as pd
    rows = []
    for sweep, level, p in e2.settings():
        k = e2.key(p)
        for rep in range(e2.REPS):
            d = OUT / "e2_cache" / k / f"rep{rep}"
            if not (d / "adapre.npz").exists():
                continue
            W = np.load(e2.CACHE / k / f"rep{rep}" / "W.npy")
            kd = np.abs(np.log1p(-e2.planted(e2.N_GENES, p["cycle_frac"], p["rho"], rep)[2]))
            strong = kd > np.median(kd)
            off = ~np.eye(len(W), dtype=bool); pos = (W != 0) & off
            genes = [f"G{g:03d}" for g in range(e2.N_GENES)]
            scores = {}
            for m in FIT_METHODS:
                if (d / f"{m}.npz").exists():
                    path, meta = load_fit(d / f"{m}.npz")
                    ch, _ = _choose_e2(path, meta, W, genes)
                    for rule in ("dbar", "min_eps_g", "min_train", "oracle_f1"):
                        scores[f"{m}:{rule}"] = ch[rule][1]
            for m in ("isem", "inspre_dag", "plugin", "mean_diff"):
                f = e2.CACHE / k / f"rep{rep}" / f"{m}.npy"
                if f.exists():
                    scores[m] = np.load(f)
            isem = scores.get("isem")
            for half, hm in (("strong", strong), ("weak", ~strong)):
                cols = np.flatnonzero(hm); colm = np.zeros(len(W), bool); colm[cols] = True
                sel = off & colm[None, :]
                for m, S in scores.items():
                    auc, ap, prev = _half_metrics_pairs(S, pos, cols)
                    est = (np.abs(S) > EDGE_THR) & sel
                    r = {"sweep": sweep, "level": level, "rep": rep, "half": half, "method": m, "auroc_direct": auc,
                         "auprc_direct": ap, "auprc_lift": ap / prev,
                         "mean_true_outdeg": float(pos[:, cols].sum(0).mean()),
                         "mean_est_outdeg": float(est[:, cols].sum(0).mean()) if m not in ("isem", "plugin", "mean_diff") else np.nan}
                    if m not in ("isem", "plugin", "mean_diff"):
                        r.update({f"{x}": v for x, v in _prf(est, pos & sel).items()})
                        # ADAPRE Fig. 2a: group-wise OLS of estimated on true out-degree (slope 1 = calibrated)
                        r["slope_est_on_true_outdeg"] = _slope(est[:, cols].sum(0).astype(float), pos[:, cols].sum(0).astype(float))["slope"]
                        if isem is not None and r["k"] > 0:
                            top = _topk_mask(np.where(sel, np.abs(isem), 0), r["k"]) & sel & (np.abs(isem) > 0)
                            r.update({f"isem_topk_{x}": v for x, v in _prf(top, pos & sel).items() if x != "k"})
                    rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "e2_split.csv", index=False)
    print(f"wrote e2_split.csv ({len(rows)} rows)", flush=True)


def split_real(ds, n_boot=500):
    """Strong / weak |delta_jj| halves on real data: AUROC, AUPRC lift and precision at ADAPRE's within-half k
    for I-SEM, ADAPRE and official inspre; node bootstrap (score_eval's scheme) paired with I-SEM."""
    import score_eval as se
    c = np.load(PAPER / f"cache/{ds}.npz", allow_pickle=True)
    genes = [str(g) for g in c["gene_names"]]; n = len(genes)
    S = {"I-SEM": c["S_current"], "ADAPRE": np.load(OUT / f"{ds}_adapre.npy"),
         "inspre (official)": np.load(PAPER / f"e1b_inspre_official/{ds}.npy")}
    kd = np.abs(c["delta_jj"]); strong = kd > np.median(kd)
    off = ~np.eye(n, dtype=bool)
    out = {}
    for ref, G in se.references(genes).items():
        for half, hm in (("strong", strong), ("weak", ~strong)):
            sel = off & hm[None, :]
            ka = int(((np.abs(S["ADAPRE"]) > 0) & sel).sum())
            # top-k within the half, never padded with zero-score pairs (sparse inspre may have < k edges there)
            topk = {m: (_topk_mask(np.where(sel, np.abs(A), 0), ka) & sel & (np.abs(A) > 0)) for m, A in S.items() if m != "ADAPRE"}
            topk["ADAPRE"] = (np.abs(S["ADAPRE"]) > 0) & sel

            def stats(Y, A, mask, hsel):
                yy = Y[hsel]
                if yy.sum() == 0 or yy.sum() == yy.size:
                    return {"auroc": np.nan, "auprc_lift": np.nan, "precision_at_k": np.nan}
                auc, ap = se.auc_ap_multi(yy[None, :].astype(np.int8), A[hsel])
                return {"auroc": float(auc[0]), "auprc_lift": float(ap[0] / yy.mean()),
                        "precision_at_k": float(Y[mask].mean()) if mask.any() else np.nan}
            pos = G.astype(bool)
            point = {m: stats(pos, np.abs(A), topk[m], sel) for m, A in S.items()}
            rng = np.random.default_rng(0)
            draws = {m: {x: [] for x in ("auroc", "auprc_lift", "precision_at_k")} for m in S}
            for _ in range(n_boot):
                idx = rng.choice(n, n, replace=True)
                Pb = pos[np.ix_(idx, idx)]; offb = idx[:, None] != idx[None, :]
                selb = offb & hm[idx][None, :]
                for m, A in S.items():
                    v = stats(Pb, np.abs(A)[np.ix_(idx, idx)], topk[m][np.ix_(idx, idx)] & offb, selb)
                    for x in v:
                        draws[m][x].append(v[x])
            res = {"k_adapre_in_half": ka, "n_regulators": int(hm.sum())}
            for m in S:
                res[m] = {**point[m], **{f"{x}_ci": [float(np.nanpercentile(draws[m][x], 2.5)), float(np.nanpercentile(draws[m][x], 97.5))] for x in draws[m]}}
                if m != "I-SEM":
                    res[m]["isem_minus_this"] = {x: {"difference": point["I-SEM"][x] - point[m][x],
                                                     "ci": [float(np.nanpercentile(np.subtract(draws["I-SEM"][x], draws[m][x]), q)) for q in (2.5, 97.5)]}
                                                 for x in draws[m]}
            out[f"{ref}@{half}"] = res
            print(ds, ref, half, {m: {x: round(v, 3) for x, v in point[m].items()} for m in S}, flush=True)
    (OUT / f"split_{ds}.json").write_text(json.dumps(out, indent=1, allow_nan=True))


# ------------------------------------------------------------------------------------------- tables
def _ms(s):
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else (f"{s.mean():.3f}" if len(s) else "–")


def tables():
    """TABLES.md: every number REPORT.md quotes, generated from the result files."""
    import pandas as pd
    out = ["# E8 tables (generated by `e8_adapre.py tables`; do not edit by hand)\n"]
    e2r = pd.read_csv(OUT / "e2_adapre.csv")
    ref = pd.read_csv(PAPER / "e2_planted.csv")
    dflt = (e2r.sweep == "rho") & (e2r.level == e2.DEFAULT["rho"])
    d = e2r[dflt]
    out.append(f"## E2 default setting (rho 0.8, noise 0.3, cycle_frac 0.5, 100 cells), mean ± sd over {d.rep.nunique()} reps\n")
    out.append("Rules: dbar = PRIMARY (official inspre rule); min_eps_g = CV error; min_train = demo's non-oracle rule; "
               "oracle_f1 = ORACLE (true graph); matched = lambda whose edge count is closest to the TRUE edge count. "
               "Leakage = auprc_total_only (lower = less indirect-effect leakage). I-SEM top-k = I-SEM's top-k pairs, k = that row's edge count.\n")
    cols = ["auprc_direct", "auroc_direct", "auprc_total_only", "k", "precision", "recall", "f1", "shd", "isem_topk_precision", "isem_topk_f1"]
    out.append("| method | rule | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2))
    for (m, r), g in d.groupby(["method", "rule"], sort=False):
        out.append(f"| {m} | {r} | " + " | ".join(_ms(g[c].dropna()) if c in g else "–" for c in cols) + " |")
    rd = ref[(ref.sweep == "rho") & (ref.level == e2.DEFAULT["rho"])]
    for m, g in rd.groupby("method"):
        out.append(f"| {m} (e2_planted.csv) | as published | {_ms(g.auprc_direct)} | {_ms(g.auroc_direct)} | {_ms(g.auprc_total_only)} |" + " – |" * 7)
    # paired per-rep differences vs I-SEM on the default setting
    out.append("\n### Paired per-replicate differences vs I-SEM (default setting): method − I-SEM, mean [min, max], #reps method > I-SEM\n")
    out.append("| method | rule | Δ auprc_direct | Δ auroc_direct | Δ leakage |\n|---|---|---|---|---|")
    isem = rd[rd.method == "isem"].set_index("rep")
    for (m, r), g in d.groupby(["method", "rule"], sort=False):
        g = g.set_index("rep"); cells = []
        for c in ("auprc_direct", "auroc_direct", "auprc_total_only"):
            x = (g[c] - isem[c].reindex(g.index)).dropna()
            cells.append(f"{x.mean():+.3f} [{x.min():+.3f}, {x.max():+.3f}], {int((x > 0).sum())}/{len(x)}")
        out.append(f"| {m} | {r} | " + " | ".join(cells) + " |")
    # sweep
    out.append("\n## E2 sweeps: direct AUPRC (mean over reps)\n")
    show = [("adapre", "dbar"), ("adapre", "min_eps_g"), ("adapre", "min_train"), ("adapre", "oracle_f1"),
            ("inspre_default", "dbar"), ("inspre_default", "min_train"), ("inspre_default", "oracle_f1")]
    heads = [f"{m}:{r}" for m, r in show] + ["isem", "plugin", "mean_diff", "inspre (official dag)"]
    for metric in ("auprc_direct", "auprc_total_only"):
        out.append(f"\n### {metric}\n\n| sweep | level | n reps | " + " | ".join(heads) + " |\n|" + "---|" * (len(heads) + 3))
        for (sw, lv), g in e2r.groupby(["sweep", "level"], sort=False):
            vals = [g[(g.method == m) & (g.rule == r)][metric].mean() for m, r in show]
            rr = ref[(ref.sweep == sw) & (ref.level == lv)]
            vals += [rr[rr.method == m][metric].mean() for m in ("isem", "plugin", "mean_diff", "inspre")]
            out.append(f"| {sw} | {lv} | {g.rep.nunique()} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    secs = e2r[e2r.method.isin(FIT_METHODS)].drop_duplicates(["sweep", "level", "rep", "method"]).groupby("method").fit_seconds
    out.append("\nFit seconds per replicate (2 threads, incl. 5-fold CV): " +
               "; ".join(f"{m} median {s.median():.0f} s (max {s.max():.0f})" for m, s in secs))
    # E2 bias
    b = pd.read_csv(OUT / "e2_bias.csv")
    bd = b[(b.sweep == "rho") & (b.level == e2.DEFAULT["rho"])]
    out.append("\n## E6 on E2 (default setting): Spearman of estimated out-strength with knockdown strength |log(1-kd)| "
               "and with TRUE out-degree; kd is drawn independently of the network, so any out ~ kd correlation is bias\n")
    out.append("| method | ρ(out, kd) | ρ(out, true out-degree) | partial ρ(out, kd · true out-degree) | # regulators with out-edges |\n|---|---|---|---|---|")
    for m, g in bd.groupby("method", sort=False):
        out.append(f"| {m} | {_ms(g.spearman_out_vs_kd)} | {_ms(g.spearman_out_vs_true_outdeg)} | "
                   f"{_ms(g.partial_out_kd_given_true_outdeg)} | {g.n_nonzero_out.mean():.0f} |")
    out.append("\n### E6 on E2 across all settings (mean over settings × reps)\n\n| method | ρ(out, kd) | ρ(out, true out-degree) |\n|---|---|---|")
    for m, g in b.groupby("method", sort=False):
        out.append(f"| {m} | {g.spearman_out_vs_kd.mean():.3f} | {g.spearman_out_vs_true_outdeg.mean():.3f} |")
    # real data
    out.append("\n## Real data (E1): node bootstrap 500, paired with I-SEM\n")
    for ds in DATASETS:
        f = OUT / f"score_{ds}.json"
        if not f.exists():
            out.append(f"\n### {ds}: not scored\n"); continue
        sc = json.loads(f.read_text()); am = sc["meta"]["adapre_meta"]
        out.append(f"\n### {ds} (ADAPRE kept {am['K']} genes, {am['n_nonzero']} edges, dbar fallback {am['selection']['dbar_fallback']}, "
                   f"fit {am['fit_seconds']:.0f} s)\n")
        mets = ["auroc", "auprc_lift", "p_at_1000", "edge_specific_auroc", "edge_specific_auprc_ratio", "precision_at_k"]
        out.append("| reference@genes | method | k | " + " | ".join(mets) + " | I-SEM − method (AUROC, AUPRC lift) |\n|" + "---|" * (len(mets) + 4))
        for part, r in sc.items():
            if part == "meta":
                continue
            for m in ("I-SEM", "ADAPRE", "inspre (official, E1b)", "I-SEM (top k = ADAPRE edge count)", "I-SEM (top k = official inspre edge count)"):
                row = r.get(m, {})
                cell = []
                for k in mets:
                    v = row.get(k)
                    ci = row.get(f"{k}_ci")
                    cell.append("–" if v is None else (f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{v:.3f}"))
                diff = row.get("isem_minus_this", {})
                dtxt = "; ".join(f"{diff[k]['difference']:+.3f} [{diff[k]['difference_ci'][0]:+.3f}, {diff[k]['difference_ci'][1]:+.3f}]"
                                 for k in ("auroc", "auprc_lift") if k in diff) or "–"
                out.append(f"| {part} | {m} | {row.get('k', '–')} | " + " | ".join(cell) + f" | {dtxt} |")
            mk = r.get("I-SEM (top k = ADAPRE edge count)", {}).get("isem_minus_inspre_official", {}).get("precision_at_k")
            if mk:
                out.append(f"| {part} | **I-SEM top-k − ADAPRE, precision at ADAPRE's k** | | {mk['difference']:+.3f} "
                           f"[{mk['difference_ci'][0]:+.3f}, {mk['difference_ci'][1]:+.3f}], P(≤0) = {mk['p_le0']:.3f} |" + " |" * (len(mets) - 1) + " |")
    f = OUT / "e6_adapre_bias.json"
    if f.exists():
        bj = json.loads(f.read_text())
        out.append("\n## E6 on real data: Spearman of out-strength with |delta_jj| (knockdown) and |Δlog UMI| (library size), 500 gene resamples\n")
        out.append("| dataset | method | ρ(out, \\|δ_jj\\|) [95% CI] | ρ(out, \\|Δlog UMI\\|) [CI] | on ADAPRE-kept genes ρ(out, \\|δ_jj\\|) | # genes with out-edges |\n|---|---|---|---|---|---|")
        for ds, d in bj.items():
            if isinstance(d, str):
                out.append(f"| {ds} | {d} | | | | |"); continue
            for m, v in d["methods"].items():
                c1, c2 = v["spearman_out_vs_absdeltajj_ci"], v["spearman_out_vs_absdlogUMI_ci"]
                out.append(f"| {ds} | {m} | {v['spearman_out_vs_absdeltajj']:.3f} [{c1[0]:.3f}, {c1[1]:.3f}] | "
                           f"{v['spearman_out_vs_absdlogUMI']:.3f} [{c2[0]:.3f}, {c2[1]:.3f}] | "
                           f"{v['on_adapre_kept_genes']['spearman_out_vs_absdeltajj']:.3f} | {v['n_nonzero_out_strength_genes']} |")
            for k, v in d["paired"].items():
                out.append(f"| {ds} | paired {k} | {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | | | |")
    (OUT / "TABLES.md").write_text("\n".join(out) + "\n")
    print("wrote", OUT / "TABLES.md", flush=True)


# ------------------------------------------------------------------------------------------- checks
def check():
    lam = np.array([1.0, .5, .25, .12])
    assert rule_dbar([0, .005, .02, .001], lam) == (1, False)      # cummax: .02 blocks the smallest lambda
    assert rule_dbar([.02, 0, 0, 0], lam) == (0, True)             # none qualifies -> largest, flagged
    assert rule_dbar([0, 0, 0, 0], lam[::-1]) == (0, False)        # ascending grid: index of the smallest lambda
    W = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    em = edge_metrics(W.T, W)                                      # every edge reversed
    assert em["k"] == 3 and em["precision"] == 0 and em["shd"] == 6, em
    em = edge_metrics(W * 0.5, W)
    assert em["f1"] == 1 and em["shd"] == 0
    assert topk_metrics(W + 1e-3 * np.random.default_rng(0).random((3, 3)), W, 3)["f1"] == 1
    path = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    S = embed(path[:, :, 1], ["b", "c"], ["a", "b", "c"])
    assert embed(np.array([[0, np.nan], [np.inf, 0]]), ["a", "b"], ["a", "b"]).sum() == 0
    assert S[1, 2] == path[0, 1, 1] and S[2, 1] == path[1, 0, 1] and S[0].sum() == 0 and np.all(np.diag(S) == 0)
    # analyze.py out-strength = column sums (column j = regulator j, the I-SEM orientation)
    assert np.allclose(outstrength(np.array([[0, 2], [3, 0]])), [3, 2])
    # matched sparsity on a path: lambda 0 has 2 nonzeros, lambda 1 has 4; k = 3 -> lambda 1, top 3 |G|
    P = np.zeros((3, 3, 2), np.float32)
    P[0, 1, 0], P[1, 2, 0] = .9, .8
    P[0, 1, 1], P[1, 2, 1], P[2, 0, 1], P[1, 0, 1] = .9, .8, .7, .1
    m, kk, ok = _path_at_k(P, ["a", "b", "c"], ["a", "b", "c"], 3)
    assert kk == 3 and ok and m[2, 0] and not m[1, 0]
    m, kk, ok = _path_at_k(P, ["a", "b", "c"], ["a", "b", "c"], 9)  # unreachable: densest lambda, flagged
    assert kk == 4 and not ok
    s = _slope(np.array([3., 2, 1, 0]), np.array([-3., -2, -1, 0]))
    assert np.isclose(s["slope"], -1) and np.isclose(s["spearman"], -1)
    print("check ok", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "e2", "e2collect", "real", "score", "bias", "tables", "isemb", "tii", "slopes", "split"])
    ap.add_argument("--default-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ncores", type=int, default=2)
    ap.add_argument("--methods", nargs="+", choices=FIT_METHODS, default=list(FIT_METHODS))
    ap.add_argument("--reps", type=int, nargs="+", default=list(range(e2.REPS)))
    ap.add_argument("--ds", choices=DATASETS)
    ap.add_argument("--its", type=int, default=100)                # released code's default (paper text: 300; not used,
    ap.add_argument("--solve-its", type=int, default=10)           # baselines run at their code defaults); default 10
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.cmd == "check":
        check()
    elif a.cmd == "e2":
        run_e2(a.default_only, a.workers, a.ncores, a.methods, a.reps)
    elif a.cmd == "e2collect":
        e2_collect()
    elif a.cmd == "real":
        for m in a.methods:
            run_real(a.ds, m, a.ncores, a.its, a.solve_its, a.cv, a.overwrite)
    elif a.cmd == "score":
        score_real(a.ds, a.n_boot)
    elif a.cmd == "bias":
        bias_real()
    elif a.cmd == "tables":
        tables()
    elif a.cmd == "isemb":
        isem_b_e2(a.default_only)
    elif a.cmd == "tii":
        tii_e2()
    elif a.cmd == "slopes":
        slopes_e2(); slopes_real()
    elif a.cmd == "split":
        split_e2() if a.ds is None else split_real(a.ds, a.n_boot)


if __name__ == "__main__":
    main()
