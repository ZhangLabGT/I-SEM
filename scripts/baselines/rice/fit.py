"""RICE (Ge & Li 2026) on one real Perturb-seq dataset, following the authors' real-data script
upstream/perturbseq/run_Replogle_2022.py (@ 036b414) statement by statement. Runs inside .envs/rice (CuPy, GPU).

  python fit.py RAW.h5ad PANEL.txt OUT.npz [--jobs 16] [--processed P.npz]

RAW.h5ad   the dataset's raw single-cell h5ad (read only; dense X, obs gene_id/gem_group/mitopercent, var gene_name)
PANEL.txt  our panel, one Ensembl ID per line (data/dataset_<ds>.npz var_names, in that order)
OUT.npz    W (upstream orientation, W[i, j] = effect of gene i on gene j, rows/columns in PANEL order),
           W_c, genes and a JSON meta string. scripts/baselines/runners.py::rice transposes W to the repo orientation.

The only intended differences from the script (README.md):
1. Genes = our panel; the script's gene filter (lines 26-78: >= 100 cells, |z| >= 5 knockdown, mean >= 0.5) is
   not run. Everything from line 80 on is the script's code.
2. UMI_size (line 58) uses obs['core_adjusted_UMI_count'] as the script does when present (K562, RPE1). HepG2 and
   Jurkat lack it: obs['UMI_count'] (each cell's total UMI count) if present, else X.sum(axis=1).
Engineering only (same arithmetic): the raw file is read with obs/var from anndata backed mode and only the
selected cells' rows of X (the script loads the whole file); `--jobs` runs the independent per-gene regressions of
lines 155-159 in parallel processes (--jobs 1 = the script's serial loop); the script's np.savez of the processed
inputs (line 164) goes to --processed instead of data/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(__file__).resolve().parent / "upstream"
COMMIT = "036b4140d7514682c626dbc84ded98b50f542566"
# model.fit(...) of run_Replogle_2022.py line 174 (binsum_k = data_X.shape[1] = p is set at the call)
FIT_ARGS = dict(regularizer="binsum", pen1=1e-4, pen2=0, pen1_c=0, pen2_c=1e-2, lr=0.0001, lr_coef_multiplier=2,
                max_iter=100, checkpoint=2000, h_tol=1e-4, loss_tol=1e-5, rho_init=1, verbose=True)


def umi_column(obs):
    """The per-cell UMI total behind UMI_size (script line 58): the script's column if present, else a substitute."""
    for c in ("core_adjusted_UMI_count", "UMI_count"):
        if c in obs.columns:
            return c
    return "X.sum(axis=1)"


def read_rows(path, rows, block=20000):
    """Dense rows of X (sorted indices) from an h5ad without loading the whole matrix."""
    import h5py
    with h5py.File(path, "r") as h:
        X = h["X"]
        assert isinstance(X, h5py.Dataset), "RICE wrapper expects a dense X"
        out = np.empty((len(rows), X.shape[1]), X.dtype)
        k = 0
        for s in range(0, X.shape[0], block):
            r = rows[(rows >= s) & (rows < s + block)]
            if len(r):
                out[k:k + len(r)] = X[s:s + block][r - s]
                k += len(r)
    assert k == len(rows)
    return out


def row_sums(path, block=20000):
    import h5py
    with h5py.File(path, "r") as h:
        X = h["X"]
        return np.concatenate([X[s:s + block].sum(axis=1) for s in range(0, X.shape[0], block)])


def load(path, panel):
    """adata (obs/var of every cell, UMI_size set) and adata_real_inc (the script's line-80 subset, with X).
    The X-less AnnData is subset exactly as the script's adata[...].copy() (unused categories dropped)."""
    import anndata as ad
    b = ad.read_h5ad(path, backed="r")
    adata = ad.AnnData(obs=b.obs.copy(), var=b.var.copy())
    n_vars = b.shape[1]                                     # = the script's adata.X.shape[1]
    b.file.close()
    umi = umi_column(adata.obs)
    total = row_sums(path) if umi == "X.sum(axis=1)" else adata.obs[umi]
    # run_Replogle_2022.py line 58 (core_adjusted_UMI_count, or the substitute above)
    adata.obs['UMI_size'] = total/n_vars
    # our panel in place of lines 26-78; line 78
    gene_real_inc = list(panel)
    gene_real_inc.append('non-targeting')
    # line 80 (obs/var), then the selected rows of X
    adata_real_inc = adata[adata.obs.gene_id.isin(gene_real_inc)].copy()
    rows = np.where(adata.obs.gene_id.isin(gene_real_inc).values)[0]
    adata_real_inc.X = read_rows(path, rows)
    return adata, adata_real_inc, umi


def _residualize(args):
    """Lines 155-159 for a block of columns (Z, mask, other_gene_exp inherited by fork)."""
    cols, threads = args
    from threadpoolctl import threadpool_limits
    from sklearn.linear_model import LinearRegression
    Z, mask, other_gene_exp = _G["Z"], _G["mask"], _G["other_gene_exp"]
    out = np.empty((other_gene_exp.shape[0], len(cols)), other_gene_exp.dtype)
    with threadpool_limits(threads):
        for k, i in enumerate(cols):
            lr = LinearRegression()
            predictors = np.hstack((Z,mask))
            lr.fit(predictors, other_gene_exp[:,i])
            out[:, k] = other_gene_exp[:,i] - lr.predict(predictors)
    return cols, out


_G = {}


def preprocess(adata_real_inc, panel, jobs=1, log=print, cached=None):
    """run_Replogle_2022.py lines 80-166, verbatim except the line-155 loop's parallel execution (jobs > 1).
    Returns the arrays the fit uses (lines 170-174)."""
    import pandas
    try:
        from tqdm.auto import tqdm                                  # line 112 (in .envs/rice)
    except ImportError:                                             # the CPU test env has no tqdm
        tqdm = lambda x, **_: x  # noqa: E731
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA

    # %% lines 83-86 (library_size and weights are computed but never used by the script; mirrored)
    adata_real_inc.obs['library_size'] = adata_real_inc.X.mean(axis=1)
    gene_counts = adata_real_inc.obs['gene_id'].value_counts()
    adata_real_inc.obs['weights'] = adata_real_inc.X.shape[0]/len(panel + ['non-targeting'])/gene_counts[adata_real_inc.obs['gene_id']].values
    gem_group_onehot = pandas.get_dummies(adata_real_inc.obs['gem_group'], prefix='gem_group', drop_first=True).astype(np.float32)
    # %% lines 88-89
    gene_targeting = gene_counts.index.categories[:-1]
    gene_name_targeting = adata_real_inc.var.gene_name[gene_targeting].values
    assert sorted(gene_targeting) == sorted(panel), "categories[:-1] is not the panel ('non-targeting' not last?)"

    # %% lines 92-95
    gene_columns = []
    for gene in gene_targeting:
        gene_columns.append(np.where(adata_real_inc.var.index == gene)[0][0])
    gene_columns = np.array(gene_columns)

    # %% lines 98-110
    data_X = adata_real_inc.X[:, gene_columns]
    mask = np.zeros_like(data_X, dtype=bool)
    for i in range(data_X.shape[0]):
        if adata_real_inc.obs['gene_id'].values[i] == 'non-targeting':
            continue
        j = np.where(adata_real_inc.obs['gene_id'].values[i] == gene_targeting)[0][0]
        if j.size > 0:
            mask[i, j] = True
    weights = adata_real_inc.obs['weights'].values.reshape(-1,1)  # noqa: F841 (unused, as in the script)
    Z = adata_real_inc.obs[['UMI_size','mitopercent']].values
    Z[:,0] = np.log(Z[:,0])
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    Z = np.hstack((Z, gem_group_onehot.values))

    # lines 112-121
    Z_front_size = Z.shape[1]

    Offset = adata_real_inc.obs['UMI_size'].values.reshape(-1, 1)

    normalized_X = data_X / Offset
    normalized_X = np.log(1 + normalized_X)
    normalized_X = normalized_X - normalized_X.mean(axis=0)
    normalized_X = normalized_X / normalized_X.std(axis=0)

    # lines 123-140
    unique_genes = list(gene_targeting) + ['non-targeting']
    np.random.seed(0)
    train_cell_idx = np.array([])
    test_cell_idx = np.array([])
    train_gene_idx = dict()
    test_gene_idx = dict()
    train_cell_idx_dict = dict()
    for gene in unique_genes:
        gene_cell_idx = np.where(adata_real_inc.obs['gene_id'].values == gene)[0]
        np.random.shuffle(gene_cell_idx)
        split_idx = int(0.8 * len(gene_cell_idx))
        train_cell_idx = np.concatenate((train_cell_idx, gene_cell_idx[:split_idx]))
        test_cell_idx = np.concatenate((test_cell_idx, gene_cell_idx[split_idx:]))
        train_cell_idx_dict[gene] = gene_cell_idx[:split_idx]
        train_gene_idx[gene] = (gene_cell_idx[:split_idx]).astype(int)
        test_gene_idx[gene] = (gene_cell_idx[split_idx:]).astype(int)
    train_cell_idx = train_cell_idx.astype(int)
    test_cell_idx = test_cell_idx.astype(int)

    if cached is not None:   # resume: lines 143-162 from a previous run's line-164 arrays (the rest recomputed)
        for k, v in (("normalized_X", normalized_X), ("mask", mask), ("train_cell_idx", train_cell_idx), ("Z", Z)):
            assert np.array_equal(cached[k], v), f"cached {k} differs from the recomputed one"
        con_res, other_gene_exp_pca = cached["con_res"], cached["other_gene_exp_pca"]
        n_other = int((~adata_real_inc.var.index.isin(gene_targeting)).sum())
        log("  con_res and other_gene_exp_pca reused from the processed file")
    else:
        # lines 143-148
        t0 = time.time()
        con_res = np.zeros_like(data_X)
        for i in tqdm(range(data_X.shape[1]), mininterval=30):
            lr = LinearRegression()
            predictors = np.hstack((Z, mask[:,i].reshape(-1,1)))
            lr.fit(predictors, normalized_X[:,i])
            con_res[:,i] = normalized_X[:,i] - lr.predict(predictors)
        log(f"  con_res: {data_X.shape[1]} regressions in {time.time() - t0:.0f} s")

        # lines 151-159
        t0 = time.time()
        adata_not_inc = adata_real_inc[:,~adata_real_inc.var.index.isin(gene_targeting)].copy()
        Offset = adata_real_inc.obs['UMI_size'].values.reshape(-1, 1)
        other_gene_exp = np.log(1+adata_not_inc.X/Offset)
        other_gene_exp = (other_gene_exp - other_gene_exp.mean(axis=0)) / other_gene_exp.std(axis=0)
        if jobs <= 1:
            for i in tqdm(range(other_gene_exp.shape[1]), mininterval=30):
                lr = LinearRegression()
                predictors = np.hstack((Z,mask))
                lr.fit(predictors, other_gene_exp[:,i])
                other_gene_exp[:,i] = other_gene_exp[:,i] - lr.predict(predictors)
        else:   # the same per-column regressions, columns split over forked processes
            import multiprocessing as mp
            import os
            _G.update(Z=Z, mask=mask, other_gene_exp=other_gene_exp)
            threads = max(1, (os.cpu_count() or jobs) // jobs // 2)
            blocks = [(c, threads) for c in np.array_split(np.arange(other_gene_exp.shape[1]), jobs * 8) if len(c)]
            with mp.get_context("fork").Pool(jobs) as pool:
                res = pool.map(_residualize, blocks, chunksize=1)
            _G.clear()
            for cols, out in res:
                other_gene_exp[:, cols] = out
        log(f"  other genes: {other_gene_exp.shape[1]} regressions in {time.time() - t0:.0f} s (jobs {jobs})")

        # lines 161-162
        other_gene_exp_pca = PCA(n_components=50).fit_transform(other_gene_exp)
        n_other = other_gene_exp.shape[1]

    processed = dict(other_gene_exp_pca=other_gene_exp_pca, con_res=con_res, Z=Z, normalized_X=normalized_X,
                     mask=mask, train_cell_idx=train_cell_idx, test_cell_idx=test_cell_idx)   # line 164's np.savez
    # line 166
    Z = np.hstack((Z, other_gene_exp_pca))
    return dict(data_X=data_X, normalized_X=normalized_X, con_res=con_res, mask=mask, Z=Z, Z_front_size=Z_front_size,
                train_cell_idx=train_cell_idx, gene_targeting=[str(g) for g in gene_targeting],
                gene_name_targeting=[str(g) for g in gene_name_targeting], n_other_genes=n_other,
                processed=processed)


def fit(d):
    """run_Replogle_2022.py lines 170-176 (imports as lines 6-9). Returns the model's result() dict."""
    sys.path.append(str(UPSTREAM / "source"))
    from scRICE_CF import scRICE_CF
    import utils  # noqa: F401  (the script's `from utils import *`; imports castle.metrics, as the script does)
    import cupy as cp
    import scRICE_CF as _m
    assert all(Path(x.__file__).parent == UPSTREAM / "source" for x in (_m, utils)), "not the upstream modules"
    data_X, normalized_X, con_res, mask, Z = d["data_X"], d["normalized_X"], d["con_res"], d["mask"], d["Z"]
    train_cell_idx, Z_front_size = d["train_cell_idx"], d["Z_front_size"]
    need = 16 * data_X[train_cell_idx].size * 8          # measured: hepg2 pool 13x, rpe1 15x one n x p float64
    t0 = time.time()
    while cp.cuda.runtime.memGetInfo()[0] < need and time.time() - t0 < 12 * 3600:   # shared GPU: wait, never trim
        if int(time.time() - t0) % 600 < 30:
            print(f"  waiting for {need / 2**30:.1f} GiB free on the GPU", flush=True)
        time.sleep(30)
    gpu_wait = time.time() - t0
    # lines 170-176
    cp.random.seed(0)
    model = scRICE_CF(dtype=cp.float64)
    model.prep(X=data_X[train_cell_idx], predictor = normalized_X[train_cell_idx], confounding_res = con_res[train_cell_idx], intervention_effect = mask[train_cell_idx], intervention_type='soft', add_intercept=True, Zm_g=Z[train_cell_idx], Zr_g=Z[train_cell_idx][:,:Z_front_size])

    model.fit(binsum_k=data_X.shape[1], **FIT_ARGS)

    result = model.result()
    gpu = dict(gpu_wait_seconds=gpu_wait, device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
               cupy_pool_gib=cp.get_default_memory_pool().total_bytes() / 2**30,
               h=float(model._h_value(model.W, reg=FIT_ARGS["regularizer"])), rho=float(model.rho),
               alpha=float(model.alpha))
    result["W_c"] = cp.asnumpy(result["W_c"])                 # result() leaves W_c on the GPU
    return result, gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("panel")
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=16, help="processes for the lines-155-159 regressions (1 = serial)")
    ap.add_argument("--processed", help="where to write the script's line-164 processed arrays (.npz)")
    ap.add_argument("--reuse-processed", action="store_true", help="resume: read --processed instead of redoing "
                    "lines 143-162 (checked against the recomputed normalized_X, mask, split and Z)")
    a = ap.parse_args()
    from importlib.metadata import version
    panel = [g for g in Path(a.panel).read_text().split() if g]
    t0 = time.time()
    adata, adata_real_inc, umi = load(a.raw, panel)
    mapped = sorted(set(panel) & set(adata.var.index) & set(adata.obs.gene_id.astype(str)))
    print(f"RICE: {adata.n_obs} cells x {adata.n_vars} genes in {a.raw}; panel {len(panel)}, {len(mapped)} mapped "
          f"(var and gene_id); {adata_real_inc.n_obs} cells selected; UMI_size from {umi}", flush=True)
    if len(mapped) != len(panel):
        raise SystemExit(f"panel genes missing from the raw file: {sorted(set(panel) - set(mapped))[:10]}")
    del adata
    cached = dict(np.load(a.processed)) if a.reuse_processed else None
    d = preprocess(adata_real_inc, panel, a.jobs, log=lambda s: print(s, flush=True), cached=cached)
    if a.processed and cached is None:
        np.savez(a.processed, **d["processed"])
    t_pre = time.time() - t0
    print(f"  preprocessing {t_pre:.0f} s; fit on {len(d['train_cell_idx'])} training cells, Z {d['Z'].shape[1]} "
          f"columns ({d['Z_front_size']} before PCA)", flush=True)
    t1 = time.time()
    result, gpu = fit(d)
    t_fit = time.time() - t1
    W = result["W"]
    if not np.isfinite(W).all():
        raise SystemExit(f"RICE fit produced {int((~np.isfinite(W)).sum())} non-finite W entries")
    order = [d["gene_targeting"].index(g) for g in panel]       # script's gene order (sorted ids) -> panel order
    W = W[np.ix_(order, order)]
    W_c = result["W_c"][np.ix_(order, order)]
    meta = {"method": "RICE", "upstream": f"github.com/HowardGech/RICE @ {COMMIT}",
            "protocol": "perturbseq/run_Replogle_2022.py, genes = our panel", "fit_args": {**FIT_ARGS, "binsum_k": len(panel)},
            "prep_args": {"intervention_type": "soft", "add_intercept": True, "dtype": "float64"},
            "result_w_threshold": 0.0, "umi_column": umi, "panel_genes": len(panel), "panel_genes_mapped": len(mapped),
            "n_cells_selected": int(adata_real_inc.n_obs), "n_cells_train": int(len(d["train_cell_idx"])),
            "n_other_genes": int(d["n_other_genes"]), "z_columns": int(d["Z"].shape[1]),
            "z_front_size": int(d["Z_front_size"]), "regression_jobs": a.jobs,
            "reused_processed": bool(a.reuse_processed), **gpu,
            "preprocess_seconds": t_pre, "fit_seconds": t_fit - gpu["gpu_wait_seconds"], "wall_seconds": time.time() - t0,
            "n_nonzero_offdiag": int(((W != 0) & ~np.eye(len(W), dtype=bool)).sum()),
            "versions": {k: version(k) for k in ("cupy-cuda12x", "numpy", "scipy", "pandas", "scikit-learn", "anndata")},
            "orientation": "W[i, j] = effect of gene i on gene j (m = ... + (gX @ W) * mask, scRICE_CF.py line 71)"}
    np.savez(a.out, W=W, W_c=W_c, genes=np.array(panel), meta=json.dumps(meta))
    print(f"  fit {t_fit:.0f} s; {meta['n_nonzero_offdiag']} nonzero off-diagonal W; pool {gpu['cupy_pool_gib']:.1f} GiB",
          flush=True)


if __name__ == "__main__":
    main()
