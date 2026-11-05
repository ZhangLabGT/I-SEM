#!/usr/bin/env python3
"""Build data/dataset_<name>.npz from the published single-cell h5ad files.

    python -u scripts/prepare_screens.py                 all four datasets
    python -u scripts/prepare_screens.py hepg2 jurkat

Same steps as preprocess_dataset in CausalBench 1.1.2 (Chevalley et al., Apache-2.0), without its strong-perturbation
filter: scale each cell to the median total count, log1p, keep the knockdowns with more than 100 cells whose target is a
measured gene, and keep only those genes as columns. Cells of any other knockdown stay in the file, labelled "excluded".
"""
import sys
from collections import Counter
from pathlib import Path

import anndata
import numpy as np
from scipy.sparse import issparse

DATA = Path(__file__).resolve().parent.parent / "data"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/"
RAW = {  # local path under data/, download URL
    "k562": ("k562.h5ad", "https://plus.figshare.com/ndownloader/files/35773219"),
    "rpe1": ("rpe1.h5ad", "https://plus.figshare.com/ndownloader/files/35775606"),
    "hepg2": ("nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad", GEO + "GSE264667_hepg2_raw_singlecell_01.h5ad"),
    "jurkat": ("nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad", GEO + "GSE264667_jurkat_raw_singlecell_01.h5ad"),
}


def prepare(path):
    """Returns (expression, var_names, interventions) for one h5ad file."""
    adata = anndata.read_h5ad(path)
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    counts = X.sum(axis=1)
    keep = counts >= 1
    X, counts, targets = X[keep], counts[keep], adata.obs["gene_id"].to_numpy()[keep]
    X = np.log1p(X / (counts / np.median(counts))[:, None])

    genes = adata.var_names.to_numpy()
    cells = Counter(targets)
    kept = {t for t, n in cells.items() if n > 100 and (t in set(genes) or t == "non-targeting")}
    columns = np.array([g in kept for g in genes])
    return X[:, columns], genes[columns], [t if t in kept else "excluded" for t in targets]


def main():
    for screen in sys.argv[1:] or RAW:
        name, url = RAW[screen]
        src = DATA / name
        if not src.exists():
            sys.exit(f"{src} not found. Download it from {url}")
        expr, var_names, interventions = prepare(src)
        out = DATA / f"dataset_{screen}.npz"
        np.savez(out, expression_matrix=expr, var_names=var_names, interventions=interventions)
        print(f"{screen}: {expr.shape[0]} cells x {expr.shape[1]} genes -> {out}", flush=True)


if __name__ == "__main__":
    main()
