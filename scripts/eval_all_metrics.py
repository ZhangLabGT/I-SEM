#!/usr/bin/env python3
"""Data and reference-network loaders shared by the experiments.

load_dataset(name) -> (expression, interventions, gene_names) for the processed datasets in data/;
gene_names_to_symbols, load_string, load_corum(_complexes), edges_to_adj;
build_delta_baseline (the mean-difference control).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

DATASET_REGISTRY = {  # processed Perturb-seq datasets, see scripts/prepare_screens.py
    name: {"npz": f"dataset_{name}.npz", "gene_key": "var_names", "has_ensg": True}
    for name in ("k562", "k562_filtered", "rpe1", "rpe1_filtered", "hepg2", "jurkat")
}

# ---------------------------------------------------------------------------
# ENSG mapping
# ---------------------------------------------------------------------------
_ENSG_MAP: Optional[Dict[str, str]] = None



def load_ensg_map() -> Dict[str, str]:
    global _ENSG_MAP
    if _ENSG_MAP is not None:
        return _ENSG_MAP
    import pandas as pd
    path = Path(__file__).resolve().parent / "ensembl_to_hgnc.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    _ENSG_MAP = {}
    for _, row in df.iterrows():
        eid = str(row["ensembl_gene_id"])
        sym = str(row["hgnc_symbol"])
        if sym and sym != "nan" and sym != "":
            _ENSG_MAP[eid] = sym
    return _ENSG_MAP


def gene_names_to_symbols(gene_names: List[str]) -> Tuple[List[str], Dict[int, str]]:
    if not gene_names or not gene_names[0].startswith("ENSG"):
        idx_to_sym = {i: g for i, g in enumerate(gene_names)}
        return gene_names, idx_to_sym
    ensg_map = load_ensg_map()
    idx_to_sym = {}
    symbols = []
    for i, g in enumerate(gene_names):
        sym = ensg_map.get(g)
        if sym:
            idx_to_sym[i] = sym
            symbols.append(sym)
        else:
            symbols.append(g)
    return symbols, idx_to_sym


# ---------------------------------------------------------------------------
# Ground truth loaders
# ---------------------------------------------------------------------------

def load_string(min_score: int = 400) -> Dict[Tuple[str, str], float]:
    import pandas as pd
    info_path = DATA_ROOT / "9606.protein.info.v11.5.txt"
    links_path = DATA_ROOT / "9606.protein.links.detailed.v11.5.txt"
    if not info_path.exists() or not links_path.exists():
        return {}
    info = pd.read_csv(info_path, sep="\t")
    prot2gene = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    edges = {}
    for chunk in pd.read_csv(links_path, sep=" ", chunksize=500_000):
        chunk = chunk[chunk["combined_score"] >= min_score]
        for _, row in chunk.iterrows():
            g1 = prot2gene.get(row["protein1"])
            g2 = prot2gene.get(row["protein2"])
            if g1 and g2 and g1 != g2:
                score = row["combined_score"] / 1000.0
                edges[(g1, g2)] = max(edges.get((g1, g2), 0), score)
                edges[(g2, g1)] = max(edges.get((g2, g1), 0), score)
    return edges


# Full CORUM 5.3. The old data/humanComplexes.txt was truncated to 2 complexes
# (ribosome + 26S proteasome).
CORUM_PATH = DATA_ROOT / "corum_human_v5.3.txt"


def load_corum_complexes() -> List[List[str]]:
    """Human CORUM 5.3 complexes with >=2 subunits, as lists of gene symbols.

    The file quotes fields that contain tabs or newlines (comments), so it is read with csv, not line by line: a
    line split loses the 564 complexes whose records span several lines."""
    import csv
    complexes = []
    with open(CORUM_PATH, encoding="utf-8", errors="replace", newline="") as f:
        rows = csv.reader(f, delimiter="\t")
        header = next(rows)
        gi, oi = header.index("subunits_gene_name"), header.index("organism")
        for parts in rows:
            if len(parts) <= gi or parts[oi].strip() != "Human":
                continue
            subunits = [g.strip() for g in parts[gi].split(";") if g.strip()]
            if len(subunits) >= 2:
                complexes.append(subunits)
    return complexes


def load_corum() -> Dict[Tuple[str, str], float]:
    edges = {}
    for subunits in load_corum_complexes():
        for i, g1 in enumerate(subunits):
            for g2 in subunits[i + 1:]:
                if g1 != g2:
                    edges[(g1, g2)] = 1.0
                    edges[(g2, g1)] = 1.0
    return edges


def edges_to_adj(
    edges: Dict[Tuple[str, str], float],
    gene_names: List[str],
    idx_to_sym: Dict[int, str],
) -> np.ndarray:
    """Convert edge dict to adjacency matrix aligned with gene_names indexing."""
    n = len(gene_names)
    adj = np.zeros((n, n), dtype=float)
    sym_to_idx = {sym: idx for idx, sym in idx_to_sym.items()}
    for (src, tgt), val in edges.items():
        src_idx = sym_to_idx.get(src)
        tgt_idx = sym_to_idx.get(tgt)
        if src_idx is not None and tgt_idx is not None and src_idx != tgt_idx:
            # Edge (src, tgt) means src regulates tgt: W[tgt_idx, src_idx]
            adj[tgt_idx, src_idx] = max(adj[tgt_idx, src_idx], val)
    return adj


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

def load_dataset(name: str):
    """Load dataset and return expression, interventions, gene_names."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    info = DATASET_REGISTRY[name]
    npz_path = DATA_ROOT / info["npz"]
    data = np.load(str(npz_path), allow_pickle=True)

    if "expression_matrix" in data:
        expression = data["expression_matrix"]
    else:
        expression = data["expression"]
    interventions = list(data[info["gene_key"].replace("gene_names", "interventions") if "intervention" in str(data.files) else "interventions"])
    gene_names = list(data[info["gene_key"]])

    # Fix: properly handle interventions key
    if "interventions" in data.files:
        interventions = list(data["interventions"])
    elif "intervention" in data.files:
        interventions = list(data["intervention"])

    return expression, interventions, [str(g) for g in gene_names]


# ---------------------------------------------------------------------------
# Build delta baseline
# ---------------------------------------------------------------------------

def build_delta_baseline(expression, interventions, gene_names, idx_to_sym):
    """Build adjacency from raw mean differences |mean_KO - mean_control|."""
    from src.models.interventional_nb_sem import infer_control_label
    interventions_arr = np.array(interventions)
    control_label = infer_control_label(list(interventions))
    control_mask = interventions_arr == control_label
    control_mean = expression[control_mask].mean(axis=0)

    n = len(gene_names)
    delta_mat = np.zeros((n, n), dtype=float)

    # Map symbols to indices
    sym_to_idx = {sym: idx for idx, sym in idx_to_sym.items()}

    unique_conditions = np.unique(interventions_arr)
    for label in unique_conditions:
        if label == control_label:
            continue
        label_str = str(label).strip()
        # Try direct gene name match and symbol match
        j = None
        if label_str in sym_to_idx:
            j = sym_to_idx[label_str]
        else:
            gene_to_idx = {g: i for i, g in enumerate(gene_names)}
            if label_str in gene_to_idx:
                j = gene_to_idx[label_str]
        if j is None:
            continue
        cond_mask = interventions_arr == label
        cond_mean = expression[cond_mask].mean(axis=0)
        delta_mat[:, j] = np.abs(cond_mean - control_mean)

    np.fill_diagonal(delta_mat, 0.0)
    return delta_mat
