#!/usr/bin/env python3
"""Fit I-SEM on a small simulated screen and score it against the planted network. Needs no data.

    bash scripts/experiments/run_env.sh python -u scripts/demo.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import GRNInferencePipeline  # noqa: E402

GENES, CONTROLS, CELLS_PER_KNOCKDOWN, KNOCKDOWN = 30, 500, 100, -2.0

rng = np.random.default_rng(0)
genes = [f"G{i}" for i in range(GENES)]

# Planted network: W[i, j] is the effect of gene j on gene i, cycles allowed.
W = (rng.random((GENES, GENES)) < 0.06) * rng.choice([-0.6, 0.6], (GENES, GENES))
np.fill_diagonal(W, 0)
assert np.abs(np.linalg.eigvals(W)).max() < 1

# Cells at steady state: x = (I - W)^-1 (knockdown + noise).
labels = ["non-targeting"] * CONTROLS + [g for g in genes for _ in range(CELLS_PER_KNOCKDOWN)]
inputs = rng.normal(size=(len(labels), GENES))
for j, g in enumerate(genes):
    inputs[np.array(labels) == g, j] += KNOCKDOWN
X = (inputs @ np.linalg.inv(np.eye(GENES) - W).T).astype(np.float32)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
result = GRNInferencePipeline(device=device).run(X, labels, genes)

truth, off_diagonal = W != 0, ~np.eye(GENES, dtype=bool)
auroc = roc_auc_score(truth[off_diagonal], result.adjacency[off_diagonal])
print(f"{int(truth.sum())} planted edges, AUROC {auroc:.3f} on {device}")
assert auroc > 0.8
