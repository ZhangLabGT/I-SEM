"""I-SEM pipeline: condition-mean shifts, then a half-split ensemble, then the mean edge score.

    result = GRNInferencePipeline(device="cuda:0").run(expression, interventions, gene_names)
    result.adjacency   # S[i, j]: score of the edge j -> i
    result.edge_list   # [(source, target, score), ...], highest score first
    result.metadata    # {"n_models": 120, "elapsed_seconds": ..., ...}
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models.interventional_nb_sem import (
    InterventionalSEMConfig,
    build_intervention_targets,
    train_batched,
)

logger = logging.getLogger(__name__)


@dataclass
class GRNResult:
    adjacency: np.ndarray
    edge_list: List[Tuple[str, str, float]]
    metadata: Dict


def _mean_ensemble(adj_list: List[np.ndarray]) -> np.ndarray:
    result = np.mean(adj_list, axis=0)
    np.fill_diagonal(result, 0)
    return result


def _symmetrize(adj: np.ndarray) -> np.ndarray:
    result = (adj + adj.T) / 2.0
    np.fill_diagonal(result, 0)
    return result


def _build_edge_list(
    adj: np.ndarray,
    gene_names: List[str],
) -> List[Tuple[str, str, float]]:
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] > 0:
                edges.append((gene_names[j], gene_names[i], float(adj[i, j])))
    edges.sort(key=lambda e: -e[2])
    return edges


def _sem_config(
    epochs: int,
    warmup_epochs: int,
    l1_weight: float,
    w_dropout: float,
) -> InterventionalSEMConfig:
    return InterventionalSEMConfig(
        epochs=epochs, lr=0.05, lr_weights=0.01, delta_weight=10.0,
        l1_weight=l1_weight, l2_weight=1e-4,
        stability_weight=1.0, tau=1.5,
        warmup_epochs=warmup_epochs,
        w_dropout=w_dropout,
    )


def _complementary_pairs(n_conditions: int, n_resamples: int) -> List[Tuple[np.ndarray, int]]:
    """(train_idx, seed) per model, in training order: resample b -> half A, half B."""
    half = n_conditions // 2
    pairs = []
    for b in range(n_resamples):
        perm = np.random.RandomState(42 + b).permutation(n_conditions)
        pairs.append((perm[:half], 42 + b * 2))
        pairs.append((perm[half:2 * half], 42 + b * 2 + 1))
    return pairs


def _fit_ensemble(
    u_mat: np.ndarray,
    delta_obs: np.ndarray,
    se_obs: np.ndarray,
    n_resamples: int,
    device: str,
    epochs: int,
    warmup_epochs: int,
    l1_weight: float,
    w_dropout: float,
    models_per_chunk: Optional[int] = None,
    readout: str = "total",
) -> Tuple[List[np.ndarray], Dict]:
    """Train the complementary-pair models, models_per_chunk at a time. Returns their edge-score
    matrices in training order and train_batched's per-model diagnostics (fallback "model" = index in that order)."""
    jobs = _complementary_pairs(u_mat.shape[0], n_resamples)
    chunk = models_per_chunk or len(jobs)
    cfg = _sem_config(epochs, warmup_epochs, l1_weight, w_dropout)

    out: List[np.ndarray] = []
    diagnostics: Dict = {}
    for start in range(0, len(jobs), chunk):
        part = jobs[start:start + chunk]
        t0 = time.time()
        causal, loss, *params, diag = train_batched(
            u_mat, delta_obs, se_obs,
            folds=[j[0] for j in part], seeds=[j[1] for j in part],
            config=cfg, device=device, return_params=readout == "direct", return_diagnostics=True,
        )
        for f in diag["fallbacks"]:
            f["model"] += start
        for k, v in diag.items():
            diagnostics.setdefault(k, []).extend(v)
        if readout == "direct":
            W, B = params
            # |W[i,j] B[j]|; B[j] stays 1.0 if j's knockdown is not in the half
            causal = np.abs(W * B[:, None, :])
        out.extend(causal)
        logger.info(
            "[batched %d-%d/%d] %d models in %.1fs, mean loss=%.6f",
            start + 1, start + len(part), len(jobs), len(part), time.time() - t0, float(loss.mean()),
        )
    return out, diagnostics


class GRNInferencePipeline:
    """The defaults are the paper's model: direct readout, not symmetrized, 60 resamples = 120 models.

    Args:
        n_resamples: Complementary-pair resamples; each trains 2 models.
        symmetrize: Apply (S + S^T) / 2 to the final matrix.
        epochs: Training epochs per model.
        warmup_epochs: Epochs that train B with W frozen.
        l1_weight: L1 penalty on W.
        w_dropout: Dropout rate on W during training.
        device: Torch device string.
        models_per_chunk: Models trained at once (None = all). Each chunk restarts the dropout and batch-order
            random streams from its first model's seed, so the chunking changes the scores. The default trains
            all 120 models in one chunk (K562 peak about 5.2 GiB).
        readout: Per-model edge score. "direct" = |W[i,j] b[j]|, where b[j] is the learned B[j] if gene j's
            knockdown is in the model's half of the conditions and 1 otherwise. "total" = |(I-W)^-1 B|.
    """

    def __init__(
        self,
        n_resamples: int = 60,
        symmetrize: bool = False,
        epochs: int = 250,
        warmup_epochs: int = 50,
        l1_weight: float = 0.005,
        w_dropout: float = 0.0,
        device: str = "cpu",
        models_per_chunk: Optional[int] = 120,
        readout: str = "direct",
    ):
        if readout not in ("total", "direct"):
            raise ValueError(f"readout must be 'total' or 'direct', got {readout!r}")
        self.n_resamples = n_resamples
        self.symmetrize = symmetrize
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.l1_weight = l1_weight
        self.w_dropout = w_dropout
        self.device = device
        self.models_per_chunk = models_per_chunk
        self.readout = readout

    def run(
        self,
        expression: np.ndarray,
        interventions: List[str],
        gene_names: List[str],
    ) -> GRNResult:
        """expression: (n_cells, n_genes); interventions: per-cell knockdown labels; gene_names: (n_genes,)."""
        t0 = time.time()
        u_mat, delta_obs, _, _, _, se_obs = build_intervention_targets(
            expression=expression,
            interventions=list(interventions),
            gene_names=list(gene_names),
        )
        logger.info("Preprocessed: %d conditions, %d genes", u_mat.shape[0], len(gene_names))

        scores, diagnostics = _fit_ensemble(
            u_mat=u_mat,
            delta_obs=delta_obs,
            se_obs=se_obs,
            n_resamples=self.n_resamples,
            device=self.device,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
            l1_weight=self.l1_weight,
            w_dropout=self.w_dropout,
            models_per_chunk=self.models_per_chunk,
            readout=self.readout,
        )
        adjacency = _mean_ensemble(scores)
        if self.symmetrize:
            adjacency = _symmetrize(adjacency)
        edge_list = _build_edge_list(adjacency, gene_names)

        elapsed = time.time() - t0
        logger.info("Pipeline complete: %d models, %.1fs, %d edges", len(scores), elapsed, len(edge_list))
        return GRNResult(
            adjacency=adjacency,
            edge_list=edge_list,
            metadata={
                "n_models": len(scores),
                "n_resamples": self.n_resamples,
                "epochs": self.epochs,
                "symmetrized": self.symmetrize,
                "readout": self.readout,
                "elapsed_seconds": round(elapsed, 1),
                "n_genes": len(gene_names),
                "n_conditions": u_mat.shape[0],
                "model_diagnostics": diagnostics,  # per-model norms, min LU pivot, failing-pivot records
            },
        )
