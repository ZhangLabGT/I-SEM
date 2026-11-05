"""Scoring used by every experiment: exact tie-aware AUROC/AP, node bootstrap, degree-preserving nulls.

Every method is scored exactly as it outputs: |A| with the diagonal zeroed, NOT symmetrized. Scoring runs over
all ordered gene pairs (i, j), i != j; STRING and CORUM are undirected, so (i, j) is positive if the genes are
linked in either order. For a symmetric matrix this gives the same AUROC/AUPRC as unordered pairs (asserted in
__main__). Uncertainty: gene-level (node) bootstrap, B_BOOT resamples, pairs of the same resampled gene dropped.
Edge specificity: AUROC/AUPRC against N_REWIRE degree-preserving rewirings of each ground truth.
(Formerly scripts/experiments/paper_eval.py; these functions are unchanged.)
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

B_BOOT, N_REWIRE = 200, 5


def auc_ap(y, s):
    """Exact AUROC and average precision with tied scores grouped (matches sklearn)."""
    o = np.argsort(-s, kind="mergesort"); s, y = s[o], y[o]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[last].astype(float); fps = last + 1 - tps
    P, N = tps[-1], fps[-1]
    ap = np.sum(np.diff(np.r_[0.0, tps / P]) * tps / (tps + fps))
    return float(np.trapz(np.r_[0.0, tps / P], np.r_[0.0, fps / N])), float(ap)


def absd(A):
    """A method's score as provided: magnitudes, diagonal zeroed, direction kept."""
    A = np.abs(np.asarray(A, float)); np.fill_diagonal(A, 0); return A


def offdiag(n):
    """All ordered pairs (i, j), i != j."""
    return np.nonzero(~np.eye(n, dtype=bool))


def metrics(S, G, iu):
    y, s = G[iu], S[iu]
    auroc, auprc = auc_ap(y, s)
    k = 1000; top = np.argsort(-s, kind="mergesort")[:k]
    return {"auroc": auroc, "auprc": auprc, "pos_rate": float(y.mean()),
            "auprc_lift": auprc / float(y.mean()), "p_at_1000": float(y[top].mean())}


def boot(scores, G, n, rng):
    """Node bootstrap: per-resample AUROC/AUPRC for every score (same resample for all -> paired)."""
    out = {m: [] for m in scores}
    for _ in range(B_BOOT):
        idx = rng.choice(n, n, replace=True)
        p, q = offdiag(n); keep = idx[p] != idx[q]
        a, b = idx[p[keep]], idx[q[keep]]
        y = G[a, b]
        for m, S in scores.items():
            out[m].append(auc_ap(y, S[a, b]))
    return {m: np.array(v) for m, v in out.items()}


def rewire(G, reps, seed):
    """Degree-preserving rewirings (double edge swaps, 10 swaps per edge)."""
    n = G.shape[0]; iu = np.triu_indices(n, 1)
    edges = [(i, j) for i, j in zip(*iu) if G[i, j]]
    out = []
    for r in range(reps):
        g = nx.Graph(); g.add_nodes_from(range(n)); g.add_edges_from(edges)
        nx.double_edge_swap(g, nswap=10 * len(edges), max_tries=200 * len(edges), seed=seed + r)
        H = np.zeros((n, n), np.int8)
        for i, j in g.edges(): H[i, j] = H[j, i] = 1
        out.append(H)
    return out


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score, average_precision_score
    _r = np.random.default_rng(1); _y = _r.integers(0, 2, 5000); _s = _r.integers(0, 20, 5000).astype(float)
    assert np.allclose(auc_ap(_y, _s), (roc_auc_score(_y, _s), average_precision_score(_y, _s))), "auc_ap mismatch"
    _A = _r.random((40, 40)); _A = _A + _A.T; _G = _r.random((40, 40)) < .2; _G = (_G | _G.T).astype(int)
    _o, _u = offdiag(40), np.triu_indices(40, 1)  # symmetric scores: ordered pairs == unordered pairs
    assert np.allclose(auc_ap(_G[_o], _A[_o]), auc_ap(_G[_u], _A[_u])), "ordered/unordered mismatch"
    print("paper_eval self-check ok")
