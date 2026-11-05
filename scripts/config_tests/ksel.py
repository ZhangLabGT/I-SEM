"""Graph-size (k) selection on held-out TRAINING cells, for CausalBench submissions of ranking methods.

CausalBench hands a model only its training split. Inside that split we hold out VAL_FRACTION of the cells of
every condition (control included), fit the method on the rest, and grow its ranked edge list in blocks of
BLOCK edges. An edge source -> target counts as a validated effect when the target's expression in the
source-knockdown validation cells differs from control validation cells by Welch's t-test (p < P_EDGE).
The benchmark uses a different statistic (Wasserstein distance) and a different test (Mann-Whitney U) on
its own held-out test cells, neither of which is used here. Edges are added while a block's validated
fraction is significantly higher than that of uniformly random ordered pairs (one-sided Fisher exact test,
alpha = ALPHA); k* is the size before the first block that is not. k* is also capped so the graph keeps
>= OMISSION_SIZE ordered pairs with no directed path (CausalBench's FOR is undefined otherwise).
The method is then refit on the full training split and its top k* edges are submitted.

Only functions here (CausalBench picks the AbstractInferenceModel subclass of a custom file).
"""
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import fisher_exact, ttest_ind

CONTROL = "non-targeting"
VAL_FRACTION = float(os.getenv("KSEL_VAL_FRACTION", "0.2"))
BLOCK = int(os.getenv("KSEL_BLOCK", "500"))
P_EDGE = 0.05
ALPHA = 0.05
N_NULL = int(os.getenv("KSEL_N_NULL", "5000"))
OMISSION_SIZE = 500  # must match --omission_estimation_size
SENSITIVITY_BLOCKS = (250, 500, 1000)
OUT = Path(os.getenv("KSEL_OUT", Path(__file__).resolve().parent / "out"))


def split_cells(interventions: Sequence[str], frac: float, seed: int) -> np.ndarray:
    """Boolean validation mask: round(frac * n_c) random cells of every condition c (fewer than 1 -> none)."""
    iv = np.asarray(interventions)
    rng = np.random.RandomState(seed)
    val = np.zeros(len(iv), bool)
    for c in np.unique(iv):
        idx = np.flatnonzero(iv == c)
        rng.shuffle(idx)
        val[idx[:int(round(frac * len(idx)))]] = True
    return val


class EdgeTester:
    """Welch t-test of target expression, source-knockdown vs control, on validation cells (cached per edge)."""

    def __init__(self, X: np.ndarray, interventions: Sequence[str], gene_names: Sequence[str]):
        iv = np.asarray(interventions)
        self.g2i = {g: i for i, g in enumerate(gene_names)}
        self.ctrl = X[iv == CONTROL]
        self.kd = {g: X[iv == g] for g in np.unique(iv) if g != CONTROL and g in self.g2i}

    def testable(self, src: str) -> bool:
        return len(self.kd.get(src, ())) >= 2 and len(self.ctrl) >= 2

    def significant(self, edges: Sequence[Tuple[str, str]]) -> np.ndarray:
        """1 = validated, 0 = not, -1 = untestable (source has < 2 validation knockdown cells)."""
        out = np.full(len(edges), -1, np.int8)
        for e, (s, t) in enumerate(edges):
            if self.testable(s):
                a, b = self.kd[s][:, self.g2i[t]], self.ctrl[:, self.g2i[t]]
                p = ttest_ind(a, b, equal_var=False).pvalue
                out[e] = int(np.isfinite(p) and p < P_EDGE)
        return out

    def null_edges(self, n: int, seed: int) -> List[Tuple[str, str]]:
        """n uniformly random ordered pairs (s != t) with a testable source."""
        rng = np.random.RandomState(seed)
        srcs = sorted(s for s in self.kd if self.testable(s))
        genes = list(self.g2i)
        out = []
        while len(out) < n:
            s, t = srcs[rng.randint(len(srcs))], genes[rng.randint(len(genes))]
            if s != t:
                out.append((s, t))
        return out


def unconnected_pairs(edges: Sequence[Tuple[str, str]], gene_names: Sequence[str]) -> int:
    """Ordered pairs (a, b), a != b, with no directed path a -> b (CausalBench's FOR negative pool)."""
    g2i = {g: i for i, g in enumerate(gene_names)}
    n = len(gene_names)
    if not edges:
        return n * (n - 1)
    A = csr_matrix((np.ones(len(edges)), ([g2i[a] for a, _ in edges], [g2i[b] for _, b in edges])), shape=(n, n))
    return int((~np.isfinite(shortest_path(A, unweighted=True))).sum())


def select_k(ranked: Sequence[Tuple[str, str]], sig: Callable[[Sequence[Tuple[str, str]]], np.ndarray],
             null_sig: np.ndarray, gene_names: Sequence[str], block: int = BLOCK) -> Tuple[int, Dict]:
    """Stopping rule. Returns (k*, record with the per-block curve and why it stopped)."""
    null = null_sig[null_sig >= 0]
    x0, n0 = int(null.sum()), len(null)
    curve, k, reason = [], 0, "exhausted ranking"
    for start in range(0, len(ranked), block):
        blk = ranked[start:start + block]
        s = sig(blk)
        s = s[s >= 0]
        x, n = int(s.sum()), len(s)
        p = fisher_exact([[x, n - x], [x0, n0 - x0]], alternative="greater")[1] if n else 1.0
        curve.append({"start": start, "end": start + len(blk), "validated": x, "tested": n,
                      "rate": x / n if n else float("nan"), "p_vs_random": float(p)})
        if p >= ALPHA:
            reason = "block not better than random pairs"
            break
        if unconnected_pairs(ranked[:start + len(blk)], gene_names) < OMISSION_SIZE:
            reason = "FOR cap (fewer than %d unconnected pairs)" % OMISSION_SIZE
            break
        k = start + len(blk)
    return k, {"k": k, "block": block, "stop_reason": reason, "random_rate": x0 / n0 if n0 else float("nan"),
               "random_validated": x0, "random_tested": n0, "curve": curve}


def choose_k(ranked_on_fit: Sequence[Tuple[str, str]], X_val: np.ndarray, iv_val: Sequence[str],
             gene_names: Sequence[str], seed: int) -> Tuple[int, Dict]:
    """k* at BLOCK, plus k* at SENSITIVITY_BLOCKS for the supplement (edge tests are cached and shared)."""
    tester = EdgeTester(X_val, iv_val, gene_names)
    cache: Dict[Tuple[str, str], int] = {}

    def sig(edges):
        todo = [e for e in edges if e not in cache]
        for e, v in zip(todo, tester.significant(todo)):
            cache[e] = int(v)
        return np.array([cache[e] for e in edges], np.int8)

    null_sig = tester.significant(tester.null_edges(N_NULL, seed))
    k, rec = select_k(ranked_on_fit, sig, null_sig, gene_names, BLOCK)
    rec["sensitivity"] = {str(b): (k if b == BLOCK else select_k(ranked_on_fit, sig, null_sig, gene_names, b)[0])
                          for b in SENSITIVITY_BLOCKS}
    return k, rec


def save_record(name: str, rec: Dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{name}.json"
    json.dump(rec, open(f, "w"), indent=1)
    return f
