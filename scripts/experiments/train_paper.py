"""Fresh post-Part-A fits for E3, E6/E6b and later E1; all writes stay in scripts/.

No E4 timings are produced: this script is allowed on the shared GPU.
Cached training durations are provenance only, never a runtime benchmark.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.stats import spearmanr
from src.models.interventional_nb_sem import build_intervention_targets, train_batched
from src.pipeline import GRNInferencePipeline, _complementary_pairs, _mean_ensemble, _sem_config, _symmetrize
from scripts.eval_all_metrics import load_dataset

PAPER = ROOT / "scripts/results/paper"
CACHE = PAPER / "cache"
REFERENCE = PAPER / "reference_option_a"  # {ds}.npz["S_a"]: full fit of the current model (option-A solver)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def provenance():
    return {
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "date": datetime.now(timezone.utc).isoformat(),
        "hardware": {"gpu": torch.cuda.get_device_name(0), "device": "cuda:0", "dedicated": False},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT / "src/pipeline.py", ROOT / "src/models/interventional_nb_sem.py", ROOT / "src/models/batched_linalg.py"]},
        "note": "Shared GPU training only; these durations are not E4 measurements.",
    }


def fit(ds, variant, write_reference=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    out_path = CACHE / (f"{ds}.npz" if variant == "full" else f"{ds}_{variant}.npz")
    meta_path = out_path.with_suffix(".json")
    meta = provenance()
    if out_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["code_commit"] == meta["code_commit"] and old["source_sha256"] == meta["source_sha256"]:
            if write_reference:
                raise RuntimeError(f"--write-reference needs a fresh fit, but the current-code cache {out_path} exists: "
                                   "move it (and its .json) away first")
            print(f"reuse validated current-code cache {out_path}", flush=True)
            return
        raise RuntimeError(f"Refusing to overwrite cache from another source revision: {out_path}")

    reference = REFERENCE / f"{ds}.npz"
    if variant == "full" and reference.exists() == write_reference:
        raise RuntimeError(f"Refusing to overwrite the reference {reference}" if write_reference else
                           f"Missing reference {reference}. Create it from the committed src (clean tree): "
                           f"python -u scripts/experiments/train_paper.py --datasets {ds} --write-reference")
    print(f"START {ds} {variant} {meta['date']}", flush=True)
    expression, interventions, genes = load_dataset(ds)
    u, delta, n_c, control, cmean, se = build_intervention_targets(expression, interventions, genes)
    n = len(genes)
    del expression
    pipe = GRNInferencePipeline(device="cuda:0")
    assert (pipe.n_resamples, pipe.epochs, pipe.warmup_epochs, pipe.l1_weight,
            pipe.w_dropout, pipe.models_per_chunk, pipe.readout, pipe.symmetrize) == (60, 250, 50, .005, 0.0, 120, "direct", False)
    cfg = _sem_config(pipe.epochs, pipe.warmup_epochs, pipe.l1_weight, pipe.w_dropout)
    pairs = _complementary_pairs(len(u), 1 if variant == "two_models" else pipe.n_resamples)
    if variant == "uniform":
        # Inverse-SE weights on ones are exactly uniform, without modifying production code.
        fit_se = np.ones_like(se)
    else:
        fit_se = se
    if variant == "no_stability":
        cfg = dataclasses.replace(cfg, stability_weight=0.0)
    start = time.monotonic()
    outputs = train_batched(u, delta, fit_se, folds=[p[0] for p in pairs], seeds=[p[1] for p in pairs],
                            config=cfg, device="cuda:0", return_params=True,
                            return_diagnostics=variant == "full")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    total, loss, W, B = outputs[:4]
    direct = _mean_ensemble(np.abs(W * B[:, None, :]))
    assert np.isfinite(direct).all()
    payload = {"scores": direct, "gene_names": np.asarray(genes)}
    if variant == "full":
        membership = np.stack([(u[p[0]] > .5).any(0) for p in pairs])
        assert np.all(B[~membership] == 1.0), "Out-of-half B drifted"
        counts = membership.sum(0)
        # Odd condition counts leave one condition out of each complementary pair (HepG2: 227 -> 58-60 halves).
        assert np.all(counts > 0), "a gene's knockdown is in no model's half"
        fixed = np.zeros((n, n), dtype=np.float64)
        for m in range(len(pairs)):
            fixed += np.abs(W[m].astype(np.float64) * B[m][None, :]) * membership[m][None, :]
        fixed /= counts[None, :]
        np.fill_diagonal(fixed, 0)
        W_only = _mean_ensemble(np.abs(W))
        ensemble_W = W.mean(0)
        target = (u > .5).argmax(1)
        assert np.all((u > .5).sum(1) == 1)
        assert sorted(target.tolist()) == list(range(n))
        delta_jj = np.zeros(n)
        delta_jj[target] = delta[np.arange(len(delta)), target]
        inhalf_abs_B = (np.abs(B) * membership).sum(0) / counts
        # Reference = a full fit of this config (tau 1.5, dropout 0, half-split) with the option-A solver, rule a =
        # this readout. (E7's tau1.5_do0.0_half fits used the pre-option-A solver, src_old.)
        if write_reference:
            reference.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(reference, S_a=direct)
            json_write(reference.with_suffix(".json"), meta)
            print(f"wrote reference {reference}", flush=True)
        expected = np.load(reference, allow_pickle=True)["S_a"]
        offdiag = ~np.eye(n, dtype=bool)
        maxrel = float(np.max(np.abs(direct - expected)) / np.max(np.abs(expected)))
        rho = float(spearmanr(direct[offdiag], expected[offdiag]).statistic)
        meta["part_a_acceptance"] = {"max_relative_difference": maxrel, "spearman_offdiag": rho}
        # Tolerance as in Part A: float32 averaging can reorder near-ties, so Spearman may fall just below 1.
        if maxrel > 1e-5 or rho < 1 - 1e-10:
            raise RuntimeError(f"Current fit differs from the pinned reference scores: {meta['part_a_acceptance']}")
        payload.update(S_current=direct, S_fixed=fixed.astype(np.float32), W_only=W_only,
                       ensemble_W=ensemble_W, delta_jj=delta_jj,
                       inhalf_mean_abs_B=inhalf_abs_B, cond_target_gene_idx=target,
                       n_c=n_c, u=u, delta=delta, se=se, control_mean=cmean,
                       total_readout=_mean_ensemble(total), symmetrized=_symmetrize(direct),
                       B_models=B, membership=membership, inhalf_counts=counts)   # membership[m, j]: j's knockdown in model m's half
        # Retain actual per-model |W| for auditing E6b, rather than reconstructing it from B.
        np.save(CACHE / f"{ds}_W_models.npy", W)
        json_write(CACHE / f"{ds}_norms.json", outputs[-1])
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(out_path)
    meta.update(dataset=ds, variant=variant, n_genes=n, n_cells=len(interventions),
                n_conditions=len(u), n_models=len(pairs), seeds=[p[1] for p in pairs],
                config=dataclasses.asdict(cfg), training_seconds_provenance_only=elapsed,
                final_loss=loss.tolist(), W_only_definition="mean_m(abs(W_m)), diagonal zero")
    json_write(meta_path, meta)
    print(f"DONE {ds} {variant}: {out_path}, {elapsed:.1f}s (not E4)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hepg2", "rpe1", "jurkat", "k562"])
    ap.add_argument("--variants", nargs="+", choices=["full", "uniform", "no_stability", "two_models"], default=["full"])
    ap.add_argument("--write-reference", action="store_true",
                    help="save this full fit as the acceptance reference REFERENCE/{ds}.npz; the run then checks the "
                         "fit against itself (trivially passes). Needs --variants full, src/ equal to git HEAD and "
                         "no current-code cache")
    args = ap.parse_args()
    if args.write_reference:
        if args.variants != ["full"]:
            ap.error("--write-reference only applies to --variants full")
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src"], cwd=ROOT).returncode != 0:
            ap.error("--write-reference needs src/ equal to git HEAD (commit first): the reference records that commit")
    if not torch.cuda.is_available():
        raise RuntimeError("The paper fits require CUDA; CPU training has different penalty arithmetic")
    torch.set_num_threads(8)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for variant in args.variants:
        for ds in args.datasets:
            fit(ds, variant, args.write_reference)


if __name__ == "__main__":
    main()
