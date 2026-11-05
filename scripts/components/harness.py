#!/usr/bin/env python3
"""I-SEM ablation + plug-in. src/ imported unchanged; all writes under this folder.

  bash run.sh harness.py fit --kind sims --arms shipped --workers 4 --limit 16
  bash run.sh harness.py fit --kind real --ds hepg2 --arms shipped no_se --workers 2
  bash run.sh harness.py plugin
Cache name `no_push` = the table row `no_push_abs`.
"""
import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/config_tests"), str(ROOT / "scripts/experiments"),
                str(ROOT / "scripts/experiments/e9_author_sims")]
import run_sims  # noqa: E402
import score  # noqa: E402

e9, e8 = score.e9, score.e9.e8
e2 = e9.e2
SIM = ROOT / "scripts/fresh_seeds"
e9.DATA, e9.OUT, e9.CACHE, e9.LOGS, e9.TMP = SIM / "data", HERE, HERE / "cache", HERE / "logs", HERE / "tmp"
e8.TMP = run_sims.TMP = HERE / "tmp"
run_sims.OUT = HERE / "cache"
SHIPPED_REAL = ROOT / "scripts/results/config_tests/real"
SCREENS = ("hepg2", "rpe1", "jurkat", "k562")
ARMS = ("shipped", "no_push", "no_se", "no_stability", "no_half", "no_push_signed", "wald")
SEEDS = tuple(range(12, 22))
OOM_WAIT, OOM_MAX = 30, 480
GPU_FULL = ("out of memory", "alloc_failed", "cusolver")


def block(name="weak"):
    """A block by e9.describe, seeds 12-21: weak (small effects, weak instruments) or contrast (large, strong)."""
    want = {"weak": ("small", "weak"), "contrast": ("large", "strong")}[name]
    out = []
    for s in sorted((e9.DATA / "inspre").iterdir()):
        d = e9.describe(json.loads((s / "rep12/meta.json").read_text()))
        if (d["effects"], d["instruments"]) == want:
            assert name != "weak" or s.name.endswith(("_7", "_8")), s.name
            out += [s / f"rep{r}" for r in SEEDS]
    assert len(out) == 160, len(out)
    return out


def wald_targets(u, delta, se):
    """`wald`: d' = d / d_jj and s' = s / |d_jj| per condition, float64 then float32. No filter, clip or guard;
    an exact-zero divisor is a failed fit."""
    assert (u.sum(1) == 1).all(), "a condition is not one panel gene"
    dj = np.asarray(delta, np.float64)[np.arange(len(u)), u.argmax(1)][:, None]
    if (dj == 0).any():
        raise ValueError(f"wald: {int((dj == 0).sum())} exact-zero divisor(s)")
    return (np.asarray(delta, np.float64) / dj).astype(np.float32), (np.asarray(se, np.float64) / np.abs(dj)).astype(np.float32)


def signed_init(model, u_mat, delta_obs):
    """`no_push_signed` start: in-half B_j = sign(d_jj) clip(|d_jj|, 0.01, 100), others 1."""
    import torch
    b = np.ones(model.n_genes, dtype=np.float32)
    for j in range(model.n_genes):
        mask = u_mat[:, j] > 0.5
        if mask.any():
            d = np.mean(delta_obs[mask, j])
            b[j] = np.sign(d) * np.clip(np.abs(d), .01, 100.)
    with torch.no_grad():
        model.direct.copy_(torch.from_numpy(b))


def arm_inputs(arm, u, delta, se):
    """(delta, se, cfg, folds, seeds) of an arm: one change from the pipeline's defaults."""
    from src.pipeline import GRNInferencePipeline, _complementary_pairs, _sem_config
    p = GRNInferencePipeline()
    assert (p.n_resamples, p.epochs, p.warmup_epochs, p.l1_weight, p.w_dropout, p.models_per_chunk, p.readout, p.symmetrize) == \
        (60, 250, 50, .005, 0.0, 120, "direct", False), "not the shipped defaults"
    cfg = _sem_config(p.epochs, p.warmup_epochs, p.l1_weight, p.w_dropout)
    jobs = _complementary_pairs(u.shape[0], p.n_resamples)
    folds, seeds = [j[0] for j in jobs], [j[1] for j in jobs]
    if arm == "no_se":          # inverse-SE weights on ones are exactly uniform (train_paper.py "uniform")
        se = np.ones_like(se)
    elif arm == "no_stability":
        cfg = dataclasses.replace(cfg, stability_weight=0.0)
    elif arm in ("no_push", "no_push_signed"):   # B stays at its start: lr 0 (the scheduler's eta_min moves it <= 1e-6 per step; drift is measured)
        cfg = dataclasses.replace(cfg, lr=0.0)
    elif arm == "no_half":      # every condition in every model, same seeds
        folds = [np.arange(u.shape[0])] * len(jobs)
    elif arm == "wald":
        delta, se = wald_targets(u, delta, se)
    else:
        assert arm == "shipped", arm
    return delta, se, cfg, folds, seeds


def b_start(u, delta, folds, signed=False):
    """The start of B per model (M, n), and the mask of in-half genes: initialize_direct_from_deltas, or signed_init."""
    B = np.ones((len(folds), u.shape[1]), np.float32)
    inh = np.zeros_like(B, bool)
    for m, f in enumerate(folds):
        for j in range(u.shape[1]):
            mask = u[f, j] > .5
            if mask.any():
                d = np.mean(delta[f][mask, j]) if signed else np.mean(np.abs(delta[f][mask, j]))
                B[m, j] = (np.sign(d) if signed else 1.) * np.clip(np.abs(d), .01, 100.)
                inh[m, j] = True
    return B, inh


def fit_arm(arm, u, delta, se, device="cuda:0"):
    """Returns S (the shipped readout: mean |W*B|, diagonal 0) and a diagnostics dict."""
    import torch
    import src.models.interventional_nb_sem as isem
    from src.pipeline import _mean_ensemble
    fit_delta, fit_se, cfg, folds, seeds = arm_inputs(arm, u, delta, se)
    signed = arm == "no_push_signed"
    init = isem.initialize_direct_from_deltas
    if signed:
        isem.initialize_direct_from_deltas = signed_init      # train_batched's module-global lookup; restored below
    try:
        for attempt in range(OOM_MAX):
            try:
                _, loss, W, B, diag = isem.train_batched(u, fit_delta, fit_se, folds=folds, seeds=seeds, config=cfg, device=device,
                                                         return_params=True, return_diagnostics=True)
                break
            except RuntimeError as e:     # torch.cuda.OutOfMemoryError is a RuntimeError; never CPU, never a smaller chunk
                if not any(x in str(e).lower() for x in GPU_FULL) or attempt == OOM_MAX - 1:
                    raise
                torch.cuda.empty_cache()
                time.sleep(OOM_WAIT)
    finally:
        isem.initialize_direct_from_deltas = init
    S = _mean_ensemble(np.abs(W * B[:, None, :]))
    B0, inh = b_start(u, fit_delta, folds, signed)
    dB = np.abs(B - B0)
    own = np.stack([np.where(inh[m], np.sign(np.array([np.mean(fit_delta[f][u[f, j] > .5, j]) if inh[m, j] else 0. for j in range(u.shape[1])])), 0.)
                    for m, f in enumerate(folds)]) if arm in ("no_push", "no_push_signed") else None
    rho, fb = np.asarray(diag["rho_W"], float), diag["fallbacks"]
    return S, {"arm": arm, "n_models": len(folds), "oom_retries": attempt, "loss_mean": float(np.mean(loss)),
               "models_nonfinite_W": int((~np.isfinite(W).all(axis=(1, 2))).sum()),
               "models_nonfinite_B": int((~np.isfinite(B).all(axis=1)).sum()), "S_nonfinite": int((~np.isfinite(S)).sum()),
               "rho_gt1": int(np.nansum(rho > 1)), "rho_nan": int(np.isnan(rho).sum()), "rho_max": float(np.nanmax(rho)) if np.isfinite(rho).any() else None,
               "rho_median": float(np.nanmedian(rho)) if np.isfinite(rho).any() else None,
               "sigma_abs_max": float(np.nanmax(diag["sigma_abs"])) if np.isfinite(diag["sigma_abs"]).any() else None,
               "fallback_records": len(fb), "fallback_models": len({f["model"] for f in fb}),
               "pivoted_models": len({f["model"] for f in fb if f["pivoted"]}),
               "B_drift_max": float(np.nanmax(dB)), "B_drift_rel_max": float(np.nanmax(dB / np.abs(B0))), "B_drift_median": float(np.nanmedian(dB)),
               "B_sign_mismatch_in_half": None if own is None else int((np.sign(B)[inh] != own[inh]).sum()),
               "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}


def save(f, S, meta):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, S=np.asarray(S, np.float32), meta=np.asarray(json.dumps(meta, default=float)))
    tmp.replace(f)
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))


def real_targets(ds):
    """u, delta, se, genes of a dataset (run_real.load + build_intervention_targets), kept in this folder after the first build."""
    f = HERE / "cache/real" / ds / "targets.npz"
    if not f.exists():
        import run_real
        from src.models.interventional_nb_sem import build_intervention_targets
        X, iv, genes = run_real.load(ds)
        u, delta, _, _, _, se = build_intervention_targets(X, list(iv), list(genes))
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, u=u, delta=delta, se=se, genes=np.asarray([str(g) for g in genes]))
        tmp.replace(f)
    with np.load(f) as z:
        return z["u"], z["delta"], z["se"], [str(g) for g in z["genes"]]


def job(kind, name, arm):
    """kind sims: name = simulated dataset dir; kind real: name = real dataset. Resumable; a failure is recorded, never dropped."""
    out = HERE / "cache" / kind / (Path(name).relative_to(e9.DATA / "inspre") if kind == "sims" else name) / f"{arm}.npz"
    if out.exists():
        return f"{kind} {name} {arm} cached", 0.
    claim = out.with_suffix(".claim")           # several processes may run one arm: exactly one fits each dataset
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)      # owner, so a1.py can tell a dead claim from a live one
    except FileExistsError:
        return f"{kind} {name} {arm} claimed elsewhere", 0.
    try:
        if out.exists():
            return f"{kind} {name} {arm} cached", 0.
        return _job(kind, name, arm, out)
    finally:
        claim.unlink(missing_ok=True)


def _job(kind, name, arm, out):
    t0 = time.time()
    try:
        if kind == "sims":
            from src.models.interventional_nb_sem import build_intervention_targets
            _, Y, labels, genes, _ = e9.load(Path(name))
            u, delta, _, _, _, se = build_intervention_targets(Y, list(labels), list(genes))
            ref = SIM / "cache" / Path(name).relative_to(e9.DATA) / "isem.npz"
        else:
            u, delta, se, genes = real_targets(name)
            ref = SHIPPED_REAL / name / "isem_c.npz"
        S, meta = fit_arm(arm, u, delta, se)
    except Exception as e:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".err").write_text(repr(e))
        return f"FAIL {kind} {name} {arm}: {repr(e)[-300:]}", time.time() - t0
    out.with_suffix(".err").unlink(missing_ok=True)
    meta["wall_seconds"] = time.time() - t0
    if arm == "shipped":
        with np.load(ref) as z:
            meta["control_max_abs_diff"] = float(np.abs(np.asarray(S, np.float32) - z["S"]).max())
    save(out, S, meta)
    return f"{kind} {Path(name).parent.name if kind == 'sims' else ''}/{Path(name).name} {arm} {meta['wall_seconds']:.0f}s " \
           f"ctrl={meta.get('control_max_abs_diff', '-')} rho>1={meta['rho_gt1']} piv={meta['pivoted_models']} " \
           f"dB={meta['B_drift_max']:.2e} nf={meta['S_nonfinite']}", meta["wall_seconds"]


def gpu_watch(ex, tag, stop):
    """Every 15 s: all compute-app memory on the GPU, and this run's share. Logged, never acted on."""
    log = HERE / "logs" / f"gpu_{tag}.csv"
    with log.open("a") as fh:
        while not stop.wait(15):
            mine = {int(p) for p in (ex._processes or {})}
            try:
                r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=30).stdout
                rows = [tuple(map(int, x.split(","))) for x in r.strip().splitlines()]
            except Exception:
                continue
            fh.write(f"{time.time():.0f},{sum(m for _, m in rows)},{sum(m for p, m in rows if p in mine)},{len(mine)}\n")
            fh.flush()


def fit(a):
    arms = a.arms
    names = [str(d) for d in block(a.block)] if a.kind == "sims" else (a.ds or list(SCREENS))
    todo = []
    for arm in arms:
        for n in names:
            rel = Path(n).relative_to(e9.DATA / "inspre") if a.kind == "sims" else n
            if not (HERE / "cache" / a.kind / rel / f"{arm}.npz").exists():
                todo.append((a.kind, n, arm))
    if a.spread and a.kind == "sims":       # one rep per setting first: a sample across the block
        todo.sort(key=lambda t: (Path(t[1]).name, t[2]))
    if a.kind == "real":                   # small datasets first, control first within a dataset
        todo.sort(key=lambda t: (SCREENS.index(t[1]), ARMS.index(t[2])))
    todo = todo[:a.limit]
    tag = f"{a.kind}_{a.block}_{a.workers}w_{int(time.time())}"
    t0 = time.time(); done = 0; fails = 0
    stop = threading.Event()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        threading.Thread(target=gpu_watch, args=(ex, tag, stop), daemon=True).start()
        for f in as_completed([ex.submit(job, *t) for t in todo]):
            msg, _ = f.result()
            done += 1; fails += msg.startswith("FAIL")
            print(f"[{done}/{len(todo)} {time.time() - t0:.0f}s] {msg}", flush=True)
    stop.set()
    el = time.time() - t0
    print(f"DONE {tag}: {done} fits, {fails} failed, {el:.0f}s, {done / max(el, 1e-9) * 60:.1f} fits/min", flush=True)


def plugin_S(u, delta):
    """e2_planted's plug-in on the genes with exactly one knockdown; other genes stay zero. No regularisation.
    Returns S (n x n), cond(T), genes kept."""
    tgt = u.argmax(1)
    assert (u.sum(1) == 1).all(), "a condition is not one panel gene"
    assert len(set(tgt.tolist())) == len(tgt), "a gene has two conditions"
    keep = np.sort(tgt)
    n = u.shape[1]
    T = np.zeros((n, n)); T[:, tgt] = delta.T
    T = T[np.ix_(keep, keep)]
    cond = float(np.linalg.cond(T))
    Ti = np.linalg.inv(T)
    sub = np.eye(len(keep)) - (1 / np.diag(Ti))[:, None] * Ti
    S = np.zeros((n, n)); S[np.ix_(keep, keep)] = sub
    return S, cond, len(keep)


def plugin(_a):
    """CPU: both blocks' simulated datasets and all real datasets."""
    from src.models.interventional_nb_sem import build_intervention_targets
    for d in block("weak") + block("contrast"):
        out = HERE / "cache/sims" / d.relative_to(e9.DATA / "inspre") / "plugin.npz"
        if out.exists():
            continue
        _, Y, labels, genes, _ = e9.load(d)
        u, delta, _, _, _, _ = build_intervention_targets(Y, list(labels), list(genes))
        S, cond, K = plugin_S(u, delta)
        save(out, S, {"arm": "plugin", "cond_T": cond, "K": K, "n": len(genes), "S_nonfinite": int((~np.isfinite(S)).sum())})
    for ds in SCREENS:
        out = HERE / "cache/real" / ds / "plugin.npz"
        if out.exists():
            continue
        u, delta, _, genes = real_targets(ds)
        t0 = time.time()
        try:
            S, cond, K = plugin_S(u, delta)
        except Exception as e:      # singular T: reported, no regularisation
            out.with_suffix(".err").write_text(repr(e)); print(f"plugin {ds} FAILED {e!r}", flush=True); continue
        save(out, S, {"arm": "plugin", "cond_T": cond, "K": K, "n": len(genes), "S_nonfinite": int((~np.isfinite(S)).sum()), "wall_seconds": time.time() - t0})
        print(f"plugin {ds}: K={K}/{len(genes)} cond={cond:.3g} nonfinite={int((~np.isfinite(S)).sum())}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "plugin", "one"])
    ap.add_argument("--name")
    ap.add_argument("--kind", choices=["sims", "real"], default="sims")
    ap.add_argument("--block", choices=["weak", "contrast"], default="weak")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--ds", nargs="+", choices=SCREENS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--spread", action="store_true")
    a = ap.parse_args()
    if a.cmd == "one":      # one fit in its own process, under a claim a1.py holds; an abort here costs only this fit
        out = HERE / "cache" / a.kind / (Path(a.name).relative_to(e9.DATA / "inspre") if a.kind == "sims" else a.name) / f"{a.arms[0]}.npz"
        print(_job(a.kind, a.name, a.arms[0], out)[0], flush=True)
    else:
        {"fit": fit, "plugin": plugin}[a.cmd](a)
