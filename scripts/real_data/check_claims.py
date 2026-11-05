"""Check the four claims of the "positive biology result" draft (CLAIM_CHECK.md) against every method's saved
real-data output. Reads only (fits, baselines, papers' tables); writes only into this folder.

Run (no writes outside this folder: -B, TMPDIR here, run_env.sh not used because it writes caches under scripts/):
  TMPDIR=scripts/real_data/tmp .envs/isem/bin/python -I -B scripts/real_data/check_claims.py \
      > scripts/real_data/check_claims.log 2>&1
Self-check of the AUROC code:  ... check_claims.py --selftest
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CC = ROOT / "scripts/celltype_comparison"
sys.path.insert(0, str(CC))
import run_tests as rt                                     # read-only reuse: loaders, sign_test, gene lists
FITS = HERE / "fits"                                # shipped-solver (option A) I-SEM fits, build_fits.py
rt.HERE = HERE                                      # rt.load reads rt.HERE / "fits" / f"{ds}.npz"

PAPER = ROOT / "scripts/results/paper"
INSPRE_THR = 0.015                                         # authors' GWPS |G| zeroing (scripts/config_tests/score.py)
RP = ("RPL", "RPS", "MRPL", "MRPS")
# coefficient-type outputs (effect per unit regulator change): a knockdown that LOWERS a target = positive entry
COEF = {"I-SEM W", "inspre official", "inspre default", "ADAPRE", "LiNGAM", "RICE"}
RESP = {"raw z": "Z", "mean difference": "D"}               # knockdown responses: lowers = negative entry
OUT = HERE / "out"


# ---------------------------------------------------------------- loading
def path_info(ds, genes):
    """Per lambda-path method: matrices along the path (panel-embedded) and per-gene status."""
    info = {}
    z = np.load(PAPER / f"e1b_inspre_official/{ds}_Gpath.npz")
    info["inspre official"] = (z["W_path"].copy(), [str(g) for g in z["kept_genes"]], True)
    for m, lab in [("inspre_default", "inspre default"), ("adapre", "ADAPRE")]:
        z = np.load(PAPER / f"e8_adapre/{ds}_{m}_Gpath.npz")
        meta = json.loads((PAPER / f"e8_adapre/{ds}_{m}.json").read_text())
        info[lab] = (z["W_path"].copy(), meta["kept_genes"], False)
    out = {}
    pos = {g: i for i, g in enumerate(genes)}
    for lab, (P, kept, thr) in info.items():
        P = np.where(np.isfinite(P), P, 0.0)                # non-finite -> 0, as e8.embed (nan_to_num would make inf huge)
        if thr:
            P[np.abs(P) < INSPRE_THR] = 0.0
        k = np.array([pos[g] for g in kept])
        mats = []
        for l in range(P.shape[2]):
            A = np.zeros((len(genes), len(genes))); A[np.ix_(k, k)] = P[:, :, l]; np.fill_diagonal(A, 0); mats.append(A)
        out[lab] = dict(mats=mats, kept=set(kept))
    return out


def methods(ds, d):
    """name -> W-oriented matrix (diag 0). d = run_tests.load() line."""
    M = {"I-SEM S": d["M"]["S"], "I-SEM |W|": d["M"]["Wabs"], "I-SEM W": d["M"]["W"],
         "raw z": d["M"]["Z"], "mean difference": d["M"]["D"],
         "inspre official": d["M"]["inspre"], "ADAPRE": d["M"]["adapre"],
         "inspre default": np.load(PAPER / f"e8_adapre/{ds}_inspre_default.npy")}
    for m, lab in [("lingam", "LiNGAM"), ("igsp", "IGSP"), ("rice", "RICE"), ("notears", "NOTEARS"), ("golem", "GOLEM")]:
        M[lab] = np.load(ROOT / f"results/{m}_adj_{ds}.npy")
    M = {k: np.nan_to_num(np.array(v, float)) for k, v in M.items()}
    for v in M.values():
        np.fill_diagonal(v, 0.0)
    return M


def activation(name, A):
    """Signed so that larger = knockdown lowers the target (regulator activates it); magnitude-only methods as |A|."""
    if name in COEF:
        return A
    if name in RESP:
        return -A
    return np.abs(A)


# ---------------------------------------------------------------- AUROC helpers
def auroc(score, lab):
    lab = np.asarray(lab, bool); n1 = lab.sum(); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def col_auroc(A, P, rows=None):
    """Per-column AUROC of A[:, j] for positives P[:, j], diagonal excluded, ties averaged; NaN where undefined.
    rows: optional boolean row mask (stratum). Returns (auc per column, n1*n0 weight per column)."""
    A = np.array(A, float); P = np.array(P, bool); n = A.shape[0]
    keep = ~np.eye(n, dtype=bool) if rows is None else (~np.eye(n, dtype=bool)) & rows[:, None]
    A[~keep] = -np.inf; P = P & keep
    R = rankdata(A, axis=0)                                 # excluded entries tie at the bottom
    nx = (~keep).sum(0)
    n1 = P.sum(0); n0 = keep.sum(0) - n1
    s = np.where(P, R - nx, 0).sum(0)                       # ranks among kept entries (excluded are strictly lowest)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (s - n1 * (n1 + 1) / 2) / (n1 * n0)
    w = (n1 * n0).astype(float)
    a[(n1 == 0) | (n0 == 0)] = np.nan; w[(n1 == 0) | (n0 == 0)] = 0
    return a, w


def strat_reg_expr(A, P, expr_bin):
    """Within-regulator AUROC stratified by target-expression decile (removes regulator scale and target expression)."""
    num = den = 0.0
    for b in np.unique(expr_bin):
        a, w = col_auroc(A, P, rows=expr_bin == b)
        ok = w > 0; num += (a[ok] * w[ok]).sum(); den += w[ok].sum()
    return num / den


def boot_ci(vals, rng, B=2000):
    v = np.asarray(vals)[np.isfinite(vals)]
    return np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)], [2.5, 97.5]).tolist()


# ---------------------------------------------------------------- claim A: MYC -> Muhar targets (Jurkat)
def claim_myc(L, rng):
    d = L["jurkat"]; g = d["genes"]; ix = d["idx"]; n = len(g)
    ens = [str(x) for x in np.load(FITS / "jurkat.npz", allow_pickle=True)["genes"]]
    M = methods("jurkat", d); paths = path_info("jurkat", ens)
    j = ix["MYC"]; pos = np.array([ix[x] for x in rt.MUHAR_JURKAT]); deep = np.nonzero(-d["relkd"] >= rt.KD_FLOOR)[0]
    others = np.setdiff1d(np.arange(n), [j]); y = np.isin(others, pos)
    resp = d["M"]["Z"][:, j] < -3                           # genes that fall after MYC knockdown
    rows = []

    def summarise(name, A):
        S = activation(name, A)
        col = S[:, j]
        zero_col = not np.any(A[:, j])
        a_all = auroc(col[others], y)
        a_resp = auroc(col[others][resp[others]], y[resp[others]])
        bs = []
        p_i, n_i = np.nonzero(y)[0], np.nonzero(~y)[0]
        for _ in range(2000):
            sel = np.r_[rng.choice(p_i, len(p_i)), rng.choice(n_i, len(n_i))]
            bs.append(auroc(col[others][sel], y[sel]))
        v = np.array([np.mean([S[t, r] for t in pos if t != r]) for r in deep])
        vm = v[list(deep).index(j)]
        return dict(method=name, myc_column_all_zero=zero_col, auroc_all=a_all, auroc_all_ci95=np.percentile(bs, [2.5, 97.5]).tolist(),
                    auroc_among_responders=a_resp, n_responders=int(resp[others].sum()),
                    myc_rank_best=int((v > vm).sum()) + 1, myc_rank_worst=int((v >= vm).sum()), n_regulators=len(deep))

    for name, A in M.items():
        rows.append(summarise(name, A))
    myc_ens = ens[j]
    for lab, p in paths.items():
        best = max((summarise(lab, A) for A in p["mats"]), key=lambda r: np.nan_to_num(r["auroc_all"]))
        nz = [bool(np.any(A[:, j])) for A in p["mats"]]
        rows.append(dict(best, method=f"{lab} (best lambda on path)", myc_kept_by_filter=myc_ens in p["kept"],
                         myc_column_nonzero_at_n_of_lambdas=f"{sum(nz)}/{len(nz)}"))
    # paired bootstrap: I-SEM W minus raw z / mean difference / best baseline, same resampled genes
    S_w = activation("I-SEM W", M["I-SEM W"])[:, j][others]
    diffs = {}
    for ref in ["raw z", "mean difference", "inspre default", "RICE", "LiNGAM"]:
        S_r = activation(ref, M[ref])[:, j][others]
        p_i, n_i = np.nonzero(y)[0], np.nonzero(~y)[0]; bs = []
        for _ in range(2000):
            sel = np.r_[rng.choice(p_i, len(p_i)), rng.choice(n_i, len(n_i))]
            bs.append(auroc(S_w[sel], y[sel]) - auroc(S_r[sel], y[sel]))
        diffs[f"I-SEM W - {ref}"] = dict(diff=auroc(S_w, y) - auroc(S_r, y), ci95=np.percentile(bs, [2.5, 97.5]).tolist())
    return dict(rows=rows, paired=diffs, myc_relkd=float(-d["relkd"][j]))


# ---------------------------------------------------------------- claim B: P3 ER entry -> UPR genes
def claim_p3(L, rng):
    out = {}
    sp = rt.SPEC["P3_UPR_ER_entry"]
    for ds in sp["lines"]:
        d = dict(L[ds]); d["M"] = dict(d["M"])
        ens = [str(x) for x in np.load(FITS / f"{ds}.npz", allow_pickle=True)["genes"]]
        M = methods(ds, d); paths = path_info(ds, ens)
        keys = []
        for name, A in M.items():
            if name in ("I-SEM S", "I-SEM |W|", "IGSP", "NOTEARS", "GOLEM"):
                continue                                     # unsigned or empty: no sign test
            k = f"m:{name}"; d["M"][k] = A; keys.append(k)
        rt.SIGN_KEYS = keys
        rt.COEF_KEYS = {f"m:{x}" for x in COEF}
        regs, reads = sp["regs"][ds], sp["reads"][ds]
        res = rt.sign_test(d, regs, reads, sp["response"], (), rng, sp["pool"]())
        used = res["regs"]
        status = {}
        for name, A in M.items():
            zero = [r for r in used if not np.any(A[:, d["idx"][r]])]
            row = dict(n_regs_all_zero_column=len(zero), all_zero=zero)
            for lab, p in paths.items():
                if name == lab:
                    row["not_kept_by_filter"] = [r for r in zero if ens[d["idx"][r]] not in p["kept"]]
                    row["kept_but_zero_on_whole_path"] = [r for r in zero if ens[d["idx"][r]] in p["kept"] and
                                                          not any(np.any(B[:, d["idx"][r]]) for B in p["mats"])]
            status[name] = row
        out[ds] = dict(regs_used=used, reads=res["reads"], n_pool=res["n_pool"],
                       scores={k[2:]: v for k, v in res["scores"].items()}, zero_columns=status)
        print("P3 done", ds, flush=True)
    return out


# ---------------------------------------------------------------- claim C: CORUM co-complex recovery
def corum_sets():
    cor = pd.read_csv(ROOT / "data/corum_human_v5.3.txt", sep="\t", usecols=["subunits_gene_name"])
    return [set(s.split(";")) for s in cor.subunits_gene_name.dropna()]


def claim_corum(L, rng):
    cplx = corum_sets(); out = {}
    for ds, d in L.items():
        g = d["genes"]; ens = [str(x) for x in np.load(FITS / f"{ds}.npz", allow_pickle=True)["genes"]]
        M = methods(ds, d); paths = path_info(ds, ens)
        res = {}
        for tag, keep in [("all genes", np.ones(len(g), bool)),
                          ("ribosomal proteins removed", np.array([not x.startswith(RP) for x in g]))]:
            k = np.nonzero(keep)[0]; ix = {g[i]: a for a, i in enumerate(k)}; n = len(k)
            P = np.zeros((n, n), bool)
            for c in cplx:
                m = [ix[s] for s in c if s in ix]; P[np.ix_(m, m)] = True
            np.fill_diagonal(P, False)
            ex = np.log(d["cmean"][k]); exb = np.digitize(ex, np.quantile(ex, np.linspace(0, 1, 11)[1:-1]))
            off = ~np.eye(n, dtype=bool)
            per_reg, rows = {}, []

            def score(name, A):
                A = np.abs(A[np.ix_(k, k)])
                a_col, _ = col_auroc(A, P)
                per_reg[name] = a_col
                return dict(method=name, global_auroc=auroc(A[off], P[off]), per_regulator_auroc=float(np.nanmean(a_col)),
                            per_regulator_ci95=boot_ci(a_col, rng), reg_x_target_expr_auroc=strat_reg_expr(A, P, exb),
                            p_at_500=float(P[off][np.argsort(-A[off], kind="stable")[:500]].mean()))

            for name, A in M.items():
                rows.append(score(name, A))
            for lab, p in paths.items():
                cand = [score(f"{lab} @lambda{l}", A) for l, A in enumerate(p["mats"])]
                best = max(cand, key=lambda r: r["reg_x_target_expr_auroc"])
                rows.append(dict(best, method=f"{lab} (best lambda on path, by stratified AUROC)"))
                for c in cand:
                    per_reg.pop(c["method"], None)
            # paired regulator bootstrap: I-SEM S minus each other method (per-regulator AUROC)
            paired = {}
            for name in per_reg:
                if name == "I-SEM S":
                    continue
                dlt = per_reg["I-SEM S"] - per_reg[name]
                paired[name] = dict(diff=float(np.nanmean(dlt)), ci95=boot_ci(dlt, rng))
            # profile-similarity reference (unordered pairs of regulator columns)
            iu = np.triu_indices(n, 1)
            prof = {}
            for name in ["raw z", "mean difference", "I-SEM S", "I-SEM W"]:
                X = M[name][np.ix_(k, k)] if name in ("raw z", "mean difference", "I-SEM W") else np.abs(M[name][np.ix_(k, k)])
                X = X - X.mean(0); X = X / (np.linalg.norm(X, axis=0) + 1e-300)
                prof[name] = auroc(np.nan_to_num((X.T @ X)[iu]), P[iu])
            res[tag] = dict(n_genes=n, n_cocomplex_pairs=int(P.sum()), base_rate=float(P[off].mean()), rows=rows,
                            paired_vs_isem_S=paired, profile_correlation_auroc=prof)
        out[ds] = res
        print("CORUM done", ds, flush=True)
    return out


# ---------------------------------------------------------------- claim D: conserved hubs (from the run, read only)
def claim_hubs():
    q = json.loads((CC / "results/q2_conserved_hubs.json").read_text())["tier1"]
    return {k: dict(genes=v["genes"], stability=v.get("stability"), jaccard_with_null=v.get("jaccard_with_null"))
            for k, v in q.items()}


def selftest():
    rng = np.random.default_rng(0)
    n = 60; A = rng.normal(size=(n, n)); P = rng.random((n, n)) < .2; np.fill_diagonal(P, False)
    A[P] += 1.0
    a, _ = col_auroc(A, P)
    for j in range(5):
        m = np.arange(n) != j
        assert abs(a[j] - auroc(A[m, j], P[m, j])) < 1e-12
    assert auroc(np.r_[np.ones(5), np.zeros(5)], np.r_[np.ones(5), np.zeros(5)]) == 1.0
    assert auroc(np.zeros(10), np.r_[np.ones(5), np.zeros(5)]) == 0.5       # all-zero column -> chance
    print("selftest ok")


def main():
    rng = np.random.default_rng(0)
    OUT.mkdir(exist_ok=True)
    sym = rt._symbols()
    L = {ds: rt.load(ds, sym) for ds in rt.ALL4}
    dump = lambda name, obj: (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
    dump("A_myc.json", claim_myc(L, rng)); print("A done", flush=True)
    dump("B_p3.json", claim_p3(L, rng)); print("B done", flush=True)
    dump("D_hubs.json", claim_hubs())
    dump("C_corum.json", claim_corum(L, rng)); print("C done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else main()
