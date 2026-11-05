"""CORUM co-complex recovery: significance and figure.

Statistic: within-regulator AUROC stratified by target-expression decile, pooled over regulators with weights
n1*n0 (check_claims.strat_reg_expr). Uncertainty: paired bootstrap over regulator columns (2,000 draws, the same
draws for every method); p = two-sided normal approximation from the bootstrap SE, Holm over all comparisons. I-SEM S (the shipped readout) is the pre-stated primary I-SEM score.
Baselines appear at the authors' selection rules; ADAPRE also at the oracle lambda (best stratified AUROC on its path,
chosen on the full data with the CORUM labels; an upper bound for ADAPRE, not a selection rule).

Run (writes only into this folder):
  MPLCONFIGDIR=scripts/real_data/tmp TMPDIR=scripts/real_data/tmp \
    .envs/isem/bin/python -I -B scripts/real_data/corum_sig.py > scripts/real_data/corum_sig.log 2>&1
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_claims as cc                                  # same folder: loaders, col_auroc, CORUM sets
rt = cc.rt

B = 2000
OUT = HERE / "out"
FIG = HERE / "figures"
ORDER = ["I-SEM S", "I-SEM |W|", "I-SEM W", "raw z", "mean difference", "inspre official (D-bar)",
         "inspre official (min CV)", "inspre default (D-bar)", "inspre default (min CV)", "ADAPRE (D-bar)",
         "ADAPRE (min CV)", "ADAPRE (oracle lambda)", "RICE", "IGSP", "LiNGAM", "NOTEARS", "GOLEM"]


def which_min(x):
    x = np.array([np.nan if (v is None or v == "NaN") else float(v) for v in x])
    return int(np.nanargmin(x))


def selected(ds, d, M, paths):
    """name -> matrix at each authors' selection rule (and the ADAPRE oracle is added later)."""
    out = {k: M[k] for k in ["I-SEM S", "I-SEM |W|", "I-SEM W", "raw z", "mean difference", "RICE", "IGSP",
                             "LiNGAM", "NOTEARS", "GOLEM"]}
    out["inspre official (D-bar)"] = M["inspre official"]
    z = np.load(cc.PAPER / f"e1b_inspre_official/{ds}_Gpath.npz")
    out["inspre official (min CV)"] = paths["inspre official"]["mats"][which_min(z["eps_hat_G"])]
    for m, lab in [("inspre_default", "inspre default"), ("adapre", "ADAPRE")]:
        sel = json.loads((cc.PAPER / f"e8_adapre/{ds}_{m}.json").read_text())["selection"]
        out[f"{lab} (D-bar)"] = paths[lab]["mats"][sel["dbar"]]
        out[f"{lab} (min CV)"] = paths[lab]["mats"][sel["min_eps_g"]]
    assert np.array_equal(out["ADAPRE (D-bar)"], M["ADAPRE"]), "stored ADAPRE .npy is not the D-bar path matrix"
    return out


def num_den(A, P, exb):
    """Per-regulator numerator / denominator of the stratified AUROC (sum over target-expression strata)."""
    num = np.zeros(A.shape[1]); den = np.zeros(A.shape[1])
    for b in np.unique(exb):
        a, w = cc.col_auroc(A, P, rows=exb == b)
        ok = w > 0; num[ok] += a[ok] * w[ok]; den[ok] += w[ok]
    return num, den


def analyse(L, rng):
    cplx = cc.corum_sets(); res = {}
    for ds, d in L.items():
        g = d["genes"]; ens = [str(x) for x in np.load(cc.FITS / f"{ds}.npz", allow_pickle=True)["genes"]]
        M = cc.methods(ds, d); paths = cc.path_info(ds, ens); S = selected(ds, d, M, paths)
        res[ds] = {}
        for tag, keep in [("all", np.ones(len(g), bool)), ("noRP", np.array([not x.startswith(cc.RP) for x in g]))]:
            k = np.nonzero(keep)[0]; ix = {g[i]: a for a, i in enumerate(k)}; n = len(k)
            P = np.zeros((n, n), bool)
            for c in cplx:
                m = [ix[s] for s in c if s in ix]; P[np.ix_(m, m)] = True
            np.fill_diagonal(P, False)
            ex = np.log(d["cmean"][k]); exb = np.digitize(ex, np.quantile(ex, np.linspace(0, 1, 11)[1:-1]))
            ND = {name: num_den(np.abs(A[np.ix_(k, k)]), P, exb) for name, A in S.items()}
            orc = [num_den(np.abs(A[np.ix_(k, k)]), P, exb) for A in paths["ADAPRE"]["mats"]]
            li = int(np.argmax([nu.sum() / de.sum() for nu, de in orc]))
            ND["ADAPRE (oracle lambda)"] = orc[li]
            den = ND["I-SEM S"][1]
            assert all(np.allclose(v[1], den) for v in ND.values())   # same strata weights for every method
            cols = np.nonzero(den > 0)[0]
            draws = [cols[rng.integers(0, len(cols), len(cols))] for _ in range(B)]
            stat = lambda nd, idx: nd[0][idx].sum() / nd[1][idx].sum()
            boot = {name: np.array([stat(nd, i) for i in draws]) for name, nd in ND.items()}
            rows = {}
            for name in ORDER:
                est = stat(ND[name], cols); bs = boot[name]
                diff = boot["I-SEM S"] - bs; dhat = stat(ND["I-SEM S"], cols) - est
                # two-sided p from the bootstrap SE (normal approximation): a bootstrap-count p cannot go below 1/B,
                # which Holm over ~112 comparisons would turn into a floor of 0.056
                p = float(2 * norm.sf(abs(dhat) / diff.std())) if name != "I-SEM S" and diff.std() > 0 else np.nan
                rows[name] = dict(auroc=est, ci95=np.percentile(bs, [2.5, 97.5]).tolist(), diff_isemS_minus=dhat,
                                  diff_ci95=np.percentile(diff, [2.5, 97.5]).tolist(), p_boot_se=p)
            res[ds][tag] = dict(n_genes=n, n_regulators_with_partners=len(cols), adapre_oracle_lambda_index=li, rows=rows)
            print("done", ds, tag, flush=True)
    # Holm over every I-SEM-S-vs-baseline comparison (all lines, both gene sets)
    flat = [(ds, tag, m, r["p_boot_se"]) for ds in res for tag in res[ds] for m, r in res[ds][tag]["rows"].items()
            if m not in ("I-SEM S", "I-SEM |W|", "I-SEM W")]
    order = sorted(range(len(flat)), key=lambda i: flat[i][3]); run = 0.0
    for rank, i in enumerate(order):
        ds, tag, m, p = flat[i]; run = max(run, min(1.0, (len(flat) - rank) * p))
        res[ds][tag]["rows"][m]["p_holm"] = run
    return res


def figure(res, tag, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ACC, CTX, INK, GRID = "#2a78d6", "#8a8984", "#0b0b0b", "#e4e3df"
    lines = ["k562", "rpe1", "jurkat", "hepg2"]; names = {"k562": "K562", "rpe1": "RPE1", "jurkat": "Jurkat", "hepg2": "HepG2"}
    meth = [m for m in ORDER if m not in ("NOTEARS", "GOLEM")]          # all-zero outputs: AUROC exactly 0.5 (caption)
    comp = [m for m in meth if not m.startswith("I-SEM")]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": CTX, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(2, 4, figsize=(11, 7.2), sharey="row", sharex="row",
                           gridspec_kw={"height_ratios": [len(meth), len(comp)], "hspace": 0.25, "wspace": 0.08})
    for c, ds in enumerate(lines):
        R = res[ds][tag]["rows"]
        a = ax[0, c]
        for y, m in enumerate(meth):
            col = ACC if m.startswith("I-SEM") else CTX
            lo, hi = R[m]["ci95"]
            a.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
            a.plot(R[m]["auroc"], y, "o", ms=6, color=col, mec="#fcfcfb", mew=1.2)
        a.axvline(0.5, color=CTX, lw=1, ls=(0, (3, 3)))
        a.set_title(names[ds], fontsize=9, color=INK)
        b = ax[1, c]
        for y, m in enumerate(comp):
            lo, hi = R[m]["diff_ci95"]
            b.plot([lo, hi], [y, y], color=ACC, lw=2, solid_capstyle="round")
            sig = R[m]["p_holm"] < 0.05                    # open marker = not significant (caption)
            b.plot(R[m]["diff_isemS_minus"], y, "o", ms=6, color=ACC, mew=1.5,
                   mec="#fcfcfb" if sig else ACC, mfc=ACC if sig else "#fcfcfb")
        b.axvline(0.0, color=CTX, lw=1, ls=(0, (3, 3)))
        for z in (a, b):
            z.grid(axis="x", color=GRID, lw=0.6); z.set_axisbelow(True)
            for s in ("top", "right"):
                z.spines[s].set_visible(False)
            z.tick_params(length=2)
    from matplotlib.ticker import MaxNLocator
    for a in ax.flat:
        a.xaxis.set_major_locator(MaxNLocator(4))
    ax[0, 0].set_yticks(range(len(meth))); ax[0, 0].set_yticklabels(meth); ax[0, 0].invert_yaxis()
    ax[1, 0].set_yticks(range(len(comp))); ax[1, 0].set_yticklabels(comp); ax[1, 0].invert_yaxis()
    for c in range(4):
        ax[0, c].set_xlabel("Stratified AUROC"); ax[1, c].set_xlabel("ΔAUROC (I-SEM S − method)")
    for t, a in zip("ab", (ax[0, 0], ax[1, 0])):
        a.text(-0.62, 1.0, t, transform=a.transAxes, fontsize=11, fontweight="bold", va="bottom")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    rng = np.random.default_rng(0)
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    L = {ds: rt.load(ds, rt._symbols()) for ds in rt.ALL4}
    res = analyse(L, rng)
    (OUT / "corum_sig.json").write_text(json.dumps(res, indent=1, default=float))
    figure(res, "all", FIG / "fig_corum_cocomplex")
    figure(res, "noRP", FIG / "fig_corum_cocomplex_noRP")


if __name__ == "__main__":
    main()
