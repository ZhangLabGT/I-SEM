"""MYC -> Muhar 2018 targets (Jurkat): I-SEM against the published network methods, significance and figure.

Statistic: AUROC of MYC's column (activation-signed, check_claims.activation) for the 20 Muhar 2018 Table S3 targets
against all other genes. Baselines at the authors' selection rules (corum_sig.selected); ADAPRE also at the oracle
lambda (best AUROC on its path, chosen with these labels: an upper bound). Uncertainty: paired bootstrap over genes,
positives and negatives resampled separately (2,000 draws, same draws for every method); p = two-sided normal
approximation from the bootstrap SE, Holm over every I-SEM (S and W) vs baseline comparison. raw z and mean difference
are kept in the JSON as reference rows only (not network methods; not in the figure).

Run (writes only into this folder):
  MPLCONFIGDIR=scripts/real_data/tmp TMPDIR=scripts/real_data/tmp \
    .envs/isem/bin/python -I -B scripts/real_data/myc_sig.py > scripts/real_data/myc_sig.log 2>&1
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_claims as cc
import corum_sig as cs
rt = cc.rt

B = 2000
ISEM = ["I-SEM S", "I-SEM W"]                                 # S = shipped readout (primary); W = signed
ORDER = ["I-SEM S", "I-SEM |W|", "I-SEM W", "inspre official (D-bar)", "inspre official (min CV)",
         "inspre default (D-bar)", "inspre default (min CV)", "ADAPRE (D-bar)", "ADAPRE (min CV)",
         "ADAPRE (oracle lambda)", "RICE", "IGSP", "LiNGAM", "NOTEARS", "GOLEM"]
REF = ["raw z", "mean difference"]


def base(name):
    """Method whose sign convention a selected/path matrix follows (paths are W-oriented coefficients)."""
    return name.split(" (")[0]


def analyse(L, rng):
    d = L["jurkat"]; g = d["genes"]; ix = d["idx"]
    ens = [str(x) for x in np.load(cc.FITS / "jurkat.npz", allow_pickle=True)["genes"]]
    M = cc.methods("jurkat", d); paths = cc.path_info("jurkat", ens); S = cs.selected("jurkat", d, M, paths)
    j = ix["MYC"]; others = np.setdiff1d(np.arange(len(g)), [j])
    y = np.isin(others, [ix[x] for x in rt.MUHAR_JURKAT])
    col = {name: cc.activation(base(name), A)[:, j][others] for name, A in S.items()}
    orc = [cc.activation("ADAPRE", A)[:, j][others] for A in paths["ADAPRE"]["mats"]]
    li = int(np.nanargmax([cc.auroc(c, y) for c in orc])); col["ADAPRE (oracle lambda)"] = orc[li]
    p_i, n_i = np.nonzero(y)[0], np.nonzero(~y)[0]
    draws = [np.r_[rng.choice(p_i, len(p_i)), rng.choice(n_i, len(n_i))] for _ in range(B)]
    boot = {m: np.array([cc.auroc(c[s], y[s]) for s in draws]) for m, c in col.items()}
    rows = {}
    for m in ORDER + REF:
        est = cc.auroc(col[m], y)
        rows[m] = dict(auroc=est, ci95=np.percentile(boot[m], [2.5, 97.5]).tolist(),
                       myc_column_all_zero=not np.any(S[m][:, j]) if m in S else False)
        for i in ISEM:
            if m.startswith("I-SEM"):
                continue
            diff = boot[i] - boot[m]; dhat = cc.auroc(col[i], y) - est
            p = float(2 * norm.sf(abs(dhat) / diff.std())) if diff.std() > 0 else np.nan
            rows[m][f"vs {i}"] = dict(diff=dhat, ci95=np.percentile(diff, [2.5, 97.5]).tolist(), p_boot_se=p)
    # Holm over I-SEM S and W vs every network baseline (reference rows excluded)
    flat = [(m, i, rows[m][f"vs {i}"]["p_boot_se"]) for m in ORDER if not m.startswith("I-SEM") for i in ISEM]
    order = sorted(range(len(flat)), key=lambda k: np.nan_to_num(flat[k][2], nan=1.0)); run = 0.0
    for rank, k in enumerate(order):
        m, i, p = flat[k]; run = max(run, min(1.0, (len(flat) - rank) * (1.0 if np.isnan(p) else p)))
        rows[m][f"vs {i}"]["p_holm"] = run
    return dict(n_targets=int(y.sum()), n_genes=int(len(y)), adapre_oracle_lambda_index=li, n_holm=len(flat), rows=rows)


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    ACC, CTX, INK, GRID = "#2a78d6", "#8a8984", "#0b0b0b", "#e4e3df"
    R = res["rows"]
    meth = [m for m in ORDER if m not in ("NOTEARS", "GOLEM")]          # all-zero outputs (caption)
    comp = [m for m in meth if not m.startswith("I-SEM")]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": CTX, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(8.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[len(meth), len(comp)], hspace=0.32, wspace=0.08)
    a = fig.add_subplot(gs[0, :]); b = [fig.add_subplot(gs[1, 0])]; b.append(fig.add_subplot(gs[1, 1], sharey=b[0], sharex=b[0]))
    for y, m in enumerate(meth):
        c = ACC if m.startswith("I-SEM") else CTX
        a.plot(R[m]["ci95"], [y, y], color=c, lw=2, solid_capstyle="round")
        a.plot(R[m]["auroc"], y, "o", ms=6, color=c, mec="#fcfcfb", mew=1.2)
    a.axvline(0.5, color=CTX, lw=1, ls=(0, (3, 3)))
    a.set_yticks(range(len(meth))); a.set_yticklabels(meth); a.invert_yaxis(); a.set_xlabel("AUROC")
    for k, (z, i) in enumerate(zip(b, ISEM)):
        for y, m in enumerate(comp):
            r = R[m][f"vs {i}"]; sig = r["p_holm"] < 0.05           # open marker = not significant (caption)
            z.plot(r["ci95"], [y, y], color=ACC, lw=2, solid_capstyle="round")
            z.plot(r["diff"], y, "o", ms=6, color=ACC, mew=1.5, mec="#fcfcfb" if sig else ACC, mfc=ACC if sig else "#fcfcfb")
        z.axvline(0.0, color=CTX, lw=1, ls=(0, (3, 3)))
        z.set_title(i, fontsize=9, color=INK); z.set_xlabel(f"ΔAUROC ({i} − method)")
        if k:
            plt.setp(z.get_yticklabels(), visible=False)
    b[0].set_yticks(range(len(comp))); b[0].set_yticklabels(comp); b[0].invert_yaxis()
    for z in [a] + b:
        z.grid(axis="x", color=GRID, lw=0.6); z.set_axisbelow(True); z.xaxis.set_major_locator(MaxNLocator(5))
        for s in ("top", "right"):
            z.spines[s].set_visible(False)
        z.tick_params(length=2)
    for t, z in zip("ab", (a, b[0])):
        z.text(-0.62 if z is b[0] else -0.31, 1.0, t, transform=z.transAxes, fontsize=11, fontweight="bold", va="bottom")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    rng = np.random.default_rng(0)
    L = {"jurkat": rt.load("jurkat", rt._symbols())}
    res = analyse(L, rng)
    (cc.OUT / "myc_sig.json").write_text(json.dumps(res, indent=1, default=float))
    figure(res, cs.FIG / "fig_myc_muhar")
    for m, r in res["rows"].items():
        print(f"{m:28s} {r['auroc']:.3f} {np.round(r['ci95'], 3)}",
              *[f"| vs {i}: {r[f'vs {i}']['diff']:+.3f} {np.round(r[f'vs {i}']['ci95'], 3)} holm={r[f'vs {i}'].get('p_holm', np.nan):.2g}"
                for i in ISEM if f"vs {i}" in r])


if __name__ == "__main__":
    main()
