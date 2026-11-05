"""Cross-cell-line biology tests. Reads fits/ (current-src I-SEM fits), the saved comparators and lit/;
writes results/ next to this file. Nothing else is written.

Run:  scripts/experiments/run_env.sh python -u scripts/celltype_comparison/run_tests.py > .../results/run.log 2>&1
Self-check of the statistics on planted data:  ... run_tests.py --selftest

Orientation everywhere: M[i, j] = effect of regulator j on target i. Diagonals are zeroed.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
LIT = HERE / "lit"
TIER1 = ["k562", "rpe1", "jurkat"]
ALL4 = ["k562", "rpe1", "jurkat", "hepg2"]
HUB_KEYS = ["S", "Wabs", "absZ", "MD", "inspre", "adapre"]     # magnitudes (hubs, similarity)
SIGN_KEYS = ["W", "Z", "D", "inspre", "adapre"]                # signed (sign tests); D = signed mean difference
# A knockdown is a negative change of the regulator, so a readout RAISED by the knockdown (response +1: z, D > 0) has a
# NEGATIVE coefficient in W, inspre G and ADAPRE G (effect per unit regulator change). Checked on the fits: sign(W) =
# -sign(z) for 82-89% of |z| > 5 edges (inspre 95-100%).
COEF_KEYS = {"W", "inspre", "adapre"}
B_HUB, B_EDGE, N_NULL = 1000, 300, 5000


def _symbols():
    sym = pickle.load(open(ROOT / "scripts/ensembl_to_symbol.pkl", "rb"))
    h = pd.read_csv(ROOT / "scripts/ensembl_to_hgnc.csv").dropna()
    for a, b in zip(h.ensembl_gene_id, h.hgnc_symbol):
        sym.setdefault(a, b)
    return sym


def load(ds, sym):
    f = np.load(HERE / "fits" / f"{ds}.npz", allow_pickle=True)
    genes = [sym.get(str(g), str(g)) for g in f["genes"]]
    n, t = len(genes), f["cond_target_gene_idx"]
    D = np.zeros((n, n)); D[:, t] = f["delta"].T
    Z = np.zeros((n, n)); Z[:, t] = (f["delta"] / f["se"]).T
    M = dict(S=f["S"], Wabs=f["W_abs"], W=f["W_mean"], Z=Z, D=D,
             inspre=np.load(ROOT / f"scripts/results/paper/e1b_inspre_official/{ds}.npy"),
             adapre=np.load(ROOT / f"scripts/results/paper/e8_adapre/{ds}_adapre.npy"))
    M = {k: np.array(v, float) for k, v in M.items()}
    for v in M.values():
        np.fill_diagonal(v, 0.0)
    M["absZ"], M["MD"] = np.abs(M["Z"]), np.abs(M["D"])
    cm = np.asarray(f["control_mean"], float)
    return dict(ds=ds, genes=genes, idx={g: i for i, g in enumerate(genes)}, M=M, cmean=cm,
                relkd=np.asarray(f["delta_jj"], float) / cm,            # negative = knocked down
                impact=(np.abs(Z) > 3).sum(0) / (n - 1))                # fraction of measured genes with |z| > 3


def pct(x):
    return rankdata(x) / len(x)


def resid(y, covs):
    """Rank regression of y on covariates; returns (residual, R^2)."""
    ry = rankdata(y)
    X = np.column_stack([np.ones(len(ry))] + [rankdata(c) for c in covs])
    r = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    return r, 1 - r.var() / ry.var()


def sub(d, genes):
    return np.array([d["idx"][g] for g in genes])


def hubs(d, key, s, rows=None):
    """Out-strength (column sums over target rows) of regulators s, raw and adjusted for relative knockdown and
    baseline expression. rows: target indices (default s) - resampled in the bootstrap."""
    rows = s if rows is None else rows
    out = np.abs(d["M"][key][np.ix_(rows, s)]).sum(0)
    adj, r2 = resid(out, [d["relkd"][s], np.log(d["cmean"][s])])
    return out, adj, r2


def edges(d, key, s):
    """Edge scores on the s x s sub-network, raw and adjusted (regulator relative KD and expression, target expression)."""
    E = np.abs(d["M"][key][np.ix_(s, s)]) if key != "W" else d["M"][key][np.ix_(s, s)]
    off = ~np.eye(len(s), dtype=bool)
    I, J = np.nonzero(off)
    adj, r2 = resid(E[off], [d["relkd"][s][J], np.log(d["cmean"][s][J]), np.log(d["cmean"][s][I])])
    A = np.zeros_like(E); A[off] = adj
    return E, A, off, r2


def rank1_null(d, s):
    """Expression-only edge null: target expression x regulator absolute knockdown size."""
    return np.outer(d["cmean"][s], np.abs(d["relkd"][s] * d["cmean"][s]))


# ---------------------------------------------------------------- 1a: confounders
def confounders(L):
    rows = []
    for d in L.values():
        for key in HUB_KEYS:
            s = np.arange(len(d["genes"]))
            out, _, r2 = hubs(d, key, s)
            rows.append(dict(line=d["ds"], score=key,
                             rho_hub_vs_abs_kd=spearmanr(out, np.abs(d["relkd"] * d["cmean"])).statistic,
                             rho_hub_vs_expression=spearmanr(out, d["cmean"]).statistic,
                             rho_hub_vs_relative_kd=spearmanr(out, -d["relkd"]).statistic,
                             r2_both_covariates=r2))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Q1: similarity between lines
def similarity(L, lines, shared, rng):
    S = {ds: sub(L[ds], shared) for ds in lines}
    pairs = [(a, b) for i, a in enumerate(lines) for b in lines[i + 1:]]
    rows = []
    for key in HUB_KEYS + ["W", "expression_null"]:
        H, E = {}, {}
        for ds in lines:
            d, s = L[ds], S[ds]
            if key == "expression_null":
                H[ds] = (d["cmean"][s], d["cmean"][s])
                N = rank1_null(d, s); off = ~np.eye(len(s), dtype=bool)
                E[ds] = (N, N, off)
            else:
                if key != "W":
                    o, a, _ = hubs(d, key, s); H[ds] = (o, a)
                Ee, A, off, _ = edges(d, key, s); E[ds] = (Ee, A, off)
        n = len(shared)
        for a, b in pairs:
            row = dict(score=key, pair=f"{a}-{b}")
            if H:                                    # W (signed) has no hub score
                for j, tag in enumerate(["hub_raw", "hub_adj"]):
                    x, y = H[a][j], H[b][j]
                    row[tag] = spearmanr(x, y).statistic
                    bs = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, n, n) for _ in range(B_HUB))]
                    row[tag + "_ci"] = np.percentile(bs, [2.5, 97.5]).round(3).tolist()
            for j, tag in enumerate(["edge_raw", "edge_adj"]):
                X, Y, off = E[a][j], E[b][j], E[a][2]
                if tag == "edge_adj" and min((E[a][0][off] == 0).mean(), (E[b][0][off] == 0).mean()) > 0.5:
                    row[tag] = np.nan                # sparse output: residuals of tied zeros carry the covariates
                    continue
                row[tag] = spearmanr(X[off], Y[off]).statistic
                bs = []
                for _ in range(B_EDGE):
                    i = rng.integers(0, n, n); o = off[np.ix_(i, i)]
                    bs.append(spearmanr(X[np.ix_(i, i)][o], Y[np.ix_(i, i)][o]).statistic)
                row[tag + "_ci"] = np.percentile(bs, [2.5, 97.5]).round(3).tolist()
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Q2: conserved hubs, Q4-style line-specific hubs
def conserved_hubs(L, lines, shared, rng, top=0.10):
    S = {ds: sub(L[ds], shared) for ds in lines}
    n = len(shared)
    res = {}
    null = np.all([pct(L[ds]["cmean"][S[ds]]) > 1 - top for ds in lines], 0)
    res["null_expression_only"] = dict(genes=[shared[i] for i in np.nonzero(null)[0]])
    for key in HUB_KEYS:
        for tag in ["raw", "adj"]:
            j = 0 if tag == "raw" else 1
            P = np.array([pct(hubs(L[ds], key, S[ds])[j]) for ds in lines])
            call = np.all(P > 1 - top, 0)
            hits = np.zeros(n)
            for _ in range(B_HUB):
                rows = rng.integers(0, n, n)
                Pb = np.array([pct(hubs(L[ds], key, S[ds], rows=S[ds][rows])[j]) for ds in lines])
                hits += np.all(Pb > 1 - top, 0)
            g = [shared[i] for i in np.nonzero(call)[0]]
            res[f"{key}_{tag}"] = dict(genes=g, stability={shared[i]: hits[i] / B_HUB for i in np.nonzero(call)[0]},
                                       jaccard_with_null=len(set(g) & set(res["null_expression_only"]["genes"]))
                                       / max(1, len(set(g) | set(res["null_expression_only"]["genes"]))))
    return res


def specific_hubs(L, lines, shared, rng, key, k=10):
    S = {ds: sub(L[ds], shared) for ds in lines}
    n = len(shared)
    P = {ds: pct(hubs(L[ds], key, S[ds])[1]) for ds in lines}
    boot = {ds: [] for ds in lines}
    for _ in range(B_HUB):
        rows = rng.integers(0, n, n)
        Pb = {ds: pct(hubs(L[ds], key, S[ds], rows=S[ds][rows])[1]) for ds in lines}
        for ds in lines:
            boot[ds].append(Pb[ds] - np.mean([Pb[o] for o in lines if o != ds], 0))
    rows = []
    for ds in lines:
        spec = P[ds] - np.mean([P[o] for o in lines if o != ds], 0)
        lo, hi = np.percentile(np.array(boot[ds]), [2.5, 97.5], axis=0)
        for i in np.argsort(-spec)[:k]:
            rows.append(dict(line=ds, gene=shared[i], specificity=spec[i], ci=[lo[i], hi[i]], ci_excludes_0=lo[i] > 0,
                             **{f"adjpct_{o}": P[o][i] for o in lines},
                             **{f"rawpct_{o}": pct(hubs(L[o], key, S[o])[0])[i] for o in lines}))
    return pd.DataFrame(rows)


def specific_edges(L, lines, shared, key, top=0.99):
    S = {ds: sub(L[ds], shared) for ds in lines}
    A = {}
    for ds in lines:
        _, adj, off, _ = edges(L[ds], key, S[ds])
        P = np.zeros_like(adj); P[off] = pct(adj[off]); A[ds] = P
    N = {ds: rank1_null(L[ds], S[ds]) for ds in lines}
    rows = []
    for ds in lines:
        others = [o for o in lines if o != ds]
        I, J = np.nonzero((A[ds] >= top) & np.all([A[o] < 0.5 for o in others], 0))
        for i, j in zip(I, J):
            off = ~np.eye(len(shared), dtype=bool)
            rows.append(dict(line=ds, regulator=shared[j], target=shared[i], adjpct=A[ds][i, j],
                             **{f"adjpct_{o}": A[o][i, j] for o in others},
                             null_pct=(N[ds][off] < N[ds][i, j]).mean(),
                             **{f"z_{o}": L[o]["M"]["Z"][S[o][i], S[o][j]] for o in lines},
                             **{f"W_{o}": L[o]["M"]["W"][S[o][i], S[o][j]] for o in lines}))
    df = pd.DataFrame(rows)
    # Chance count under independence of lines: n_edges * (1 - top) * 0.5^(n_lines - 1) per line.
    df["expected_by_chance_per_line"] = len(shared) * (len(shared) - 1) * (1 - top) * 0.5 ** (len(lines) - 1)
    zl = np.array([r[f"z_{r['line']}"] for r in rows])
    zo = np.array([max(abs(r[f"z_{o}"]) for o in lines if o != r["line"]) for r in rows])
    df["raw_z_supported"] = (np.abs(zl) > 3) & (zo < 1)              # the data alone show the line-specificity
    return df.sort_values(["line", "adjpct"], ascending=[True, False])


# ---------------------------------------------------------------- set specificity (P2, P6)
def set_specificity(L, lines, shared, line, genes, key, rng, readouts=()):
    """Mean adjusted edge percentile within `genes` (+ genes -> readouts) in `line` minus the mean in the other
    lines; null = random shared-gene sets of the same size matched on expression and relative knockdown in `line`."""
    S = {ds: sub(L[ds], shared) for ds in lines}
    A = {}
    for ds in lines:
        _, adj, off, _ = edges(L[ds], key, S[ds])
        P = np.full_like(adj, np.nan); P[off] = pct(adj[off]); A[ds] = P
    pos = {g: i for i, g in enumerate(shared)}
    gi = np.array([pos[g] for g in genes if g in pos])
    ri = np.array([pos[g] for g in readouts if g in pos], int)

    def stat(gi, ri):
        tg = np.union1d(gi, ri)
        return np.nanmean(A[line][np.ix_(tg, gi)]) - np.mean(
            [np.nanmean(A[o][np.ix_(tg, gi)]) for o in lines if o != line])

    obs = stat(gi, ri)
    d = L[line]; s = S[line]
    feat = np.column_stack([pct(np.log(d["cmean"][s])), pct(-d["relkd"][s])])
    pool = np.setdiff1d(np.arange(len(shared)), np.union1d(gi, ri))
    nn = [pool[np.argsort(((feat[pool] - feat[g]) ** 2).sum(1))[:15]] for g in gi]
    null = []
    for _ in range(2000):
        draw = np.unique([rng.choice(c) for c in nn])
        if len(draw) == len(gi):
            null.append(stat(draw, ri))
    null = np.array(null)
    return dict(key=key, line=line, genes=[shared[i] for i in gi], readouts=[shared[i] for i in ri], stat=obs,
                null_mean=null.mean(), p=(1 + (null >= obs).sum()) / (1 + len(null)), n_null=len(null),
                per_line={ds: float(np.nanmean(A[ds][np.ix_(np.union1d(gi, ri), gi)])) for ds in lines})


# ---------------------------------------------------------------- sign tests (P3, P4, P5, P7, P8, P9)
def sign_test(d, regs, reads, response, exclude, rng, pool_genes=None):
    """Mean of sign * M[readouts, regulators] (sign = response for z / D, -response for coefficients) against matched
    control regulators (nearest 15 on relative knockdown, baseline expression and transcriptome-wide impact; drawn one
    per regulator, N_NULL sets). Regulators and controls must reach KD_FLOOR relative knockdown; controls come from
    pool_genes (e.g. a published dataset's null genes) when given, else from every other regulator."""
    gi = lambda gs: np.array([d["idx"][g] for g in gs if g in d["idx"]], int)
    deep = -d["relkd"] >= KD_FLOOR
    r0 = gi(regs); r, t = r0[deep[r0]], gi(reads)
    excl = np.union1d(np.union1d(r, t), gi(exclude))
    feat = np.column_stack([pct(-d["relkd"]), pct(np.log(d["cmean"])), pct(d["impact"])])
    cand = np.nonzero(deep)[0] if pool_genes is None else np.intersect1d(gi(pool_genes), np.nonzero(deep)[0])
    pool = np.setdiff1d(cand, excl)
    nn = [pool[np.argsort(((feat[pool] - feat[j]) ** 2).sum(1))[:15]] for j in r]
    match_gap = np.median([np.abs(feat[c] - feat[j]).mean(0) for c, j in zip(nn, r)], 0)
    draws = np.array([[rng.choice(c) for c in nn] for _ in range(N_NULL)])

    def mean_edge(M, regs_, sign=1):
        X = sign * M[np.ix_(t, regs_)]
        X[t[:, None] == regs_[None, :]] = np.nan
        return np.nanmean(X)

    out = {}
    for key in SIGN_KEYS:
        M, sign = d["M"][key], (-response if key in COEF_KEYS else response)
        obs = mean_edge(M, r, sign)
        null = np.array([mean_edge(M, dr, sign) for dr in draws])
        out[key] = dict(obs=obs, z=(obs - null.mean()) / null.std(), p=(1 + (null >= obs).sum()) / (1 + N_NULL))
    Spct = {key: np.apply_along_axis(pct, 0, d["M"][key]) for key in ["S", "Wabs"]}
    edge_rows = [dict(regulator=d["genes"][j], target=d["genes"][i], W=d["M"]["W"][i, j], z=d["M"]["Z"][i, j],
                      rel_change=d["M"]["D"][i, j] / d["cmean"][i], S_colpct=Spct["S"][i, j], Wabs_colpct=Spct["Wabs"][i, j])
                 for j in r for i in t if i != j]
    return dict(n_regs=len(r), n_reads=len(t), regs=[d["genes"][j] for j in r], reads=[d["genes"][i] for i in t],
                regs_below_kd_floor={d["genes"][j]: float(-d["relkd"][j]) for j in r0[~deep[r0]]},
                regs_relkd={d["genes"][j]: float(-d["relkd"][j]) for j in r}, n_pool=len(pool),
                readout_cmean_pct={d["genes"][i]: float(pct(d["cmean"])[i]) for i in t},
                match_gap_relkd_expr_impact=match_gap.round(3).tolist(), scores=out, edges=edge_rows)


def contrast(La, Lb, regs, reads, response, rng, bound=0.5, kd_gap=0.05):
    """Raw-data contrast between two lines on the regulators/readouts present in both: per-line mean relative change
    response * delta / control_mean with a regulator bootstrap; equivalence of line b with +-bound * effect in line a;
    the same restricted to regulators whose relative knockdown differs by <= kd_gap; transcriptome impact per line."""
    common_r = [g for g in regs if g in La["idx"] and g in Lb["idx"]]
    common_t = [g for g in reads if g in La["idx"] and g in Lb["idx"]]
    shared = sorted(set(La["genes"]) & set(Lb["genes"]))

    def eff(d, rs):
        r, t = sub(d, rs), sub(d, common_t)
        X = response * d["M"]["D"][np.ix_(t, r)] / d["cmean"][t][:, None]
        X[t[:, None] == r[None, :]] = np.nan
        return np.nanmean(X, 0)                                      # per regulator

    def summarise(rs):
        if not rs:
            return None
        ea, eb = eff(La, rs), eff(Lb, rs)
        ba = [np.nanmean(ea[i]) for i in (rng.integers(0, len(rs), len(rs)) for _ in range(2000))]
        bb = [np.nanmean(eb[i]) for i in (rng.integers(0, len(rs), len(rs)) for _ in range(2000))]
        ma, mb = np.nanmean(ea), np.nanmean(eb)
        lo, hi = np.percentile(bb, [5, 95])
        return dict(n=len(rs), eff_a=ma, ci95_a=np.percentile(ba, [2.5, 97.5]).tolist(), eff_b=mb,
                    ci90_b=[lo, hi], equivalence_b=bool(-bound * abs(ma) < lo and hi < bound * abs(ma)))

    kd = {g: (La["relkd"][La["idx"][g]], Lb["relkd"][Lb["idx"][g]]) for g in common_r}
    matched = [g for g in common_r if abs(kd[g][0] - kd[g][1]) <= kd_gap]

    def shared_impact(d, g):
        s = sub(d, shared); j = d["idx"][g]
        return float((np.abs(d["M"]["Z"][s, j]) > 3).mean())

    return dict(lines=[La["ds"], Lb["ds"]], regs=common_r, reads=common_t, all=summarise(common_r),
                kd_matched=summarise(matched), relkd=kd,
                impact_on_shared={g: (shared_impact(La, g), shared_impact(Lb, g)) for g in common_r})


# ---------------------------------------------------------------- the tests (gene lists fixed before running; TESTS.md v3)
KD_FLOOR = 0.6     # minimum relative knockdown for a regulator or control (P3 judge rule, applied to every sign test)


def adamson_s4():
    """Adamson 2016 Table S4 (CRISPRi-v2, K562 UPRE-mCherry/EF1a-GFP reporter), ave of replicates: p, phenotype."""
    t = pd.read_csv(LIT / "adamson2016_tableS4_crispri_v2_reporter.csv", skiprows=1)
    t = t.rename(columns={t.columns[0]: "gene", t.columns[10]: "p", t.columns[11]: "phen"}).iloc[1:]
    return t.assign(p=pd.to_numeric(t["p"], errors="coerce"), phen=pd.to_numeric(t["phen"], errors="coerce"))


def adamson_null():
    t = adamson_s4()
    return set(t.loc[(t.p > 0.5) & (t.phen.abs() < 0.25), "gene"])


# Old -> current HGNC symbols where Adamson's table and our panels differ (aminoacyl-tRNA synthetases).
_OLD = {f"{x}ARS": f"{x}ARS1" for x in "ACDEGHKMNQRSTVWY"}


def adamson_null_current():
    return {_OLD.get(g, g) for g in adamson_null()}


RP_BIOGENESIS = ["RPL23", "RPL7", "RPL35A", "RPS13", "RPS29", "BOP1", "POLR1C", "RSL1D1", "TSR1"]
NON_RP = lambda d: [g for g in d["genes"] if not g.startswith(("RPL", "RPS", "MRPL", "MRPS"))]
MUHAR_JURKAT = ["DDX21", "DHX33", "DIMT1", "GNL3", "GRPEL1", "GSPT1", "MYBBP1A", "NHP2", "NIFK", "NLE1", "NOB1", "NOLC1",
                "NOP16", "POLR1C", "PPA1", "RRS1", "RSL1D1", "URB2", "WDR3", "WDR43"]
CYTO_AARS_K562 = ["FARSA", "GARS1", "HARS1", "NARS1", "QARS1", "RARS1", "TARS1", "VARS1", "YARS1"]

# response: +1 = the knockdown raises the readouts (z, D > 0; W, inspre G, ADAPRE G < 0), -1 = lowers them.
SPEC = {
    "P3_UPR_ER_entry": dict(
        response=+1, lines=["k562", "rpe1", "jurkat"], pool=adamson_null_current,
        regs={"k562": ["SPCS3", "SRP19", "SLC39A7", "UFL1", "SRPRB", "TMED2", "SRP72", "SPCS2", "FAF2", "DERL2"],
              "rpe1": ["SPCS3", "UFL1", "SRPRB", "TMED10", "SRP72", "FAF2", "DERL2"],
              "jurkat": ["SEC61G", "SPCS3", "SLC39A7", "UFL1", "SRPRB", "SEC63", "TMED10", "SRP72", "RPN1"]},
        reads={"k562": ["HSP90B1", "MANF", "HSPA5"], "rpe1": ["HSP90B1", "MANF"], "jurkat": ["HSP90B1", "MANF"]}),
    "P6_HepG2_SRP72_HSPA5": dict(response=+1, lines=["hepg2"], pool=adamson_null_current, regs=["SRP72"], reads=["HSPA5"]),
    "P6b_HepG2_SRP_class_HSPA5": dict(response=+1, lines=["hepg2"], pool=adamson_null_current,
                                      regs=["SRP72", "SRP54", "SRP68"], reads=["HSPA5"]),
    "P4_p53_targets_up": dict(
        response=+1, lines=["rpe1", "k562"], pool=NON_RP, regs=RP_BIOGENESIS,
        reads={"rpe1": ["FDXR", "ASCC3", "FBXW7"], "k562": ["FDXR"]}, contrast=("rpe1", "k562"), contrast_reads=["FDXR"]),
    "P4b_p53_table_S3_secondary": dict(response=+1, lines=["rpe1", "k562"], pool=NON_RP, regs=RP_BIOGENESIS,
                                       reads=["BCL2L1", "CCNK", "KRT8", "RABGGTA"], contrast=("rpe1", "k562")),
    "P5_DREAM_down": dict(response=-1, lines=["rpe1", "k562"], pool=NON_RP, regs=RP_BIOGENESIS,
                          reads=["MYBL2", "CCNB1", "MCM3", "MCM6", "RRM2", "CDK2"], contrast=("rpe1", "k562")),
    "P7_Jurkat_MYC_Muhar": dict(response=-1, lines=["jurkat"], regs=["MYC"], reads=MUHAR_JURKAT),
    "P9_ISR_aaRS_SLC7A5": dict(
        response=+1, lines=["k562", "jurkat"], pool=adamson_null_current,
        regs={"k562": CYTO_AARS_K562, "jurkat": ["FARSA", "GARS1", "HARS1", "QARS1", "RARS1", "VARS1"]},
        reads={"k562": ["SLC7A5"], "jurkat": ["SLC7A5"]}),
    "P9b_ISR_secondary_readouts": dict(
        response=+1, lines=["k562", "jurkat"], pool=adamson_null_current,
        regs={"k562": CYTO_AARS_K562, "jurkat": ["FARSA", "GARS1", "HARS1", "QARS1", "RARS1", "VARS1"]},
        reads={"k562": ["SLC1A5"], "jurkat": ["SLC1A5", "PSAT1"]}),
    "P9c_mito_aaRS_descriptive": dict(       # no pre-stated sign; reported with response +1 only as a convention
        response=+1, lines=["k562"], pool=adamson_null_current,
        regs=["AARS2", "EARS2", "IARS2", "LARS2", "RARS2", "SARS2", "VARS2", "YARS2"], reads=["SLC7A5", "SLC1A5"]),
}

# Exploratory centrality only (no pre-stated direction, no claim): dropped P2 and the P6 centrality half.
SET_TESTS = [
    dict(id="P2_explore_Jurkat_mTOR", tier=1, line="jurkat",
         genes=["MTOR", "MIOS", "SLC7A5", "SLC1A5", "LAMTOR1", "LAMTOR2", "DEPDC5", "PTEN"]),
    dict(id="P6_explore_HepG2_secretory", tier=2, line="hepg2", genes=["SEC16A", "TRAPPC5", "TRAPPC11", "SRP72"]),
]


def run_spec(L, rng):
    s4 = adamson_s4().set_index("gene")
    res = {}
    for tid, sp in SPEC.items():
        print("running", tid, flush=True)
        per = lambda v, ds: v[ds] if isinstance(v, dict) else v
        r = dict(response=sp["response"])
        for ds in sp["lines"]:
            d = L[ds]
            pool = sp["pool"]() if sp.get("pool") is adamson_null_current else (sp["pool"](d) if "pool" in sp else None)
            r[ds] = sign_test(d, per(sp["regs"], ds), per(sp["reads"], ds), sp["response"], (), rng, pool)
            r[ds]["adamson_s4"] = {g: s4.loc[g, ["p", "phen"]].tolist() if g in s4.index else None for g in r[ds]["regs"]}
        if "contrast" in sp:
            a, b = sp["contrast"]
            regs = [g for g in sp["regs"] if all(g in L[x]["idx"] and -L[x]["relkd"][L[x]["idx"][g]] >= KD_FLOOR
                                                 for x in (a, b))]
            r["contrast"] = contrast(L[a], L[b], regs, sp.get("contrast_reads", per(sp["reads"], a)), sp["response"], rng)
        res[tid] = r
    return res


def selftest():
    """Planted check: a regulator set with +edges to readouts is detected by sign_test; a random set is not."""
    rng = np.random.default_rng(1)
    n = 200
    W = rng.normal(0, 1, (n, n)); np.fill_diagonal(W, 0)
    genes = [f"g{i}" for i in range(n)]
    regs, reads = genes[:10], genes[10:20]
    W[10:20, :10] += 1.5
    d = dict(ds="sim", genes=genes, idx={g: i for i, g in enumerate(genes)}, cmean=rng.uniform(.1, 4, n),
             relkd=-rng.uniform(.5, 1, n), impact=rng.uniform(0, .3, n),
             M={k: W.copy() for k in SIGN_KEYS + ["S", "Wabs"]})
    hit = sign_test(d, regs, reads, -1, (), rng)["scores"]     # W > 0 planted = knockdown lowers the readouts
    miss = sign_test(d, genes[30:40], reads, -1, (), rng)["scores"]["W"]
    assert hit["W"]["p"] < 0.01 and hit["Z"]["p"] > 0.99 and miss["p"] > 0.01, (hit, miss)
    y = rng.normal(size=500); c = y + rng.normal(size=500)
    r, r2 = resid(y, [c])
    assert abs(np.corrcoef(r, rankdata(c))[0, 1]) < 1e-9 and 0.3 < r2 < 0.5   # residual is linear-orthogonal to rank(c)
    print("selftest ok", hit, miss)


def main(parts):
    rng = np.random.default_rng(0)
    OUT.mkdir(exist_ok=True)
    sym = _symbols()
    L = {ds: load(ds, sym) for ds in ALL4}
    shared3 = sorted(set.intersection(*[set(L[ds]["genes"]) for ds in TIER1]))
    shared4 = sorted(set.intersection(*[set(L[ds]["genes"]) for ds in ALL4]))
    print("shared", len(shared3), len(shared4), flush=True)
    dump = lambda name, obj: (OUT / name).write_text(json.dumps(obj, indent=1, default=float))

    if "1a" in parts:
        confounders(L).to_csv(OUT / "1a_confounders.csv", index=False)
        pd.DataFrame([dict(line=ds, median_relative_kd=float(np.median(-L[ds]["relkd"])), n_genes=len(L[ds]["genes"]))
                      for ds in ALL4]).to_csv(OUT / "1a_kd_depth.csv", index=False)
    if "sign" in parts:
        dump("sign_tests.json", run_spec(L, rng))
    if "sets" in parts:                  # magnitudes only: a percentile of signed W ranks "most positive", not "strongest"
        res = {}
        for t in SET_TESTS:
            lines, shared = (TIER1, shared3) if t["tier"] == 1 else (ALL4, shared4)
            res[t["id"]] = {key: set_specificity(L, lines, shared, t["line"], t["genes"], key, rng,
                                                 t.get("readouts", ())) for key in ["S", "Wabs"]}
        dump("set_tests.json", res)
    if "q1" in parts:
        similarity(L, TIER1, shared3, rng).to_csv(OUT / "q1_similarity_tier1.csv", index=False)
        similarity(L, ALL4, shared4, rng).to_csv(OUT / "q1_similarity_tier2.csv", index=False)
    if "q2" in parts:
        dump("q2_conserved_hubs.json", dict(n_shared3=len(shared3), n_shared4=len(shared4),
                                            tier1=conserved_hubs(L, TIER1, shared3, rng),
                                            tier2=conserved_hubs(L, ALL4, shared4, rng)))
    if "q3" in parts:
        pd.concat([specific_hubs(L, TIER1, shared3, rng, k).assign(score=k) for k in ["S", "Wabs"]]).to_csv(
            OUT / "q3_specific_hubs_tier1.csv", index=False)
    if "q4" in parts:
        pd.concat([specific_edges(L, TIER1, shared3, k).assign(score=k) for k in ["S", "Wabs"]]).to_csv(
            OUT / "q4_specific_edges_tier1.csv", index=False)
    print("done", parts, flush=True)


if __name__ == "__main__":
    PARTS = ["1a", "sign", "sets", "q1", "q2", "q3", "q4"]
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS)
    a = ap.parse_args()
    selftest() if a.selftest else main(a.parts)
