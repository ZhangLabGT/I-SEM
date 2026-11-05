"""ADAPRE across its whole penalty path on the Fig. 2(b) and (c) tests: same statistics and matrices as corum_sig.py and ../myc_size_1102."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/real_data"))
import check_claims as cc, corum_sig as cs
cplx = cc.corum_sets(); out = {}
for ds in ["k562", "rpe1", "hepg2", "jurkat"]:
    d = cc.rt.load(ds, cc.rt._symbols()); g = d["genes"]
    ens = [str(x) for x in np.load(cc.FITS / f"{ds}.npz", allow_pickle=True)["genes"]]
    paths = cc.path_info(ds, ens); ix = {s: i for i, s in enumerate(g)}; n = len(g)
    P = np.zeros((n, n), bool)
    for c in cplx:
        m = [ix[s] for s in c if s in ix]; P[np.ix_(m, m)] = True
    np.fill_diagonal(P, False)
    ex = np.log(d["cmean"]); exb = np.digitize(ex, np.quantile(ex, np.linspace(0, 1, 11)[1:-1]))
    v = []
    for A in paths["ADAPRE"]["mats"]:
        nu, de = cs.num_den(np.abs(A), P, exb); v.append(nu.sum() / de.sum())
    e = [int((A != 0).sum()) for A in paths["ADAPRE"]["mats"]]
    out[ds] = dict(min=min(v), median=float(np.median(v)), max=max(v), n=len(v), edges_median=float(np.median(e)))
    print(ds, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in out[ds].items()}, flush=True)
    if ds == "jurkat":
        j = ix["MYC"]; oth = np.setdiff1d(np.arange(n), [j]); y = np.isin(oth, [ix[t] for t in cc.rt.MUHAR_JURKAT])
        w = [cc.auroc(np.abs(A)[:, j][oth], y) for A in paths["ADAPRE"]["mats"]]
        nz = [int((A[:, j][oth] != 0).sum()) for A in paths["ADAPRE"]["mats"]]
        out["myc"] = dict(min=min(w), median=float(np.median(w)), max=max(w), edges_from_myc_median=float(np.median(nz)))
        print("myc", out["myc"], flush=True)
(Path(__file__).parent / "path.json").write_text(json.dumps(out, indent=1, default=float))
