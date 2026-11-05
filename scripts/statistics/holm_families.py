#!/usr/bin/env python3
"""The co-complex and MYC Holm families without
ADAPRE's D-hat and lowest-CV-error rows, which Fig. 2 no longer draws.

    python3 holm_families.py            prints every Holm-dependent macro (\\renewcommand) and marks the changed ones
    python3 holm_families.py --test     self-check of the Holm step

No bootstrap is rerun: the saved raw p values (p_boot_se) of
scripts/real_data/out/corum_sig.json and scripts/myc/myc_size.json
(read only) are Holm-adjusted again over the smaller families. The macros follow the number ledger
(F2.summary.holm.*, F2.*.oracle.p, F3.n_holm, F3.*.p.*, F3.selected.*). The selected-rule values are
taken over the three inspre rule rows Fig. 2 draws (inspre-GWPS at both rules, inspre-default at the
lowest-CV-error rule; fig2_real.py:165-171), not four.
"""
import json, math, re, sys
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parents[2]
CS, MS = ROOT / "scripts/real_data/out/corum_sig.json", ROOT / "scripts/myc/myc_size.json"
DROP = {"ADAPRE (D-bar)", "ADAPRE (min CV)"}
ORC = "ADAPRE (oracle lambda)"
SEL = ["inspre official (D-bar)", "inspre official (min CV)", "inspre default (min CV)"]  # drawn rule rows
DS = {"hepg2": "H", "rpe1": "R", "jurkat": "J", "k562": "K"}
GSET = {"all": "", "noRP": "NoRP"}
MINUS = r"\ensuremath{-}"


def holm(ps):
    """Holm step-down adjusted p values, {key: p} -> {key: p_adj}."""
    out, run, items = {}, 0.0, sorted(ps.items(), key=lambda kv: kv[1])
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * p))
        out[k] = run
    return out


def rnd(x, d):
    s = f"{Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP):f}"
    return s.lstrip("-") if float(s) == 0 else s


def num(x, d):
    s = rnd(x, d)
    neg, s = s.startswith("-"), s.lstrip("-")
    i, _, f = s.partition(".")
    return (MINUS if neg else "") + f"{int(i):,}".replace(",", "{,}") + ("." + f if f else "")


def sci(p, d=1):
    e = math.floor(math.log10(p))
    m = rnd(p / 10 ** e, d)
    if float(m) >= 10:
        e, m = e + 1, rnd(p / 10 ** (e + 1), d)
    return rf"\ensuremath{{{m}\times10^{{{e}}}}}"


def pv(x):
    return sci(x) if x < 0.001 else num(x, 3 if x < 0.1 else 2)


def macros():
    c, m = json.load(open(CS)), json.load(open(MS))
    out = {}
    raw = {(ds, g, r): v["p_boot_se"] for ds in DS for g in GSET for r, v in c[ds][g]["rows"].items()
           if "p_holm" in v and r not in DROP}
    adj = holm(raw)
    for ds, D in DS.items():
        for g, G in GSET.items():
            out[f"num{D}CorumAdapreOracleHolmP{G}"] = pv(adj[(ds, g, ORC)])
    excl = [k for k in adj if k[2] != ORC]
    out["numCorumHolmTests"] = str(len(adj))
    out["numCorumHolmSigExclOracle"] = str(sum(adj[k] < 0.05 and c[k[0]][k[1]]["rows"][k[2]]["diff_isemS_minus"] > 0
                                               for k in excl))
    out["numCorumMaxHolmPExclOracle"] = pv(max(adj[k] for k in excl))
    mraw = {(r, v): x[v]["p_boot_se"] for r, x in m["rows"].items() for v in x
            if v.startswith("vs ") and x[v].get("p_holm") is not None and r not in DROP}
    madj = holm(mraw)
    out["numMycHolmTests"] = str(len(madj))
    for r, nm, i, key in [("RICE", "Rice", "S", "S"), ("IGSP", "Igsp", "S", "S"), ("IGSP", "Igsp", "W", "|W|"),
                          (ORC, "AdapreOracle", "S", "S")]:
        out[f"numMyc{nm}HolmP{i}"] = pv(madj[(r, f"vs I-SEM {key}")])
    out["numMycSelectedMaxAuroc"] = num(max(m["rows"][r]["auroc"] for r in SEL), 3)
    d = min((m["rows"][r]["vs I-SEM S"]["diff"], *m["rows"][r]["vs I-SEM S"]["ci95"]) for r in SEL)
    out["numMycSelectedDeltaMin"] = " ".join([("+" if d[0] > 0 else "") + num(d[0], 3),
                                              f"[{('+' if d[1] > 0 else '') + num(d[1], 3)},",
                                              f"{('+' if d[2] > 0 else '') + num(d[2], 3)}]"])
    out["numMycSelectedHolmPMax"] = pv(max(madj[(r, "vs I-SEM S")] for r in SEL))
    return out


def test():
    a = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert abs(a["x"] - 0.03) < 1e-12 and abs(a["z"] - 0.06) < 1e-12 and abs(a["y"] - 0.06) < 1e-12, a
    assert pv(1.28e-8) == r"\ensuremath{1.3\times10^{-8}}" and pv(0.0493) == "0.049" and pv(0.582) == "0.58"
    new = macros()
    assert new["numCorumHolmTests"] == "96" and new["numMycHolmTests"] == "20", new
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        test()
    else:
        for k, v in macros().items():
            print(f"\\renewcommand{{\\{k}}}{{{v}}}")
