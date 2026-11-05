import json,pathlib,csv
R=pathlib.Path(__file__).resolve().parents[2]/'scripts/results/paper/e7_stability'
DS=['hepg2','jurkat','rpe1','k562']; CFG=['tau0.9_do0.2_half','tau0.9_do0.0_half','tau1.2_do0.0_half','tau1.5_do0.0_half','tau2.0_do0.0_half','tau2.5_do0.0_half']
rows=[]
for ds in DS:
    ctl=json.load(open(R/f'real_select/{ds}_seed0/tau0.9_do0.2_half.json'))['heldout_mse']
    for c in CFG:
        s=json.load(open(R/f'real_select/{ds}_seed0/{c}.json')); f=json.load(open(R/f'real_full/{ds}/{c}.json'))
        rows.append(dict(screen=ds,config=c,tau=s['tau'],w_dropout=s['w_dropout'],n_models=s['n_models'],heldout_mse=s['heldout_mse'],mse_rel_control=s['heldout_mse']/ctl,rho_median_sel=s['rho_W_median'],rho_max_sel=s['rho_W_max'],rho_median_allcells=f['rho_W_median'],rho_max_allcells=f['rho_W_max']))
w=csv.DictWriter(open(pathlib.Path(__file__).parent/'tau_table.csv','w'),rows[0].keys());w.writeheader();w.writerows(rows)
L=["# tau sweep, half-split fits only (appendix app:tau)","",
"Source: raw `scripts/results/paper/e7_stability/real_select/<ds>_seed0/<cfg>.json` (80% of cells fit, seed 0; held-out = other 20% of cells) and `real_full/<ds>/<cfg>.json` (all cells). Held-out error = held-out MSE / MSE of the tau 0.9, W-dropout 0.2 half-split fit (the control row of `stage3_tausweep.md`). rho(W) = median over the 120 models. Cross-check: every value below equals `stage3_tausweep.md` at its printed 2-decimal / 3-decimal precision (checked in script output, see REPORT.md).","",
"Solver: these fits used the pre-option-A solver (whole-batch pivoted-LU fallback; fallback counts in the json). The reference_option_a refits at tau 1.5 are in stability.md.","",
"| dataset | tau | W-dropout | held-out error (rel. control) | median rho(W), 80% fit | median rho(W), all-cells fit |","|---|---|---|---|---|---|"]
for r in rows: L.append(f"| {r['screen']} | {r['tau']} | {r['w_dropout']} | {r['mse_rel_control']:.3f} | {r['rho_median_sel']:.3f} | {r['rho_median_allcells']:.3f} |")
L+=["","## Claim check: tau 1.5 is the largest tau whose median model is stable (median rho < 1) on all four datasets, among half-split configs (dropout 0)",""]
for t in [1.2,1.5,2.0,2.5]:
    rs=[r for r in rows if r['tau']==t and r['w_dropout']==0]
    a=all(r['rho_median_sel']<1 for r in rs); b=all(r['rho_median_allcells']<1 for r in rs)
    L.append(f"- tau {t}: all four <1 (80% fit): {a}; (all-cells fit): {b}; max of medians {max(r['rho_median_sel'] for r in rs):.4f} / {max(r['rho_median_allcells'] for r in rs):.4f}")
open(pathlib.Path(__file__).parent/'tau_table.md','w').write("\n".join(L)+"\n"); print("\n".join(L))
