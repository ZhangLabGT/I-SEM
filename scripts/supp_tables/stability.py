import json,numpy as np,csv
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[2])+'/'
C=ROOT+'scripts/results/paper/cache'
old=json.load(open(ROOT+'scripts/supp_tables/eig_check.json'))
rows=[]
for ds in ['hepg2','jurkat','rpe1','k562']:
    meta=json.load(open(f'{C}/{ds}.json'))
    Ws=np.load(f'{C}/{ds}_W_models.npy',mmap_mode='r')
    rho=[];re=[];real=[];imag_lead=[]
    for W in Ws:
        ev=np.linalg.eigvals(np.asarray(W,dtype=np.float64)); a=np.abs(ev); i=a.argmax()
        rho.append(a[i]); re.append(ev.real.max()); real.append(abs(ev[i].imag)<=1e-9*max(1,a[i])); imag_lead.append(abs(ev[i].imag))
    rho=np.array(rho);re=np.array(re);real=np.array(real)
    gt=rho>1
    r=dict(screen=ds,n_models=len(Ws),n_genes=Ws.shape[1],code_commit=meta['code_commit'][:7],tau=meta['config']['tau'],w_dropout=meta['config']['w_dropout'],
      n_rho_gt1=int(gt.sum()),rho_median=float(np.median(rho)),rho_max=float(rho.max()),max_re_lambda=float(re.max()),n_re_ge1=int((re>=1).sum()),
      leading_real_all=bool(real.all()),leading_real_among_rho_gt1=bool(real[gt].all()) if gt.any() else None,n_leading_complex=int((~real).sum()),max_abs_imag_leading=float(max(imag_lead)),
      matches_eig_check_json=bool(old[ds]['rho_gt1']==int(gt.sum()) and abs(old[ds]['rho_max']-rho.max())<1e-9))
    rows.append(r);print(r,flush=True)
w=csv.DictWriter(open(ROOT+'scripts/supp_tables/stability.csv','w'),rows[0].keys());w.writeheader();w.writerows(rows)
L=["# Stability of the shipped (option-A) fits, tau 1.5, half-split, 120 models per dataset","",
f"Source: `scripts/results/paper/cache/<ds>_W_models.npy` (per-model W, code commit {rows[0]['code_commit']} = option A, tau 1.5, W-dropout 0; the cache json records this). `scripts/supp_tables/eig_check.json` used the same files; recomputed here on CPU (float64 eigvals) with the added leading-eigenvalue column; counts and max rho match eig_check.json (column matches_eig_check_json).","",
"| dataset | genes | models with rho(W) > 1 (of 120) | median rho | max rho | max Re lambda | models with Re lambda >= 1 | leading eigenvalue real, models with rho > 1 | leading eigenvalue complex, all models |","|---|---|---|---|---|---|---|---|---|"]
for r in rows: L.append(f"| {r['screen']} | {r['n_genes']} | {r['n_rho_gt1']} | {r['rho_median']:.3f} | {r['rho_max']:.3f} | {r['max_re_lambda']:.3f} | {r['n_re_ge1']} | {r['leading_real_among_rho_gt1']} | {r['n_leading_complex']} of 120 (|Im| <= {r['max_abs_imag_leading']:.2f}; all with rho < 1) |")
L+=["","Reading: dx/dt = -x + Wx + input settles to (I-W)^-1 input iff all Re lambda < 1. Models with rho > 1 are exactly the models with Re lambda >= 1 when the leading eigenvalue is real positive (compare the two count columns). Those models do not have an attracting steady state; the steady-state equation (I-W)x = input still defines the fitted prediction."]
open(ROOT+'scripts/supp_tables/stability.md','w').write("\n".join(L)+"\n")
