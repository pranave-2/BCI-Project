import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0,'.')
from common import load
D=load(); keep=D['RT']>=750
Z=np.load('results_scores.npz')
sk=D['sub'][keep]; assert (sk==Z['sub']).all() and (D['confidence'][keep]==Z['conf']).all()
run=D['run'][keep]; orig=D['orig'][keep]; ak=Z['acc']; ck=Z['conf']; rk=Z['rt'].astype(float); sc=Z['sc_conf']
dirk=D['dotdirection'][keep]; chn=D['choice'][keep]
key={(s,r,o):i for i,(s,r,o) in enumerate(zip(sk,run,orig))}
pr=[(i,key[(s,r,o+1)]) for i,(s,r,o) in enumerate(zip(sk,run,orig)) if (s,r,o+1) in key and o>0]
a=np.array([p[0] for p in pr]); b=np.array([p[1] for p in pr])
z=lambda v:(v-v.mean())/v.std()
zs=np.zeros(len(sk)); zc=np.zeros(len(sk))
for s in np.unique(sk):
    m=sk==s; zs[m]=z(sc[m]); zc[m]=z(ck[m].astype(float))
T=pd.DataFrame(dict(sub=sk[a],rep=(chn[a]==chn[b]).astype(int),stimrep=(dirk[a]==dirk[b]).astype(int),accN=ak[a],S=zs[a],Cn=zc[a],lrtN=np.log(rk[a])))
out={}
for nm,f in (("neural","rep ~ S + stimrep*accN + lrtN"),("behav","rep ~ Cn + stimrep*accN + lrtN"),("neural_x","rep ~ S*stimrep + stimrep*accN + lrtN"),("behav_x","rep ~ Cn*stimrep + stimrep*accN + lrtN")):
    m=sm.GEE.from_formula(f,groups="sub",data=T,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
    out[nm]={k:dict(b=float(m.params[k]),se=float(m.bse[k]),p=float(m.pvalues[k])) for k in m.params.index}
    print(nm,{k:(round(v['b'],3),round(v['p'],4)) for k,v in out[nm].items() if k!='Intercept'})
json.dump(out,open('results_rq4_interaction.json','w'),indent=1)
