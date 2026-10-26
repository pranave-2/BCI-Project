import sys, json, numpy as np
sys.path.insert(0,'.')
from common import *
from sklearn.metrics import roc_auc_score
D=load(); keep=D['RT']>=750; Z=np.load('results_scores.npz')
sk=D['sub'][keep]; assert (sk==Z['sub']).all()
dirk=(D['dotdirection'][keep]==0).astype(int)   # 0 = rightward? check coding below
dv=np.unique(D['dotdirection'][keep]); chn=Z['choice']; sc=Z['sc_choice']; acc=Z['acc']
out={'dot_values':[int(x) for x in dv]}
# stimulus-direction decoding (same window, features) for reference
X,k=window_features(D,600,150); X=zscore_by_subject(X,D['sub'])[k]
yd=(D['dotdirection'][k]==dv[1]).astype(int)
a,p,_=loso_auc(X,yd,sk,np.ones(len(yd),bool)); out['direction_decoding']=dict(mean=float(np.mean(list(a.values()))),pooled=float(p))
# choice scores within direction strata
rng=np.random.default_rng(3)
def strat_auc(ylab,s,m): 
    return roc_auc_score(ylab[m],s[m])
res={}
for v in dv:
    m=D['dotdirection'][keep]==v
    obs=roc_auc_score(chn[m],sc[m]); 
    # permutation: shuffle choice labels within subject inside stratum
    cnt=0; N=2000; idx=np.where(m)[0]
    for _ in range(N):
        y=chn.copy()
        for s in np.unique(sk[idx]):
            q=idx[sk[idx]==s]; y[q]=rng.permutation(chn[q])
        if roc_auc_score(y[m],sc[m])>=obs: cnt+=1
    res[int(v)]=dict(auc=float(obs),n=int(m.sum()),p=(cnt+1)/(N+1))
out['choice_within_direction']=res
# per-subject mean of within-direction AUC
ps=[]
for s in np.unique(sk):
    vals=[]
    for v in dv:
        m=(sk==s)&(D['dotdirection'][keep]==v)
        if len(np.unique(chn[m]))==2 and min(chn[m].sum(),(1-chn[m]).sum())>=3: vals.append(roc_auc_score(chn[m],sc[m]))
    if vals: ps.append(np.mean(vals))
out['choice_within_direction_subject_mean']=dict(mean=float(np.mean(ps)),n=len(ps))
out['choice_dir_agree']=float((chn==(D['dotdirection'][keep]==dv[1]).astype(int)).mean())
print(json.dumps(out,indent=1)); json.dump(out,open('results_dir_control.json','w'),indent=1)
