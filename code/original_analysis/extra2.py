import sys, json, numpy as np
sys.path.insert(0,'.')
from common import *
D=load(); keep=D['RT']>=750; Z=np.load('results_scores.npz')
sk=D['sub'][keep]; assert (sk==Z['sub']).all()
X,k=window_features(D,600,150); X=zscore_by_subject(X,D['sub'])[k]; assert (D['sub'][k]==sk).all()
acc=Z['acc']; conf=Z['conf']
out={}
def contrast(m,ypos,nperm=300):
    y=ypos.astype(int); a,pooled,sc=loso_auc(X,y,sk,m)
    pm,pp,r=perm_test(X[m],y[m],sk[m],np.mean(list(a.values())),pooled,n=nperm,seed0=9000)
    return dict(n=int(m.sum()),n_pos=int(y[m].sum()),pooled=float(pooled),p=float(pp),n_subj=int(len(np.unique(sk[m]))))
for nm,(lo,hi) in {'primary 1-3 vs 7-9':(3,7),'strict 1-2 vs 8-9':(2,8),'liberal 1-4 vs 6-9':(4,6)}.items():
    low=conf<=lo; high=conf>=hi
    inc=(acc==0)&(low|high); cor=(acc==1)&(low|high)
    out[nm]=dict(incorrect=contrast(inc,high),correct=contrast(cor,high))
    print(nm,out[nm],flush=True)
# per-participant table data
Zs=np.load('results_scores.npz'); R=json.load(open('results.json'))
out['per_subject']={int(s):dict(choice=R['RQ1']['per_subject_auc'].get(str(s)),accuracy=R['RQ2']['per_subject_auc'].get(str(s)),confidence=R['conf_decoding']['per_subject_auc'].get(str(s)),
    n=int((sk==s).sum()),ih=int(((sk==s)&(acc==0)&(conf>=7)).sum())) for s in np.unique(sk)}
# behaviour: rating distribution by accuracy (all trials)
c=D['confidence']; a=D['accuracy']
out['rating_dist']={'correct':[float(np.mean(c[a==1]==r)) for r in range(1,10)],'incorrect':[float(np.mean(c[a==0]==r)) for r in range(1,10)]}
json.dump(out,open('results_extra2.json','w'),indent=1)
