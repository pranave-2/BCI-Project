import sys, json, numpy as np
sys.path.insert(0,'.')
from common import *
from scipy import stats
from sklearn.metrics import roc_auc_score
D=load(); keep=D['RT']>=750; Z=np.load('results_scores.npz')
sk=D['sub'][keep]; assert (sk==Z['sub']).all()
pre=D['pre'][keep]                       # (n,53,300), time of sample j = -1200+4j ms
rt=D['RT'][keep].astype(float); lrt=np.log(rt)
acc=Z['acc']; conf=Z['conf']; chn=Z['choice']
hl=(conf<=3)|(conf>=7); yhl=(conf>=7).astype(int)
targets={'choice':(chn,np.ones(len(chn),bool)),'accuracy':(acc,np.ones(len(acc),bool)),'confidence':(yhl,hl)}
def resid(X):
    Xr=X.copy()
    for s in np.unique(sk):
        m=sk==s; A=np.c_[np.ones(m.sum()),(lrt[m]-lrt[m].mean())/lrt[m].std()]
        Xr[m]=X[m]-A[:,1:]@np.linalg.lstsq(A,X[m],rcond=None)[0][1:]; Xr[m]=(Xr[m]-Xr[m].mean(0))/(Xr[m].std(0)+1e-12)
    return Xr
starts=list(range(-750,-99,50)); out={'starts':starts,'win_ms':100}
for t,(y,v) in targets.items(): out[t]={'raw':[], 'resid':[]}
for s0 in starts:
    a=(s0+1200)//4; X=pre[:,:,a:a+25].mean(2).astype(float); X=zscore_by_subject(X,sk); Xr=resid(X)
    for t,(y,v) in targets.items():
        for nm,XX in (('raw',X),('resid',Xr)):
            aa,pooled,_=loso_auc(XX,y,sk,v); vals=np.array(list(aa.values()))
            tt=stats.ttest_1samp(vals,0.5); p=tt.pvalue/2 if tt.statistic>0 else 1-tt.pvalue/2
            out[t][nm].append(dict(mean=float(vals.mean()),se=float(vals.std(ddof=1)/np.sqrt(len(vals))),p=float(p)))
    print(s0,{t:round(out[t]['raw'][-1]['mean'],3) for t in targets},flush=True)
# Holm across the 14 windows per target/variant
for t in targets:
    for nm in ('raw','resid'):
        ps=np.array([w['p'] for w in out[t][nm]]); o=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0
        for rank,i in enumerate(o): run=max(run,min(1,ps[i]*(m-rank))); adj[i]=run
        for w,a_ in zip(out[t][nm],adj): w['p_holm']=float(a_)
# RT-tercile-stratified AUC of the main LOSO scores
strat={}
for nm,sc,(y,v) in (('choice',Z['sc_choice'],targets['choice']),('accuracy',Z['sc_acc'],targets['accuracy']),('confidence',Z['sc_conf'],targets['confidence'])):
    per=[]
    for s in np.unique(sk):
        for q in range(3):
            m=(sk==s)&v
            idx=np.where(m)[0]; cuts=np.quantile(rt[idx],[1/3,2/3]); b=np.digitize(rt[idx],cuts)
            ii=idx[b==q]
            if len(np.unique(y[ii]))==2 and min(y[ii].sum(),(1-y[ii]).sum())>=5: per.append((s,q,roc_auc_score(y[ii],sc[ii])))
    arr=np.array(per)
    sub_mean=[np.mean(arr[arr[:,0]==s,2]) for s in np.unique(arr[:,0])]
    tt=stats.ttest_1samp(sub_mean,0.5)
    strat[nm]=dict(mean=float(np.mean(sub_mean)),sd=float(np.std(sub_mean,ddof=1)),n=len(sub_mean),p_one=float(tt.pvalue/2 if tt.statistic>0 else 1-tt.pvalue/2),
                   by_tercile=[float(np.mean(arr[arr[:,1]==q,2])) for q in range(3)])
    print(nm,strat[nm],flush=True)
out['rt_tercile']=strat
json.dump(out,open('results_extra.json','w'),indent=1)
