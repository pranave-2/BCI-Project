import sys, json, time, numpy as np
sys.path.insert(0,'.')
from common import *
from scipy import stats
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
D=load(); keep=D['RT']>=750; Z=np.load('results_scores.npz')
sk=D['sub'][keep]; assert (sk==Z['sub']).all()
pre=D['pre'][keep]; rt=D['RT'][keep].astype(float); lrt=np.log(rt)
acc=Z['acc']; conf=Z['conf']; chn=Z['choice']
hl=(conf<=3)|(conf>=7); yhl=(conf>=7).astype(int)
targets={'choice':(chn,np.ones(len(chn),bool)),'accuracy':(acc,np.ones(len(acc),bool)),'confidence':(yhl,hl)}
end=300-int(round(150*250/1000)); start=end-int(round(600*250/1000)); seg=pre[:,:,start:end].astype(np.float64)
def zs(X): return zscore_by_subject(X,sk)
amp=zs(seg.mean(2))
parts=np.array_split(np.arange(seg.shape[2]),4); amp4=zs(np.concatenate([seg[:,:,p].mean(2) for p in parts],1))
logvar=zs(np.log(seg.var(2)+1e-12))
feats={'amp':amp,'amp4':amp4,'logvar':logvar,'amp+logvar':np.concatenate([amp,logvar],1)}
def resid(X):
    Xr=X.copy()
    for s in np.unique(sk):
        m=sk==s; A=np.c_[np.ones(m.sum()),(lrt[m]-lrt[m].mean())/lrt[m].std()]
        Xr[m]=X[m]-A[:,1:]@np.linalg.lstsq(A,X[m],rcond=None)[0][1:]; Xr[m]=(Xr[m]-Xr[m].mean(0))/(Xr[m].std(0)+1e-12)
    return Xr
models={
 'lr':lambda: LogisticRegression(C=0.1,class_weight='balanced',max_iter=300),
 'lda':lambda: LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'),
 'svm_lin':lambda: LinearSVC(C=0.01,class_weight='balanced',dual=False),
 'svm_rbf':lambda: SVC(C=1.0,kernel='rbf',gamma='scale',class_weight='balanced'),
 'rf':lambda: RandomForestClassifier(n_estimators=300,min_samples_leaf=20,class_weight='balanced_subsample',n_jobs=1,random_state=0),
 'mlp':lambda: MLPClassifier(hidden_layer_sizes=(32,),alpha=0.1,early_stopping=True,max_iter=200,random_state=0)}
def score(clf,Xte):
    return clf.predict_proba(Xte)[:,1] if hasattr(clf,'predict_proba') and not isinstance(clf,SVC) else clf.decision_function(Xte)
def fold(mk,X,y,tr,te):
    clf=models[mk]().fit(X[tr],y[tr]); return te,score(clf,X[te])
configs=[('amp',m) for m in models]+[('amp4','lr'),('amp4','lda'),('logvar','lr'),('amp+logvar','lr')]
out={}; t0=time.time()
for variant in ('raw','resid'):
    F={k:(v if variant=='raw' else resid(v)) for k,v in feats.items()}
    for tname,(y,v) in targets.items():
        idx=np.where(v)[0]; ys=y[idx]; ss=sk[idx]
        for fk,mk in configs:
            X=F[fk][idx]; res=Parallel(n_jobs=8)(delayed(fold)(mk,X,ys,np.where(ss!=s)[0],np.where(ss==s)[0]) for s in np.unique(ss))
            sc=np.zeros(len(ys))
            for te,p in res: sc[te]=p
            a=subj_auc(sc,ys,ss,np.ones(len(ys),bool))
            out.setdefault(variant,{}).setdefault(tname,{})[f'{fk}|{mk}']=dict(per_subject={int(k):float(x) for k,x in a.items()},mean=float(np.mean(list(a.values()))),pooled=float(roc_auc_score(ys,sc)))
            print(variant,tname,fk,mk,round(out[variant][tname][f'{fk}|{mk}']['mean'],4),round(time.time()-t0),'s',flush=True)
# paired Wilcoxon vs primary (amp|lr), Holm within target & variant
for variant in out:
    for tname in out[variant]:
        base=out[variant][tname]['amp|lr']['per_subject']; names=[k for k in out[variant][tname] if k!='amp|lr']; ps=[]
        for k in names:
            cur=out[variant][tname][k]['per_subject']; ks=sorted(set(base)&set(cur))
            d=np.array([cur[s]-base[s] for s in ks]); w=stats.wilcoxon(d) if np.any(d!=0) else None
            out[variant][tname][k]['diff_vs_primary']=float(d.mean()); out[variant][tname][k]['p_wilcoxon']=float(w.pvalue) if w else 1.0; ps.append(out[variant][tname][k]['p_wilcoxon'])
        o=np.argsort(ps); m=len(ps); run=0
        for r,i in enumerate(o): run=max(run,min(1,ps[i]*(m-r))); out[variant][tname][names[i]]['p_holm']=float(run)
json.dump(out,open('results_compare.json','w'),indent=1); print('FINISHED',flush=True)
