import os, sys, warnings, numpy as np, scipy.io as sio, scipy.signal as ss, json
warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import dl
OUT="epochs"; TMP="raw_tmp"; os.makedirs(OUT,exist_ok=True); os.makedirs(TMP,exist_ok=True)
PRE=300   # 1200 ms at 250 Hz, response-locked, ends at tresp
STIM0,STIMN=25,325  # 100 ms before stimulus .. 1200 ms after, 250 Hz
def run(sub,r):
    out=f"{OUT}/sub-{sub:02d}_run-{r:02d}.npz"
    if os.path.exists(out): return
    ev=f"{TMP}/ev_{sub}_{r}.mat"; dt=f"{TMP}/dat_{sub}_{r}.mat"
    base=f"sub-{sub:02d}/EEG/"
    dl.get(base+f"EEG_events_sub-{sub:02d}_run-{r:02d}.mat",ev,workers=1)
    dl.get(base+f"EEG_data_sub-{sub:02d}_run-{r:02d}.mat",dt)
    e=sio.loadmat(ev); d=sio.loadmat(dt,struct_as_record=False,squeeze_me=True)
    s=d["EEGdata"]; Y=np.asarray(s.Y,dtype=np.float64); assert s.fd==1000
    exch=np.atleast_1d(d["excludedchannels"]).ravel().astype(int)  # 1-based indices into 64
    valid=[c for c in range(1,65) if c not in set(exch.tolist())]
    assert len(valid)==Y.shape[0],(len(valid),Y.shape,exch)
    Yd=ss.decimate(Y,4,axis=1,zero_phase=True).astype(np.float32); del Y
    g=lambda k:e[k].ravel()
    tstim,tresp=g("tstim").astype(int),g("tresp").astype(int)
    tot=int(g("totaltrials")[0]); exc=set(np.atleast_1d(g("excludedtrials")).astype(int).tolist())
    orig=np.array([i for i in range(1,tot+1) if i not in exc]); n=len(tstim)
    if len(orig)!=n: orig=np.arange(1,n+1)*-1  # flag: original index unknown
    pre=np.full((n,64,PRE),np.nan,np.float32); st=np.full((n,64,STIMN),np.nan,np.float32); ok=np.zeros(n,bool)
    vi=np.array(valid)-1
    for i in range(n):
        b=tresp[i]//4; a=b-PRE; s0=int(np.ceil(tstim[i]/4))-STIM0
        if a<0 or s0<0 or s0+STIMN>Yd.shape[1] or b>Yd.shape[1]: continue
        pre[i][vi]=Yd[:,a:b]; st[i][vi]=Yd[:,s0:s0+STIMN]; ok[i]=True
    np.savez_compressed(out,pre=pre,stim=st,ok=ok,sub=sub,run=r,orig=orig,valid_ch=np.array(valid),
        **{k:g(k) for k in ["RT","choice","accuracy","confidence","dotdirection","tstim","tresp"]})
    os.remove(ev); os.remove(dt)
    print("done",sub,r,n,int(ok.sum()),"missing_ch",sorted(set(range(1,65))-set(valid)),flush=True)
if __name__=="__main__":
    for sub in range(1,25):
        for r in (1,2):
            for t in range(3):
                try: run(sub,r); break
                except Exception as ex: print("retry",sub,r,repr(ex)[:200],flush=True)
    print("ALL DONE",flush=True)
