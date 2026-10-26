import json, sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
sys.path.insert(0, ".")
from common import *

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 500
R = {}
D = load()
sub = D["sub"]; N = len(sub)
subs = np.unique(sub)
R["n_subjects"] = int(len(subs)); R["n_trials_total"] = int(N)
R["n_runs"] = int(len(set(zip(D["sub"], D["run"]))))
R["n_channels_common"] = int(len(D["channels"])); R["excluded_channels"] = sorted(set(range(1, 65)) - set(D["channels"].tolist()))
cnt = pd.Series(sub).value_counts().sort_index()
R["trials_per_subject"] = dict(min=int(cnt.min()), median=float(cnt.median()), max=int(cnt.max()))

# ---------- behaviour ----------
acc, conf, rt = D["accuracy"], D["confidence"], D["RT"].astype(float)
df = pd.DataFrame(dict(sub=sub, acc=acc, conf=conf, rt=rt, choice=D["choice"]))
g = df.groupby("sub")
R["beh"] = dict(acc_mean=float(g.acc.mean().mean()), acc_sd=float(g.acc.mean().std()), acc_min=float(g.acc.mean().min()), acc_max=float(g.acc.mean().max()),
                rt_mean=float(rt.mean()), rt_sd=float(rt.std()), rt_median=float(np.median(rt)),
                conf_mean=float(conf.mean()), conf_sd=float(conf.std()),
                conf_counts={int(k): int(v) for k, v in pd.Series(conf).value_counts().sort_index().items()},
                right_choice_frac=float((D["choice"] == 2).mean()),
                conf_correct=float(conf[acc == 1].mean()), conf_incorrect=float(conf[acc == 0].mean()))
low, high = conf <= 3, conf >= 7
R["cells_all"] = dict(CL=int(((acc == 1) & low).sum()), CH=int(((acc == 1) & high).sum()), IL=int(((acc == 0) & low).sum()),
                      IH=int(((acc == 0) & high).sum()), med=int((~low & ~high).sum()))
ih = (acc == 0) & high
s_ih = pd.Series(sub[ih]).value_counts()
R["IH_diversity"] = dict(n_subjects=int(len(s_ih)), max_share=float(s_ih.max() / s_ih.sum()), top3=[int(x) for x in s_ih.index[:3]], per_subject=s_ih.sort_index().to_dict())
R["IH_diversity"]["per_subject"] = {int(k): int(v) for k, v in R["IH_diversity"]["per_subject"].items()}
# per-subject Spearman of RT with accuracy / confidence (behavioural leakage reference)
R["beh"]["rt_conf_rho_mean"] = float(np.mean([stats.spearmanr(rt[sub == s], conf[sub == s])[0] for s in subs]))
R["beh"]["rt_acc_rho_mean"] = float(np.mean([stats.spearmanr(rt[sub == s], acc[sub == s])[0] for s in subs]))

# ---------- window grid / selection (retention only) ----------
Ls = [200, 300, 400, 500, 600, 700, 800]; Bs = [0, 50, 100, 150, 200, 250, 300, 350]
ret = {(L, B): float((rt >= L + B).mean()) for L in Ls for B in Bs}
R["retention_grid"] = {f"{L}_{B}": v for (L, B), v in ret.items()}
cand = [(L, B) for (L, B), v in ret.items() if B >= 100 and v >= 0.80]
L0, B0 = sorted(cand, key=lambda x: (-x[0], -x[1]))[0]
R["window"] = dict(L=L0, B=B0, retention_pooled=ret[(L0, B0)],
                   retention_per_subject_mean=float(np.mean([(rt[sub == s] >= L0 + B0).mean() for s in subs])),
                   retention_sub01=float((rt[(sub == 1)] >= L0 + B0).mean()))
print("window", R["window"], flush=True)

X, keep = window_features(D, L0, B0)
X = zscore_by_subject(X, sub)
Xk = X[keep]; sk = sub[keep]
ak = acc[keep]; ck = conf[keep]; rk = rt[keep]; chk = (D["choice"][keep] == 2).astype(int)
R["n_trials_analysed"] = int(keep.sum())
# RT-residualised features (label-free, per subject): removes linear dependence of every feature on log RT
lrt_k = np.log(rk); Xr = Xk.copy()
for s_ in np.unique(sk):
    m_ = sk == s_; A_ = np.c_[np.ones(m_.sum()), (lrt_k[m_] - lrt_k[m_].mean()) / lrt_k[m_].std()]
    Xr[m_] = Xk[m_] - A_[:, 1:] @ np.linalg.lstsq(A_, Xk[m_], rcond=None)[0][1:]
    Xr[m_] = (Xr[m_] - Xr[m_].mean(0)) / (Xr[m_].std(0) + 1e-12)


def summarize(a):
    v = np.array(list(a.values()))
    t = stats.ttest_1samp(v, 0.5)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=int(len(v)), t=float(t.statistic), p_t=float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2),
                n_above=int((v > 0.5).sum()), min=float(v.min()), max=float(v.max()))


def decode(name, y, valid, perms=True):
    out = {}
    a, pooled, sc = loso_auc(Xk, y, sk, valid)
    out["lr"] = summarize(a); out["lr"]["pooled_auc"] = float(pooled)
    out["per_subject_auc"] = {int(k): float(v) for k, v in a.items()}
    if perms:
        pm, pp, r = perm_test(Xk[valid], y[valid], sk[valid], out["lr"]["mean"], pooled, n=NPERM)
        out["lr"].update(p_perm_mean=float(pm), p_perm_pooled=float(pp), perm_mean_of_mean=float(r[:, 0].mean()), perm_sd_of_mean=float(r[:, 0].std()), perm_p95=float(np.percentile(r[:, 0], 95)))
    for C in (0.01, 1.0):
        a2, p2, _ = loso_auc(Xk, y, sk, valid, C=C); out[f"lr_C{C}"] = dict(mean=float(np.mean(list(a2.values()))), pooled=float(p2))
    a3, p3, _ = loso_auc(Xk, y, sk, valid, kind="lda"); out["lda"] = dict(mean=float(np.mean(list(a3.values()))), pooled=float(p3), sd=float(np.std(list(a3.values()), ddof=1)))
    w = within_subject_cv(Xk[valid], y[valid], sk[valid]); out["within_subject"] = summarize(w) if len(w) > 2 else {}
    ar, pr, _ = rt_only_auc({"RT": rk}, y, sk, valid)
    out["rt_only"] = dict(mean=float(np.mean(list(ar.values()))), pooled=float(pr))
    ar2, pr2, _ = loso_auc(Xr, y, sk, valid); out["rt_resid"] = summarize(ar2); out["rt_resid"]["pooled_auc"] = float(pr2)
    if perms:
        pm2, pp2, _ = perm_test(Xr[valid], y[valid], sk[valid], out["rt_resid"]["mean"], pr2, n=NPERM, seed0=5000); out["rt_resid"].update(p_perm_mean=float(pm2), p_perm_pooled=float(pp2))
    print(name, {k: out[k].get("mean", None) for k in ["lr", "lda", "within_subject", "rt_only"] if out[k]}, out["lr"].get("p_perm_mean"), flush=True)
    return out, sc


allv = np.ones(len(sk), bool)
R["RQ1"], sc_choice = decode("RQ1 choice", chk, allv)
R["RQ2"], sc_acc = decode("RQ2 accuracy", ak, allv)
# sub-bin spatiotemporal variant (supplementary)
Xb, _ = window_features(D, L0, B0, nbins=4); Xb = zscore_by_subject(Xb, sub)[keep]
for nm, y in (("choice", chk), ("accuracy", ak)):
    a, p, _ = loso_auc(Xb, y, sk, allv); R[f"suppl_4bin_{nm}"] = dict(mean=float(np.mean(list(a.values()))), pooled=float(p))

# ---------- confidence decoder (high vs low), reference + RQ4 state ----------
hl = low[keep] | high[keep]; yhl = high[keep].astype(int)
ahl, phl, _ = loso_auc(Xk, yhl, sk, hl)
R["conf_decoding"] = summarize(ahl); R["conf_decoding"]["pooled_auc"] = float(phl); R["conf_decoding"]["per_subject_auc"] = {int(k): float(v) for k, v in ahl.items()}
pm, pp, r = perm_test(Xk[hl], yhl[hl], sk[hl], R["conf_decoding"]["mean"], phl, n=NPERM)
R["conf_decoding"].update(p_perm_mean=float(pm), p_perm_pooled=float(pp))
ahr, phr, _ = loso_auc(Xr, yhl, sk, hl); R["conf_decoding"]["rt_resid"] = summarize(ahr); R["conf_decoding"]["rt_resid"]["pooled_auc"] = float(phr)
pm, pp, _ = perm_test(Xr[hl], yhl[hl], sk[hl], R["conf_decoding"]["rt_resid"]["mean"], phr, n=NPERM, seed0=7000); R["conf_decoding"]["rt_resid"].update(p_perm_mean=float(pm), p_perm_pooled=float(pp))
arc, prc, _ = rt_only_auc({"RT": rk}, yhl, sk, hl); R["conf_decoding"]["rt_only_mean"] = float(np.mean(list(arc.values())))
sc_conf = loso_scores(Xk, yhl, sk, train_ok=hl)  # applied to ALL analysed trials
print("conf decoding", R["conf_decoding"], flush=True)

# ---------- RQ3 ----------
lowk, highk = low[keep], high[keep]
cell = dict(CL=(ak == 1) & lowk, CH=(ak == 1) & highk, IL=(ak == 0) & lowk, IH=(ak == 0) & highk)
R["RQ3"] = dict(cells={k: int(v.sum()) for k, v in cell.items()})
sih = pd.Series(sk[cell["IH"]]).value_counts()
R["RQ3"]["IH_n_subjects"] = int(len(sih)); R["RQ3"]["IH_max_share"] = float(sih.max() / sih.sum())
feasible = cell["IH"].sum() >= 100 and len(sih) >= 12 and sih.max() / sih.sum() <= 0.20
R["RQ3"]["four_way_feasible"] = bool(feasible)


def pair(name, m, ypos):
    """m: rows used; ypos: boolean label (1 = positive class) on those rows."""
    y = ypos[m].astype(int)
    a, pooled, sc = loso_auc(Xk, ypos.astype(int), sk, m)
    pm, pp, r = perm_test(Xk[m], y, sk[m], np.mean(list(a.values())) if a else 0.5, pooled, n=NPERM)
    res = dict(n=int(m.sum()), n_pos=int(y.sum()), n_subjects=int(len(np.unique(sk[m]))), pooled_auc=float(pooled), p_perm_pooled=float(pp),
               subj_mean=float(np.mean(list(a.values()))) if a else None, n_subj_auc=len(a))
    print(name, res, flush=True)
    return res


R["RQ3"]["IH_vs_IL"] = pair("IH vs IL", cell["IH"] | cell["IL"], cell["IH"])
R["RQ3"]["IH_vs_CH"] = pair("IH vs CH", cell["IH"] | cell["CH"], cell["IH"])
R["RQ3"]["IL_vs_CL"] = pair("IL vs CL", cell["IL"] | cell["CL"], cell["IL"])
if feasible:
    lab = np.full(len(sk), -1); [lab.__setitem__(np.where(v)[0], i) for i, (k, v) in enumerate(cell.items())]
    m4 = lab >= 0
    pr = np.zeros((m4.sum(), 4)); X4, y4, s4 = Xk[m4], lab[m4], sk[m4]
    for s in np.unique(s4):
        te = s4 == s
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=300).fit(X4[~te], y4[~te]); pr[te] = clf.predict_proba(X4[te])
    R["RQ3"]["four_way"] = dict(balanced_acc=float(balanced_accuracy_score(y4, pr.argmax(1))), macro_auc=float(roc_auc_score(y4, pr, multi_class="ovr", average="macro")))
# two-stage: accuracy-decoder score, "correct-likeness" (higher = more correct-like)
inc = ak == 0
ts = {}
d_ih_il = sc_acc[cell["IH"]], sc_acc[cell["IL"]]
ts["auc_IH_gt_IL"] = float(roc_auc_score(np.r_[np.ones(len(d_ih_il[0])), np.zeros(len(d_ih_il[1]))], np.r_[d_ih_il[0], d_ih_il[1]]))
rng = np.random.default_rng(7); mm = cell["IH"] | cell["IL"]; idx = np.where(mm)[0]; labs = cell["IH"][idx].astype(int); ss_ = sk[idx]; scs = sc_acc[idx]
cnt_ge = 0
for i in range(10000):
    yp = labs.copy()
    for s in np.unique(ss_):
        q = np.where(ss_ == s)[0]; yp[q] = rng.permutation(labs[q])
    if roc_auc_score(yp, scs) >= ts["auc_IH_gt_IL"]: cnt_ge += 1
ts["p_perm_one_sided"] = (cnt_ge + 1) / 10001
ts["mean_score_IH"] = float(d_ih_il[0].mean()); ts["mean_score_IL"] = float(d_ih_il[1].mean())
ts["mean_score_CH"] = float(sc_acc[cell["CH"]].mean()); ts["mean_score_CL"] = float(sc_acc[cell["CL"]].mean())
rho_c, rho_i = [], []
for s in np.unique(sk):
    for msk, store in (((sk == s) & (ak == 1), rho_c), ((sk == s) & (ak == 0), rho_i)):
        if msk.sum() >= 10 and len(np.unique(ck[msk])) > 2:
            store.append(np.arctanh(np.clip(stats.spearmanr(ck[msk], sc_acc[msk])[0], -0.99, 0.99)))
for nm, st in (("correct", rho_c), ("incorrect", rho_i)):
    t = stats.ttest_1samp(st, 0); ts[f"fisher_z_{nm}"] = dict(mean=float(np.mean(st)), t=float(t.statistic), p=float(t.pvalue), n=len(st))
R["RQ3"]["two_stage"] = ts; print("two-stage", ts, flush=True)

# ---------- RQ4 ----------
orig = D["orig"][keep]; run = D["run"][keep]
key = {(s, r_, o): i for i, (s, r_, o) in enumerate(zip(sk, run, orig))}
pairs = [(i, key[(s, r_, o + 1)]) for i, (s, r_, o) in enumerate(zip(sk, run, orig)) if (s, r_, o + 1) in key and o > 0]
a_i = np.array([p[0] for p in pairs]); b_i = np.array([p[1] for p in pairs])
dirk = D["dotdirection"][keep]; chn = D["choice"][keep]
z_ = lambda v: (v - v.mean()) / v.std()
zs = np.zeros(len(sk)); zc = np.zeros(len(sk))
for s in np.unique(sk):
    m = sk == s; zs[m] = z_(sc_conf[m]); zc[m] = z_(ck[m].astype(float))
T = pd.DataFrame(dict(sub=sk[a_i], rep=(chn[a_i] == chn[b_i]).astype(int), stimrep=(dirk[a_i] == dirk[b_i]).astype(int), accN=ak[a_i], accN1=ak[b_i],
                      S=zs[a_i], Cn=zc[a_i], lrtN=np.log(rk[a_i]), lrtN1=np.log(rk[b_i])))
R["RQ4"] = dict(n_pairs=int(len(T)), n_subjects=int(T["sub"].nunique()), rep_rate=float(T.rep.mean()), stimrep_rate=float(T.stimrep.mean()))


def gee(formula, fam):
    m = sm.GEE.from_formula(formula, groups="sub", data=T, family=fam, cov_struct=sm.cov_struct.Exchangeable()).fit()
    return {k: dict(b=float(m.params[k]), se=float(m.bse[k]), p=float(m.pvalues[k])) for k in m.params.index}


B_ = sm.families.Binomial(); G_ = sm.families.Gaussian()
R["RQ4"]["rep_neural"] = gee("rep ~ S + stimrep + accN + lrtN", B_)
R["RQ4"]["rep_behav"] = gee("rep ~ Cn + stimrep + accN + lrtN", B_)
R["RQ4"]["lrt_neural"] = gee("lrtN1 ~ S + lrtN + accN", G_)
R["RQ4"]["lrt_behav"] = gee("lrtN1 ~ Cn + lrtN + accN", G_)
R["RQ4"]["acc_neural"] = gee("accN1 ~ S + accN + lrtN", B_)
R["RQ4"]["acc_behav"] = gee("accN1 ~ Cn + accN + lrtN", B_)
R["RQ4"]["rep_joint"] = gee("rep ~ S + Cn + stimrep + accN + lrtN", B_)
print("RQ4", {k: v.get("S", v.get("Cn")) for k, v in R["RQ4"].items() if isinstance(v, dict) and ("S" in v or "Cn" in v)}, flush=True)

# ---------- window sensitivity grid ----------
def grid_one(L, B):
    Xg, kg = window_features(D, L, B); Xg = zscore_by_subject(Xg, sub)[kg]; sg = sub[kg]; r = {}
    for nm, y, v in (("choice", (D["choice"][kg] == 2).astype(int), None), ("accuracy", acc[kg], None), ("confidence", (conf[kg] >= 7).astype(int), (conf[kg] <= 3) | (conf[kg] >= 7))):
        v = np.ones(len(y), bool) if v is None else v
        a, p, _ = loso_auc(Xg, y, sg, v); r[nm] = (float(np.mean(list(a.values()))), float(p))
    return (L, B), r


gr = Parallel(n_jobs=8)(delayed(grid_one)(L, B) for L in Ls for B in Bs if ret[(L, B)] >= 0.5)
R["grid"] = {f"{L}_{B}": r for (L, B), r in gr}
np.savez("results_scores.npz", sc_choice=sc_choice, sc_acc=sc_acc, sc_conf=sc_conf, sub=sk, acc=ak, conf=ck, choice=chk, rt=rk)
json.dump(R, open("results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("FINISHED", flush=True)
