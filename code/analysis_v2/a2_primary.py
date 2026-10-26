"""A2: primary decoding (choice, accuracy, confidence) with permutation inference, and the response-time control suite.
Every p-value uses the same code path: 1000 label permutations of the whole LOSO procedure, p = (1 + #{null >= obs}) / (1 + 1000)."""
import sys, json, time, numpy as np
sys.path.insert(0, "v2"); from common2 import *
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
D = load(); keep = D["RT"] >= L0 + B0; sub = D["sub"][keep]; rt = D["RT"][keep]; lrt = np.log(rt)
X = zscore_within(window_mean(D, rows=keep), sub); T = targets(D, keep)
R = dict(n_perm=NPERM, n_analysed=int(keep.sum())); S = {}
Blin = rt_basis(lrt, sub, kind="linear"); Bflex = rt_basis(lrt, sub, df=5, kind="spline")
Xlin = residualize(X, sub, Blin); Xflex = residualize(X, sub, Bflex)
decile = np.zeros(len(sub), int)
for s in np.unique(sub):
    m = np.where(sub == s)[0]; decile[m] = np.minimum((np.argsort(np.argsort(lrt[m])) * 10) // len(m), 9)
tercile = np.zeros(len(sub), int)
for s in np.unique(sub):
    m = np.where(sub == s)[0]; tercile[m] = np.minimum((np.argsort(np.argsort(lrt[m])) * 3) // len(m), 2)
strata_dec = sub * 100 + decile                     # participant x RT decile
t0 = time.time(); seed = 100000
for ti, (name, (y, valid)) in enumerate(T.items()):
    out = {}; seed0 = 100000 + 10000 * ti
    # --- uncorrected primary decoder
    a, pooled, sc = loso_auc(X, y, sub, valid); S[f"{name}_raw"] = sc
    o = summarize_auc(a); o.update(pooled_auc=pooled, per_participant={int(k): v for k, v in a.items()})
    pt = perm_test(X, y, sub, valid, o["mean"], pooled, n=NPERM, seed0=seed0); S[f"{name}_perm_null_means"] = np.array(pt.pop("null_means")); S[f"{name}_perm_null_pooled"] = np.array(pt.pop("null_pooled")); o["perm"] = pt
    for C in (0.01, 1.0):
        a2, p2, _ = loso_auc(X, y, sub, valid, C=C); o[f"C{C}"] = dict(mean=mean_auc(a2), pooled=p2)
    sc_lda = np.full(len(y), np.nan); idx = np.where(valid)[0]
    for s in np.unique(sub[idx]):
        te = idx[sub[idx] == s]; tr = idx[sub[idx] != s]; sc_lda[te] = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], y[tr]).decision_function(X[te])
    a3 = subj_auc(sc_lda[valid], y[valid], sub[valid]); o["lda"] = dict(mean=mean_auc(a3), sd=float(np.std(list(a3.values()), ddof=1)), pooled=float(roc_auc_score(y[valid], sc_lda[valid])))
    wcv = {}
    for s in np.unique(sub[idx]):
        m = idx[sub[idx] == s]; ys = y[m]
        if min((ys == 0).sum(), (ys == 1).sum()) < 10: continue
        sc_ = np.zeros(len(m))
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(X[m], ys): sc_[te] = make_clf().fit(X[m][tr], ys[tr]).decision_function(X[m][te])
        wcv[int(s)] = float(roc_auc_score(ys, sc_))
    o["within_participant_10fold"] = dict(mean=float(np.mean(list(wcv.values()))), n=len(wcv))
    out["uncorrected"] = o
    # --- RT only (linear and flexible) --------------------------------------------------------------------------------------------------
    for nm, Bm in (("rt_only_linear", Blin), ("rt_only_flex", Bflex)):
        a_, p_, sc_ = loso_auc(Bm, y, sub, valid); S[f"{name}_{nm}"] = sc_; out[nm] = dict(mean=mean_auc(a_), pooled=p_, per_participant={int(k): v for k, v in a_.items()})
    # --- residualized (transductive; label-free) ------------------------------------------------------------------------------------------
    for nm, Xr in (("resid_linear", Xlin), ("resid_flex", Xflex)):
        a_, p_, sc_ = loso_auc(Xr, y, sub, valid); S[f"{name}_{nm}"] = sc_; oo = summarize_auc(a_); oo.update(pooled_auc=p_, per_participant={int(k): v for k, v in a_.items()})
        pt = perm_test(Xr, y, sub, valid, oo["mean"], p_, n=NPERM, seed0=seed0 + 1000 * (1 if nm == "resid_linear" else 2)); pt.pop("null_means"); pt.pop("null_pooled"); oo["perm"] = pt; out[nm] = oo
    # --- EEG + flexible RT versus RT only: incremental prediction, RT-stratified permutation -------------------------------------------
    XB = np.hstack([X, Bflex]); a_, p_, sc_ = loso_auc(XB, y, sub, valid); S[f"{name}_eeg_plus_rt"] = sc_
    a_rt = {int(k): v for k, v in subj_auc(S[f"{name}_rt_only_flex"][valid], y[valid], sub[valid]).items()}
    common = sorted(set(a_) & set(a_rt)); dif = np.array([a_[k] - a_rt[k] for k in common]); obs_delta = float(mean_auc(a_) - mean_auc(a_rt))
    def _inc(seed):
        rng = np.random.default_rng(seed); yp = y.copy()
        for g in np.unique(strata_dec[valid]):
            ii = np.where((strata_dec == g) & valid)[0]; yp[ii] = rng.permutation(y[ii])
        a1, _, _ = loso_auc(XB, yp, sub, valid); a0, _, _ = loso_auc(Bflex, yp, sub, valid); return mean_auc(a1) - mean_auc(a0)
    nulls = np.array(Parallel(n_jobs=NJOBS)(delayed(_inc)(seed0 + 5000 + i) for i in range(NPERM)))
    out["increment_eeg_over_rt"] = dict(auc_eeg_plus_rt=mean_auc(a_), auc_rt_only=mean_auc(a_rt), delta=obs_delta, delta_ci95_boot=boot_mean_ci(dif), n_participants_improved=int((dif > 0).sum()), n=len(dif),
                                         p_perm_rt_stratified=float((1 + np.sum(nulls >= obs_delta)) / (1 + NPERM)), null_mean=float(nulls.mean()), null_sd=float(nulls.std(ddof=1)), null_p95=float(np.percentile(nulls, 95)),
                                         per_participant_eeg_plus_rt={int(k): v for k, v in a_.items()})
    # --- RT-tercile stratified AUC of the uncorrected scores (permutation keeps the label-RT relation by shuffling within participant x tercile) ------------------
    def strat_auc(score, yy):
        per = []
        for s in np.unique(sub[valid]):
            vals = []
            for q in range(3):
                m = np.where((sub == s) & valid & (tercile == q))[0]
                if len(np.unique(yy[m])) == 2 and min((yy[m] == 0).sum(), (yy[m] == 1).sum()) >= 5: vals.append(roc_auc_score(yy[m], score[m]))
            if vals: per.append(np.mean(vals))
        return float(np.mean(per)), per
    obs_s, per = strat_auc(sc, y)
    strata_ter = sub * 10 + tercile
    def _tp(seed):
        rng = np.random.default_rng(seed); yp = y.copy()
        for g in np.unique(strata_ter[valid]):
            ii = np.where((strata_ter == g) & valid)[0]; yp[ii] = rng.permutation(y[ii])
        _, _, scp = loso_auc(X, yp, sub, valid); return strat_auc(scp, yp)[0]
    nulls = np.array(Parallel(n_jobs=NJOBS)(delayed(_tp)(seed0 + 8000 + i) for i in range(NPERM)))
    out["rt_tercile_stratified"] = dict(mean=obs_s, n=len(per), ci95_boot=boot_mean_ci(per), p_perm=float((1 + np.sum(nulls >= obs_s)) / (1 + NPERM)), null_mean=float(nulls.mean()), null_p95=float(np.percentile(nulls, 95)),
                                        by_tercile=[float(np.mean([roc_auc_score(y[(sub == s) & valid & (tercile == q)], sc[(sub == s) & valid & (tercile == q)]) for s in np.unique(sub[valid])
                                                                   if len(np.unique(y[(sub == s) & valid & (tercile == q)])) == 2])) for q in range(3)])
    # --- matched-trial sensitivity (equalize the classes within participant x RT decile [x stimulus direction for choice]) ---------------------------------------
    dirn = D["dotdirection"][keep]
    def matched_mask(seed):
        rng = np.random.default_rng(seed); sel = np.zeros(len(y), bool)
        cells = (sub * 1000 + decile * 10 + ((dirn == 180).astype(int) if name == "choice" else 0))
        for g in np.unique(cells[valid]):
            ii = np.where((cells == g) & valid)[0]; i1 = ii[y[ii] == 1]; i0 = ii[y[ii] == 0]; k = min(len(i1), len(i0))
            if k > 0: sel[rng.choice(i1, k, replace=False)] = True; sel[rng.choice(i0, k, replace=False)] = True
        return sel
    ms = []; sel1 = None
    for dr in range(10):
        sel = matched_mask(777 + dr); a_, p_, _ = loso_auc(X, y, sub, sel); ms.append((mean_auc(a_), p_, int(sel.sum()), len(a_)))
        if dr == 0: sel1 = sel; a_m1 = mean_auc(a_)
    pt = perm_test(X, y, sub, sel1, a_m1, ms[0][1], n=NPERM, seed0=seed0 + 9000, strata=strata_dec if name != "choice" else (sub * 1000 + decile * 10 + (dirn == 180).astype(int)))
    pt.pop("null_means"); pt.pop("null_pooled")
    out["matched"] = dict(auc_mean_over_draws=float(np.mean([m[0] for m in ms])), auc_sd_over_draws=float(np.std([m[0] for m in ms], ddof=1)), n_trials_per_draw=ms[0][2], n_participants=ms[0][3],
                          matched_on="participant x RT decile" + (" x stimulus direction" if name == "choice" else ""), perm_draw1=pt, auc_draw1=a_m1)
    R[name] = out
    print(name, "raw", round(out["uncorrected"]["mean"], 4), out["uncorrected"]["perm"]["p_mean"], "| lin", round(out["resid_linear"]["mean"], 4), out["resid_linear"]["perm"]["p_mean"], "| flex", round(out["resid_flex"]["mean"], 4),
          out["resid_flex"]["perm"]["p_mean"], "| inc", round(out["increment_eeg_over_rt"]["delta"], 4), out["increment_eeg_over_rt"]["p_perm_rt_stratified"], "| terc", round(out["rt_tercile_stratified"]["mean"], 4),
          out["rt_tercile_stratified"]["p_perm"], "| matched", round(out["matched"]["auc_mean_over_draws"], 4), out["matched"]["perm_draw1"]["p_mean"], round(time.time() - t0), "s", flush=True)
# Holm across the three primary targets for each variant
for var in ("uncorrected", "resid_linear", "resid_flex"):
    ps = [R[n][var]["perm"]["p_mean"] for n in T]; adj = holm(ps)
    for n, a_ in zip(T, adj): R[n][var]["perm"]["p_holm_3targets"] = a_
np.savez("v2/results/scores_primary.npz", sub=sub, run=D["run"][keep], orig=D["orig"][keep], rt=rt, acc=D["accuracy"][keep], conf=D["confidence"][keep], choice=T["choice"][0], dotdirection=dirn, **S)
json.dump(R, open("v2/results/a2_primary.json", "w"), indent=1); print("FINISHED a2", flush=True)
