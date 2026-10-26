"""A8: deployment-setting sensitivity. Compares the canonical (transductive) normalization with variants that do not use the held-out participant's data
(training-only pooled statistics), use a separate unlabeled calibration run, or use only earlier trials (causal running statistics); and RT regression with training-only coefficients."""
import sys, json, numpy as np
sys.path.insert(0, "v2"); from common2 import *
from scipy import stats
D = load(); keep = D["RT"] >= L0 + B0; sub = D["sub"][keep]; run = D["run"][keep]; orig = D["orig"][keep]; rt = D["RT"][keep]; lrt = np.log(rt)
Xraw = window_mean(D, rows=keep); Xall = zscore_within(window_mean(D), D["sub"])[keep]; Xcan = zscore_within(Xraw, sub); T = targets(D, keep)
Bflex = rt_basis(lrt, sub, 5); Blin = rt_basis(lrt, sub, kind="linear")
R = dict(descriptions={
    "canonical": "per-participant z-score over the participant's retained trials (target participant's unlabeled batch used; labels never used)",
    "all_trials_batch": "per-participant z-score over ALL valid trials of the participant, including excluded fast trials (normalization of the first submitted manuscript's Table 4)",
    "training_only_pooled": "no target-participant data used: every feature standardized with the mean/SD of the pooled training participants' raw features",
    "separate_calibration_run": "held-out participant's run r standardized with statistics of the participant's OTHER run (unlabeled calibration session); training participants as canonical",
    "causal_running": "each trial standardized with the mean/SD of the participant's earlier retained trials in the same run (>= 20 earlier trials required; trials with fewer are dropped from training and testing)",
    "rt_linear_transductive": "canonical features residualized on the participant's own z(ln RT) (needs completed RTs of the target participant)",
    "rt_linear_training_only": "residualization coefficients estimated on training participants only and applied to the held-out participant (no target RT regression)",
    "rt_flex_training_only": "as above with the 5-df natural-spline RT basis"})
def finish(name, tn, sc, y, valid, ss):
    a = subj_auc(sc[valid], y[valid], ss[valid]); o = summarize_auc(a); o["pooled_auc"] = float(roc_auc_score(y[valid], sc[valid])); o["per_participant"] = a; return o
res = {}
subs = np.unique(sub)
for tn, (y, valid) in T.items():
    res[tn] = {}
    a, p, sc = loso_auc(Xcan, y, sub, valid); res[tn]["canonical"] = dict(summarize_auc(a), pooled_auc=p, per_participant=a)
    a, p, sc = loso_auc(Xall, y, sub, valid); res[tn]["all_trials_batch"] = dict(summarize_auc(a), pooled_auc=p, per_participant=a)
    # training-only pooled standardization
    sc = np.full(len(y), np.nan)
    for s in subs:
        te = sub == s; tr = (~te) & valid; mu, sd = Xraw[tr].mean(0), Xraw[tr].std(0) + 1e-12
        sc[te] = make_clf().fit((Xraw[tr] - mu) / sd, y[tr]).decision_function((Xraw[te] - mu) / sd)
    res[tn]["training_only_pooled"] = finish("", tn, sc, y, valid, sub)
    # separate calibration run
    sc = np.full(len(y), np.nan)
    for s in subs:
        te = sub == s; tr = (~te) & valid; clf = make_clf().fit(Xcan[tr], y[tr]); Xt = np.zeros((te.sum(), Xraw.shape[1])); idx = np.where(te)[0]; rr = run[te]
        for r_ in np.unique(rr):
            other = idx[run[idx] != r_]; cur = rr == r_
            if len(other) >= 30: mu, sd = Xraw[other].mean(0), Xraw[other].std(0) + 1e-12
            else: mu, sd = Xraw[tr].mean(0), Xraw[tr].std(0) + 1e-12
            Xt[cur] = (Xraw[idx[cur]] - mu) / sd
        sc[te] = clf.decision_function(Xt)
    res[tn]["separate_calibration_run"] = finish("", tn, sc, y, valid, sub)
    # causal running normalization (training and test)
    Xc = np.full_like(Xraw, np.nan); ok = np.zeros(len(y), bool)
    for s in subs:
        for r_ in np.unique(run[sub == s]):
            idx = np.where((sub == s) & (run == r_))[0]; idx = idx[np.argsort(orig[idx])]
            for k, i in enumerate(idx):
                if k >= 20: prev = Xraw[idx[:k]]; Xc[i] = (Xraw[i] - prev.mean(0)) / (prev.std(0) + 1e-12); ok[i] = True
    v2 = valid & ok; sc = np.full(len(y), np.nan)
    for s in subs:
        te = (sub == s) & v2; tr = (sub != s) & v2
        if te.sum(): sc[te] = make_clf().fit(Xc[tr], y[tr]).decision_function(Xc[te])
    a = subj_auc(sc[v2], y[v2], sub[v2]); res[tn]["causal_running"] = dict(summarize_auc(a), pooled_auc=float(roc_auc_score(y[v2], sc[v2])), n_trials=int(v2.sum()), per_participant=a)
    # RT regression with training-only coefficients
    for nm, Bm in (("rt_linear", Blin), ("rt_flex", Bflex)):
        sc = np.full(len(y), np.nan)
        for s in subs:
            te = sub == s; tr = (~te)
            A = np.column_stack([np.ones(tr.sum()), Bm[tr]]); coef = np.linalg.lstsq(A, Xcan[tr], rcond=None)[0]                  # pooled over training participants (features already z-scored within participant)
            Xr_tr = Xcan[tr] - Bm[tr] @ coef[1:]; Xr_te = Xcan[te] - Bm[te] @ coef[1:]
            sd = Xr_tr.std(0) + 1e-12; mu = Xr_tr.mean(0)
            sc[te] = make_clf().fit(((Xr_tr - mu) / sd)[valid[tr]], y[tr][valid[tr]]).decision_function((Xr_te - mu) / sd)
        res[tn][f"{nm}_training_only"] = finish("", tn, sc, y, valid, sub)
    # differences vs canonical (paired, participant-level) for the RT-free variants
    for k in ("all_trials_batch", "training_only_pooled", "separate_calibration_run", "causal_running"):
        base = res[tn]["canonical"]["per_participant"]; cur = res[tn][k]["per_participant"]; ks = sorted(set(base) & set(cur)); d = np.array([cur[q] - base[q] for q in ks])
        res[tn][k]["diff_vs_canonical"] = dict(mean=float(d.mean()), ci95=boot_mean_ci(d), n=len(d))
    print(tn, {k: round(v["mean"], 4) for k, v in res[tn].items()}, flush=True)
R["results"] = res
json.dump(R, open("v2/results/a8_normalization.json", "w"), indent=1); print("FINISHED a8", flush=True)
