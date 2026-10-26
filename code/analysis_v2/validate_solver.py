"""Validate the exact Newton solver against scikit-learn's lbfgs LogisticRegression on the real data (coefficients, scores, AUCs)."""
import sys, json, time, numpy as np
sys.path.insert(0, "v2"); from common2 import *
D = load(); keep = D["RT"] >= L0 + B0; sub = D["sub"][keep]; X = zscore_within(window_mean(D, rows=keep), sub); T = targets(D, keep); out = {}
for name, (y, valid) in T.items():
    Xv, yv, sv = X[valid], y[valid], sub[valid]
    t = time.time(); f = loso_scores(Xv, yv, sv); t_fast = time.time() - t
    t = time.time(); g = loso_scores(Xv, yv, sv, make=make_clf_sklearn); t_sk = time.time() - t
    a_f = subj_auc(f, yv, sv); a_s = subj_auc(g, yv, sv)
    full_f = FastLR().fit(Xv, yv).beta; sk = make_clf_sklearn().fit(Xv, yv); full_s = np.r_[sk.coef_.ravel(), sk.intercept_]
    out[name] = dict(mean_auc_fast=mean_auc(a_f), mean_auc_sklearn=mean_auc(a_s), abs_diff_mean_auc=abs(mean_auc(a_f) - mean_auc(a_s)), max_abs_diff_participant_auc=max(abs(a_f[k] - a_s[k]) for k in a_f),
                     score_corr=float(np.corrcoef(f, g)[0, 1]), coef_max_abs_diff_full_data=float(np.max(np.abs(full_f - full_s))), coef_max_abs_full=float(np.max(np.abs(full_s))), seconds_fast=t_fast, seconds_sklearn=t_sk)
    print(name, out[name], flush=True)
json.dump(out, open("v2/results/solver_validation.json", "w"), indent=1)
