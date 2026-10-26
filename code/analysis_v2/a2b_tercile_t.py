"""A2b: participant-level RT-tercile-stratified AUCs of the uncorrected out-of-participant scores (from a2) with the one-sided one-sample t-test used in the first manuscript, reported next to the permutation test."""
import sys, json, numpy as np
sys.path.insert(0, "v2"); from common2 import *
from scipy import stats
P = np.load("v2/results/scores_primary.npz"); sub = P["sub"]; rt = P["rt"]; lrt = np.log(rt); conf = P["conf"]; acc = P["acc"]; ch = P["choice"]
tercile = np.zeros(len(sub), int)
for s in np.unique(sub):
    m = np.where(sub == s)[0]; tercile[m] = np.minimum((np.argsort(np.argsort(lrt[m])) * 3) // len(m), 2)
R = {}
for nm, y, valid in (("choice", ch, np.ones(len(ch), bool)), ("accuracy", acc.astype(int), np.ones(len(ch), bool)), ("confidence", (conf >= 7).astype(int), (conf <= 3) | (conf >= 7))):
    sc = P[f"{nm}_raw"]; per = []; byq = [[], [], []]
    for s in np.unique(sub[valid]):
        vals = []
        for q in range(3):
            m = np.where((sub == s) & valid & (tercile == q))[0]
            if len(np.unique(y[m])) == 2 and min((y[m] == 0).sum(), (y[m] == 1).sum()) >= 5: a = roc_auc_score(y[m], sc[m]); vals.append(a); byq[q].append(a)
        if vals: per.append(np.mean(vals))
    t = stats.ttest_1samp(per, 0.5); R[nm] = dict(mean=float(np.mean(per)), n=len(per), t=float(t.statistic), p_t_onesided=float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2), ci95=boot_mean_ci(per), by_tercile=[float(np.mean(b)) for b in byq])
    print(nm, R[nm])
json.dump(R, open("v2/results/a2b_tercile_t.json", "w"), indent=1)
