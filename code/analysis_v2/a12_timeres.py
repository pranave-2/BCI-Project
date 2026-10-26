"""A12: time-resolved response-locked decoding (100 ms windows, 50 ms step, windows from -750 to 0 ms relative to ~response) on the retained trials, canonical normalization,
with and without flexible RT residualization; one-sided one-sample t-test of participant AUCs per window (descriptive) (same procedure as the primary test) and Holm correction over the 14 windows per target/variant."""
import sys, json, numpy as np
from scipy import stats
sys.path.insert(0, "v2"); from common2 import *
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 500
D = load(); keep = D["RT"] >= 750; sub = D["sub"][keep]; lrt = np.log(D["RT"][keep]); pre = D["pre"][keep]; T = targets(D, keep); Bflex = rt_basis(lrt, sub, 5)
starts = list(range(-750, -99, 50)); R = dict(starts=starts, win_ms=100, n_perm=NPERM, note="window k covers [start, start+100) ms relative to 4b (b = floor(tresp/4)); window centre = start + 50")
for ti, name in enumerate(T):
    R[name] = {"raw": [], "resid_flex": []}
for wi, s0 in enumerate(starts):
    a0 = (s0 + 1200) // 4; Xw = zscore_within(pre[:, :, a0:a0 + 25].mean(2).astype(np.float64), sub); Xr = residualize(Xw, sub, Bflex)
    for ti, (name, (y, valid)) in enumerate(T.items()):
        for vi, (vn, XX) in enumerate((("raw", Xw), ("resid_flex", Xr))):
            a, p, _ = loso_auc(XX, y, sub, valid); m = mean_auc(a); se = float(np.std(list(a.values()), ddof=1) / np.sqrt(len(a)))
            tt = stats.ttest_1samp(list(a.values()), 0.5); pt_ = float(tt.pvalue / 2 if tt.statistic > 0 else 1 - tt.pvalue / 2)      # descriptive one-sided t-test over participants (no permutations: 84 analyses)
            R[name][vn].append(dict(start=s0, mean=m, se=se, p_t=pt_, pooled=p))
    print(s0, {n: round(R[n]["raw"][-1]["mean"], 3) for n in T}, flush=True)
for name in T:
    for vn in ("raw", "resid_flex"):
        adj = holm([w["p_t"] for w in R[name][vn]])
        for w, a_ in zip(R[name][vn], adj): w["p_holm"] = a_
json.dump(R, open("v2/results/a12_timeres.json", "w"), indent=1); print("FINISHED a12", flush=True)
