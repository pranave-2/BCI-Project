"""A13: window sensitivity grid (mean held-out-participant AUC, canonical normalization on each grid window's own retained trials; descriptive, not used for selection)."""
import sys, json, numpy as np
sys.path.insert(0, "v2"); from common2 import *
D = load(); rt = D["RT"]; Ls = [200, 300, 400, 500, 600, 700, 800]; Bs = [0, 50, 100, 150, 200, 250, 300, 350]; R = dict(L=Ls, B=Bs, auc={n: [[None] * len(Bs) for _ in Ls] for n in ("choice", "accuracy", "confidence")}, n_trials=[[None] * len(Bs) for _ in Ls])
for i, L in enumerate(Ls):
    for j, B in enumerate(Bs):
        keep = rt >= L + B; R["n_trials"][i][j] = int(keep.sum())
        if keep.mean() < 0.5: continue
        sub = D["sub"][keep]; X = zscore_within(window_mean(D, L, B, rows=keep), sub)
        for n, (y, v) in targets(D, keep).items():
            a, p, _ = loso_auc(X, y, sub, v); R["auc"][n][i][j] = mean_auc(a)
    print(L, flush=True)
json.dump(R, open("v2/results/a13_grid.json", "w"), indent=1); print("FINISHED a13", flush=True)
