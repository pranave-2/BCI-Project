"""A7: stimulus-locked, causally-available evaluation. At each decision time tau after stimulus onset the monitor sees only [tau-300, tau) ms of EEG (mean amplitude per channel) and predicts choice, accuracy or confidence.
It is evaluated on trials in which no response had yet been given by tau + delta (delta = 0 or 150 ms processing latency), i.e. the prediction is genuinely 'in advance' of the response. No response-locked information is used."""
import sys, json, numpy as np
from scipy import stats
sys.path.insert(0, "v2"); from common2 import *
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 300
D = load(); sub_all = D["sub"]; rt_all = D["RT"]; stim = D["stim"]       # (n, 53, 325): sample j at about -100 + 4 j ms after stimulus onset
taus = [400, 500, 600, 700, 800, 900, 1000, 1100]; deltas = [0, 150]; R = dict(taus=taus, deltas=deltas, window_ms=300, n_perm=NPERM, note="sample j is at (-100 + 4 j) ms (+0-3 ms jitter from decimation alignment) relative to stimulus onset")
conf_all = D["confidence"]; acc_all = D["accuracy"]; ch_all = (D["choice"] == 2).astype(int)
t0 = 0
for delta in deltas:
    for tau in taus:
        m = rt_all > tau + delta; j0 = (tau - 300 + 100) // 4; j1 = (tau + 100) // 4
        X = zscore_within(stim[m][:, :, j0:j1].mean(2).astype(np.float64), sub_all[m]); sub = sub_all[m]; hl = (conf_all[m] <= 3) | (conf_all[m] >= 7)
        for ti, (name, y, valid) in enumerate((("choice", ch_all[m], np.ones(m.sum(), bool)), ("accuracy", acc_all[m].astype(int), np.ones(m.sum(), bool)), ("confidence", (conf_all[m] >= 7).astype(int), hl))):
            a, p, sc = loso_auc(X, y, sub, valid); o = summarize_auc(a); o.update(pooled_auc=p, n_trials=int(valid.sum()), n_participants=len(a))
            tt = stats.ttest_1samp(list(a.values()), 0.5); o["p_t"] = float(tt.pvalue / 2 if tt.statistic > 0 else 1 - tt.pvalue / 2)      # descriptive one-sided t-test over participants
            R.setdefault(f"delta{delta}", {}).setdefault(name, {})[str(tau)] = o
            print(delta, tau, name, round(o["mean"], 4), round(o["p_t"], 4), o["n_trials"], flush=True)
for dk in [f"delta{d}" for d in deltas]:
    for name in ("choice", "accuracy", "confidence"):
        ps = [R[dk][name][str(t)]["p_t"] for t in taus]; adj = holm(ps)
        for t, a_ in zip(taus, adj): R[dk][name][str(t)]["p_holm_over_tau"] = a_
json.dump(R, open("v2/results/a7_stimlocked.json", "w"), indent=1); print("FINISHED a7", flush=True)
