"""A1: behaviour, window grid, retention counts and the window-selection rule (retention only), per-participant retention and class counts, trial/pair manifests."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "v2"); from common2 import *
from scipy import stats
D = load(); sub = D["sub"]; N = len(sub); rt = D["RT"]; subs = np.unique(sub)
R = {}
R["n_subjects"] = int(len(subs)); R["n_runs"] = int(len(set(zip(D["sub"], D["run"])))); R["n_trials_total"] = int(N)
R["channels_common"] = D["channel_names"]; R["channel_indices_1based"] = D["channels"].tolist(); R["n_channels_common"] = int(len(D["channels"]))
R["channels_excluded_in_any_participant"] = [ELEC[c - 1] for c in sorted(set(range(1, 65)) - set(D["channels"].tolist()))]
acc, conf = D["accuracy"], D["confidence"]
df = pd.DataFrame(dict(sub=sub, acc=acc, conf=conf, rt=rt))
g = df.groupby("sub")
R["beh"] = dict(acc_mean=float(g.acc.mean().mean()), acc_sd=float(g.acc.mean().std()), rt_mean=float(rt.mean()), rt_sd=float(rt.std()), rt_median=float(np.median(rt)),
                conf_correct=float(conf[acc == 1].mean()), conf_incorrect=float(conf[acc == 0].mean()),
                rt_conf_rho=float(np.mean([stats.spearmanr(rt[sub == s], conf[sub == s])[0] for s in subs])), rt_acc_rho=float(np.mean([stats.spearmanr(rt[sub == s], acc[sub == s])[0] for s in subs])),
                right_choice_frac=float((D["choice"] == 2).mean()), trials_per_subject=dict(min=int(df.groupby("sub").size().min()), max=int(df.groupby("sub").size().max())))
# window grid (retention only) and the selection rule, stated step by step
Ls = [200, 300, 400, 500, 600, 700, 800]; Bs = [0, 50, 100, 150, 200, 250, 300, 350]
ret = {(L, B): float((rt >= L + B).mean()) for L in Ls for B in Bs}
R["grid_L"] = Ls; R["grid_B"] = Bs
R["retention_grid"] = [[ret[(L, B)] for B in Bs] for L in Ls]; R["retention_trials_grid"] = [[int((rt >= L + B).sum()) for B in Bs] for L in Ls]
cand = sorted([(L, B) for (L, B), v in ret.items() if B >= 100 and v >= 0.80], key=lambda x: (-x[0], -x[1]))
R["rule"] = dict(text="Among grid windows with B >= 100 ms and pooled retention >= 80 percent, choose the largest L; ties are broken by the largest B.",
                 n_candidates=len(cand), top5=[dict(L=L, B=B, retention=ret[(L, B)], n_trials=int((rt >= L + B).sum())) for L, B in cand[:5]],
                 L600_row=[dict(B=B, retention=ret[(600, B)], n_trials=int((rt >= 600 + B).sum()), meets_80=bool(ret[(600, B)] >= .8)) for B in Bs],
                 L700_row=[dict(B=B, retention=ret[(700, B)], meets_80=bool(ret[(700, B)] >= .8) and B >= 100) for B in Bs])
L, B = cand[0]; assert (L, B) == (L0, B0)
R["window"] = dict(L=L, B=B, n_retained=int((rt >= L + B).sum()), retention=ret[(L, B)])
s, e = win_idx(L, B)
R["window"].update(epoch_samples=[int(s), int(e)], note="window = decimated samples [b-188, b-38) with b = floor(tresp/4); i.e. 1 kHz time [4b-752, 4b-152) ms; 4b is 0-3 ms before the response, so the window ends 152-155 ms before the response and spans about -755 to -152 ms")
keep = rt >= L + B
# retained vs excluded trials
R["retained_vs_excluded"] = dict(acc_ret=float(acc[keep].mean()), acc_exc=float(acc[~keep].mean()), conf_ret=float(conf[keep].mean()), conf_exc=float(conf[~keep].mean()), n_exc=int((~keep).sum()))
# per participant retention and class counts (retained trials)
rows = []
for s_ in subs:
    m = sub == s_; k = m & keep; c = conf[k]; a = acc[k]; ch = D["choice"][k]
    rows.append(dict(participant=int(s_), n_valid=int(m.sum()), n_retained=int(k.sum()), retention=float(k.sum() / m.sum()), acc_retained=float(a.mean()), n_right=int((ch == 2).sum()), n_left=int((ch == 1).sum()),
                     n_correct=int((a == 1).sum()), n_incorrect=int((a == 0).sum()), n_lowconf=int((c <= 3).sum()), n_highconf=int((c >= 7).sum()),
                     n_incorrect_high=int(((a == 0) & (c >= 7)).sum()), n_incorrect_low=int(((a == 0) & (c <= 3)).sum()), n_correct_high=int(((a == 1) & (c >= 7)).sum()), n_correct_low=int(((a == 1) & (c <= 3)).sum())))
P = pd.DataFrame(rows); P.to_csv("v2/results/per_participant_counts.csv", index=False)
R["retention_per_participant"] = dict(min=float(P.retention.min()), median=float(P.retention.median()), max=float(P.retention.max()), argmin=int(P.participant[P.retention.idxmin()]),
                                       n_below_50=int((P.retention < .5).sum()), n_retained_min=int(P.n_retained.min()), n_retained_max=int(P.n_retained.max()), n_retained_median=float(P.n_retained.median()))
# manifests for the reproducibility package
pd.DataFrame(dict(participant=sub, run=D["run"], original_trial_index=D["orig"], RT=rt, retained=keep.astype(int), choice=D["choice"], accuracy=acc, confidence=conf, dotdirection=D["dotdirection"])).to_csv("v2/results/trial_manifest.csv", index=False)
# adjacent trial pairs among retained trials (RQ4)
sk, rk_, ok_ = sub[keep], D["run"][keep], D["orig"][keep]
key = {(a_, b_, c_): i for i, (a_, b_, c_) in enumerate(zip(sk, rk_, ok_))}
pairs = [(i, key[(a_, b_, c_ + 1)]) for i, (a_, b_, c_) in enumerate(zip(sk, rk_, ok_)) if (a_, b_, c_ + 1) in key and c_ > 0]
pd.DataFrame(dict(participant=[sk[i] for i, j in pairs], run=[rk_[i] for i, j in pairs], trial_N=[ok_[i] for i, j in pairs], trial_N1=[ok_[j] for i, j in pairs])).to_csv("v2/results/adjacent_pairs.csv", index=False)
R["n_pairs"] = len(pairs)
json.dump(R, open("v2/results/a1_design.json", "w"), indent=1)
print(json.dumps({k: R[k] for k in ("n_trials_total", "window", "retention_per_participant", "n_pairs", "retained_vs_excluded")}, indent=1)); print(R["rule"]["L600_row"]); print(R["rule"]["top5"])
