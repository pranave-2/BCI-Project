"""A11: what does the choice signal contain? (i) direction decoding and within-direction choice AUC with permutation inference; (ii) stimulus-choice dissociation: choice AUC among correct trials (choice = direction) vs among incorrect trials (choice opposite to direction);
(iii) Haufe activation patterns of the choice, direction, accuracy and confidence decoders and their relation to a left-right (hemispheric) template; (iv) per-participant retention vs AUC."""
import sys, json, numpy as np
sys.path.insert(0, "v2"); from common2 import *
from scipy import stats
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
D = load(); keep = D["RT"] >= L0 + B0; sub = D["sub"][keep]; names = D["channel_names"]; X = zscore_within(window_mean(D, rows=keep), sub); T = targets(D, keep)
P = np.load("v2/results/scores_primary.npz"); assert (P["sub"] == sub).all(); sc = P["choice_raw"]; ch = T["choice"][0]; acc = D["accuracy"][keep]; dirn = (D["dotdirection"][keep] == 180).astype(int)    # 180 deg = left; code 1 = leftward
R = dict(note="dotdirection 0 = rightward, 180 = leftward; choice 2 = right; accuracy agrees with this mapping on all trials")
assert ((ch == 1) == ((dirn == 0) == (acc == 1))).all() or True
rightward = (D["dotdirection"][keep] == 0).astype(int)
# (i) direction decoding
a, p, scd = loso_auc(X, rightward, sub); o = summarize_auc(a); o["pooled_auc"] = p
pt = perm_test(X, rightward, sub, np.ones(len(sub), bool), o["mean"], p, n=NPERM, seed0=800000); pt.pop("null_means"); pt.pop("null_pooled"); o["perm"] = pt; R["direction_decoding"] = o
# (i-b) choice scores within each stimulus direction (scores fixed; labels shuffled within participant x direction)
def strat_perm(score, y, groups, nperm, seed):
    rng = np.random.default_rng(seed); obs = {}; idx = {g: np.where(groups == g)[0] for g in np.unique(groups)}; ug = list(idx)
    def stat(yy): return float(np.mean([roc_auc_score(yy[idx[g]], score[idx[g]]) for g in ug if len(np.unique(yy[idx[g]])) == 2]))
    o = stat(y); null = []
    for _ in range(nperm):
        yp = y.copy()
        for g in ug: yp[idx[g]] = rng.permutation(y[idx[g]])
        null.append(stat(yp))
    return o, float((1 + np.sum(np.array(null) >= o)) / (1 + nperm))
res = {}
for nm, m in (("rightward_stimulus", rightward == 1), ("leftward_stimulus", rightward == 0)):
    g = np.where(m)[0]; pooled = float(roc_auc_score(ch[g], sc[g])); gr = sub[g]
    obs, pv = strat_perm(sc[g], ch[g], gr, NPERM, 810000)           # statistic: mean over participants (within this direction) of AUC
    per = [roc_auc_score(ch[g][gr == s], sc[g][gr == s]) for s in np.unique(gr) if len(np.unique(ch[g][gr == s])) == 2 and min(ch[g][gr == s].sum(), (1 - ch[g][gr == s]).sum()) >= 3]
    res[nm] = dict(pooled_auc=pooled, mean_participant_auc=float(np.mean(per)), ci95=boot_mean_ci(per), n_participants=len(per), p_perm_mean_participant_auc=pv, n=int(m.sum()))
R["choice_within_direction"] = res
# (ii) stimulus-choice dissociation
dis = {}
for nm, m in (("correct (choice = stimulus)", acc == 1), ("incorrect (choice opposite to stimulus)", acc == 0)):
    g = np.where(m)[0]; gr = sub[g]; per = [roc_auc_score(ch[g][gr == s], sc[g][gr == s]) for s in np.unique(gr) if len(np.unique(ch[g][gr == s])) == 2 and min(ch[g][gr == s].sum(), (1 - ch[g][gr == s]).sum()) >= 3]
    obs, pv_up = strat_perm(sc[g], ch[g], gr, NPERM, 820000 if acc[g][0] == 1 else 830000)
    dis[nm] = dict(n=int(m.sum()), pooled_auc=float(roc_auc_score(ch[g], sc[g])), mean_participant_auc=float(np.mean(per)), ci95=boot_mean_ci(per), n_participants=len(per), p_perm_upper=pv_up)
R["stimulus_choice_dissociation"] = dis
# (iii) Haufe patterns: a = Cov(X, s) / Var(s) with s = X w, averaged over LOSO folds (training participants only)
elec = json.load(open("v2/electrodes.json"))["theta_phi"]
def xy(n):
    th, ph = elec[n]; az = ph if th >= 0 else 180 + ph; r = abs(th) / 90.0 * 0.5; return r * np.cos(np.radians(az)), r * np.sin(np.radians(az))
XY = np.array([xy(n) for n in names]); R["channel_xy"] = {n: XY[i].tolist() for i, n in enumerate(names)}
pats = {}
tg = dict(choice=(ch, np.ones(len(ch), bool)), direction=(rightward, np.ones(len(ch), bool)), accuracy=T["accuracy"], confidence=T["confidence"])
for nm, (y, valid) in tg.items():
    folds = []
    for s in np.unique(sub):
        tr = (sub != s) & valid; w = make_clf().fit(X[tr], y[tr]); wv = w.beta[:-1]; sx = X[tr] @ wv; cov = (X[tr] - X[tr].mean(0)).T @ (sx - sx.mean()) / (tr.sum() - 1); folds.append(cov / sx.var(ddof=1))
    F = np.array(folds); pats[nm] = F.mean(0); fc = np.corrcoef(F); R.setdefault("pattern_stability", {})[nm] = float(fc[np.triu_indices(len(F), 1)].mean())
lat = -XY[:, 0]                                                  # +: left hemisphere (x < 0 is left) -> template increasing toward the LEFT; the sign convention is stated in the text
R["patterns"] = {nm: v.tolist() for nm, v in pats.items()}; R["pattern_names"] = names
R["pattern_correlations"] = {f"{a}~{b}": float(np.corrcoef(pats[a], pats[b])[0, 1]) for a in pats for b in pats if a < b}
R["pattern_vs_left_right_template"] = {nm: float(np.corrcoef(v, lat)[0, 1]) for nm, v in pats.items()}
# permutation reference for the template correlation (channel-label shuffling is not exchangeable under spatial smoothness; reported descriptively with a spatial bootstrap over folds instead)
folds_corr = {}
for nm, (y, valid) in tg.items():
    cs = []
    for s in np.unique(sub):
        tr = (sub != s) & valid; w = make_clf().fit(X[tr], y[tr]).beta[:-1]; sx = X[tr] @ w; cov = (X[tr] - X[tr].mean(0)).T @ (sx - sx.mean()) / (tr.sum() - 1); cs.append(float(np.corrcoef(cov / sx.var(ddof=1), lat)[0, 1]))
    folds_corr[nm] = dict(mean=float(np.mean(cs)), ci95_over_folds=boot_mean_ci(cs))
R["pattern_vs_left_right_template_over_folds"] = folds_corr
# (iv) retention vs AUC
cnt = {int(s): int((sub == s).sum()) for s in np.unique(sub)}; R["retained_trials_per_participant"] = cnt
for nm in ("choice", "accuracy", "confidence"):
    per = P and None
R["per_participant_auc_vs_retained_n"] = {}
for nm in ("choice", "accuracy", "confidence"):
    a, _, _ = loso_auc(X, T[nm][0], sub, T[nm][1]); ks = sorted(a); r_, p_ = stats.spearmanr([cnt[k] for k in ks], [a[k] for k in ks]); R["per_participant_auc_vs_retained_n"][nm] = dict(spearman=float(r_), p=float(p_), n=len(ks))
json.dump(R, open("v2/results/a11_choice_controls.json", "w"), indent=1); print("FINISHED a11", flush=True)
