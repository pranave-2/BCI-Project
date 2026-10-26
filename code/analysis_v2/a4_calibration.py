"""A4: calibration contrasts (RQ3), fully documented: class counts per participant, evaluable participants, missing-class handling, statistic, permutation count/seeds, participant-bootstrap CIs, two-stage analysis (also RT-controlled)."""
import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "v2"); from common2 import *
from scipy import stats
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
D = load(); keep = D["RT"] >= L0 + B0; sub = D["sub"][keep]; rt = D["RT"][keep]
X = zscore_within(window_mean(D, rows=keep), sub); acc = D["accuracy"][keep]; conf = D["confidence"][keep]
P = np.load("v2/results/scores_primary.npz"); assert (P["sub"] == sub).all()
sc_acc = P["accuracy_raw"]; sc_acc_flex = P["accuracy_resid_flex"]
R = dict(n_perm=NPERM, statistic_primary="pooled out-of-participant AUC (all held-out scores pooled over participants)", statistic_secondary="mean over evaluable participants of the participant-level AUC (>= 10 trials and >= 3 per class)",
         missing_class_rule="participants with fewer than 3 trials in a class are not evaluable for the participant-level AUC; their scores still enter the pooled AUC and their trials still train the other folds",
         permutation="labels shuffled within participant inside the contrast subset; the whole LOSO procedure re-fitted for every permutation; p = (1 + #{null >= observed}) / (1 + n_perm); seeds = seed0 + permutation index")
cnt_rows = []
def boot_pooled(sc, y, ss, n=2000, seed=5):
    rng = np.random.default_rng(seed); us = np.unique(ss); idx = {u: np.where(ss == u)[0] for u in us}; vals = []
    for _ in range(n):
        pick = rng.choice(us, len(us)); ii = np.concatenate([idx[u] for u in pick])
        if len(np.unique(y[ii])) == 2: vals.append(roc_auc_score(y[ii], sc[ii]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
def contrast(name, m, ypos, seed0):
    y = ypos.astype(int); a, pooled, sc = loso_auc(X, y, sub, m); mm = mean_auc(a) if a else float("nan")
    pt = perm_test(X, y, sub, m, mm, pooled, n=NPERM, seed0=seed0); pt.pop("null_means"); pt.pop("null_pooled")
    ss = sub[m]; yy = y[m]; sc_ = sc[m]
    per = []
    for s in np.unique(sub):
        k = (sub == s) & m; per.append(dict(contrast=name, participant=int(s), n=int(k.sum()), n_pos=int(y[k].sum()), n_neg=int((1 - y[k]).sum()), evaluable=bool(int(s) in a)))
    cnt_rows.extend(per); sel = np.where(m)[0]
    res = dict(n_trials=int(m.sum()), n_pos=int(y[m].sum()), n_neg=int((1 - y[m]).sum()), n_participants_with_trials=int(len(np.unique(ss))), n_evaluable_participants=len(a), pooled_auc=pooled, pooled_auc_ci95_participant_bootstrap=boot_pooled(sc[m], y[m], ss),
               mean_participant_auc=mm, mean_participant_auc_ci95_boot=boot_mean_ci(list(a.values())) if a else None, perm=pt, min_class_per_participant=int(min(r["n_pos"] for r in per if r["n"] > 0)), max_share_of_positive_class=float(max(r["n_pos"] for r in per) / max(1, y[m].sum())))
    print(name, {k: res[k] for k in ("n_trials", "n_pos", "n_evaluable_participants", "pooled_auc")}, res["perm"]["p_pooled"], flush=True); return res, sc
low, high = conf <= 3, conf >= 7
cell = dict(CL=(acc == 1) & low, CH=(acc == 1) & high, IL=(acc == 0) & low, IH=(acc == 0) & high)
R["cells"] = {k: int(v.sum()) for k, v in cell.items()}; R["cells_n_participants"] = {k: int(len(np.unique(sub[v]))) for k, v in cell.items()}
s_ih = pd.Series(sub[cell["IH"]]).value_counts(); R["IH_max_share"] = float(s_ih.max() / s_ih.sum()); R["IH_participants"] = int(len(s_ih))
R["feasible_four_way"] = bool(cell["IH"].sum() >= 100 and len(s_ih) >= 12 and s_ih.max() / s_ih.sum() <= 0.2)
seed = 300000; t0 = time.time(); main = {}
for nm, m, ypos in (("IH_vs_IL", cell["IH"] | cell["IL"], cell["IH"]), ("IH_vs_CH", cell["IH"] | cell["CH"], cell["IH"]), ("IL_vs_CL", cell["IL"] | cell["CL"], cell["IL"]), ("CH_vs_CL", cell["CH"] | cell["CL"], cell["CH"])):
    main[nm], _ = contrast(nm, m, ypos, seed); seed += 20000
R["primary_split_1-3_vs_7-9"] = main
# alternative definitions of the confidence classes (same code path and the same seeds for the primary split, so identical numbers appear wherever it is reported)
alt = {}
for nm, (lo, hi) in {"strict_1-2_vs_8-9": (2, 8), "liberal_1-4_vs_6-9": (4, 6)}.items():
    l_, h_ = conf <= lo, conf >= hi; inc = (acc == 0) & (l_ | h_); cor = (acc == 1) & (l_ | h_); alt[nm] = {}
    alt[nm]["incorrect_high_vs_low"], _ = contrast(f"{nm}:IH_vs_IL", inc, (acc == 0) & h_, seed); seed += 20000
    alt[nm]["correct_high_vs_low"], _ = contrast(f"{nm}:CH_vs_CL", cor, (acc == 1) & h_, seed); seed += 20000
R["alternative_class_definitions"] = alt
# four-way multinomial (balanced accuracy / macro AUC) with permutation inference
lab = np.full(len(sub), -1)
for i, k in enumerate(("CL", "CH", "IL", "IH")): lab[cell[k]] = i
m4 = lab >= 0; X4, y4, s4 = X[m4], lab[m4], sub[m4]
def four(yv):
    pr = np.zeros((len(yv), 4))
    for s in np.unique(s4):
        te = s4 == s; pr[te] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=300).fit(X4[~te], yv[~te]).predict_proba(X4[te])
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(yv, pr.argmax(1)), roc_auc_score(yv, pr, multi_class="ovr", average="macro")
ob = four(y4)
def _fp(seed):
    rng = np.random.default_rng(seed); yp = y4.copy()
    for s in np.unique(s4): q = np.where(s4 == s)[0]; yp[q] = rng.permutation(y4[q])
    return four(yp)
nl = np.array(Parallel(n_jobs=NJOBS)(delayed(_fp)(700000 + i) for i in range(500)))
R["four_way"] = dict(n=int(m4.sum()), balanced_accuracy=float(ob[0]), macro_auc=float(ob[1]), chance_balanced_accuracy=0.25, n_perm=500, p_bal_acc=float((1 + np.sum(nl[:, 0] >= ob[0])) / 501), p_macro_auc=float((1 + np.sum(nl[:, 1] >= ob[1])) / 501),
                     null_bal_acc_mean=float(nl[:, 0].mean()), null_bal_acc_p95=float(np.percentile(nl[:, 0], 95)))
# two-stage analysis with the out-of-participant accuracy decoder (higher score = more correct-like): uncorrected and RT-flexible-residualized decoder
def two_stage(sc, tag):
    mm = cell["IH"] | cell["IL"]; idx = np.where(mm)[0]; labs = cell["IH"][idx].astype(int); ss_ = sub[idx]; scs = sc[idx]
    obs = roc_auc_score(labs, scs); rng = np.random.default_rng(7); cnt = 0; NP = 10000
    groups = [np.where(ss_ == s)[0] for s in np.unique(ss_)]
    for _ in range(NP):
        yp = labs.copy()
        for g in groups: yp[g] = rng.permutation(labs[g])
        cnt += roc_auc_score(yp, scs) >= obs
    # participant-level: AUC per participant (evaluable) and bootstrap CI
    per = {}
    for s in np.unique(ss_):
        g = np.where(ss_ == s)[0]
        if min(labs[g].sum(), (1 - labs[g]).sum()) >= 3: per[int(s)] = roc_auc_score(labs[g], scs[g])
    z_ = {}
    for nm_, mk in (("correct", acc == 1), ("incorrect", acc == 0)):
        zs_, zr_ = [], []
        for s in np.unique(sub):
            k = (sub == s) & mk
            if k.sum() >= 10 and len(np.unique(conf[k])) > 2:
                rho = stats.spearmanr(conf[k], sc[k])[0]; zs_.append(np.arctanh(np.clip(rho, -.99, .99)))
                # partial Spearman controlling for ln RT (rank-based): residualize ranks of confidence and score on ranks of RT
                rc, rs, rr = stats.rankdata(conf[k]), stats.rankdata(sc[k]), stats.rankdata(rt[k])
                A = np.column_stack([np.ones(k.sum()), rr]); e1 = rc - A @ np.linalg.lstsq(A, rc, rcond=None)[0]; e2 = rs - A @ np.linalg.lstsq(A, rs, rcond=None)[0]
                zr_.append(np.arctanh(np.clip(np.corrcoef(e1, e2)[0, 1], -.99, .99)))
        t1, t2 = stats.ttest_1samp(zs_, 0), stats.ttest_1samp(zr_, 0)
        z_[nm_] = dict(fisher_z_mean=float(np.mean(zs_)), t=float(t1.statistic), p=float(t1.pvalue), n=len(zs_), partial_rt_fisher_z_mean=float(np.mean(zr_)), partial_t=float(t2.statistic), partial_p=float(t2.pvalue))
    return dict(auc_IH_gt_IL=float(obs), n_perm=NP, p_perm_onesided=float((cnt + 1) / (NP + 1)), ci95_pooled_participant_bootstrap=boot_pooled(scs, labs, ss_), mean_participant_auc=float(np.mean(list(per.values()))), n_evaluable=len(per),
                mean_score_IH=float(sc[cell["IH"]].mean()), mean_score_IL=float(sc[cell["IL"]].mean()), mean_score_CH=float(sc[cell["CH"]].mean()), mean_score_CL=float(sc[cell["CL"]].mean()), spearman_confidence_score=z_)
R["two_stage"] = two_stage(sc_acc, "uncorrected"); R["two_stage_rt_flexible_decoder"] = two_stage(sc_acc_flex, "rt-flex")
pd.DataFrame(cnt_rows).to_csv("v2/results/calibration_counts_per_participant.csv", index=False)
json.dump(R, open("v2/results/a4_calibration.json", "w"), indent=1); print("FINISHED a4", round(time.time() - t0), "s", flush=True)
