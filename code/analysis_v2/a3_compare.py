"""A3: comparison decoders on the SAME canonical features/trials/splits as the primary decoder, fully specified, with equivalence tests and below-chance diagnostics."""
import sys, json, time, warnings, numpy as np
sys.path.insert(0, "v2"); from common2 import *
from scipy import stats
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
D = load(); keep = D["RT"] >= L0 + B0; sub = D["sub"][keep]; rt = D["RT"][keep]; lrt = np.log(rt)
T = targets(D, keep); s0, e0 = win_idx(L0, B0); seg = D["pre"][keep][:, :, s0:e0].astype(np.float64)
z = lambda M: zscore_within(M, sub)
amp = z(seg.mean(2)); parts = np.array_split(np.arange(seg.shape[2]), 4); amp4 = z(np.concatenate([seg[:, :, p].mean(2) for p in parts], 1)); logvar = z(np.log(seg.var(2) + 1e-12))
feats = dict(amp=amp, amp4=amp4, logvar=logvar, **{"amp+logvar": np.concatenate([amp, logvar], 1)})
Bflex = rt_basis(lrt, sub, 5)
MODELS = {
    "lr": lambda: FastLR(C=0.1),                                                                           # primary decoder (exact Newton solver for the same objective as sklearn LogisticRegression(C=0.1, class_weight='balanced'))
    "lda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),                             # Ledoit-Wolf shrinkage
    "svm_lin": lambda: LinearSVC(C=0.01, class_weight="balanced", dual=False, tol=1e-4, max_iter=1000),
    "svm_rbf": lambda: SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced"),                      # gamma = 1 / (n_features * X.var())
    "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=1, random_state=0),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=200, random_state=0)}   # internal 10% validation split for early stopping only
CONFIGS = [("amp", m) for m in MODELS] + [("amp4", "lr"), ("amp4", "lda"), ("logvar", "lr"), ("amp+logvar", "lr")]
def score(clf, X): return clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") and not isinstance(clf, SVC) else clf.decision_function(X)
def fold(mk, X, y, tr, te):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning); clf = MODELS[mk]().fit(X[tr], y[tr]); sc = score(clf, X[te])
    info = dict(conv_warn=int(sum(issubclass(x.category, ConvergenceWarning) for x in w)), n_iter=int(np.max(getattr(clf, "n_iter_", 0))) if hasattr(clf, "n_iter_") else None,
                classes=clf.classes_.tolist() if hasattr(clf, "classes_") else None,
                w=(clf.coef_.ravel().astype(float) if hasattr(clf, "coef_") else None))
    return te, sc, info
R = dict(spec={k: v().__class__.__name__ + str(v().get_params()) for k, v in MODELS.items()}, normalization="within-participant z-score over retained trials; for logvar the z-score is applied to log variance", seeds=dict(rf=0, mlp=0, cv=0))
t0 = time.time(); SC = {}
for variant in ("raw", "resid"):
    F = {k: (v if variant == "raw" else residualize(v, sub, Bflex)) for k, v in feats.items()}
    for tname, (y, valid) in T.items():
        idx = np.where(valid)[0]; ys = y[idx]; ss = sub[idx]
        for fk, mk in CONFIGS:
            X = F[fk][idx]; res = Parallel(n_jobs=NJOBS)(delayed(fold)(mk, X, ys, np.where(ss != s)[0], np.where(ss == s)[0]) for s in np.unique(ss))
            sc = np.zeros(len(ys)); weights = []; cw = 0; nit = []
            for te, p, info in res:
                sc[te] = p; cw += info["conv_warn"]
                if info["w"] is not None: weights.append(info["w"])
                if info["n_iter"] is not None: nit.append(info["n_iter"])
            a = subj_auc(sc, ys, ss); key = f"{fk}|{mk}"
            e = R.setdefault(variant, {}).setdefault(tname, {})[key] = dict(per_participant=a, mean=mean_auc(a), pooled=float(roc_auc_score(ys, sc)), n_conv_warnings=cw, class_order=res[0][2]["classes"], max_iter_used=int(max(nit)) if nit else None)
            if weights and len(weights) > 1:
                Wm = np.array(weights); Wn = Wm / np.linalg.norm(Wm, axis=1, keepdims=True); cs = Wn @ Wn.T; e["fold_weight_cosine_mean"] = float(cs[np.triu_indices(len(Wn), 1)].mean())
            SC[f"{variant}|{tname}|{key}"] = sc
            print(variant, tname, key, round(e["mean"], 4), round(time.time() - t0), "s", flush=True)
json.dump(R, open('v2/results/a3_compare_partial.json', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))      # checkpoint before the statistics stage
# equivalence (TOST via bootstrap CI) and superiority tests against the primary decoder, margin +-0.02 AUC
MARGIN = 0.02; R["margin"] = MARGIN
for variant in ("raw", "resid"):
    for tname in T:
        base = R[variant][tname]["amp|lr"]["per_participant"]; names = [k for k in R[variant][tname] if k != "amp|lr"]; ps = []
        for k in names:
            cur = R[variant][tname][k]["per_participant"]; ks = sorted(set(base) & set(cur)); d = np.array([cur[s] - base[s] for s in ks])
            w = stats.wilcoxon(d) if np.any(d != 0) else None; ci90 = boot_mean_ci(d, alpha=0.10, seed=11); ci95 = boot_mean_ci(d, alpha=0.05, seed=12)
            e = R[variant][tname][k]; e.update(diff_vs_primary=float(d.mean()), diff_ci95=ci95, diff_ci90=ci90, p_wilcoxon=float(w.pvalue) if w else 1.0, n_pairs=len(d))
            e["verdict"] = (f"equivalent (90 percent CI within +-{MARGIN:.2f})") if (ci90[0] > -MARGIN and ci90[1] < MARGIN) else ("worse (CI below -margin)" if ci90[1] < -MARGIN else ("better (CI above +margin)" if ci90[0] > MARGIN else "inconclusive"))
            ps.append(e["p_wilcoxon"])
        for k, a_ in zip(names, holm(ps)): R[variant][tname][k]["p_holm"] = a_
# ---- diagnostics for the below-chance log-variance AUCs -----------------------------------------------------------------------------------------
Dg = {}
for tname in ("accuracy", "confidence"):
    y, valid = T[tname]; idx = np.where(valid)[0]; ys = y[idx]; ss = sub[idx]; X = logvar[idx]
    a = R["raw"][tname]["logvar|lr"]["per_participant"]; v = np.array(list(a.values()))
    # permutation null of the same decoder (whole LOSO repeated with shuffled labels): is the below-chance mean outside the null?
    def _p(seed):
        rng = np.random.default_rng(seed); yp = ys.copy()
        for s in np.unique(ss):
            m = np.where(ss == s)[0]; yp[m] = rng.permutation(ys[m])
        return mean_auc(subj_auc(loso_scores(X, yp, ss), yp, ss))
    nul = np.array(Parallel(n_jobs=NJOBS)(delayed(_p)(900000 + i) for i in range(500)))
    obs = mean_auc(a)
    # within-participant 10-fold CV of the same feature (is there any information inside a participant?)
    from sklearn.model_selection import StratifiedKFold
    wcv = []
    for s in np.unique(ss):
        m = np.where(ss == s)[0]
        if min((ys[m] == 0).sum(), (ys[m] == 1).sum()) < 10: continue
        sc_ = np.zeros(len(m))
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(X[m], ys[m]): sc_[te] = make_clf().fit(X[m][tr], ys[m][tr]).decision_function(X[m][te])
        wcv.append(roc_auc_score(ys[m], sc_))
    # pattern consistency: feature-wise AUC vector of the held-out participant versus that of the training participants
    def fvec(m): return np.array([roc_auc_score(ys[m], X[m][:, j]) for j in range(X.shape[1])]) - 0.5
    cors = []
    for s in np.unique(ss):
        te = np.where(ss == s)[0]; tr = np.where(ss != s)[0]
        if len(np.unique(ys[te])) < 2: continue
        cors.append(float(np.corrcoef(fvec(te), fvec(tr))[0, 1]))
    # label orientation check: training-set AUC of the full model and the sign of the mean score by class on the training data
    full = make_clf().fit(X, ys); tr_auc = float(roc_auc_score(ys, full.decision_function(X)))
    Dg[tname] = dict(observed_mean_auc=obs, n_participants=len(v), n_participants_below_05=int((v < 0.5).sum()), null_mean=float(nul.mean()), null_sd=float(nul.std(ddof=1)), null_p05=float(np.percentile(nul, 5)), null_p95=float(np.percentile(nul, 95)),
                     p_lower_tail=float((1 + np.sum(nul <= obs)) / (1 + len(nul))), n_perm=len(nul), within_participant_10fold_mean=float(np.mean(wcv)), n_within=len(wcv),
                     pattern_corr_heldout_vs_train_mean=float(np.mean(cors)), pattern_corr_frac_negative=float(np.mean(np.array(cors) < 0)), training_auc_full_model=tr_auc, orientation=f"score = decision_function, positive class = {'correct' if tname == 'accuracy' else 'high confidence'}; classes_={full.classes_.tolist()}")
    print("diag", tname, Dg[tname], flush=True)
R["logvar_diagnostics"] = Dg
np.savez("v2/results/scores_compare.npz", **SC)
json.dump(R, open("v2/results/a3_compare.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("FINISHED a3", flush=True)
