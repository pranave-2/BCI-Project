import glob, warnings, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
FS = 250  # Hz after decimation
PRE = 300  # samples, response-locked epoch [-1200, 0) ms


def load(dirp="epochs"):
    files = sorted(glob.glob(f"{dirp}/sub-*_run-*.npz"))
    cols = {k: [] for k in ["RT", "choice", "accuracy", "confidence", "dotdirection", "orig", "sub", "run"]}
    pre, chs = [], []
    for f in files:
        z = np.load(f)
        ok = z["ok"]
        pre.append(z["pre"][ok])
        for k in ["RT", "choice", "accuracy", "confidence", "dotdirection", "orig"]:
            cols[k].append(np.asarray(z[k])[ok])
        cols["sub"].append(np.full(ok.sum(), int(z["sub"])))
        cols["run"].append(np.full(ok.sum(), int(z["run"])))
        chs.append(set(z["valid_ch"].tolist()))
    D = {k: np.concatenate(v) for k, v in cols.items()}
    common = sorted(set.intersection(*chs))
    D["channels"] = np.array(common)
    pre = np.concatenate(pre)  # (N,64,300)
    D["pre"] = pre[:, np.array(common) - 1, :]
    return D


def window_features(D, L, B, nbins=1):
    """Mean amplitude per channel over [tresp-B-L, tresp-B); nbins>1 gives sub-bin means. Returns X, keep mask."""
    end = PRE - int(round(B * FS / 1000)); start = end - int(round(L * FS / 1000))
    keep = D["RT"] >= L + B
    seg = D["pre"][:, :, start:end]
    if nbins == 1:
        X = seg.mean(axis=2)
    else:
        parts = np.array_split(np.arange(seg.shape[2]), nbins)
        X = np.concatenate([seg[:, :, p].mean(axis=2) for p in parts], axis=1)
    return X.astype(np.float64), keep


def zscore_by_subject(X, sub):
    X = X.copy()
    for s in np.unique(sub):
        m = sub == s
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-12)
    return X


def make_clf(kind="lr", C=0.1):
    if kind == "lr":
        return LogisticRegression(C=C, class_weight="balanced", max_iter=300)
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def loso_scores(X, y, sub, train_ok=None, kind="lr", C=0.1):
    """Out-of-subject decision scores for ALL rows; training uses rows with train_ok (labels valid)."""
    if train_ok is None:
        train_ok = np.ones(len(y), bool)
    out = np.full(len(y), np.nan)
    for s in np.unique(sub):
        te = sub == s; tr = (~te) & train_ok
        clf = make_clf(kind, C).fit(X[tr], y[tr])
        out[te] = clf.decision_function(X[te])
    return out


def subj_auc(score, y, sub, valid):
    res = {}
    for s in np.unique(sub):
        m = (sub == s) & valid
        if m.sum() >= 10 and len(np.unique(y[m])) == 2 and min(np.sum(y[m] == 0), np.sum(y[m] == 1)) >= 3:
            res[s] = roc_auc_score(y[m], score[m])
    return res


def loso_auc(X, y, sub, valid, kind="lr", C=0.1):
    """valid: rows that carry a label in this analysis. Returns per-subject AUC dict, pooled AUC, scores."""
    Xv, yv, sv = X[valid], y[valid], sub[valid]
    sc = loso_scores(Xv, yv, sv, kind=kind, C=C)
    a = subj_auc(sc, yv, sv, np.ones(len(yv), bool))
    return a, roc_auc_score(yv, sc), sc


def _perm_once(X, y, sub, seed, kind, C):
    rng = np.random.default_rng(seed)
    yp = y.copy()
    for s in np.unique(sub):
        m = np.where(sub == s)[0]
        yp[m] = rng.permutation(y[m])
    a, pooled, _ = loso_auc(X, yp, sub, np.ones(len(y), bool), kind, C)
    return np.mean(list(a.values())), pooled


def perm_test(X, y, sub, obs_mean, obs_pooled, n=500, kind="lr", C=0.1, seed0=1000):
    r = Parallel(n_jobs=8)(delayed(_perm_once)(X, y, sub, seed0 + i, kind, C) for i in range(n))
    r = np.array(r)
    p_mean = (1 + np.sum(r[:, 0] >= obs_mean)) / (1 + n)
    p_pool = (1 + np.sum(r[:, 1] >= obs_pooled)) / (1 + n)
    return p_mean, p_pool, r


def within_subject_cv(X, y, sub, kind="lr", C=0.1, k=10, seed=0):
    res = {}
    for s in np.unique(sub):
        m = sub == s
        Xs, ys = X[m], y[m]
        if min(np.sum(ys == 0), np.sum(ys == 1)) < k:
            continue
        sc = np.zeros(len(ys))
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xs, ys):
            sc[te] = make_clf(kind, C).fit(Xs[tr], ys[tr]).decision_function(Xs[te])
        res[s] = roc_auc_score(ys, sc)
    return res


def rt_only_auc(D, y, sub, valid):
    X = np.log(D["RT"].astype(float))[:, None]
    X = zscore_by_subject(X, sub)
    return loso_auc(X, y, sub, valid)
