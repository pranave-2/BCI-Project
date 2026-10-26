"""Shared code for the revision analyses (single code path for every table and p-value).
Conventions (documented in the manuscript, Section 4):
  * trials: valid trials of ds002739 (as released); analysis set = trials with RT >= L + B (retained set)
  * features: mean amplitude per channel over the window, 53 channels present in every participant
  * canonical normalization: z-score every feature within participant over the RETAINED trials (unsupervised; uses the target participant's own unlabeled batch)
  * decoder: L2 logistic regression, C = 0.1, class_weight = 'balanced', lbfgs, max_iter = 300, tol = 1e-4; score = decision_function (positive class = 1)
  * positive classes: choice = right (code 2); accuracy = correct; confidence = high (rating 7-9; low = 1-3; 4-6 excluded from the confidence analysis only)
"""
import glob, json, os, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(_v, "1")      # one BLAS thread per worker: parallelism is over permutations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FS = 250; PRE = 300
L0, B0 = 600, 150
NJOBS = int(os.environ.get('NJOBS', '8'))
C_DEFAULT = 0.1
ELEC = json.load(open(os.path.join(HERE, "electrodes.json")))["names"]      # 64 names, file order = channel indices 1..64


def load(dirp=None):
    dirp = dirp or os.path.join(ROOT, "epochs")
    files = sorted(glob.glob(f"{dirp}/sub-*_run-*.npz"))
    keys = ["RT", "choice", "accuracy", "confidence", "dotdirection", "orig", "tstim", "tresp"]
    cols = {k: [] for k in keys + ["sub", "run", "pos"]}
    pre, stim, chs = [], [], []
    for f in files:
        z = np.load(f); ok = z["ok"]
        pre.append(z["pre"][ok]); stim.append(z["stim"][ok])
        for k in keys: cols[k].append(np.asarray(z[k])[ok])
        cols["sub"].append(np.full(ok.sum(), int(z["sub"]))); cols["run"].append(np.full(ok.sum(), int(z["run"]))); cols["pos"].append(np.where(ok)[0])
        chs.append(set(z["valid_ch"].tolist()))
    D = {k: np.concatenate(v) for k, v in cols.items()}
    common = np.array(sorted(set.intersection(*chs))); D["channels"] = common; D["channel_names"] = [ELEC[c - 1] for c in common]
    D["pre"] = np.concatenate(pre)[:, common - 1, :]; D["stim"] = np.concatenate(stim)[:, common - 1, :]
    D["RT"] = D["RT"].astype(float)
    return D


def win_idx(L, B):
    end = PRE - int(round(B * FS / 1000)); start = end - int(round(L * FS / 1000)); return start, end


def window_mean(D, L=L0, B=B0, rows=None):
    s, e = win_idx(L, B); X = D["pre"][:, :, s:e].mean(axis=2).astype(np.float64)
    return X if rows is None else X[rows]


def zscore_within(X, sub):
    X = X.copy()
    for s in np.unique(sub):
        m = sub == s; X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-12)
    return X


def targets(D, keep):
    acc = D["accuracy"][keep]; conf = D["confidence"][keep]; ch = (D["choice"][keep] == 2).astype(int)
    hl = (conf <= 3) | (conf >= 7)
    return dict(choice=(ch, np.ones(len(ch), bool)), accuracy=(acc.astype(int), np.ones(len(acc), bool)), confidence=((conf >= 7).astype(int), hl))


def make_clf_sklearn(C=C_DEFAULT): return LogisticRegression(C=C, class_weight="balanced", max_iter=300)


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Exact solver for the primary decoder. The objective is that of scikit-learn's LogisticRegression(C, class_weight='balanced'):
#     0.5 * ||w||^2 + C * sum_i s_i * log(1 + exp(-y_i (w.x_i + b))),   s_i = n_train / (2 n_class(i)),  intercept b unpenalized.
# It is minimized by Newton iterations to a gradient tolerance of 1e-9 (scikit-learn's lbfgs stops at tol=1e-4); LOSO folds are warm-started from the full-data solution with the
# fold Hessian approximated by (H_full - H_participant) evaluated at that solution, which makes the 1000-permutation tests tractable. Validated against scikit-learn in v2/validate_solver.py.
from scipy.linalg import cho_factor, cho_solve


def _sigmoid(z): return 0.5 * (1.0 + np.tanh(0.5 * z))


def _class_w(y):
    n = len(y); n1 = max(1, int(y.sum())); n0 = max(1, n - int(y.sum())); return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))


def _grad(X1, y, sw, beta, C, reg):
    p = _sigmoid(X1 @ beta); return reg * beta + C * (X1.T @ (sw * (p - y)))


def newton_lr(X1, y, sw, C, beta0=None, Hfix=None, tol=1e-9, maxit=60):
    """X1: design matrix with a trailing column of ones. Returns beta (p+1,)."""
    d = X1.shape[1]; reg = np.ones(d); reg[-1] = 0.0; beta = np.zeros(d) if beta0 is None else beta0.copy()
    fixed = None
    if Hfix is not None: fixed = cho_factor(np.diag(reg) + C * Hfix + 1e-10 * np.eye(d))
    for it in range(maxit):
        g = _grad(X1, y, sw, beta, C, reg)
        if np.max(np.abs(g)) < tol: break
        if fixed is None:
            p = _sigmoid(X1 @ beta); w = sw * p * (1 - p); H = np.diag(reg) + C * (X1.T * w) @ X1 + 1e-10 * np.eye(d); beta = beta - np.linalg.solve(H, g)
        else: beta = beta - cho_solve(fixed, g)
    return beta


class FastLR:
    """Drop-in for the primary decoder (fit / decision_function); same objective as LogisticRegression(C, class_weight='balanced')."""
    def __init__(self, C=C_DEFAULT): self.C = C
    def fit(self, X, y, sample_weight=None):
        X1 = np.column_stack([X, np.ones(len(X))]); sw = _class_w(y) * (1.0 if sample_weight is None else sample_weight)
        self.beta = newton_lr(X1, y.astype(float), sw, self.C); self.classes_ = np.array([0, 1]); return self
    def decision_function(self, X): return X @ self.beta[:-1] + self.beta[-1]
    def get_params(self): return dict(C=self.C, class_weight='balanced', solver='exact Newton (gradient tol 1e-9)', penalty='l2', intercept_penalized=False)
    @property
    def coef_(self): return self.beta[:-1][None, :]


def make_clf(C=C_DEFAULT): return FastLR(C)


def loso_scores(X, y, sub, C=C_DEFAULT, make=None, weights=None):
    """Out-of-participant decision scores; every participant is predicted by a model trained on all other participants."""
    out = np.full(len(y), np.nan)
    if make is not None:
        for s in np.unique(sub):
            te = sub == s; out[te] = make().fit(X[~te], y[~te]).decision_function(X[te])
        return out
    X1 = np.column_stack([X, np.ones(len(X))]); yf = y.astype(float); us = np.unique(sub)
    sw_full = _class_w(y); beta0 = newton_lr(X1, yf, sw_full, C)
    p0 = _sigmoid(X1 @ beta0); w0 = sw_full * p0 * (1 - p0); Hs = {s: (X1[sub == s].T * w0[sub == s]) @ X1[sub == s] for s in us}; Hall = sum(Hs.values())
    for s in us:
        te = sub == s; tr = ~te; swf = _class_w(y[tr]); b = newton_lr(X1[tr], yf[tr], swf, C, beta0=beta0, Hfix=Hall - Hs[s]); out[te] = X1[te] @ b
    return out


def subj_auc(score, y, sub, min_n=10, min_class=3):
    res = {}
    for s in np.unique(sub):
        m = sub == s
        if m.sum() >= min_n and len(np.unique(y[m])) == 2 and min((y[m] == 0).sum(), (y[m] == 1).sum()) >= min_class: res[int(s)] = float(roc_auc_score(y[m], score[m]))
    return res


def mean_auc(a): return float(np.mean(list(a.values())))


def loso_auc(X, y, sub, valid=None, C=C_DEFAULT):
    valid = np.ones(len(y), bool) if valid is None else valid
    sc = loso_scores(X[valid], y[valid], sub[valid], C=C); a = subj_auc(sc, y[valid], sub[valid])
    full = np.full(len(y), np.nan); full[valid] = sc
    return a, float(roc_auc_score(y[valid], sc)), full


def _perm_stat(seed, X, y, sub, valid, C, strata):
    rng = np.random.default_rng(seed); yp = y.copy()
    groups = sub if strata is None else strata
    for g in np.unique(groups[valid]):
        idx = np.where((groups == g) & valid)[0]; yp[idx] = rng.permutation(y[idx])
    a, pooled, _ = loso_auc(X, yp, sub, valid, C)
    return mean_auc(a), pooled


def perm_test(X, y, sub, valid, obs_mean, obs_pooled, n=1000, C=C_DEFAULT, seed0=100000, strata=None):
    """Label-permutation test of the whole LOSO procedure. Labels are shuffled within participant (or within `strata`, e.g. participant x RT bin); one-sided p = (1 + #{null >= obs}) / (1 + n)."""
    r = np.array(Parallel(n_jobs=NJOBS)(delayed(_perm_stat)(seed0 + i, X, y, sub, valid, C, strata) for i in range(n)))
    return dict(n_perm=n, seed0=seed0, p_mean=float((1 + np.sum(r[:, 0] >= obs_mean)) / (1 + n)), p_pooled=float((1 + np.sum(r[:, 1] >= obs_pooled)) / (1 + n)),
                null_mean=float(r[:, 0].mean()), null_sd=float(r[:, 0].std(ddof=1)), null_p95=float(np.percentile(r[:, 0], 95)), null_pooled_mean=float(r[:, 1].mean()), null_pooled_sd=float(r[:, 1].std(ddof=1)),
                null_pooled_p95=float(np.percentile(r[:, 1], 95)), null_means=r[:, 0].tolist(), null_pooled=r[:, 1].tolist())


def boot_mean_ci(vals, n=10000, seed=1, alpha=0.05):
    v = np.asarray(vals, float); rng = np.random.default_rng(seed); idx = rng.integers(0, len(v), (n, len(v)))
    m = v[idx].mean(1); return [float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))]


def summarize_auc(a, n_boot=10000):
    v = np.array(list(a.values())); from scipy import stats
    t = stats.ttest_1samp(v, 0.5)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=int(len(v)), n_above=int((v > 0.5).sum()), min=float(v.min()), max=float(v.max()), ci95_boot=boot_mean_ci(v, n_boot),
                t=float(t.statistic), p_t_onesided=float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1.0, ps[i] * (m - r))); adj[i] = run
    return adj.tolist()


def rt_basis(lrt, sub, df=5, kind="spline"):
    """Within-participant flexible RT basis. Natural-cubic-spline-like truncated-power basis on the within-participant rank of ln RT (df columns), or linear (1 column)."""
    B = np.zeros((len(lrt), df if kind == "spline" else 1))
    for s in np.unique(sub):
        m = np.where(sub == s)[0]; r = (np.argsort(np.argsort(lrt[m])) + 0.5) / len(m)            # rank in (0,1)
        if kind == "linear": B[m, 0] = (lrt[m] - lrt[m].mean()) / lrt[m].std(); continue
        K = df + 1; kn = np.quantile(r, np.linspace(0, 1, K))                                   # K knots at quantiles of the within-participant rank (boundary knots = min/max)
        def d(x, k): return (np.clip(x - k, 0, None) ** 3 - np.clip(x - kn[-1], 0, None) ** 3) / (kn[-1] - k)
        cols = [r] + [d(r, kn[k]) - d(r, kn[-2]) for k in range(K - 2)]                            # natural cubic spline: df = K - 1 columns
        cols = np.column_stack(cols); B[m] = (cols - cols.mean(0)) / (cols.std(0) + 1e-12)
    return B


def residualize(X, sub, basis):
    """Remove (within participant, label-free) the linear dependence of every feature on the RT basis; re-standardize."""
    Xr = X.copy()
    for s in np.unique(sub):
        m = sub == s; A = np.column_stack([np.ones(m.sum()), basis[m]])
        coef = np.linalg.lstsq(A, X[m], rcond=None)[0]; Xr[m] = X[m] - A[:, 1:] @ coef[1:]
        Xr[m] = (Xr[m] - Xr[m].mean(0)) / (Xr[m].std(0) + 1e-12)
    return Xr
