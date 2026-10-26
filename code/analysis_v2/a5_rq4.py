"""A5: sequential analysis (RQ4) with explicit models, small-sample-corrected GEE inference and a participant bootstrap that re-fits the decoding pipeline.
Models (trial pair N -> N+1 within a run, both trials retained; D = same motion direction on N and N+1, A = accuracy on N, T = ln RT on N, S = z-scored neural confidence state on N, C = z-scored reported confidence on N):
  M1  logit P(choice N+1 = choice N) = b0 + b1*S + b2*D + b3*A + b4*T          (binomial, logit link)
  M1c same with C in place of S;   M1j both S and C;   M1i adds D x A
  M2  ln RT(N+1) = b0 + b1*S + b2*T + b3*A                                       (Gaussian, identity link)
  M3  logit P(accuracy N+1 = 1) = b0 + b1*S + b2*A + b3*T                        (binomial, logit link)
GEE: participant clusters, exchangeable working correlation. Inference: (i) robust sandwich z-test, (ii) Mancl-DeRouen bias-reduced covariance with t(23) reference, (iii) participant bootstrap (resample participants, refit the LOSO confidence decoder inside each replicate)."""
import sys, json, time, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, "v2"); from common2 import *
from scipy import stats
NBOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 500
D = load(); keep = D["RT"] >= L0 + B0; sub = D["sub"][keep]; run = D["run"][keep]; orig = D["orig"][keep]; rt = D["RT"][keep]; acc = D["accuracy"][keep]; conf = D["confidence"][keep]; chn = D["choice"][keep]; dirn = D["dotdirection"][keep]
X = zscore_within(window_mean(D, rows=keep), sub); hl = (conf <= 3) | (conf >= 7); yhl = (conf >= 7).astype(int)
lrt = np.log(rt); Bflex = rt_basis(lrt, sub, 5); Xflex = residualize(X, sub, Bflex)
def neural_state(Xm, w=None):
    """out-of-participant high-vs-low confidence decoder score for ALL retained trials; training uses only high/low trials of the other participants"""
    sc = np.full(len(sub), np.nan)
    for s in np.unique(sub):
        te = sub == s; tr = (~te) & hl
        sc[te] = make_clf().fit(Xm[tr], yhl[tr], sample_weight=None if w is None else w[tr]).decision_function(Xm[te])
    return sc
zz = lambda v, s_: np.concatenate([(v[s_ == u] - v[s_ == u].mean()) / v[s_ == u].std() for u in np.unique(s_)])[np.argsort(np.argsort(np.concatenate([np.where(s_ == u)[0] for u in np.unique(s_)])))]
def zby(v):
    out = np.zeros(len(v))
    for u in np.unique(sub): m = sub == u; out[m] = (v[m] - v[m].mean()) / v[m].std()
    return out
key = {(s, r_, o): i for i, (s, r_, o) in enumerate(zip(sub, run, orig))}
pairs = [(i, key[(s, r_, o + 1)]) for i, (s, r_, o) in enumerate(zip(sub, run, orig)) if (s, r_, o + 1) in key and o > 0]
a_i = np.array([p[0] for p in pairs]); b_i = np.array([p[1] for p in pairs])
def frame(S, Cn, Sf=None):
    T = pd.DataFrame(dict(sub=sub[a_i], rep=(chn[a_i] == chn[b_i]).astype(int), D=(dirn[a_i] == dirn[b_i]).astype(int), A=acc[a_i], A1=acc[b_i], S=S[a_i], C=Cn[a_i], T=lrt[a_i], T1=lrt[b_i]))
    if Sf is not None: T["Sf"] = Sf[a_i]
    return T
S = zby(neural_state(X)); Cn = zby(conf.astype(float)); Sf = zby(neural_state(Xflex))
T = frame(S, Cn, Sf)
R = dict(n_pairs=int(len(T)), n_participants=int(T["sub"].nunique()), repeat_rate=float(T.rep.mean()), same_direction_rate=float(T.D.mean()), n_pairs_per_participant=dict(min=int(T.groupby("sub").size().min()), max=int(T.groupby("sub").size().max())),
         definition=dict(S="z-scored (within participant) out-of-participant high-vs-low confidence decoder score on trial N (generated regressor; training uses high/low trials of other participants only; scores computed for all retained trials incl. medium confidence)",
                         C="z-scored reported confidence rating on trial N", D="same motion direction on N and N+1", A="accuracy on N", T="ln RT on N"), nboot=NBOOT)
FORM = dict(M1="rep ~ S + D + A + T", M1c="rep ~ C + D + A + T", M1j="rep ~ S + C + D + A + T", M1i="rep ~ S + D*A + T", M1ci="rep ~ C + D*A + T", M1f="rep ~ Sf + D + A + T",
            M2="T1 ~ S + T + A", M2c="T1 ~ C + T + A", M3="A1 ~ S + A + T", M3c="A1 ~ C + A + T")
FAM = {k: (sm.families.Gaussian() if k.startswith("M2") else sm.families.Binomial()) for k in FORM}
def fit_gee(f, fam, data, cov_type="robust"):
    return sm.GEE.from_formula(f, groups="sub", data=data, family=fam, cov_struct=sm.cov_struct.Exchangeable()).fit(cov_type=cov_type, maxiter=100)
R["gee"] = {}
for k, f in FORM.items():
    m1 = fit_gee(f, FAM[k], T, "robust"); m2 = fit_gee(f, FAM[k], T, "bias_reduced"); df_ = T["sub"].nunique() - 1; res = {}
    for nm in m1.params.index:
        b = float(m1.params[nm]); se1 = float(m1.bse[nm]); se2 = float(m2.bse[nm]); tcrit = stats.t.ppf(0.975, df_)
        res[nm] = dict(b=b, se_robust=se1, p_robust=float(m1.pvalues[nm]), se_bias_reduced=se2, p_bias_reduced_t23=float(2 * stats.t.sf(abs(b / se2), df_)), ci95_bias_reduced=[b - tcrit * se2, b + tcrit * se2],
                       odds_ratio=float(np.exp(b)) if k.startswith(("M1", "M3")) else None, odds_ratio_ci95_bias_reduced=[float(np.exp(b - tcrit * se2)), float(np.exp(b + tcrit * se2))] if k.startswith(("M1", "M3")) else None)
    R["gee"][k] = res
print({k: (round(v.get("S", v.get("C", v.get("Sf", {}))).get("b", np.nan), 3)) for k, v in R["gee"].items()}, flush=True)
# ---- participant bootstrap re-fitting the decoder ---------------------------------------------------------------------------------------------------
subs = np.unique(sub); rows_by = {s: np.where(sub == s)[0] for s in subs}; pair_by = {s: np.where(T["sub"].values == s)[0] for s in subs}
def one_boot(seed):
    rng = np.random.default_rng(seed); pick = rng.choice(subs, len(subs)); mult = pd.Series(pick).value_counts().to_dict(); uniq = list(mult)
    w = np.zeros(len(sub))
    for s in uniq: w[rows_by[s]] = mult[s]
    Sb = np.full(len(sub), np.nan)
    for s in uniq:
        te = rows_by[s]; tr = np.concatenate([rows_by[u][hl[rows_by[u]]] for u in uniq if u != s])
        Sb[te] = make_clf().fit(X[tr], yhl[tr], sample_weight=w[tr]).decision_function(X[te])
    Zb = np.full(len(sub), np.nan)
    for s in uniq: Zb[rows_by[s]] = (Sb[rows_by[s]] - Sb[rows_by[s]].mean()) / Sb[rows_by[s]].std()
    parts = []; cid = 0
    for s in pick:                                                       # duplicates become separate clusters
        ii = pair_by[s]; d = T.iloc[ii].copy(); d["S"] = Zb[a_i[ii]]; d["sub"] = cid; cid += 1; parts.append(d)
    Tb = pd.concat(parts, ignore_index=True); out = {}
    for k in ("M1", "M1c", "M2", "M3"):
        try: m = fit_gee(FORM[k], FAM[k], Tb); out[k] = float(m.params["S" if "c" not in k else "C"])
        except Exception: out[k] = np.nan
    return out
t0 = time.time(); bs = Parallel(n_jobs=NJOBS)(delayed(one_boot)(40000 + i) for i in range(NBOOT)); R["bootstrap_seconds"] = round(time.time() - t0)
R["bootstrap"] = {}
for k in ("M1", "M1c", "M2", "M3"):
    v_all = np.array([b[k] for b in bs]); ok = np.isfinite(v_all) & (np.abs(v_all) < 5); v = v_all[ok]; nm = "S" if "c" not in k else "C"      # |beta| >= 5 or non-finite = non-converged GEE (quasi-separation in a replicate)
    R["bootstrap"][k] = dict(coef=nm, point=R["gee"][k][nm]["b"], mean=float(v.mean()), ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], p_two_sided=float(min(1.0, 2 * min((1 + np.sum(v <= 0)) / (len(v) + 1), (1 + np.sum(v >= 0)) / (len(v) + 1)))), n_valid=int(len(v)), n_failed=int((~ok).sum()))
print(json.dumps(R["bootstrap"], indent=1), flush=True)
np.savez("v2/results/rq4_scores.npz", S=S, Sf=Sf, C=Cn, a_i=a_i, b_i=b_i, boot_betas=np.array([[b[k] for k in ("M1", "M1c", "M2", "M3")] for b in bs]))
json.dump(R, open("v2/results/a5_rq4.json", "w"), indent=1); print("FINISHED a5", flush=True)
