"""A10: sensitivity (minimum detectable effects), equivalence margins and uncertainty summaries for every null/weak claim. Reads the other result files; run last.
Definitions: group tests of the mean participant AUC: MDE80 = c95 + 0.8416 * SE, c95 = 95th percentile of the permutation null of the mean AUC, SE = SD(participant AUC) / sqrt(n).
Pooled-AUC contrasts: MDE80 = smallest true pooled AUC theta with P(AUC_hat > c95_pooled) = 0.80 where AUC_hat ~ N(theta, SE_HM(theta)^2) (Hanley-McNeil), c95_pooled = 95th percentile of the permutation null of the pooled AUC.
GEE coefficients: MDE80 = 2.8 * SE (bias-reduced), reported as an odds ratio per SD of the predictor. The permutation null's 95th percentile is a rejection threshold, NOT a power estimate."""
import sys, json, numpy as np
from scipy import stats
def hm_se(theta, n1, n0):
    q1 = theta / (2 - theta); q2 = 2 * theta ** 2 / (1 + theta); return np.sqrt((theta * (1 - theta) + (n1 - 1) * (q1 - theta ** 2) + (n0 - 1) * (q2 - theta ** 2)) / (n1 * n0))
def mde_pooled(c, n1, n0):
    for th in np.arange(0.5, 0.95, 0.0005):
        if 1 - stats.norm.cdf((c - th) / hm_se(th, n1, n0)) >= 0.8: return float(th)
    return None
A2 = json.load(open("v2/results/a2_primary.json")); A4 = json.load(open("v2/results/a4_calibration.json")); A5 = json.load(open("v2/results/a5_rq4.json")); A3 = json.load(open("v2/results/a3_compare.json"))
R = dict(definitions=__doc__)
g = {}
for nm in ("choice", "accuracy", "confidence"):
    o = A2[nm]["uncorrected"]; se = o["sd"] / np.sqrt(o["n"]); c = o["perm"]["null_p95"]
    g[nm] = dict(c95=c, se=float(se), mde80=float(c + 0.8416 * se), observed=o["mean"], observed_ci95=o["ci95_boot"])
R["group_mean_auc"] = g
cal = {}
for k, v in A4["primary_split_1-3_vs_7-9"].items():
    c = v["perm"].get("null_pooled_p95"); n1, n0 = v["n_pos"], v["n_neg"]
    cal[k] = dict(n_pos=n1, n_neg=n0, pooled_auc=v["pooled_auc"], ci95=v["pooled_auc_ci95_participant_bootstrap"], null_pooled_p95=c, mde80=mde_pooled(c, n1, n0) if c else None,
                  upper_ci_below_0_55=bool(v["pooled_auc_ci95_participant_bootstrap"][1] < 0.55), analytic_null_se=float(np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))))
R["calibration_contrasts"] = cal
eq = {}
for var in ("raw", "resid"):
    for tn in ("choice", "accuracy", "confidence"):
        for k, v in A3[var][tn].items():
            if k == "amp|lr": continue
            eq.setdefault(f"{var}|{tn}", {})[k] = dict(diff=v["diff_vs_primary"], ci90=v["diff_ci90"], verdict=v["verdict"], mde80_paired=float(2.8 * (v["diff_ci95"][1] - v["diff_ci95"][0]) / (2 * 1.96)))
R["classifier_equivalence_margin_0.02"] = eq
seq = {}
for k, v in A5["gee"].items():
    for nm in ("S", "C", "Sf"):
        if nm in v:
            se = v[nm]["se_bias_reduced"]; seq[f"{k}:{nm}"] = dict(b=v[nm]["b"], se_bias_reduced=se, ci95=v[nm]["ci95_bias_reduced"], mde80_coef=float(2.8 * se), mde80_odds_ratio_per_sd=float(np.exp(2.8 * se)) if k.startswith(("M1", "M3")) else None)
R["sequential"] = seq
json.dump(R, open("v2/results/a10_sensitivity.json", "w"), indent=1); print("FINISHED a10"); print(json.dumps({k: R[k] for k in ("group_mean_auc", "calibration_contrasts")}, indent=1)[:2500])
