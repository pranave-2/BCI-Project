"""A9: temporal support of the feature computation. (i) analytic future-sample weight of every operation we control; (ii) empirical re-extraction from the 1 kHz data with operations that use no sample after the window end,
and a proxy experiment for the (undocumented) zero-phase filtering of the supplied data; (iii) change of the primary AUCs."""
import sys, json, glob, numpy as np, scipy.signal as ss
sys.path.insert(0, "v2"); from common2 import *
from scipy import stats
R = {}
# ---------- (i) analytic: how much weight can samples AFTER the window end carry? ------------------------------------------------------------------
fs = 1000; N = 600                                         # window: 600 samples at 1 kHz
def kernel_stats(h, name, causal=False):
    """Effective kernel of 'operation then 600-sample window mean' on the INPUT samples; returns the share of its L1 mass that lies on samples after the window end (and beyond 3 / 150 ms after it).
    h: impulse response (zero-phase operations: symmetric, centred; causal operations: starts at lag 0)."""
    if causal: return dict(operation=name, support_ms_abs_h_above_1e3=float(np.where(np.abs(h) > 1e-3 * np.abs(h).max())[0].max()), share_L1_after_window_end=0.0, share_L1_after_window_end_beyond_3ms=0.0, share_L1_after_window_end_beyond_150ms=0.0, note="causal: output at a window sample depends only on earlier inputs")
    c = len(h) // 2; g = np.convolve(np.ones(N) / N, h); j = np.arange(len(g)) - c; tot = np.abs(g).sum()
    half = float(np.where(np.abs(h) > 1e-3 * np.abs(h).max())[0].max() - c)
    return dict(operation=name, support_ms_abs_h_above_1e3=half, share_L1_after_window_end=float(np.abs(g)[j >= N].sum() / tot), share_L1_after_window_end_beyond_3ms=float(np.abs(g)[j >= N + 3].sum() / tot),
                share_L1_after_window_end_beyond_150ms=float(np.abs(g)[j >= N + 150].sum() / tot))
imp = np.zeros(8001); imp[4000] = 1.0
sos_dec = ss.cheby1(8, 0.05, 0.8 / 4, output="sos")                                  # scipy.signal.decimate(x, 4) design
h_zp = ss.sosfiltfilt(sos_dec, imp, padlen=0)[3000:5001]                               # effective zero-phase impulse response (forward-backward), centred
h_c = ss.sosfilt(sos_dec, imp)[4000:]
sos_bp = ss.butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos"); imp2 = np.zeros(60001); imp2[30000] = 1.0
h_bp_zp = ss.sosfiltfilt(sos_bp, imp2, padlen=0)[10000:50001]; h_bp_c = ss.sosfilt(sos_bp, imp2)[30000:]
ops = [kernel_stats(h_zp, "zero-phase decimation filter (as used for the epochs)"), kernel_stats(h_c, "causal decimation filter", causal=True),
       kernel_stats(h_bp_zp, "proxy: zero-phase 0.5-40 Hz Butterworth band-pass (order 4)"), kernel_stats(h_bp_c, "proxy: causal 0.5-40 Hz Butterworth band-pass", causal=True)]
R["analytic"] = ops
# ---------- (ii) empirical variants ----------------------------------------------------------------------------------------------------------------
D = load(); keep = D["RT"] >= L0 + B0; sub = D["sub"][keep]; names = ["V0 (zero-phase decimation of the whole recording; used throughout)", "V1_boxcar", "V2_causal_decim", "V3_trunc_decim", "P0_zerophase_full", "P1_causal", "P2_zerophase_trunc"]
files = sorted(glob.glob("v2/features/sub-*_run-*.npz") + glob.glob("v2/features_old6/sub-*_run-*.npz")); R["n_feature_files"] = len(files)
Fm = {}
for f in files:
    z = np.load(f); F_ = z["F"]
    if F_.shape[2] < 8: F_ = np.concatenate([F_, np.full(F_.shape[:2] + (8 - F_.shape[2],), np.nan, np.float32)], 2)       # files from the first extraction pass lack the P3/P4 decomposition
    Fm[(int(z["sub"]), int(z["run"]))] = F_
have = {}
for (s_, r_) in Fm: have.setdefault(s_, set()).add(r_)
complete = sorted(s_ for s_, r_ in have.items() if r_ == {1, 2}); R["participants_with_both_runs"] = complete; R["n_participants_temporal"] = len(complete)
if len(complete) >= 6:
    sel_idx = np.where(keep)[0]; chans = D["channels"] - 1; V = np.full((len(sel_idx), len(chans), 8), np.nan)
    for k, i in enumerate(sel_idx):
        if int(D["sub"][i]) in complete: V[k] = Fm[(int(D["sub"][i]), int(D["run"][i]))][D["pos"][i]][chans]
    names = names + ["P3_zerophase_inside_part", "P4_zerophase_future_part"]
    X0 = window_mean(D, rows=keep); feats = [X0] + [V[:, :, j] for j in range(8)]; T = targets(D, keep)
    R["n_trials_missing_variant"] = {nm: int((~np.isfinite(V[:, 0, j - 1])).sum()) for j, nm in enumerate(names) if j > 0}
    sim = {}; res = {t: {} for t in T}; ntr = {}
    for j, nm in enumerate(names):
        if j == 0: continue
        good = np.isfinite(feats[j]).all(axis=1) & np.isfinite(V[:, :, 0]).all(axis=1); sub_g = sub[good]; ntr[nm] = dict(n_trials=int(good.sum()), n_participants=int(len(np.unique(sub_g))))
        if good.sum() < 500 or len(np.unique(sub_g)) < 4: continue
        A_, B_ = feats[0][good], feats[j][good]; sim[nm] = dict(relative_rms_difference=float(np.sqrt(np.mean((A_ - B_) ** 2)) / np.sqrt(np.mean(A_ ** 2))), mean_within_participant_correlation=float(np.mean([np.corrcoef(A_[sub_g == s].ravel(), B_[sub_g == s].ravel())[0, 1] for s in np.unique(sub_g)])))
        for tn, (y, valid) in T.items():
            yg, vv = y[good], valid[good]; a0, _, _ = loso_auc(zscore_within(A_, sub_g), yg, sub_g, vv); a1, p1, _ = loso_auc(zscore_within(B_, sub_g), yg, sub_g, vv)
            ks = sorted(set(a0) & set(a1)); d = np.array([a1[k] - a0[k] for k in ks]); w = stats.wilcoxon(d) if np.any(d != 0) else None
            res[tn][nm] = dict(mean=mean_auc(a1), mean_V0_same_rows=mean_auc(a0), pooled_auc=p1, n_participants=len(ks), per_participant=a1, diff_vs_V0=dict(mean=float(d.mean()), ci95=boot_mean_ci(d), max_abs_participant_diff=float(np.max(np.abs(d))), p_wilcoxon=float(w.pvalue) if w else 1.0, n=len(d)))
        print(nm, {tn: round(res[tn][nm]["diff_vs_V0"]["mean"], 4) for tn in T if nm in res[tn]}, ntr[nm], flush=True)
    R["variants"] = res; R["variant_names"] = names; R["feature_similarity_to_V0"] = sim; R["variant_sample"] = ntr
json.dump(R, open("v2/results/a9_temporal.json", "w"), indent=1); print("FINISHED a9", flush=True)
