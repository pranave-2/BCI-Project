"""A6: component/slope baselines evaluated on the SAME retained trials, window and LOSO splits as the whole-scalp decoder.
CPP-like ROI = CPz, CP1, CP2 (Pz is missing in at least one participant and therefore not in the common channel set); window mean amplitude and linear slope (per second).
Lateralized-readiness-like (LRP-like) = mean over available left-minus-right pairs (C3-C4, C5-C6, CP3-CP4, CP5-CP6) of the window mean amplitude and slope."""
import sys, json, numpy as np
sys.path.insert(0, "v2"); from common2 import *
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
D = load(); keep = D["RT"] >= L0 + B0; sub = D["sub"][keep]; rt = D["RT"][keep]; lrt = np.log(rt); names = D["channel_names"]; ix = {n: i for i, n in enumerate(names)}
s0, e0 = win_idx(L0, B0); seg = D["pre"][keep][:, :, s0:e0].astype(np.float64); t = (np.arange(e0 - s0) - (e0 - s0 - 1) / 2) / FS            # seconds, centred
def feat(ch_idx, sign=None):
    w = seg[:, ch_idx, :].mean(1) if sign is None else sum(sg * seg[:, i, :] for i, sg in zip(ch_idx, sign)) / len(ch_idx)
    return np.column_stack([w.mean(1), (w * t).sum(1) / (t ** 2).sum()])                                                                         # mean (uV), slope (uV/s)
cpp = feat([ix[c] for c in ("CPz", "CP1", "CP2")])
pairs = [(a, b) for a, b in (("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")) if a in ix and b in ix]
lrp = np.column_stack([np.mean([seg[:, ix[a], :] - seg[:, ix[b], :] for a, b in pairs], axis=0).mean(1), (np.mean([seg[:, ix[a], :] - seg[:, ix[b], :] for a, b in pairs], axis=0) * t).sum(1) / (t ** 2).sum()])
Z = lambda M: zscore_within(M, sub)
X_all = Z(window_mean(D, rows=keep)); Bflex = rt_basis(lrt, sub, 5); T = targets(D, keep)
sets = {"CPP-like (mean, slope)": Z(cpp), "LRP-like (mean, slope)": Z(lrp), "CPP-like + LRP-like": Z(np.hstack([cpp, lrp])), "RT only (flexible)": Bflex,
        "CPP-like + LRP-like + RT": np.hstack([Z(np.hstack([cpp, lrp])), Bflex]), "Whole-scalp (primary)": X_all}
R = dict(pairs_used=pairs, cpp_channels=["CPz", "CP1", "CP2"], n_perm=NPERM, window_s=[(s0 - PRE) * 4, (e0 - PRE) * 4])
base = {}
for ti, (name, (y, valid)) in enumerate(T.items()):
    R[name] = {}
    for si, (sn, Xs) in enumerate(sets.items()):
        a, p, sc = loso_auc(Xs, y, sub, valid); o = summarize_auc(a); o.update(pooled_auc=p, per_participant=a)
        if sn != "RT only (flexible)":
            pt = perm_test(Xs, y, sub, valid, o["mean"], p, n=NPERM, seed0=500000 + 10000 * ti + 1000 * si); pt.pop("null_means"); pt.pop("null_pooled"); o["perm"] = pt
        R[name][sn] = o
    base[name] = R[name]["Whole-scalp (primary)"]["per_participant"]
    for sn in sets:
        if sn == "Whole-scalp (primary)": continue
        cur = R[name][sn]["per_participant"]; ks = sorted(set(cur) & set(base[name])); d = np.array([base[name][k] - cur[k] for k in ks]); R[name][sn]["whole_scalp_minus_this"] = dict(mean=float(d.mean()), ci95=boot_mean_ci(d), n=len(d))
    print(name, {k: (round(v["mean"], 4), v.get("perm", {}).get("p_mean")) for k, v in R[name].items()}, flush=True)
json.dump(R, open("v2/results/a6_components.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("FINISHED a6", flush=True)
