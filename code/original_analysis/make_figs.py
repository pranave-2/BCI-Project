import json, sys, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, ".")
from common import load
# IJIES: all text in figures 10-pt Times New Roman, figures at single-column width (~8.5 cm)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "stix"})
R = json.load(open("results.json")); Z = np.load("results_scores.npz")
OUT = "../journal_submission/figs"; os.makedirs(OUT, exist_ok=True)
L0, B0 = R["window"]["L"], R["window"]["B"]
CW = 8.1 / 2.54

# ---- Fig 1
D = load(); rt = D["RT"].astype(float)
fig, ax = plt.subplots(2, 1, figsize=(CW, 4.3), gridspec_kw={"height_ratios": [0.85, 1.2]})
a = ax[0]; a.set_xlim(-150, 1500); a.set_ylim(0, 1); a.axis("off")
a.plot([-100, 1450], [0.35, 0.35], color="k", lw=0.8)
t_resp = 800
a.add_patch(Rectangle((t_resp - B0 - L0, 0.35), L0, 0.3, fc="#9ecae1", ec="k", lw=0.6))
a.add_patch(Rectangle((t_resp - B0, 0.35), B0, 0.3, fc="#e0e0e0", ec="k", lw=0.6, hatch="////"))
for x, lab in ((0, "Stimulus\nonset"), (t_resp, "Response")):
    a.plot([x, x], [0.28, 0.42], color="k", lw=1); a.text(x, 0.22, lab, ha="center", va="top")
a.text(t_resp - B0 - L0 / 2, 0.50, f"Window\n{L0} ms", ha="center", va="center")
a.text(t_resp - B0 / 2 - 60, 0.88, f"Buffer, {B0} ms", ha="center", va="center")
a.annotate("", xy=(t_resp - B0 / 2, 0.66), xytext=(t_resp - B0 / 2 - 60, 0.80), arrowprops=dict(arrowstyle="-", lw=0.6))
a.text(1450, 0.50, "(a)", ha="right")
b = ax[1]; b.hist(rt, bins=np.arange(325, 1525, 25), color="#6baed6", ec="k", lw=0.3)
b.axvline(L0 + B0, color="r", lw=1, ls="--"); b.set_ylim(0, 640)
b.text(L0 + B0 - 20, 600, f"RT = {L0 + B0} ms:\n{100 * R['window']['retention_pooled']:.1f}% kept", color="r", va="top", ha="right")
b.set_xlabel("Response time (ms)"); b.set_ylabel("Trials"); b.text(0.98, 0.95, "(b)", transform=b.transAxes, va="top", ha="right")
plt.tight_layout(pad=0.4); plt.savefig(f"{OUT}/fig1.png", dpi=300); plt.close()

# ---- Fig 2: per-subject AUC
fig, a = plt.subplots(figsize=(CW, 2.5))
items = [("Choice", R["RQ1"]), ("Accuracy", R["RQ2"]), ("Confidence", R["conf_decoding"])]
rng = np.random.default_rng(0)
for i, (nm, r) in enumerate(items):
    v = np.array(list(r["per_subject_auc"].values()))
    a.boxplot([v], positions=[i], widths=0.45, showfliers=False, medianprops=dict(color="k"), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    a.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=7, color="#2171b5", zorder=3)
    lr = r["lr"] if "lr" in r else r
    a.hlines(lr["mean"], i - 0.3, i + 0.3, color="#d94801", lw=1.6)
a.axhline(0.5, color="gray", lw=0.7, ls="--"); a.set_xticks(range(3)); a.set_xticklabels([n for n, _ in items]); a.set_ylabel("Held-out-subject AUC")
plt.tight_layout(pad=0.4); plt.savefig(f"{OUT}/fig2.png", dpi=300); plt.close()

# ---- Fig 3: sensitivity (subset of grid so that 10-pt labels fit)
Ls3 = [200, 400, 600, 800]; Bs3 = [0, 100, 150, 200, 300]
fig, axs = plt.subplots(3, 1, figsize=(CW, 3.9), sharex=True)
for a, t, ttl in zip(axs, ("choice", "accuracy", "confidence"), ("Choice", "Accuracy", "Confidence")):
    M = np.full((len(Ls3), len(Bs3)), np.nan)
    for i, L in enumerate(Ls3):
        for j, B in enumerate(Bs3):
            k = f"{L}_{B}"
            if k in R["grid"]: M[i, j] = R["grid"][k][t][0]
    a.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0.45, vmax=0.58)
    a.set_yticks(range(len(Ls3))); a.set_yticklabels(Ls3); a.set_xticks(range(len(Bs3))); a.set_xticklabels(Bs3)
    a.spines[:].set_visible(False); a.tick_params(length=2)
    for i in range(len(Ls3)):
        for j in range(len(Bs3)):
            if not np.isnan(M[i, j]): a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.54 else "k")
    a.add_patch(Rectangle((Bs3.index(B0) - .5, Ls3.index(L0) - .5), 1, 1, fill=False, ec="r", lw=1.5))
    a.set_ylabel("L (ms)"); a.text(-0.2, 1.02, f"({'abc'[['choice','accuracy','confidence'].index(t)]}) {ttl}", transform=a.transAxes, ha="left", va="bottom")
axs[-1].set_xlabel("Buffer B (ms)")
plt.tight_layout(pad=0.4); plt.savefig(f"{OUT}/fig3.png", dpi=300); plt.close()

# ---- Fig 4
sc, sub, acc, conf = Z["sc_acc"], Z["sub"], Z["acc"], Z["conf"]
low, high = conf <= 3, conf >= 7
cells = [("Corr./\nlow", (acc == 1) & low), ("Corr./\nhigh", (acc == 1) & high), ("Incorr./\nlow", (acc == 0) & low), ("Incorr./\nhigh", (acc == 0) & high)]
fig, a = plt.subplots(figsize=(CW, 2.5))
data = [sc[m] for _, m in cells]
vp = a.violinplot(data, showmeans=False, showmedians=True, showextrema=False, widths=0.8)
for b_ in vp["bodies"]: b_.set_facecolor("#9ecae1"); b_.set_edgecolor("k"); b_.set_linewidth(0.5); b_.set_alpha(0.9)
for k in ("cmedians",): vp[k].set_color("k"); vp[k].set_linewidth(0.6)
a.set_ylim(-1.5, 1.6)
a.set_xticks(range(1, 5)); a.set_xticklabels([c[0] + f"\n({len(d)})" for c, d in zip(cells, data)]); a.set_ylabel("Decoder score")
plt.tight_layout(pad=0.4); plt.savefig(f"{OUT}/fig4.png", dpi=300); plt.close()
print("figs done")

# ---- Fig 5: time-resolved decoding (100 ms windows, response-locked)
EX = json.load(open("results_extra.json"))
fig, a = plt.subplots(figsize=(CW, 2.5))
cols = {"choice": "#1f77b4", "accuracy": "#d62728", "confidence": "#2ca02c"}
xs = np.array(EX["starts"]) + 50
for t in ("choice", "accuracy", "confidence"):
    m = np.array([w["mean"] for w in EX[t]["raw"]]); se = np.array([w["se"] for w in EX[t]["raw"]])
    a.plot(xs, m, color=cols[t], lw=1.2, label=t.capitalize()); a.fill_between(xs, m - se, m + se, color=cols[t], alpha=0.15, lw=0)
a.axhline(0.5, color="gray", lw=0.7, ls="--"); a.axvspan(-150, 0, color="#e0e0e0", alpha=0.6, lw=0)
a.set_xlabel("Window center relative to response (ms)"); a.set_ylabel("Held-out-participant AUC"); a.legend(frameon=False, loc="upper left", fontsize=10, ncol=1)
a.set_ylim(0.46, 0.60)
plt.tight_layout(pad=0.4); plt.savefig(f"{OUT}/fig5.png", dpi=300); plt.close()
print("fig5 done")

# ---- Fig 6: confidence rating distribution by accuracy (all trials)
E2 = json.load(open("results_extra2.json"))
fig, a = plt.subplots(figsize=(CW, 2.3))
x = np.arange(1, 10); wd = 0.38
a.bar(x - wd / 2, E2["rating_dist"]["correct"], wd, color="#6baed6", ec="k", lw=0.4, label="Correct")
a.bar(x + wd / 2, E2["rating_dist"]["incorrect"], wd, color="#fd8d3c", ec="k", lw=0.4, label="Incorrect")
a.set_xticks(x); a.set_xlabel("Confidence rating"); a.set_ylabel("Proportion of trials"); a.legend(frameon=False, loc="upper left")
plt.tight_layout(pad=0.4); plt.savefig(f"{OUT}/fig6.png", dpi=300); plt.close()
print("fig6 done")
