"""Merge every v2 result file into the single authoritative results_final.json (+ provenance). The manuscript, supplement and response letter read ONLY this file."""
import json, glob, os, hashlib, subprocess, sys, platform, time
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
names = ["a1_design", "a2_primary", "a2b_tercile_t", "a3_compare", "a4_calibration", "a5_rq4", "a6_components", "a7_stimlocked", "a8_normalization", "a9_temporal", "a10_sensitivity", "a11_choice_controls", "a12_timeres", "a13_grid", "solver_validation"]
RF = {}
for n in names:
    f = os.path.join(RES, n + ".json")
    if os.path.exists(f): RF[n] = json.load(open(f))
    else: print("MISSING", n)
RF["_provenance"] = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0], platform=platform.platform(),
                         files={os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:16] for f in sorted(glob.glob(RES + "/*.json"))})
json.dump(RF, open(os.path.join(RES, "results_final.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("results_final.json written; parts:", [k for k in RF if not k.startswith("_")])
