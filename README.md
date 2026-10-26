# Reproducibility package, release R1 (response to IJIES review of paper 20266918)

Contents: code (original analysis, revision analysis `analysis_v2`, manuscript build scripts), the internal analysis plan with file times (`plan/`), the environment, data manifests (channel mask, retained-trial manifest, adjacent trial pairs, per-participant counts),
normalization/RT-regression/decoder/seed configuration, participant-level out-of-participant predictions, permutation null distributions and summary statistics, all result files (`results/*.json`, merged in `results_final.json`), and the CSV of every manuscript table.
Raw EEG (38 GB) and the 250 Hz epoch files (0.7 GB) are not included; re-create them with `code/original_analysis/extract.py` (streams ds002739 from OpenNeuro; needs about 3-4 hours) and `code/analysis_v2/extract_temporal.py` (1 kHz re-extraction for the temporal-support analysis).

Order of the analysis (each script writes one JSON into `analysis/v2/results`): a1_design, a2_primary, a2b_tercile_t, a4_calibration, a5_rq4, a6_components, a8_normalization, a3_compare, a7_stimlocked, a11_choice_controls, a12_timeres, a13_grid, a9_temporal (needs extract_temporal output), a10_sensitivity, then build_results.py.
Every number in the manuscript, supplement and response letter is read from `results/results_final.json`.

Verification: `MANIFEST.sha256` lists SHA-256 digests; to publish, run `git init`, add the files, commit, tag `R1`, and push to a new GitHub repository (the editor encourages a public release; no license file is included, so choose one).


`paper/` holds the revised manuscript (marked and clean), supplement and response letter as PDF; `figures/` holds the figure images; `code/manuscript_build/` builds them from `results/results_final.json` (`MS=manuscript_short python3 build_rev.py marked|clean`).
