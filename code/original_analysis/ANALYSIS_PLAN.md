# Analysis plan (frozen before any decoding result was inspected)

Source: OpenNeuro ds002739 (Gherman & Philiastides, 2018), 24 subjects x 2 runs, preprocessed EEG
(63 channels, 0.5-40 Hz, 1000 Hz, ICA/PCA-cleaned by the dataset authors). Stored here decimated to 250 Hz.

## Trial inclusion
All "valid" trials in EEG_events files (dataset authors' exclusions already applied). Trial kept for a window
(L, B) iff tresp - B - L >= tstim, i.e. RT >= L + B.

## Window selection (RQ1-RQ3) - uses trial retention only, never decoding performance
Grid L in {200,300,400,500,600,700,800} ms, B in {0,50,100,150,200,250,300,350} ms.
Rule: among (L,B) with B >= 100 ms (motor-contamination buffer) and pooled retention >= 80%,
choose the largest L; ties -> largest B. Full grid of AUCs is reported afterwards as a sensitivity analysis only.

## Features / decoder
Spatial-integration decoder: x = mean EEG amplitude per channel over the window (63-dim; the channel missing in a
subject is excluded from all subjects, i.e. channels common to every subject). Per-subject z-scoring of features
(unsupervised, label-free). L2 logistic regression, C = 0.1 fixed (sensitivity: 0.01, 1), class_weight balanced.
Robustness: shrinkage LDA (Ledoit-Wolf, Blankertz et al. 2011). Supplementary: 4 sub-bin means (252-dim).

## Cross-validation / inference
Primary: leave-one-subject-out (LOSO); metric AUC per held-out subject, mean and SD over subjects, plus
pooled-out-of-fold AUC. Significance: group-level label-permutation test (labels permuted within subject, whole
LOSO repeated, N=500) on mean-subject AUC; one-sided p = (1 + #{perm >= obs}) / (1 + N).
Per-subject one-sample t-test of AUC vs 0.5 reported as a secondary check.
Supplementary: within-subject stratified 10-fold CV (deviation from the draft's leave-one-trial-out; chosen for
computational feasibility of permutation inference).
Control: decoding from RT alone (single-feature logistic regression, same CV) to quantify behavioural leakage.

## RQ1 target: choice (left vs right).  RQ2 target: accuracy (correct vs incorrect).
## RQ3 calibration
Confidence split: Low = ratings 1-3, High = 7-9, medium 4-6 excluded from the four-way table.
Feasibility rule for the full four-way analysis (set before looking): incorrect/high cell has >= 100 pooled trials
contributed by >= 12 subjects with no subject contributing > 20% of the cell. Otherwise only the two-stage analysis.
Always reported: (i) pairwise LOSO AUC incorrect/high vs incorrect/low; incorrect/high vs correct/high; (ii) two-stage:
accuracy decoder (LOSO) score for each trial; compare the "error evidence" score in incorrect/high vs incorrect/low
trials (Mann-Whitney on subject-wise mean differences / paired Wilcoxon across subjects); Spearman of reported
confidence with decoder score within correct and within incorrect trials (per subject, then t-test of Fisher z).
## RQ4
Consecutive valid trials N, N+1 within a run (original trial indices differ by 1). Neural confidence state on N =
out-of-subject (LOSO) high-vs-low-confidence decoder score on N's pre-response window (applied to all trials);
reported confidence on N used as behavioural benchmark. Outcomes on N+1: choice repetition (logistic GEE, subject
clusters, covariates: stimulus repetition, accuracy N), RT N+1 (log, linear GEE), accuracy N+1.
Decision bound is NOT estimated (no diffusion model was fitted); RT and accuracy of N+1 are reported as proxies.

## Deviation added after a 4-subject smoke test (no inference inspected, before the full run)
Smoke test showed RT alone is a strong predictor of confidence, so every decoding analysis additionally reports an
"RT-residualised" variant: each feature is regressed on log RT within subject (label-free) and the residual decoded.
This variant is treated as the more conservative estimate of information beyond response speed.

## Amendment 2 (comparison decoders; written before they were run, after the main results were known)
Purpose: test whether the weak and RT-dependent results are specific to the primary L2-logistic mean-amplitude decoder.
Same trials (RT >= 750 ms), same window (L=600, B=150), same LOSO splits, same per-participant z-scoring, same three targets
(choice, accuracy, confidence high-vs-low), each also on RT-residualized features. Hyperparameters FIXED in advance (no tuning, so no leakage):
- Feature sets: amp = window-mean amplitude (primary); amp4 = four sub-bin means; logvar = log variance over the window per channel; amp+logvar.
- Models: L2 logistic (C=0.1, balanced; primary); shrinkage LDA (Ledoit-Wolf); linear SVM (LinearSVC C=0.01, balanced);
  RBF SVM (C=1, gamma=scale, balanced); random forest (300 trees, min_samples_leaf=20, balanced_subsample); MLP (1 hidden layer of 32 units, alpha=0.1, early stopping, seed 0).
Metric: mean held-out-participant AUC. Comparison to the primary decoder: paired Wilcoxon signed-rank on the 24 participant AUCs, Holm-corrected within target and variant.
No permutation test is run for the comparison decoders; the primary decoder keeps its permutation inference. Deep networks (e.g., EEGNet) are not included (no deep-learning framework installed).
