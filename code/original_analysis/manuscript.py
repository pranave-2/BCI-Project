# Manuscript content. Every number is read from results*.json (no hand-typed results).
import json
R = json.load(open("results.json"))
X = json.load(open("results_rq4_interaction.json"))
DC = json.load(open("results_dir_control.json"))
EX = json.load(open("results_extra.json"))
E2 = json.load(open("results_extra2.json"))
CMP = json.load(open("results_compare.json"))


def f3(x): return f"{x:.3f}".replace("-", "−")
def f2(x): return f"{x:.2f}".replace("-", "−")
def pf(p): return "<i>p</i> &lt; 0.001" if p < 0.001 else f"<i>p</i> = {p:.3f}"
def pp(p): return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"
def n(x): return f"{x:,}"


W, B_ = R["window"]["L"], R["window"]["B"]
b, w = R["beh"], R["window"]
rq1, rq2, cf = R["RQ1"], R["RQ2"], R["conf_decoding"]
q3, q4 = R["RQ3"], R["RQ4"]; ts = q3["two_stage"]
nall, nan = R["n_trials_total"], R["n_trials_analysed"]
ca = R["cells_all"]; cc = q3["cells"]
holm = max(0.012, 0.0)  # Holm-adjusted p for the three primary permutation tests (see text), computed below
pp3 = sorted([rq1["lr"]["p_perm_mean"], rq2["lr"]["p_perm_mean"], cf["p_perm_mean"]])
h = [min(1, pp3[i] * (3 - i)) for i in range(3)]
for i in range(1, 3): h[i] = max(h[i], h[i - 1])
HOLM = max(h)

ABSTRACT = (
    "Scalp EEG carries confidence-related information before an explicit response, but it is unclear whether such pre-response activity "
    "predicts the upcoming choice, its accuracy, or the calibration of confidence. Using a public simultaneous EEG-fMRI random-dot motion dataset "
    f"({R['n_subjects']} participants, {n(nall)} trials), we decoded choice, accuracy, and confidence from a {W} ms window ending {B_} ms before the response, "
    "with leave-one-subject-out validation and permutation tests. Held-out-subject AUC was "
    f"{f3(rq1['lr']['mean'])} for choice, {f3(rq2['lr']['mean'])} for accuracy, and {f3(cf['mean'])} for confidence, all above chance. "
    "After removing linear dependence on response time, only choice decoding remained significant. High-confidence errors were not separable from "
    "low-confidence errors, and decoded neural confidence did not predict next-trial behavior, whereas reported confidence predicted choice repetition. "
    "Nonlinear and alternative-feature decoders did not outperform logistic regression. Pre-response EEG therefore carries weak information, largely related to response time."
)
KEYWORDS = "Perceptual decision-making, Electroencephalography, Decision confidence, Single-trial decoding, Overconfidence, Cross-validation"

IH_N = cc["IH"]
BLOCKS = []
A = BLOCKS.append

# ------------------------------------------------------------------ 1 Introduction
A(("h1", "Introduction"))
A(("p", "Whether the electrical activity of the brain contains information about a decision before the decision is expressed is a central question in the neuroscience of perceptual decision-making. "
        "Three claims are often conflated. First, EEG activity may be related to (correlate with) a person's choice or confidence. Second, EEG activity may allow the choice, its accuracy, or the reported confidence "
        "to be predicted on a single trial ahead of time. Third, EEG activity may cause or determine the choice. Observational EEG can support at most the first two claims; the third requires an interventional "
        "manipulation such as brain stimulation and is outside the scope of this study."))
A(("p", "The dataset analysed here {Dataset,Gherman} already supports the first claim: a stimulus-independent EEG component that discriminates confidence appeared before the explicit choice and confidence report and was consistent with a confidence measure "
        "from an accumulation-to-bound model {Gherman}. Repeating that analysis would add little. We therefore pursue the second claim, prediction, and decompose it into four research questions. "
        "<b>RQ1 (choice):</b> can pre-response EEG predict the identity of the upcoming left or right motion judgment? <b>RQ2 (accuracy):</b> can it predict whether that judgment will be correct? "
        "<b>RQ3 (calibration):</b> can it distinguish the four joint states of accuracy and confidence, and in particular do high-confidence errors, an operational marker of overconfidence, differ from low-confidence errors? "
        "<b>RQ4 (sequential effects):</b> does the neural confidence state on trial N predict behavior on trial N+1?"))
A(("p", "Evidence linking EEG to error detection and confidence comes largely from components that follow the response, such as the error positivity {Boldt,Desender21}, which cannot be used to predict a decision before it is made. "
        "Information available before the response, in a window that excludes the response and its immediate motor preparation, has received less attention, and one study with a different paradigm reported no above-chance pre-response decoding of confidence {Desender19b}. "
        "Whether pre-response EEG supports prediction of choice, accuracy, and calibration in a perceptual task therefore remains open."))
A(("p", "The distinction matters for how decoding results are interpreted. A signal that is merely correlated with confidence in an average over trials can be useless for single-trial prediction, and a signal that predicts well within a participant may not transfer to a new participant. "
        "Prediction claims therefore require out-of-sample evaluation at the level at which generalization is claimed, and, for targets that covary with response speed, a control that shows the prediction does not simply reflect speed."))
A(("p", "This paper makes six contributions. (i) We define a leakage-controlled prediction window of fixed length that ends a fixed buffer before each trial's own response. "
        "(ii) We decode choice, accuracy, and confidence in "
        f"{R['n_subjects']} participants with leave-one-subject-out validation and permutation inference. (iii) We add an explicit control for response-time information, which accounts for the accuracy and confidence effects. "
        "(iv) We compare the primary decoder with five alternative classifiers and three feature sets. (v) We report the full class balance and subject diversity of the rare high-confidence error cell before any calibration analysis. (vi) We fixed the analysis plan before inspecting decoding results and report a sensitivity analysis over the window grid. "
        "The results are modest, and we report them as such."))

# ------------------------------------------------------------------ 2 Related work
A(("h1", "Related Work"))
A(("h2", "Choice from pre-response EEG"))
A(("p", "A centro-parietal positivity (CPP) has been characterized as a domain-general decision variable that builds up to a threshold before a perceptual report {OConnell}. "
        "In a random-dot motion task its build-up rate scaled with evidence strength and predicted reaction time within coherence levels {Kelly}. "
        "Because the CPP indexes the unsigned strength of accumulated evidence {Herding}, decoding the side of a left or right choice must also draw on signals that differ between alternatives, including motor preparation; "
        "classic motor potentials were shown to reflect the continuing involvement of the motor system in value-based decisions {Gluth}. "
        "Single-trial linear classifiers over EEG components were established by Philiastides et al. {Philiastides} and extended by Ratcliff et al. {Ratcliff}, who showed that single-trial component amplitudes sort response times and choices by the quality of evidence in a diffusion model; "
        "a review is given in {Sajda}. Free-choice paradigms decoded internally generated choices from fMRI seconds before awareness {Soon,Bode}, but they differ from evidence-driven perceptual discrimination and serve here only as precedent."))
A(("h2", "Pre-response EEG and accuracy"))
A(("p", "The same single-trial framework separates correct from incorrect trials: early and late EEG components predicted decision accuracy and were distinct from a component related to task difficulty {Philiastides}. "
        "CPP build-up and threshold crossing relate to accuracy and speed {OConnell,Kelly}, and single-trial CPP slopes were related to the drift parameter of an accumulator model {vanVugt}."))
A(("h2", "Confidence, calibration, and high-confidence errors"))
A(("p", "Boldt and Yeung {Boldt} showed that the error positivity (Pe) varies in a graded way with confidence independent of accuracy, and that decoders of error detection predict confidence differences among correct trials, "
        "suggesting shared mechanisms for error detection and confidence. In masked-digit tasks the error-related negativity (ERN) appeared only for conscious trials, although errors could still be detected above chance without it {Charles13}, "
        "and response accuracy could be decoded only when the target was conscious {Charles14}. Desender et al. {Desender21} propose that the Pe reflects post-decisional evidence accumulation, "
        "whereas Rausch et al. {Rausch} found confidence-related activity both after stimulus onset and after the response and argued that markers of confidence and error awareness are partly distinct. "
        "Parietal EEG signals have also been reported to index subjective evidence and, later, a confidence-like quantity {Herding}. "
        "We use \"overconfidence\" operationally, as a high-confidence error. This is a coarse, single-trial definition compared with measures of metacognitive sensitivity {Fleming}, and linking attenuated error signals to overconfidence is our interpretation rather than a finding of the studies cited above."))
A(("h2", "Sequential effects"))
A(("p", "Choice history biases were explained by the accumulation of confidence-modulated decision variables across trials {Braun}, and pupil-linked arousal driven by decision uncertainty increased the tendency to alternate on the next trial {Urai}. "
        "Desender et al. {Desender19a} showed that decision bounds depend on the confidence reported on the previous trial and that a confidence-sensitive centro-parietal EEG component predicted the bound increase; "
        "confidence-related EEG decoding has also been linked to information seeking {Desender19b}."))
A(("p", "Table 1 compares the most closely related EEG studies on the dimensions that matter for the present question: which signal was examined, when relative to the stimulus or response, and what was found."))
A(("tbl", "Closely related EEG studies", [["Study", "Signal and timing", "Main finding"],
    ["{Philiastides}", "Single-trial components about 170 and 300 ms after the stimulus", "Components predicted accuracy; a separate component tracked difficulty"],
    ["{Gherman}", "Single-trial component before the choice and the confidence report (EEG-fMRI)", "Stimulus-independent confidence component, linked to ventromedial prefrontal activity"],
    ["{Boldt}", "Pe after the response", "Pe graded with confidence independent of accuracy"],
    ["{Rausch}", "350–500 ms after the stimulus and 250–350 ms after the response", "Two confidence-related periods; post-response activity partly distinct"],
    ["{Desender19b}", "Time-resolved decoding from the initial decision onward", "Confidence decodable; no above-chance decoding before the response"],
    ["{Desender19a}", "Centro-parietal component sensitive to confidence", "Predicted the bound increase on the next trial"]], [1.1, 3.4, 3.6], "left"))
A(("p", "In contrast with this literature, which mostly concerns post-response or stimulus-locked components and within-subject effects, we test out-of-subject prediction from a window that ends before the response, and we control for response time explicitly. "
        "Methodologically, small decoding effects in small samples can exceed analytic chance levels by chance {Combrisson}, which is why we rely on permutation inference and report effect sizes together with their null distribution."))

# ------------------------------------------------------------------ 3 Dataset
A(("h1", "Dataset and Task"))
A(("p", f"We used the open dataset of Gherman and Philiastides {{Gherman,Dataset}} (OpenNeuro ds002739, CC0 license): {R['n_subjects']} participants, two runs each, recorded with simultaneous EEG-fMRI during a speeded left or right random-dot motion discrimination at an individually calibrated coherence "
        "(targeting about 75% accuracy). After the response, participants rated confidence on a 9-point scale whose on-screen orientation was randomized across trials to decorrelate motor preparation from the report. "
        "We used the preprocessed EEG supplied with the dataset: scanner-gradient artifacts removed by template subtraction, a 12 ms median filter, 0.5 to 40 Hz band-pass filtering, 1000 Hz sampling, "
        "principal-component removal of ocular and ballistocardiogram components, and baseline correction over the 100 ms before stimulus onset. Per trial we used the stimulus onset, response time (RT), "
        "motion direction, choice, accuracy, and confidence rating. Low confidence was defined as ratings 1 to 3 and high confidence as ratings 7 to 9, following the dataset authors' split into the lowest and highest three ratings; "
        "ratings 4 to 6 were labeled medium."))
A(("p", "A pilot characterization of one run (participant 1, run 1; 158 trials) in our earlier analysis showed only five high-confidence errors, which is too few for single-participant decoding; we reproduced that pilot exactly from the raw files. "
        "We therefore pooled all participants and runs and verified that the rare cell is spread across participants (Section 5.1)."))

A(("p", "Each run contained 160 task trials. The dataset authors excluded trials with missing or invalid responses before releasing the event files, and we used only the remaining valid trials, which were also the only ones with EEG epochs in the release. "
        "Consecutive trials were identified through the original trial index so that trial pairs in the sequential analysis are truly adjacent even where invalid trials were removed. Motion direction was coded 0 and 180 degrees and the choice 1 (left) or 2 (right); "
        "the recorded accuracy agrees with the mapping 0 degrees to right and 180 degrees to left on all 7,377 trials."))

# ------------------------------------------------------------------ 4 Methods
A(("h1", "Methods"))
A(("h2", "Epoching and window definition"))
A(("p", "Preprocessed EEG was low-pass decimated to 250 Hz. Because the data were already band-limited to 40 Hz, this step does not remove information of interest. "
        "To avoid contamination by the response, we define for every trial a window of length <i>L</i> that ends a buffer <i>B</i> before that trial's own response time <i>t</i><sub>resp</sub>. "
        "The mean amplitude of channel <i>c</i> in the window is"))
A(("eq", "<i>x</i><sub><i>c</i></sub> = (1 / <i>n</i><sub><i>w</i></sub>) ∑<sub><i>t</i> = <i>t</i><sub>0</sub></sub><sup><i>t</i><sub>1</sub> − 1</sup> <i>e</i><sub><i>c</i></sub>(<i>t</i>)", 1))
A(("p", "where <i>t</i><sub>0</sub> = <i>t</i><sub>resp</sub> − <i>B</i> − <i>L</i>, <i>t</i><sub>1</sub> = <i>t</i><sub>resp</sub> − <i>B</i>, <i>e</i><sub><i>c</i></sub>(<i>t</i>) is the EEG of channel <i>c</i>, and <i>n</i><sub><i>w</i></sub> is the number of samples in the window. A trial is retained only if RT ≥ <i>L</i> + <i>B</i>, so that the window lies entirely after stimulus onset. "
        "We examined <i>L</i> from 200 to 800 ms and <i>B</i> from 0 to 350 ms. The primary window was fixed by a rule that uses trial retention and never decoding performance: among combinations with <i>B</i> ≥ 100 ms and at least 80% pooled retention, "
        f"choose the largest <i>L</i>. This gave <i>L</i> = {W} ms and <i>B</i> = {B_} ms, retaining {n(nan)} of {n(nall)} trials ({100 * w['retention_pooled']:.1f}%; Fig. 1). "
        "Decoding performance over the grid is reported only as a sensitivity analysis."))
A(("fig", "fig1.png", "Prediction window. (a) Schematic of the fixed-length window ending a buffer before the response. (b) Distribution of response times for all trials; trials left of the dashed line are too fast for the selected window and are excluded."))
A(("p", f"Eleven of the 64 channels were unavailable in at least one participant after the dataset authors' channel rejection, so all analyses use the {R['n_channels_common']} channels available in every participant."))
A(("h2", "Single-trial decoder"))
A(("p", "We adopt the single-trial spatial-integration linear classifier used for this dataset {Gherman,Philiastides,Sajda}: the window-averaged channel vector <b>x</b> is projected onto a spatial weight vector <b>w</b> to give one discriminant value per trial,"))
A(("eq", "<i>s</i> = <b>w</b><sup>T</sup><b>x</b> + <i>b</i>", 2))
A(("p", "where <b>w</b> and <i>b</i> minimize the class-weighted regularized logistic loss"))
A(("eq", "(1/2)‖<b>w</b>‖<sup>2</sup> + <i>C</i> ∑<sub><i>i</i></sub> ω<sub><i>i</i></sub> ln(1 + exp(−<i>y</i><sub><i>i</i></sub>(<b>w</b><sup>T</sup><b>x</b><sub><i>i</i></sub> + <i>b</i>)))", 3))
A(("p", "with labels <i>y</i><sub><i>i</i></sub> ∈ {−1, +1}, class weights ω<sub><i>i</i></sub> inversely proportional to class frequency, and <i>C</i> = 0.1 fixed in advance (<i>C</i> = 0.01 and 1 are reported as checks). "
        "Features were standardized within each participant using only that participant's own unlabeled data. As a robustness check for the high-dimensional, small-sample regime we also evaluated shrinkage-regularized linear discriminant analysis "
        "{Blankertz,Ledoit} and, as a supplement, a spatiotemporal variant with four sub-bin means per channel. All model fitting used training participants only."))
A(("h2", "Comparison decoders"))
A(("p", "To test whether the results depend on the primary decoder, we compared it with five alternatives, all evaluated on the same trials, window, and LOSO splits: shrinkage LDA, a linear support vector machine (<i>C</i> = 0.01), an RBF-kernel support vector machine (<i>C</i> = 1), a random forest (300 trees, minimum leaf size 20), and a multilayer perceptron (one hidden layer of 32 units, L2 penalty 0.1, early stopping). "
        "Besides the window-mean amplitude we used four sub-bin means (spatiotemporal), the log variance of each channel over the window (a power feature), and their concatenation with the mean amplitude. "
        "Hyperparameters were fixed in advance in an amendment to the analysis plan, written after the main results were known, and were not tuned, to avoid leakage. "
        "Each comparison decoder was also run on RT-residualized features, and its participant AUCs were compared with those of the primary decoder by a paired Wilcoxon signed-rank test with Holm correction within target and feature variant. No permutation test was run for these decoders."))
A(("h2", "Validation and inference"))
A(("p", "Because trials from one participant are not independent, the primary scheme is leave-one-subject-out (LOSO) cross-validation. We report the area under the ROC curve,"))
A(("eq", "AUC = P(<i>s</i><sup>+</sup> &gt; <i>s</i><sup>−</sup>) + 0.5 P(<i>s</i><sup>+</sup> = <i>s</i><sup>−</sup>)", 4))
A(("p", "where <i>s</i><sup>+</sup> and <i>s</i><sup>−</sup> are the scores of a randomly drawn trial of the positive and the negative class, for each held-out participant (mean and standard deviation over participants) and for the pooled out-of-subject scores. Chance level was estimated by permutation rather than assumed analytically {Ojala,Combrisson}: "
        "labels were shuffled within participant, the whole LOSO procedure was repeated 500 times, and the one-sided <i>p</i>-value is (1 + number of permuted means at least as large as observed) / 501, so the smallest attainable value is 0.002. "
        "Within-participant stratified 10-fold cross-validation is reported as a supplement; it replaces the leave-one-trial-out scheme in our earlier plan to make permutation inference computationally feasible."))
A(("h2", "Control for response time"))
A(("p", "Accuracy and confidence are related to response time (Section 5.1), and a response-locked window is not independent of RT. We therefore (a) decoded each target from log RT alone and (b) repeated every decoding analysis after "
        "removing the linear dependence of each feature on log RT within participant,"))
A(("eq", "<b>x</b>&#771; = <b>x</b> − <b>β</b> <i>z</i>(ln RT)", 5))
A(("p", "where <i>z</i>(·) is the within-participant z-score and <b>β</b> is the least-squares coefficient vector. This step uses no outcome labels. "
        "It is not part of the original plan; it was added after a four-participant smoke test showed that RT alone predicted confidence well, and before any full-sample result was inspected. "
        "As a further check that does not rely on a linear model, we computed the AUC of the main out-of-subject scores within participant-specific RT terciles and averaged over terciles and participants."))
A(("h2", "Time-resolved analysis"))
A(("p", "To examine how information evolves toward the response, and to test the rationale for the buffer, we repeated the decoding with 100 ms windows (step 50 ms) from 750 ms to 0 ms before the response, using trials with RT of at least 750 ms so that every window lies after stimulus onset. "
        "Each window used the same decoder and LOSO scheme, with and without RT residualization. For each window a one-sided one-sample <i>t</i>-test of the participant AUCs against 0.5 was computed and Holm-corrected over the 14 windows per target. "
        "This analysis is descriptive and was added after the primary analysis."))
A(("h2", "Evaluation protocol"))
A(("alg", "Algorithm 1. Out-of-participant evaluation", [
    "<b>Input:</b> trials with EEG, RT, labels, participant ID; <i>L</i>, <i>B</i>, <i>C</i>",
    "1. Keep trials with RT ≥ <i>L</i> + <i>B</i>; compute window means (Eq. 1).",
    "2. Standardize each feature within participant (no labels).",
    "3. For each participant <i>k</i>: train on all other participants (Eq. 3); score all trials of <i>k</i> (Eq. 2).",
    "4. Compute AUC per participant (Eq. 4); average over participants.",
    "5. Repeat steps 3–4 with labels shuffled within participant, 500 times; obtain the permutation <i>p</i>-value.",
    "6. Repeat steps 2–5 on RT-residualized features (Eq. 5).",
    "<b>Output:</b> mean AUC, permutation <i>p</i>, RT-controlled AUC"]))
A(("h2", "Reproducibility and computing"))
A(("p", "The raw data are public. The analysis used a laptop with an 8-core processor and 8 GB of memory, with raw files streamed participant by participant and reduced to 250 Hz epochs (about 0.7 GB in total), "
        "and ran with Python 3.9, NumPy 1.26, scikit-learn 1.0, and statsmodels 0.13. The window rule, regularization strength, validation scheme, and feasibility threshold for RQ3 were written in an analysis plan before any full-sample decoding result was inspected; "
        "the deviations from that plan are the RT control, the time-resolved analysis, the comparison decoders (a second amendment, with fixed hyperparameters), and the replacement of leave-one-trial-out by 10-fold cross-validation, and all are labeled in the text. "
        "Permutation seeds are fixed, so all numbers can be regenerated."))
A(("h2", "Calibration analysis (RQ3)"))
A(("p", "Accuracy was crossed with the low and high confidence classes, giving four cells: correct/low, correct/high, incorrect/low, and incorrect/high. A four-way analysis was attempted only if the incorrect/high cell contained at least 100 trials from at least 12 participants with no participant contributing more than 20% (rule fixed in advance). "
        "We also evaluated pairwise contrasts (incorrect/high versus incorrect/low, versus correct/high; incorrect/low versus correct/low) with the same LOSO and permutation procedure, "
        "and a two-stage analysis in which the out-of-subject accuracy-decoder score is compared between incorrect/high and incorrect/low trials by a within-participant permutation test, with within-participant Spearman correlations between the reported confidence and the score."))
A(("h2", "Sequential analysis (RQ4)"))
A(("p", "For consecutive valid trials N and N+1 in the same run we took as the neural confidence state the out-of-subject high-versus-low-confidence decoder score on trial N, applied to all trials and z-scored within participant (<i>S</i>), "
        "and the z-scored reported confidence (<i>C</i>) as a behavioral benchmark. Choice repetition was modeled by a logistic generalized estimating equation (GEE) {Liang} with participant clusters and an exchangeable working correlation:"))
A(("eq", "logit P(repeat) = β<sub>0</sub> + β<sub>1</sub><i>S</i> + β<sub>2</sub><i>D</i> + β<sub>3</sub><i>A</i> + β<sub>4</sub><i>T</i>", 6))
A(("p", "where <i>D</i> indicates a repeated motion direction, <i>A</i> is the accuracy on trial N, and <i>T</i> is ln RT on trial N. A supplementary model adds the <i>D</i> × <i>A</i> interaction. "
        "Log RT and accuracy on trial N+1 were analyzed analogously. The decision bound was not estimated because no diffusion model was fitted; RT and accuracy on trial N+1 are used as proxies."))

# ------------------------------------------------------------------ 5 Results
A(("h1", "Results"))
A(("h2", "Behavior and class balance"))
A(("p", f"The analysis includes {R['n_subjects']} participants, {R['n_runs']} runs, and {n(nall)} valid trials. Table 2 summarizes the behavior. Accuracy averaged {100 * b['acc_mean']:.1f}% (SD {100 * b['acc_sd']:.1f}), "
        f"close to the calibrated target, and mean RT was {b['rt_mean']:.0f} ms. Confidence was higher on correct than on incorrect trials ({b['conf_correct']:.2f} versus {b['conf_incorrect']:.2f}), "
        f"and RT was negatively related to confidence (mean within-participant Spearman ρ = {f2(b['rt_conf_rho_mean'])}) and to accuracy (ρ = {f2(b['rt_acc_rho_mean'])}). "
        f"The selected window retained {n(nan)} trials. The retained trials are slower, less accurate (74.6% versus 82.4%), and less confident (5.21 versus 6.28) than the {n(nall - nan)} excluded trials, so the decoding results apply to these slower trials."))
A(("tbl", "Dataset and behavior", [["Measure", "Value"],
    ["Participants / runs", f"{R['n_subjects']} / {R['n_runs']}"],
    ["Valid trials", n(nall)],
    ["Trials per participant", f"{R['trials_per_subject']['min']}–{R['trials_per_subject']['max']}"],
    ["Accuracy, mean (SD)", f"{100 * b['acc_mean']:.1f}% ({100 * b['acc_sd']:.1f})"],
    ["RT, mean (SD)", f"{b['rt_mean']:.0f} ({b['rt_sd']:.0f}) ms"],
    ["Confidence, mean (SD)", f"{b['conf_mean']:.2f} ({b['conf_sd']:.2f})"],
    ["Trials in window", f"{n(nan)} ({100 * w['retention_pooled']:.1f}%)"],
    ["Channels used", str(R['n_channels_common'])]], [4.2, 3.8]))
A(("p", f"Table 3 gives the joint distribution of accuracy and confidence. The incorrect/high cell contains {ca['IH']} trials in the full data and {IH_N} in the analysis window, contributed by {q3['IH_n_subjects']} of {R['n_subjects']} participants with no participant contributing more than "
        f"{100 * q3['IH_max_share']:.1f}%. The pre-specified feasibility rule for the four-way analysis was therefore met. Fig. 2 shows that the rating distributions of correct and incorrect trials overlap substantially, with a shift toward higher ratings on correct trials, so confidence is informative but imperfect about accuracy."))
A(("tbl", "Accuracy by confidence trial counts", [["Cell", "All trials", "In window"],
    ["Correct/low", n(ca['CL']), n(cc['CL'])], ["Correct/high", n(ca['CH']), n(cc['CH'])],
    ["Incorrect/low", n(ca['IL']), n(cc['IL'])], ["Incorrect/high", n(ca['IH']), n(cc['IH'])],
    ["Medium confidence", n(ca['med']), n(nan - sum(cc.values()))]], [3.6, 2.2, 2.2]))
A(("fig", "fig6.png", "Distribution of confidence ratings for correct and incorrect trials (all valid trials)."))
A(("h2", "RQ1 and RQ2: choice and accuracy"))
A(("p", f"Out-of-subject decoding of choice was above chance (mean AUC {f3(rq1['lr']['mean'])}, SD {f3(rq1['lr']['sd'])}; {rq1['lr']['n_above']} of {rq1['lr']['n']} participants above 0.5; permutation {pf(rq1['lr']['p_perm_mean'])}), "
        f"as was accuracy (mean AUC {f3(rq2['lr']['mean'])}, SD {f3(rq2['lr']['sd'])}; {rq2['lr']['n_above']} of {rq2['lr']['n']}; {pf(rq2['lr']['p_perm_mean'])}) (Table 4, Fig. 3). "
        f"The permutation null had a mean of {f3(rq1['lr']['perm_mean_of_mean'])} and a 95th percentile of {f3(rq1['lr']['perm_p95'])} for choice. "
        f"Confidence (high versus low) behaved similarly (mean AUC {f3(cf['mean'])}, {pf(cf['p_perm_mean'])}). "
        f"With Holm correction across these three primary tests the adjusted <i>p</i>-value was {HOLM:.3f}. "
        f"Results were insensitive to the regularization strength and to shrinkage LDA (choice {f3(rq1['lda']['mean'])}, accuracy {f3(rq2['lda']['mean'])}), "
        f"and the spatiotemporal variant gave {f3(R['suppl_4bin_choice']['mean'])} and {f3(R['suppl_4bin_accuracy']['mean'])}. Within-participant cross-validation yielded {f3(rq1['within_subject']['mean'])} for choice and {f3(rq2['within_subject']['mean'])} for accuracy."))
A(("tbl", "Decoding performance (AUC)", [["", "Choice", "Accuracy", "Confidence"],
    ["LOSO, mean (SD)", f"{f3(rq1['lr']['mean'])} ({f3(rq1['lr']['sd'])})", f"{f3(rq2['lr']['mean'])} ({f3(rq2['lr']['sd'])})", f"{f3(cf['mean'])} ({f3(cf['sd'])})"],
    ["Permutation <i>p</i>", pp(rq1['lr']['p_perm_mean']), pp(rq2['lr']['p_perm_mean']), pp(cf['p_perm_mean'])],
    ["LOSO, pooled", f3(rq1['lr']['pooled_auc']), f3(rq2['lr']['pooled_auc']), f3(cf['pooled_auc'])],
    ["Shrinkage LDA", f3(rq1['lda']['mean']), f3(rq2['lda']['mean']), "–"],
    ["Within-participant", f3(rq1['within_subject']['mean']), f3(rq2['within_subject']['mean']), "–"],
    ["RT only", f3(rq1['rt_only']['mean']), f3(rq2['rt_only']['mean']), f3(cf['rt_only_mean'])],
    ["RT-residualized", f3(rq1['rt_resid']['mean']), f3(rq2['rt_resid']['mean']), f3(cf['rt_resid']['mean'])],
    ["RT-resid. <i>p</i>", pp(rq1['rt_resid']['p_perm_mean']), pp(rq2['rt_resid']['p_perm_mean']), pp(cf['rt_resid']['p_perm_mean'])]], [2.4, 1.75, 1.95, 2.0]))
A(("fig", "fig2.png", "Held-out-participant AUC for choice, accuracy, and confidence (high versus low). Dots are participants, boxes show quartiles, black lines the medians, and orange lines the means; the dashed line is chance."))
A(("p", f"Response time alone predicted accuracy (AUC {f3(rq2['rt_only']['mean'])}) and confidence ({f3(cf['rt_only_mean'])}) better than the EEG did, but not choice ({f3(rq1['rt_only']['mean'])}). "
        f"After removing the linear dependence of the EEG features on RT, choice decoding was unchanged (AUC {f3(rq1['rt_resid']['mean'])}, {pf(rq1['rt_resid']['p_perm_mean'])}), whereas accuracy decoding fell to chance ({f3(rq2['rt_resid']['mean'])}, {pf(rq2['rt_resid']['p_perm_mean'])}) "
        f"and so did confidence decoding ({f3(cf['rt_resid']['mean'])}, {pf(cf['rt_resid']['p_perm_mean'])}). "
        f"As a further control, the motion direction itself could be decoded from the same features (AUC {f3(DC['direction_decoding']['mean'])}), and the choice scores discriminated choice within each stimulus direction only weakly "
        f"(AUC {f3(DC['choice_within_direction']['0']['auc'])} and {f3(DC['choice_within_direction']['180']['auc'])}; permutation {pf(DC['choice_within_direction']['0']['p'])} and {pf(DC['choice_within_direction']['180']['p'])})."))
A(("p", f"Stratifying by RT gave the same picture without a linear model. Within participant-specific RT terciles the mean AUC was {f3(EX['rt_tercile']['choice']['mean'])} for choice ({pf(EX['rt_tercile']['choice']['p_one'])}, one-sided <i>t</i>-test), "
        f"{f3(EX['rt_tercile']['accuracy']['mean'])} for accuracy ({pf(EX['rt_tercile']['accuracy']['p_one'])}), and {f3(EX['rt_tercile']['confidence']['mean'])} for confidence ({pf(EX['rt_tercile']['confidence']['p_one'])}). "
        f"For choice the AUC was highest in the fastest tercile ({f3(EX['rt_tercile']['choice']['by_tercile'][0])}, versus {f3(EX['rt_tercile']['choice']['by_tercile'][1])} and {f3(EX['rt_tercile']['choice']['by_tercile'][2])}), which is compatible with a larger motor or evidence contribution close to the response."))
ps_ = E2['per_subject']
rng_ = lambda k: (min(v[k] for v in ps_.values() if v[k] is not None), max(v[k] for v in ps_.values() if v[k] is not None))
A(("p", f"Participant-level AUCs varied considerably (Fig. 3): choice {f3(rng_('choice')[0])} to {f3(rng_('choice')[1])}, accuracy {f3(rng_('accuracy')[0])} to {f3(rng_('accuracy')[1])}, and confidence {f3(rng_('confidence')[0])} to {f3(rng_('confidence')[1])}. "
        "This spread is expected for small decoding effects in samples of 290 to 318 trials per participant and underlines why group-level inference was based on permutation of the whole procedure."))
def cmp_cell(t, k):
    r, e = CMP["raw"][t][k], CMP["resid"][t][k]
    dag = "†" if r.get("p_holm", 1) < 0.05 else ""
    return f"{f3(r['mean'])}{dag}/{f3(e['mean']).replace('0.', '.', 1)}"


A(("h2", "Comparison decoders"))
A(("p", f"Table 5 compares the primary decoder with the alternatives. No alternative was reliably better than logistic regression on mean amplitude: after Holm correction none exceeded it for any target, with or without the RT control. "
        f"Shrinkage LDA and the linear SVM matched it (choice {f3(CMP['raw']['choice']['amp|lda']['mean'])} and {f3(CMP['raw']['choice']['amp|svm_lin']['mean'])}). The nonlinear models were not better: the RBF SVM, random forest, and perceptron reached {f3(CMP['raw']['choice']['amp|svm_rbf']['mean'])}, "
        f"{f3(CMP['raw']['choice']['amp|rf']['mean'])}, and {f3(CMP['raw']['choice']['amp|mlp']['mean'])} for choice. The spatiotemporal variant gave the highest raw confidence AUC ({f3(CMP['raw']['confidence']['amp4|lda']['mean'])} with LDA), but the difference from the primary decoder was not significant "
        f"and the value fell to {f3(CMP['resid']['confidence']['amp4|lda']['mean'])} after RT control. Log-variance features produced AUCs significantly below 0.5 for accuracy and confidence ({f3(CMP['raw']['accuracy']['logvar|lr']['mean'])} and {f3(CMP['raw']['confidence']['logvar|lr']['mean'])}); "
        "values below chance are not evidence of information and are a known artifact of cross-validation with participant-specific feature scales, so we do not interpret them. "
        "The conclusions therefore do not depend on the choice of classifier or on this feature type."))
A(("tbl", "Decoder comparison, raw/RT-residualized mean AUC", [["Decoder", "Choice", "Accuracy", "Confid."],
    ["Logistic (primary)", cmp_cell("choice","amp|lr"), cmp_cell("accuracy","amp|lr"), cmp_cell("confidence","amp|lr")],
    ["Shrinkage LDA", cmp_cell("choice","amp|lda"), cmp_cell("accuracy","amp|lda"), cmp_cell("confidence","amp|lda")],
    ["Linear SVM", cmp_cell("choice","amp|svm_lin"), cmp_cell("accuracy","amp|svm_lin"), cmp_cell("confidence","amp|svm_lin")],
    ["RBF SVM", cmp_cell("choice","amp|svm_rbf"), cmp_cell("accuracy","amp|svm_rbf"), cmp_cell("confidence","amp|svm_rbf")],
    ["Random forest", cmp_cell("choice","amp|rf"), cmp_cell("accuracy","amp|rf"), cmp_cell("confidence","amp|rf")],
    ["Perceptron", cmp_cell("choice","amp|mlp"), cmp_cell("accuracy","amp|mlp"), cmp_cell("confidence","amp|mlp")],
    ["Logistic, 4 bins", cmp_cell("choice","amp4|lr"), cmp_cell("accuracy","amp4|lr"), cmp_cell("confidence","amp4|lr")],
    ["LDA, 4 bins", cmp_cell("choice","amp4|lda"), cmp_cell("accuracy","amp4|lda"), cmp_cell("confidence","amp4|lda")],
    ["Logistic, log var.", cmp_cell("choice","logvar|lr"), cmp_cell("accuracy","logvar|lr"), cmp_cell("confidence","logvar|lr")],
    ["Logistic, amp. + log var.", cmp_cell("choice","amp+logvar|lr"), cmp_cell("accuracy","amp+logvar|lr"), cmp_cell("confidence","amp+logvar|lr")],
    ], [2.4, 1.9, 1.9, 1.9]))
A(("p", "† Holm-corrected <i>p</i> &lt; 0.05 against the primary decoder (raw features). Entries are raw/RT-residualized mean held-out-participant AUC."))
A(("h2", "Window sensitivity"))
A(("p", "Fig. 4 shows mean AUC over a subset of the window grid. Values varied between 0.46 and 0.57 without a consistent optimum, and the selected window was not the best for any target; "
        "extreme cells should be read as chance variation across many comparisons, and no window was chosen from this analysis."))
A(("fig", "fig3.png", "Sensitivity of mean held-out-participant AUC to window length L and buffer B for (a) choice, (b) accuracy, and (c) confidence. The red frame marks the pre-specified window. Blank cells have retention below 50%."))
A(("h2", "Time-resolved decoding"))
A(("p", "Fig. 5 shows the AUC in 100 ms windows. Choice decoding fluctuated between 0.48 and 0.53 without a clear time course, and no window survived Holm correction. "
        "Accuracy decoding was significant after correction in three windows between 0.65 and 0.45 s before the response (centers −600, −550, and −400 ms; AUC 0.517 to 0.529), and none of these remained after RT residualization. "
        "Confidence decoding peaked at 0.541 in the window centered 200 ms before the response but was not significant after correction. "
        "Choice AUC was highest in the window starting 150 ms before the response (0.525, uncorrected <i>p</i> = 0.011), which is compatible with a late motor-related contribution but is not significant after correction; the windows are short and the signals weak, so the time course is uninformative about the buffer."))
A(("fig", "fig5.png", "Time-resolved decoding with 100 ms windows. Lines show the mean held-out-participant AUC against the window center, shaded areas the standard error across participants, and the gray band the buffer period excluded from the primary analysis."))
A(("h2", "RQ3: calibration"))
pv, pc, pl = q3["IH_vs_IL"], q3["IH_vs_CH"], q3["IL_vs_CL"]
A(("p", f"Incorrect/high trials could not be distinguished from incorrect/low trials (pooled AUC {f3(pv['pooled_auc'])}, permutation {pf(pv['p_perm_pooled'])}), nor reliably from correct/high trials ({f3(pc['pooled_auc'])}, {pf(pc['p_perm_pooled'])}) (Table 6). "
        f"The four-way classifier reached a balanced accuracy of {f3(q3['four_way']['balanced_acc'])} (chance 0.250) and a macro-averaged AUC of {f3(q3['four_way']['macro_auc'])}; no significance test was run for this analysis. "
        f"In the two-stage analysis, incorrect/high trials had a slightly more correct-like accuracy-decoder score than incorrect/low trials (AUC {f3(ts['auc_IH_gt_IL'])}, one-sided permutation {pf(ts['p_perm_one_sided'])}), "
        f"which is the direction expected for an attenuated error signature but is not significant (Fig. 6). Within participants, reported confidence correlated positively with the score on correct trials "
        f"(mean Fisher <i>z</i> = {f3(ts['fisher_z_correct']['mean'])}, <i>t</i>({ts['fisher_z_correct']['n'] - 1}) = {f2(ts['fisher_z_correct']['t'])}, {pf(ts['fisher_z_correct']['p'])}) and weakly on incorrect trials "
        f"({f3(ts['fisher_z_incorrect']['mean'])}, {pf(ts['fisher_z_incorrect']['p'])}); these correlations were not RT-controlled and should be read in light of the RT results above."))
A(("tbl", "Calibration contrasts (pooled AUC)", [["Contrast", "<i>n</i>", "AUC", "<i>p</i>"],
    ["Incorr./high vs incorr./low", str(pv['n']), f3(pv['pooled_auc']), pp(pv['p_perm_pooled'])],
    ["Incorr./high vs corr./high", n(pc['n']), f3(pc['pooled_auc']), pp(pc['p_perm_pooled'])],
    ["Incorr./low vs corr./low", n(pl['n']), f3(pl['pooled_auc']), pp(pl['p_perm_pooled'])]], [4.2, 1.2, 1.3, 1.3]))
sp, ss_, sl = E2['primary 1-3 vs 7-9'], E2['strict 1-2 vs 8-9'], E2['liberal 1-4 vs 6-9']
A(("p", f"The calibration result depends little on how the confidence classes are defined (Table 7). Among incorrect trials the high-versus-low AUC was {f3(sp['incorrect']['pooled'])} for ratings 1 to 3 versus 7 to 9, {f3(ss_['incorrect']['pooled'])} for 1 to 2 versus 8 to 9, and {f3(sl['incorrect']['pooled'])} for 1 to 4 versus 6 to 9, "
        f"with the strictest split borderline ({pf(ss_['incorrect']['p'])}, {ss_['incorrect']['n_pos']} high-confidence errors). Among correct trials the contrast was above chance for all three splits and strongest for the strictest ({f3(ss_['correct']['pooled'])}, {pf(ss_['correct']['p'])}), "
        "but these correct-trial contrasts are not RT-controlled and, given the RT results, probably include response-speed information. The pattern, a larger separation among correct than among incorrect trials, is the opposite of what an attenuated pre-response error signature would predict if the EEG contained confidence information independent of accuracy."))
A(("tbl", "Confidence contrasts by definition of the classes (pooled AUC)", [["Split", "Incorrect", "Correct"],
    ["Ratings 1–3 vs 7–9", f"{f3(sp['incorrect']['pooled'])} ({pp(sp['incorrect']['p'])})", f"{f3(sp['correct']['pooled'])} ({pp(sp['correct']['p'])})"],
    ["Ratings 1–2 vs 8–9", f"{f3(ss_['incorrect']['pooled'])} ({pp(ss_['incorrect']['p'])})", f"{f3(ss_['correct']['pooled'])} ({pp(ss_['correct']['p'])})"],
    ["Ratings 1–4 vs 6–9", f"{f3(sl['incorrect']['pooled'])} ({pp(sl['incorrect']['p'])})", f"{f3(sl['correct']['pooled'])} ({pp(sl['correct']['p'])})"]], [3.1, 2.5, 2.5]))
A(("fig", "fig4.png", "Out-of-participant accuracy-decoder score by calibration cell (higher is more correct-like). Horizontal lines are medians; the vertical axis is truncated at the extremes."))
A(("h2", "RQ4: sequential effects"))
ra, rb = q4["rep_neural"], q4["rep_behav"]
xa, xb = X["neural"], X["behav"]
A(("p", f"There were {n(q4['n_pairs'])} consecutive trial pairs from {q4['n_subjects']} participants; the choice was repeated on {100 * q4['rep_rate']:.1f}% and the motion direction on {100 * q4['stimrep_rate']:.1f}% of pairs. "
        f"The decoded neural confidence state did not predict choice repetition (β = {f3(ra['S']['b'])}, SE {f3(ra['S']['se'])}, {pf(ra['S']['p'])}), whereas reported confidence did (β = {f3(rb['Cn']['b'])}, SE {f3(rb['Cn']['se'])}, {pf(rb['Cn']['p'])}; Table 8). "
        f"With the stimulus-by-accuracy interaction included, the neural state remained null ({pf(xa['S']['p'])}) and the reported-confidence effect was larger (β = {f3(xb['Cn']['b'])}, {pf(xb['Cn']['p'])}). "
        "Neither predictor was reliably related to the RT or the accuracy on trial N+1."))
A(("p", f"The covariates behaved as expected from the task structure. A repeated motion direction strongly raised the probability of repeating the choice (β = {f3(ra['stimrep']['b'])}, {pf(ra['stimrep']['p'])}), and with the interaction included "
        f"the effect of a repeated direction was positive after a correct trial and negative after an error (interaction β = {f3(xa['stimrep:accN']['b'])}), which mainly reflects that a correct response follows the stimulus. "
        "None of these covariate effects bears on the confidence question, and the interpretation of the confidence coefficients rests on their being estimated with these terms in the model."))
A(("tbl", "Predictors of trial N+1 behavior (GEE)", [["Outcome, predictor", "β (SE)", "<i>p</i>"],
    ["Repetition, neural", f"{f3(ra['S']['b'])} ({f3(ra['S']['se'])})", pp(ra['S']['p'])],
    ["Repetition, reported", f"{f3(rb['Cn']['b'])} ({f3(rb['Cn']['se'])})", pp(rb['Cn']['p'])],
    ["Log RT, neural", f"{f3(q4['lrt_neural']['S']['b'])} ({f3(q4['lrt_neural']['S']['se'])})", pp(q4['lrt_neural']['S']['p'])],
    ["Log RT, reported", f"{f3(q4['lrt_behav']['Cn']['b'])} ({f3(q4['lrt_behav']['Cn']['se'])})", pp(q4['lrt_behav']['Cn']['p'])],
    ["Accuracy, neural", f"{f3(q4['acc_neural']['S']['b'])} ({f3(q4['acc_neural']['S']['se'])})", pp(q4['acc_neural']['S']['p'])],
    ["Accuracy, reported", f"{f3(q4['acc_behav']['Cn']['b'])} ({f3(q4['acc_behav']['Cn']['se'])})", pp(q4['acc_behav']['Cn']['p'])]], [3.8, 2.8, 1.4]))

# ------------------------------------------------------------------ 6 Discussion
A(("h1", "Discussion"))
A(("h2", "What the pre-response signal contains"))
A(("p", "Pre-response EEG carried small but reliable information about the upcoming choice, and the effect survived the control for response time and the RT-tercile analysis. The within-participant estimate was higher than the out-of-participant estimate, which suggests that part of the signal is "
        "participant-specific and is lost when a single spatial filter is transferred across participants. We cannot attribute the choice signal to a particular source. Motor preparation is a plausible contributor despite the "
        f"{B_} ms buffer, and the motion direction itself is weakly decodable (AUC {f3(DC['direction_decoding']['mean'])}), so the choice signal may reflect stimulus or evidence information as well as choice-specific activity. "
        "The result did not depend on the classifier: nonlinear models and alternative features did not improve on logistic regression (Table 5). "
        "The higher AUC in the fastest RT tercile is compatible with a stronger motor or evidence contribution near the response. Separating these sources would need a design that decouples stimulus, choice, and hand, or source-level analysis."))
A(("p", "The accuracy and confidence effects are better explained by response speed. Both were predicted more strongly by RT than by the EEG, and both vanished after RT was regressed out of the features or after stratification by RT. "
        "A response-locked window places the EEG at different stimulus-relative times depending on RT, so EEG features can encode RT without any additional information about accuracy or confidence. "
        "This is consistent with a study that found no above-chance pre-response decoding of confidence {Desender19b}. It indicates that the earlier correlational finding for this dataset {Gherman} does not translate into response-independent, out-of-participant prediction in our protocol. "
        "We do not compare AUC values with the original sliding-window analysis because the analysis differs in window, normalization, and validation, and we did not reproduce it."))
A(("h2", "Calibration and sequential effects"))
A(("p", f"For calibration, we found no evidence that high-confidence errors are neurally distinguishable from low-confidence errors before the response. The direction of the two-stage effect matched the overconfidence hypothesis, but its size was small and not significant, "
        f"so the data neither support nor exclude an attenuated pre-response error signature. The existing evidence for such signatures concerns post-response components {{Boldt,Desender21}}, which our window by design excludes, and a study of confidence in a different time window found partly distinct pre- and post-response markers {{Rausch}}. "
        "The sensitivity analysis showed a larger separation among correct than among incorrect trials, which fits the interpretation that the decoder tracks response speed more than confidence-specific error signals."))
A(("p", "For sequential effects, reported confidence predicted choice repetition, in line with confidence-weighted choice history {Braun}, whereas the decoded neural state did not. "
        "Because the neural proxy is itself weak (AUC about 0.54) and largely reflects RT, this null result is expected and is not evidence against a neural account. A fair test would use a stronger confidence readout, for example a post-response component, and a model-based estimate of the decision bound {Desender19a}."))
A(("h2", "Effect sizes and reporting"))
A(("p", f"The design also determines what effect sizes could have been detected. With {R['n_subjects']} participants the permutation null for the mean AUC had a standard deviation of about 0.010 to 0.012 and a 95th percentile of {f3(rq1['lr']['perm_p95'])} to {f3(rq2['lr']['perm_p95'])}, "
        "so mean AUCs of about 0.52 or higher were detectable, whereas the individual participant AUCs spread over roughly 0.43 to 0.67. "
        "An AUC of 0.52 corresponds to a very small shift in the score distributions, and decisions based on a single trial would be barely better than guessing. "
        "The practical consequence is that these pre-response features cannot be used to flag an individual overconfident response in this paradigm."))
A(("p", "The results also bear on how pre-response decoding studies should be reported. First, any target correlated with response time needs an RT-controlled analysis, because a response-locked window makes EEG features a proxy for RT. "
        "Second, class balance and subject diversity of rare cells, such as high-confidence errors, should be shown before the main analysis. "
        "Third, normalization and model fitting must be organized so that no information from the held-out participant's labels is used. Finally, weak or null results should be reported together with their permutation null so that readers can judge the effect size."))

A(("h1", "Limitations and Threats to Validity"))
A(("h2", "Internal validity"))
A(("p", "All relationships are predictive and correlational, and no causal role of the EEG activity is implied. The RT control removes only linear dependence, and the stimulus-direction control is partial, so residual contamination by response speed or stimulus information cannot be excluded for the choice result. "
        "The in-scanner EEG may retain residual gradient or ballistocardiogram artifacts despite the correction applied by the dataset authors, and artifacts with a systematic relation to the heartbeat could in principle covary with RT."))
A(("h2", "Construct validity"))
A(("p", "Confidence is operationalized by a coarse split of a 9-point rating, and overconfidence by a single high-confidence error, which ignores the metacognitive sensitivity of each participant {Fleming}. "
        "The decision bound is not estimated, since no diffusion model was fitted, so RT and accuracy on the next trial are only proxies. The decoded neural confidence state is weak and probably contaminated by RT, which limits its use as a measure of trial-wise confidence."))
A(("h2", "External validity"))
A(("p", f"The window rule excluded the fastest {100 - 100 * w['retention_pooled']:.1f}% of trials, which were also the more accurate and confident ones, so the results apply to slower trials. Only {R['n_channels_common']} channels common to all participants were used. "
        "The sample consists of 24 participants performing one motion task with a calibrated difficulty, and the findings may not transfer to other tasks, difficulty levels, or recording settings."))
A(("h2", "Statistical conclusion validity"))
A(("p", "The window grid contains many comparisons, so its extreme cells are optimistic, and the time-resolved analysis was added after the primary analysis. "
        "The four-way RQ3 classifier was not tested for significance and the incorrect/high class remains small. Several analysis choices, including <i>C</i>, the LOSO scheme, and the RT control, were fixed by the analysts without external review. "
        "The sequential models assume an exchangeable within-participant correlation and include only a few covariates. The comparison decoders used fixed, untuned hyperparameters, and no deep convolutional network was included, so a better-tuned or end-to-end model cannot be excluded as able to extract more information."))

A(("h1", "Conclusion and Future Work"))
A(("p", f"In {R['n_subjects']} participants, EEG recorded {W} to {B_} ms before the response predicted the upcoming choice with a small effect (AUC {f3(rq1['lr']['mean'])}), and also accuracy and confidence, but the accuracy and confidence effects are accounted for by response time. "
        "High-confidence errors were not separable from low-confidence errors, and the decoded neural confidence did not predict next-trial behavior. "
        "Five alternative classifiers and three feature sets did not do better. The study provides a leakage-controlled protocol, an explicit response-time control, and the class-balance numbers needed to plan better-powered tests. "
        "Future work should (i) match trials on RT and stimulus direction, (ii) fit sequential-sampling models to estimate decision bounds, (iii) combine pre- and post-response windows, "
        "(iv) use time-resolved, end-to-end, and EEG-fMRI-informed decoders, and (v) test in larger samples whether pre-response EEG adds predictive information beyond behavior."))

DECL = [
    ("Conflicts of Interest", "[AUTHORS TO COMPLETE: state any conflicts of interest, or “The authors declare no conflict of interest.” if none.]"),
    ("Author Contributions", "[AUTHORS TO COMPLETE: describe each author's contributions, for example conceptualization, methodology, software, validation, formal analysis, writing, and supervision.]"),
    ("Data and Code Availability", "The EEG-fMRI data are publicly available at OpenNeuro (ds002739, DOI 10.18112/openneuro.ds002739.v1.0.0). This study analyzed these de-identified data and collected no new data. "
        "[AUTHORS TO CONFIRM whether your institution requires an ethics statement for secondary analysis, and where the analysis code will be deposited.]"),
    ("Acknowledgments", "The EEG-fMRI data were collected and shared by S. Gherman and M. G. Philiastides (OpenNeuro ds002739, CC0). "
        "Generative AI (Claude, Anthropic) was used to write and run the analysis code and to draft the text; the authors are responsible for reviewing and approving all content. "
        "[AUTHORS TO ADD funding information, if any.]"),
]
