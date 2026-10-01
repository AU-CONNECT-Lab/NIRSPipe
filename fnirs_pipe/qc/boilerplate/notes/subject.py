"""Section prose for the subject reports: one run, a subject's runs, and the cohort."""

from __future__ import annotations

NOTES = {
    # ---- Fragments several paragraphs name a channel set with ----
    "channels.long": "long channels",
    "channels.long_only": "long channels only",
    "channels.every": "every channel",

    # ---- Subject report ----
    "summary.condition_scope":
        "Sliced out of the windowed pass the whole recording was measured with, so this page "
        "sits on the same window grid and the same filter as the run and as every other "
        "condition. The verdict is this condition&rsquo;s own; the recording was processed "
        "under the run&rsquo;s.",
    "summary.condition_rebuilt":
        "The epoch and topography panels are rebuilt on a crop of this condition; "
        "preprocessing and GLM are empty here and are on the run&rsquo;s page.",
    "summary.condition_empty":
        "Preprocessing, epoch, topography and GLM are empty here and are on the run&rsquo;s "
        "page.",
    "summary.condition_no_trials": "The trial panels are absent because {reason}.",
    "summary.condition_few_trials": "this condition holds too few trials to compare",
    "summary.condition_run_link": '<a href="{href}">Full report for the run</a>.',
    "steps.raw_signal":
        "Per-channel HbO/HbR timeseries and PSD, on <code>{stage}</code> converted with DPF = "
        "{dpf}. This is the recording before motion correction ({motion}) and before any "
        "filtering, the same stage the sliding-window SCI/PSP and the SNR/CV numbers are "
        "measured on.",
    "steps.raw_signal_fallback":
        "<strong>desc-sci was not readable for this run, so these are post-correction."
        "</strong>",
    "channel_detail.picker":
        "Select a channel to view its HbO/HbR timeseries and its PSD. This channel's epoch "
        "average is in the epoch section below, on the denoised signal. Pairs "
        "marked <em>(short)</em> are short-separation: too shallow to reach cortex, so what "
        "they show is scalp haemodynamics.",
    "steps.raw_quality":
        "Scalp coupling index (SCI), peak spectral power (PSP) and coefficient of variation "
        "(CV) per channel and per window, all measured on <code>desc-sci</code>, the optical "
        "density before motion correction. SCI and PSP cross-correlate the two wavelengths, so "
        "they exist only in the optical density domain and have no counterpart after "
        "Beer-Lambert. CV is the row where low is good, and its hover also gives SNR, which is "
        "1/CV exactly; its colour scale tops out at twice the <strong>{cv}</strong> line. SCI "
        "threshold: <strong>{sci}</strong>.",
    "quality.brain_views":
        "Red = rejected by the screening, whatever its SCI; every channel that survived is then "
        "graded green to amber by its own SCI against the run's line. The two are different "
        "questions: a channel is rejected on how many windows it was coupled in, so one can be "
        "dropped at a high SCI.",
    "quality.brain_views_condition":
        "Both maps carry this condition's own SCI and its own rejected channels, which is the "
        "verdict printed on this page. The recording was processed under the run's; its page "
        "has that map.",
    "steps.motion_correction":
        "Motion correction method: <strong>{method}</strong>. GVTD covers the "
        "<strong>{gvtd_set}</strong> channels, the set the analysis uses; the count is on the "
        "figure. It is an RMS across channels, so that set is part of the value.",
    "motion_detail.picker":
        "Select a channel to compare the OD signal before and after motion correction.",
    "motion_detail.condition":
        "Measured over the whole recording and viewed over this condition, like the carpet "
        "above it: the GVTD row on each figure is filtered and thresholded run-wide, so it is "
        "the run's trace with the axis narrowed.",
    "steps.beer_lambert": "Modified Beer-Lambert law applied with DPF = {dpf}.",
    "haemo.hbo_hbr_corr":
        "A genuine haemodynamic response drives HbO up and HbR down, so r near &minus;1 is the "
        "target and r near 0 or positive means a shared artifact.",
    "haemo.glm_residual":
        "The after stage on this run is the GLM residual, so the task model came out of it "
        "along with the confounds. A weaker anticorrelation there can mean the model explained "
        "part of the shared response rather than that the data got worse, so judge the run on "
        "the before stage.",
    "carpet.after":
        "Per-channel z-scored haemoglobin for {stage}, HbO above HbR in one image, a colour bar "
        "and a seam between them. One GVTD row per channel set sits above on the same time "
        "axis, each carrying the trace before and after motion correction, so a dark column "
        "can be read against what happened at that moment. The threshold, the spike shading "
        "and the spans the correction touched stay in the motion section above. The title "
        "carries how far this stage's per-channel SD has fallen; the recording before any of "
        "this is the carpet in the motion section above. Each block is scaled to itself, so it "
        "shows the structure left at this stage rather than its amplitude. The texture is "
        "finer here than on the raw carpet because the slow drift that made that one look "
        "smooth has been removed, not because noise was added: what is left is the analysis "
        "passband, and its fast end is near the limit this figure can resolve in time. The "
        "spectrum below reads that end.",
    "carpet.last_stage": "the last stage on disk",
    "carpet.glm":
        "On a GLM run this is the residual, which had the task model removed as well as the "
        "confounds: it shows what the model left unexplained, not the data the betas were "
        "measured on.",
    "carpet.condition":
        "The z-scoring is the whole recording's and only the columns drawn are this "
        "condition's, so every condition's carpet is on one greyscale and the darker stretch "
        "is the noisier one.",
    "psd.stages":
        "One row per stage file on disk, HbO and HbR together in each. Coloured bands: Mayer "
        "wave (~0.1&thinsp;Hz){bands}. Faint lines are the individual channels of the first "
        "stage; dashed verticals mark the filter cutoffs.",
    "psd.band_resp": ", respiration ({lo}–{hi}&thinsp;Hz)",
    "psd.band_cardiac": ", cardiac ({lo}–{hi}&thinsp;Hz)",
    "psd.stage_list": "Stages after <code>desc-preproc</code>: {stages}.",
    "psd.errts":
        "<code>desc-errts</code> is the confound regression's residual. Where a high-pass ran "
        "it should sit on top of the row above it, since the drift basis has little left to "
        "remove and the other regressors are measured signals rather than frequency bands; a "
        "visible gap inside the analysis band means the regression reached further than "
        "intended. Where no high-pass ran, this row carries the whole detrend.",
    "psd.simulated":
        "No post-processing output was found for this run, so the second line is the bandpass "
        "simulated in memory rather than a file the pipeline wrote.",
    "psd.too_short":
        "{holder} holds fewer samples than the transform, so no spectrum is drawn here. The "
        "run's own page carries the spectrum, and the PSP row above is sliced to this "
        "condition.",
    "psd.this_cut": "This cut",
    "psd_detail.picker": "Select a channel to compare its PSD across the stages.",
    "epoch.timeline":
        "Every event on one axis, one row per condition. The averages below cannot show a "
        "condition that stopped being delivered partway through or a block that was started "
        "twice; this can. Same panel the raw QC viewer opens with.",
    "epoch.timeline_condition":
        "<strong>This is the whole run, not just &ldquo;{label}&rdquo;</strong>: both of those "
        "failures show only against the conditions around the one you are reading.",
    "epoch.grand_mean":
        "Grand-mean HbO and HbR responses averaged across good channels, baseline-corrected to "
        "the pre-stimulus window (−5 to 0&thinsp;s). Each condition is shown separately, on one "
        "shared y scale, with the task block shaded. Short channels are drawn dotted: they are "
        "too shallow to reach cortex, so a dotted line that rises with the solid one means the "
        "response is scalp haemodynamics rather than activation.",
    "epoch.channel_maps":
        "The condition-averaged response painted along each channel's own source-detector path, "
        "one head per condition; drag the slider to step through the epoch window. Nothing is "
        "interpolated between channels, so bare scalp stays bare. Short channels get their own "
        "row on the chromophore's colour scale: they are too shallow to reach cortex, so a "
        "short row as strongly coloured as the long row above it means the response is systemic "
        "scalp signal rather than activation.",
    "epoch.trial_image_roi":
        "Per ROI, HbO channels averaged first (higher SNR), then each trial stacked as a row. "
        "Denoised signal; runs with enough repetitions are also smoothed across adjacent "
        "trials. One panel per condition, pooled panel first. Select an ROI.",
    "epoch.trial_image_single":
        "Each stimulus repetition is a row (trial × time from onset), coloured by HbO: the "
        "un-averaged view of the response, on the denoised signal. Runs with enough repetitions "
        "are also smoothed across adjacent trials. One panel per condition, pooled panel first. "
        "Select a channel.",
    "rest.alff":
        "Amplitude of Low-Frequency Fluctuations (ALFF) and fractional ALFF (fALFF), computed "
        "from the denoised residual time series. Where the amplitude is; how much it was and "
        "which channels it correlated with are in the connectivity panel below.",
    "rest.alff_layout":
        "One disc per channel at its source-detector midpoint. The amplitude row draws mALFF, "
        "each channel over its own chromophore's mean, so HbO and HbR share one scale and the "
        "measured molar value is on the hover; fALFF is already a share. Grey channels carry "
        "no value, which is what a rejected one has. Short channels are not drawn.",
    "fc.channel_matrix":
        "The run's channels in one panel: Pearson correlation between channels on top, HbO "
        "left and HbR right on one scale, and each channel's amplitude and spectral share "
        "below. All of it is computed from the denoised residual time series. The rows share "
        "one channel order and the same separation blocks, so a channel is in the same "
        "relative place throughout; they are not on one x scale, so the correspondence is by "
        "label and by block. Each matrix is drawn as a triangle, being symmetric. A rejected "
        "channel keeps its row and column, where it shows as a grey wedge, and its position "
        "in the strips, where it is a ring below the measured range.",
    "fc.roi_matrix":
        "The same correlations between ROI-averaged signals rather than between channels, on "
        "the same scale. Channels are averaged before correlating, so this is not the mean of "
        "the cells above. Needs --roi-mapping.",
    "fc.seed_topography":
        "One flat map per seed ROI, each channel coloured by its correlation with that seed's "
        "mean signal, on the same scale as the matrix above. Solid grey channels are inside the "
        "seed: their value is undefined, not zero. Faded grey channels were rejected, and are "
        "in no seed. Short channels are not drawn.",
    "glm.activation":
        "HbO betas projected onto the surface, three views per condition. The colour scale is "
        "shared across every condition, so a condition that barely activated reads as weak "
        "instead of being stretched to fill its own scale. One model over the whole recording, "
        "so every condition here rests on the same channel set. Select a condition.",
    "metrics.intro":
        'Hover a metric for what it means, which way is good and what it was measured on. <span '
        'class="ql-dot" aria-hidden="true">&#9679;</span> marks the ones that decide whether '
        'the run is usable. A pair written <b>a &rarr; b</b> is one metric measured either side '
        'of a processing step; a single number means that step did not run.',
    "metrics.long_only":
        "Long channels only. Short channels return far more light and a far stronger pulse, so "
        "averaging the two lifts SCI, PSP and SNR; they are judged on their own in the "
        "per-channel table below.",
    "metrics.set_help":
        "Long is the set the verdict is read off. Short is left uncoloured: a short channel's "
        "coupling is high by construction, so the column answers whether the regressors those "
        "channels feed are trustworthy rather than whether the run is usable. Rejection is per "
        "channel, against the screening criteria over the whole montage, and is not read off "
        "these averages.",
    "metrics.set_help_condition":
        "On this page every column is this condition's, and the channel sets are the run's, "
        "since one montage has to serve every condition.",
    "metrics.stages":
        "One panel per metric, each on its own scale, measured on the long channels alone: a "
        "short channel is a regressor rather than a measurement, and an average over both "
        "describes neither.",
    "metrics.stages_banded":
        "The quality panels are recomputed with the analysis passband applied at every stage, "
        "so the columns measure the same thing. They will not match the per-stage values in "
        "the metrics table, which are measured on each stage as stored; the gap is largest on "
        "the stage before the bandpass.",
    "metrics.stages_unbanded":
        "No passband was requested, so every panel is measured on each stage as stored and "
        "differences across a filtering step should be read with that in mind.",
    "metrics.stages_greyed":
        "The greyed panels are not quality claims: they show what left the recording. "
        "Direction is marked only across the whole chain, never on a single step, since a step "
        "can move a metric for reasons that are not quality.",
    "metrics.motion_split":
        "Optical density. Each channel set is measured on its own, in the table below.",
    "metrics.motion_unsplit":
        "Optical density. GVTD, the spike counts and the correction footprint are measured on "
        "{channels}.",
    "metrics.motion_one_side":
        "One side of the motion step, not both, and not the same side throughout: GVTD is "
        "measured on the corrected file, the spike and footprint counts on the uncorrected "
        "one. The run's own page has the before&nbsp;&rarr;&nbsp;after pair.",
    "metrics.gvtd_sets":
        "Each set is measured on its own here rather than regrouped: GVTD is an RMS across "
        "channels with its own cutoff per set, and a frame count asks how many of <i>these</i> "
        "channels were flagged at once, a bar a smaller set clears more easily. <b>Read those "
        "rows down a column, not across one.</b> Corrected per channel is the exception, being "
        "an average over the set, and is the one column two rows can be compared on.",
    "metrics.gvtd_set_help":
        "Long is the set the verdict is read off and the only one coloured. Each set is its own "
        "measurement rather than a grouping of one, with its own GVTD cutoff and its own "
        "tenth-of-the-channels bar. Corrected per channel is the exception, an average over "
        "the set.",
    "metrics.motion_extra_split":
        "What the table above has no column for. <b>Spike count</b> is a sum over channels, so "
        "a second set's would track how many channels it has; censoring names the set "
        "<code>--gvtd-censor</code> gave it.",
    "metrics.motion_extra_unsplit":
        "Counted on {channels}, and over the recording rather than per channel: a timepoint "
        "counts when at least a tenth of the set was flagged.",
    "metrics.haemo_timing":
        "Measured after Beer-Lambert, on {channels}: what the denoising did, not what the "
        "recording arrived as, so each <b>before</b> value is that channel set's row in the "
        "table above. The global correlations are the exception, their before side being the "
        "bandpassed signal rather than the unfiltered one.",
    "metrics.no_drift":
        "Low-frequency drift is not among these: it grows with the span it is fitted over. The "
        "run's own page carries it.",
    "metrics.per_channel":
        "Grouped by source-detector separation, every block screened by the same criteria, so a "
        "short channel marked BAD should not be used as a regressor. Screening is a union of "
        "SCI and PSP, so a row can be BAD with a passing SCI.",
    "channel_summary":
        "Per-channel pass/fail, long channels then short ones. Green&nbsp;=&nbsp;pass, "
        "red&nbsp;=&nbsp;fail, grey&nbsp;=&nbsp;missing data. Only Status rejects a channel: it "
        "is the share of windows in which SCI (<code>--sci-threshold</code> {sci}) and PSP both "
        "pass, against <code>--min-good-frac</code>; the other rows are reported, not enforced.",
    "trial_qc":
        "Every trial window scored on its own, over {window}, on the intensity recording. "
        "Colour is relative within a row rather than a threshold: red marks the worse end of "
        "what this recording did, so a run with a fine mean SCI can still show the few trials "
        "where the cap moved. Whole-montage numbers, so read them against the metrics table's "
        "<b>All</b> row.",
    # ---- Subject index ----
    "subject_index.runs":
        "Metrics are read from each run's <code>desc-sqm</code> record. Open a run for its "
        "figures, provenance and full metric table.",
    "subject_index.outlier":
        "A marked cell sits more than {z} median absolute deviations from this subject's other "
        "runs on that metric. It says the run differs from its neighbours, not that it failed: "
        "conditions of different length or difficulty are expected to differ.",
    "subject_index.conditions":
        "A condition is a window selection out of the whole-run pass, not a cut of the "
        "recording, so these rows are comparable with each other and with the run they open "
        "on. Long channels throughout, and every share in the whole-run row is counted over "
        "the full recording the same way a condition counts it over its own window. A tree "
        "cropped per condition before preprocessing is listed under <b>Runs</b> instead: each "
        "piece there is filtered against its own two edges and lands on its own window grid, "
        "so the two kinds do not share a table.",
    "subject_index.channel_condition":
        "Colour is the robust z inside one condition, so a cell says how far that channel sat "
        "from the others in that block rather than how the blocks rank; hover gives the value. "
        "The panels are the profile's, dropped to the ones with a per-channel form: GVTD is an "
        "RMS across channels and has none. <b>Coupled windows</b> is added at the end because "
        "it is the line a channel is rejected on, which is what the kept count above counts. "
        "One channel order over every panel, the widest mover first.",
    "subject_index.timeline":
        "Smoothed over 60 s and sampled at that step, the trend being slower than one QC "
        "window. Long and short only: <code>all</code> is a blend of the two and lands between "
        "them. Each band is the colour its condition wears on the run's own report.",
    "subject_index.rejections":
        "{n_clean} of {n_pairs} source-detector pairs survived every run. Both wavelengths of a "
        "pair are collapsed into one row: rejecting either rejects the optode. Read from each "
        "run's <code>_desc-channel_qc.tsv</code>; this is the set <code>--bads-scope "
        "subject</code> takes the union of.",
    "subject_index.rebuilt":
        "This page is assembled from the records under <code>nirs/</code> each time either "
        "command runs, so it lists whatever is on disk rather than what one command produced. "
        "The command below is the one that last rebuilt it.",

    # ---- Cohort report ----
    "group.small_cohort":
        "Only {n} runs: a box, a robust z-score and the outlier rule each measure how far a run "
        "sits from the middle of its cohort, and this cohort has no middle to measure from. "
        "Read the panels as the runs' own values side by side.",
    "group.outliers":
        "{count} of {n} runs sit more than {z} robust SDs from the cohort median on the average "
        "metric{worst}; they are listed under Outliers and drawn in colour in the panels.",
    "group.outliers_worst": "({runs} the furthest)",
    "group.no_outliers":
        "No run sits more than {z} robust SDs from the cohort median on the average metric.",
}
