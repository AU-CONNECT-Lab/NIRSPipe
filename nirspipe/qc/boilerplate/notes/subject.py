"""Section prose for the subject reports: one run, a subject's runs, and the cohort."""

from __future__ import annotations

NOTES = {
    # ---- Subject report ----
    "summary.condition_scope":
        "The coupling and motion numbers are sliced out of the windowed pass the whole "
        "recording was measured with, so they sit on the run&rsquo;s window grid and "
        "filter; the haemoglobin numbers and panels are recomputed on this "
        "condition&rsquo;s stretch. A channel marked rejected here is one the run rejected, "
        "the recording having been processed under the run&rsquo;s verdict; the count above "
        "is the channels failing on this stretch alone, a guide to choosing conditions that "
        "rejects nothing.",
    "summary.condition_failing":
        "Failing in this condition",
    "summary.condition_rebuilt":
        "The motion carpet and per-channel motion figures are the run&rsquo;s, narrowed to "
        "this condition; the SCI/PSP panel, brain maps, haemoglobin panels, spectra, epoch "
        "and topography are rebuilt for it, and the GLM section shows the run&rsquo;s one "
        "model with this condition&rsquo;s map.",
    "summary.condition_empty":
        "The motion carpet and per-channel motion figures are the run&rsquo;s, narrowed to "
        "this condition; the other preprocessing panels and the GLM map are rebuilt for "
        "it, and the epoch and topography panels are empty because no trial fits the epoch "
        "window here.",
    "summary.condition_no_trials": "The trial panels are absent because {reason}.",
    "summary.condition_few_trials": "this condition holds too few trials to compare",
    "summary.condition_run_link": '<a href="{href}">Full report for the run</a>.',
    "steps.raw_signal":
        "Per-channel HbO/HbR timeseries and PSD, on <code>{stage}</code> converted with "
        "DPF = {dpf}. This is the recording before motion correction ({motion}) and before "
        "any filtering, the stage the sliding-window SCI/PSP are measured on; SNR and CV "
        "come from the raw intensity before OD conversion.",
    "steps.raw_signal_fallback":
        "Per-channel HbO/HbR timeseries and PSD. <strong><code>desc-sci</code> was not "
        "readable for this run, so these are drawn on <code>desc-preproc</code>, after motion "
        "correction ({motion}) and Beer-Lambert with DPF = {dpf}.</strong>",
    "channel_detail.picker":
        "Select a channel to view its HbO/HbR timeseries and its PSD. This channel&rsquo;s "
        "epoch average is in the epoch section below, on the bandpassed signal before "
        "confound regression (<code>desc-preproc</code> when no passband was set). Pairs "
        "marked <em>(short)</em> are short-separation: too shallow to reach cortex, so "
        "what they show is scalp haemodynamics.",
    "steps.raw_quality":
        "Scalp coupling index (SCI) and peak spectral power (PSP) per channel and per "
        "window on <code>desc-sci</code>, the optical density before motion correction, "
        "and coefficient of variation (CV) on the raw intensity over windows of the same "
        "length. SCI and PSP cross-correlate the two wavelengths, so they exist only in "
        "the optical density domain and have no counterpart after Beer-Lambert. CV is the "
        "row where low is good, and its hover also gives SNR, which is 1/CV exactly; its "
        "colour scale tops out at twice the <strong>{cv}</strong> line. SCI threshold: "
        "<strong>{sci}</strong>.",
    "quality.brain_views":
        "Red = rejected by the screening, whatever its SCI; every channel that survived is "
        "then graded green to red by its own SCI against the run&rsquo;s line. The two are "
        "different questions: a channel is rejected on how many windows it was coupled in, "
        "so one can be dropped at a high SCI.",
    "quality.brain_views_condition":
        "Both maps carry this condition's own SCI over the run's rejected channels: the "
        "stretch is graded on its own, and only what the run rejected is drawn rejected.",
    "steps.motion_correction":
        "Motion correction method: <strong>{method}</strong>. The reported GVTD numbers "
        "cover the <strong>{gvtd_set}</strong> channels, the set the analysis uses; the "
        "panel draws one row per channel set with its count. GVTD is an RMS across "
        "channels, so the set is part of the value.",
    "motion_detail.picker":
        "Select a channel to compare the OD signal before and after motion correction.",
    "motion_detail.condition":
        "Measured over the whole recording and viewed over this condition, like the carpet "
        "above it: the GVTD row on each figure is filtered and thresholded run-wide, so it is "
        "the run's trace with the axis narrowed.",
    "steps.beer_lambert": "Modified Beer-Lambert law applied with DPF = {dpf}.",
    "haemo.hbo_hbr_corr":
        "A cortical response drives HbO up and HbR down and pushes r negative, while "
        "shared systemic or motion signals push it positive. It moves with the passband, "
        "so compare runs at the same stage rather than against a fixed value.",
    "haemo.glm_residual":
        "The after stage on this run is the GLM residual, so the task model came out of it "
        "along with the confounds. A weaker anticorrelation there can mean the model explained "
        "part of the shared response rather than that the data got worse, so judge the run on "
        "the before stage.",
    "carpet.after_named":
        "Per-channel z-scored haemoglobin for <code>{stage}</code>, HbO above HbR in one image, "
        "a colour bar and a seam between them.",
    "carpet.after":
        "One GVTD row per channel set sits above on the same time axis, each carrying the "
        "trace before and after motion correction, so a dark or light column can be read "
        "against what happened at that moment; the threshold, the spike shading and the "
        "corrected spans stay in the motion section above. Each block is z-scored per "
        "channel, so it shows the structure left at this stage rather than its amplitude; "
        "on the run page the title quotes the median SD against <code>desc-preproc</code>.",
    "carpet.glm":
        "On a GLM run this is the residual, which had the task model removed as well as the "
        "confounds: it shows what the model left unexplained, not the data the betas were "
        "measured on.",
    "carpet.condition":
        "The z-scoring is the whole recording&rsquo;s and only the columns drawn are this "
        "condition&rsquo;s, so every condition&rsquo;s carpet is on one greyscale: dark is "
        "above the channel&rsquo;s run mean, light below it, and the higher-contrast "
        "stretch is the noisier one.",
    "psd.stages":
        "One row per stage file on disk, HbO and HbR together in each. Coloured bands: "
        "Mayer wave (~0.1&thinsp;Hz){bands}. Faint lines are the individual channels "
        "behind each row&rsquo;s mean; dashed verticals mark the filter cutoffs.",
    "psd.band_resp": ", respiration ({lo}–{hi}&thinsp;Hz)",
    "psd.band_cardiac": ", cardiac ({lo}–{hi}&thinsp;Hz)",
    "psd.stage_list": "Stages after <code>desc-preproc</code>: {stages}.",
    "psd.errts":
        "<code>desc-errts</code> is the regression residual: confounds only in denoise and "
        "rest, the task model as well in glm. Short-channel and aux regressors remove "
        "in-band systemic power, so a dip at Mayer or respiration frequencies is expected; "
        "where only a drift basis was regressed and a high-pass ran, this row should sit "
        "on top of the one above. Where no high-pass ran, this row carries the whole "
        "detrend.",
    "psd.simulated":
        "No post-processing output was found for this run; when a passband was set, the "
        "second row is that bandpass simulated in memory rather than a file the pipeline "
        "wrote.",
    "psd.too_short_condition":
        "&ldquo;{label}&rdquo; holds fewer samples than the transform, so no spectrum is drawn "
        "here. The run's own page carries the spectrum, and the PSP row above is sliced to "
        "this condition.",
    "psd.too_short_cut":
        "This cut holds fewer samples than the transform, so no spectrum is drawn here. The "
        "run's own page carries the spectrum, and the PSP row above is sliced to this "
        "condition.",
    "psd_detail.picker": "Select a channel to compare its PSD across the stages.",
    "epoch.timeline":
        "Every event on one axis, one row per condition plus one for all of them. The "
        "averages below cannot show a condition that stopped being delivered partway "
        "through or a block that was started twice; this can. The same panel as the raw QC "
        "viewer&rsquo;s event timeline.",
    "epoch.timeline_condition":
        "<strong>This is the whole run, not just &ldquo;{label}&rdquo;</strong>: both of those "
        "failures show only against the conditions around the one you are reading.",
    "epoch.grand_mean":
        "Grand-mean HbO and HbR responses averaged across the good long channels, "
        "baseline-corrected to the pre-stimulus window ({tmin} to 0&thinsp;s). Each "
        "condition is shown separately, on one shared y scale, with the task block shaded. "
        "Short channels are drawn dotted: they are too shallow to reach cortex, so a "
        "dotted line that rises with the solid one means at least part of the response is "
        "scalp haemodynamics.",
    "epoch.channel_maps":
        "The condition-averaged response painted along each channel&rsquo;s own "
        "source-detector path, one head per condition; drag the slider to step through the "
        "response after onset, one second at a time. Nothing is interpolated between "
        "channels, so bare scalp stays bare. Short channels get their own row on the "
        "chromophore&rsquo;s colour scale: they are too shallow to reach cortex, so a "
        "short row as strongly coloured as the long row above it means at least part of "
        "the response is systemic scalp signal.",
    "epoch.trial_image_roi":
        "Per ROI, HbO channels averaged first (higher SNR), then each trial stacked as a "
        "row. Bandpassed signal before confound regression; runs with enough repetitions "
        "are also smoothed across adjacent trials. One panel per condition, pooled panel "
        "first when there are several. Select an ROI.",
    "epoch.trial_image_single":
        "Each stimulus repetition is a row (trial &times; time from onset), coloured by "
        "HbO: the un-averaged view of the response, on the bandpassed signal before "
        "confound regression. Runs with enough repetitions are also smoothed across "
        "adjacent trials. One panel per condition, pooled panel first when there are "
        "several. Select a channel.",
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
        "Channel-by-channel Pearson r on top (HbO left, HbR right, one scale, one triangle "
        "each), on the signal the mode leaves: the confound residual in rest and denoise "
        "(the bandpassed data when denoise ran no regression) and the task residual in "
        "glm. In rest mode each channel&rsquo;s ALFF and fALFF sit below, from the "
        "broadband residual, in the same channel order and separation blocks. A rejected "
        "channel is a grey L along its row and column in the matrix and an open ring below "
        "the range in the strips.",
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
        'Hover a metric for what it means, which way is good and what it was measured on. '
        '<span class="ql-dot" aria-hidden="true">&#9679;</span> marks the ones that decide '
        'whether the run is usable. A pair written <b>a &rarr; b</b> is one metric '
        'measured either side of a processing step; a single number is a metric with no '
        'second side on this page.',
    "metrics.long_only":
        "Long channels only: the channels outside both separation ranges are left out of "
        "these averages and listed in their own block of the per-channel table below.",
    "metrics.set_help":
        "Long is the set the verdict is read off. Short is left uncoloured: a short "
        "channel&rsquo;s coupling is high by construction, so that row answers whether the "
        "regressors those channels feed are trustworthy rather than whether the run is "
        "usable. Rejection is per channel, against the screening criteria over the whole "
        "montage, and is not read off these averages.",
    "metrics.set_help_condition":
        "On this page every column is this condition's, and the channel sets are the run's, "
        "since one montage has to serve every condition.",
    "metrics.stages":
        "One panel per metric, each on its own scale, measured on the long channels alone: a "
        "short channel is a regressor rather than a measurement, and an average over both "
        "describes neither.",
    "metrics.stages_banded":
        "The HbO&ndash;HbR, GCOR and CNR panels are recomputed with the analysis passband "
        "applied at every stage, so those columns measure the same thing; variance "
        "remaining is measured as stored, so it falls across the bandpass by construction. "
        "They will not match the per-stage values in the metrics table, which are measured "
        "on each stage as stored.",
    "metrics.stages_unbanded":
        "No passband was requested, so every panel is measured on each stage as stored and "
        "differences across a filtering step should be read with that in mind.",
    "metrics.stages_greyed":
        "The greyed panels are not quality claims: they show what left the recording. "
        "Direction is marked only across the whole chain, never on a single step, since a step "
        "can move a metric for reasons that are not quality.",
    "metrics.motion_split":
        "Optical density. Each channel set is measured on its own, in the table below.",
    "metrics.motion_unsplit_long":
        "Optical density. GVTD, the spike counts and the correction footprint are measured on "
        "long channels only.",
    "metrics.motion_unsplit_every":
        "Optical density. GVTD, the spike counts and the correction footprint are measured on "
        "every channel.",
    "metrics.motion_one_side":
        "One side of the motion step, not both, and not the same side throughout: GVTD is "
        "measured on the corrected file, the spike counts on the uncorrected one, and the "
        "correction footprint from the difference between the two. The run&rsquo;s own "
        "page has the before&nbsp;&rarr;&nbsp;after pair.",
    "metrics.gvtd_sets":
        "Each set is measured on its own here rather than regrouped: GVTD is an RMS across "
        "channels with its own cutoff per set, and a frame count asks how many of "
        "<i>these</i> channels were flagged at once, a bar a smaller set clears more "
        "easily. <b>Read each row on its own, not down a column.</b> Corrected per channel "
        "is the exception, being an average over the set, and is the one column two rows "
        "can be compared on.",
    "metrics.gvtd_set_help":
        "Long is the set the verdict is read off and the only one coloured. Each set is its own "
        "measurement rather than a grouping of one, with its own GVTD cutoff and its own bar "
        "of {pct:g}% of the channels. Corrected per channel is the exception, an average over "
        "the set.",
    "metrics.motion_extra_split":
        "What the table above has no column for. <b>Spike count</b> is a sum over "
        "channels, read against its set&rsquo;s channel count, and the per-set values are "
        "in the optical-density table above; censoring names the set "
        "<code>--gvtd-censor</code> gave it.",
    "metrics.motion_extra_long":
        "Counted on <b>long channels</b>: the frame rows count a timepoint when at least "
        "{pct:g}% of the set was flagged, while Spike count and Spike % count "
        "channel-samples.",
    "metrics.motion_extra_every":
        "Counted on every channel: the frame rows count a timepoint when at least {pct:g}% "
        "of the set was flagged, while Spike count and Spike % count channel-samples.",
    "metrics.haemo_timing_long":
        "Measured after Beer-Lambert, on long channels: what the denoising did, not what the "
        "recording arrived as, so each <b>before</b> value is that channel set's row in the "
        "table above. The global correlations are the exception, their before side being the "
        "bandpassed signal rather than the unfiltered one.",
    "metrics.haemo_timing_every":
        "Measured after Beer-Lambert: what the denoising did, not what the recording "
        "arrived as. The global correlations&rsquo; before side is the bandpassed signal "
        "rather than the unfiltered one.",
    "metrics.no_drift":
        "Low-frequency drift is not among these: it grows with the span it is fitted over. The "
        "run's own page carries it.",
    "metrics.per_channel":
        "Grouped by source-detector separation, every block screened by the same criteria, "
        "so a short channel marked BAD should not be used as a regressor. Screening counts "
        "the windows in which SCI and PSP both pass, so a row can be BAD with a passing "
        "whole-run SCI; channels rejected by hand show as BAD too.",
    "channel_summary":
        "Per-channel pass/fail, long channels then short ones. Green&nbsp;=&nbsp;pass, "
        "red&nbsp;=&nbsp;fail, grey&nbsp;=&nbsp;missing data. Only Status rejects a "
        "channel: it fails when the Coupled share, the windows in which SCI "
        "(<code>--sci-threshold</code> {sci}) and PSP both pass, falls below "
        "<code>--min-good-frac</code>, or when the channel was rejected by hand. The other "
        "rows are reported, not enforced; every row is coloured against this run&rsquo;s "
        "lines.",
    "condition_summary":
        "Each condition page's channel grid, regrouped so one metric's conditions sit in "
        "adjacent rows: a channel that fails in one condition only shows as a lone red cell "
        "in its column. Same cells and cutoffs as the condition pages (<code>--sci-threshold"
        "</code> {sci}), except the first block: <b>In condition</b> says whether each channel "
        "passes on that condition's stretch alone, a channel rejected by hand failing "
        "everywhere. It is a guide to choosing conditions and rejects nothing; Status on each "
        "condition page is the run's.",
    "trial_qc":
        "Every trial window scored on its own, over {window}, on the intensity recording. "
        "Colour is relative within a row rather than a threshold: red marks the worse end of "
        "what this recording did, so a run with a fine mean SCI can still show the few trials "
        "where the cap moved. Whole-montage numbers, so read them against the metrics table's "
        "<b>All</b> row.",
    # ---- Subject index ----
    "subject_index.runs":
        "Metrics are read from each run&rsquo;s quality record (<code>desc-sqm</code>, or "
        "<code>desc-sqmraw</code> for a run only <code>nirspipe-qc prep-raw</code> measured). "
        "Open a run for its figures, provenance and full metric table.",
    "subject_index.outlier":
        "A marked cell sits at least {z} robust SDs (1.4826 &times; MAD) from the median "
        "of this subject&rsquo;s runs on that metric. It says the run differs from its "
        "neighbours, not that it failed: runs of different length or difficulty are "
        "expected to differ.",
    "subject_index.conditions":
        "Each condition is a window of the whole-run pass, so its rows compare with each "
        "other and with the whole-run row; long channels throughout, except Channels "
        "passing, which counts every channel. On the whole-run row that is what the run "
        "kept; on a condition's row it is what passes on that stretch alone, a guide to "
        "choosing conditions that rejects nothing. Runs cropped per condition before "
        "preprocessing are listed under <b>Runs</b> instead.",
    "subject_index.channel_condition":
        "Colour is the robust z inside one condition, so a cell says how far that channel "
        "sat from the others in that block rather than how the blocks rank; hover gives "
        "the value. The panels are the profile&rsquo;s, dropped to the ones with a "
        "per-channel form: GVTD is an RMS across channels and has none. <b>Coupled "
        "windows</b> is added at the end because it is the line a channel is rejected on, "
        "which is what the kept count above counts. One channel order over every panel, "
        "sorted by how far each channel moves on the first (SCI) panel.",
    "subject_index.timeline":
        "Smoothed over {smooth:g} s and sampled at that step, the trend being slower than "
        "one QC window. Long and short only: <code>all</code> is a blend of the two and "
        "lands between them. Each band is coloured by its window, in order; the colours "
        "need not match the run report&rsquo;s event colours.",
    "subject_index.rejections":
        "{n_clean} of {n_pairs} source-detector pairs survived every run. Both wavelengths "
        "of a pair are collapsed into one row: rejecting either rejects the pair. Read "
        "from each run&rsquo;s <code>_desc-channel_qc.tsv</code>, the set "
        "<code>--bads-scope subject</code> takes the union of, or from "
        "<code>_desc-rawchannel_qc.tsv</code> for a run only prep-raw saw.",
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
        "{count} of {n} runs sit more than {z} robust SDs from the cohort median on the "
        "average metric{worst}; they are listed under Outliers, and the furthest ones are "
        "drawn in colour in the over-time and by-condition panels.",
    "group.outliers_worst": "({runs} the furthest)",
    "group.no_outliers":
        "No run sits more than {z} robust SDs from the cohort median on the average metric.",
}
