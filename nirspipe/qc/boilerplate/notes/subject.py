"""Section prose for the subject reports: one run, a subject's runs, and the cohort."""

from __future__ import annotations

NOTES = {
    # ---- Subject report ----
    "summary.condition_scope":
        "Coupling and motion numbers are sliced out of the whole recording&rsquo;s windowed "
        "pass, on the run&rsquo;s window grid and filter; haemoglobin numbers and panels are "
        "recomputed on this condition&rsquo;s stretch. Rejected channels are the "
        "run&rsquo;s. The count above is the channels failing on this stretch alone, a guide "
        "that rejects nothing.",
    "summary.condition_failing":
        "Failing in this condition",
    "summary.condition_kept":
        "{kept} of {window} s outside every BAD_ span; {windows} screening windows counted",
    "excluded.summary":
        "{kept} s of the {total} s recording lie outside every BAD_ span. Of the recording, "
        "{unselected} is BAD_unselected, outside the stretches --keep-spans named: real data "
        "left out of every estimate. {corrupted} lies under corrupted spans (BAD_gvtd, "
        "BAD_manual from --bad-spans, or spans the input carried). A stretch can be both, so the two shares can sum to more "
        "than was left out. {n} span(s) in all; each analysis section says how it leaves "
        "them out.",
    "excluded.conditions":
        "Per condition: its window, its seconds outside every BAD_ span, the screening "
        "windows its coupled share counted, and for a task GLM the trials whose interval "
        "touches no BAD_ span.",
    "excluded.skipped":
        "{output} was not computed: {reason}.",
    "summary.condition_not_assessed":
        "Every screening window of this condition touches a BAD_ span, so it is not "
        "assessed: no channel is counted as failing here, and its coupled-window shares are "
        "blank.",
    "summary.condition_rebuilt":
        "The motion figures are the run&rsquo;s, narrowed to this condition. The SCI/PSP "
        "panel, brain maps, haemoglobin panels, spectra, epoch and topography are rebuilt "
        "for it; the GLM section shows the run&rsquo;s one model with this condition&rsquo;s "
        "map.",
    "summary.condition_empty":
        "The motion figures are the run&rsquo;s, narrowed to this condition. The other "
        "preprocessing panels and the GLM map are rebuilt for it; the epoch and topography "
        "panels are empty because no trial fits the epoch window here.",
    "summary.condition_no_trials": "The trial panels are absent because {reason}.",
    "summary.condition_few_trials": "this condition holds too few trials to compare",
    "summary.condition_run_link": '<a href="{href}">Full report for the run</a>.',
    "steps.raw_signal":
        "Per-channel HbO/HbR and PSD on <code>{stage}</code>, converted with DPF {dpf}: the "
        "recording before motion correction ({motion}) and filtering, the stage windowed SCI "
        "and PSP are measured on. SNR and CV come from the raw intensity.",
    "steps.raw_signal_fallback":
        "Per-channel HbO/HbR and PSD. <strong><code>desc-sci</code> was not readable, so "
        "these are drawn on <code>desc-preproc</code>, after motion correction ({motion}) "
        "and Beer-Lambert with DPF {dpf}.</strong>",
    "channel_detail.picker":
        "Select a channel for its HbO/HbR and PSD; its epoch average is in the epoch "
        "section. Pairs marked <em>(short)</em> are too shallow to reach cortex and show "
        "scalp haemodynamics.",
    "steps.raw_quality":
        "SCI and PSP per channel and window on <code>desc-sci</code>, the optical density "
        "before motion correction; CV on the raw intensity over windows of the same length. "
        "SCI and PSP cross-correlate the two wavelengths, so they exist only for optical "
        "density. In the CV row low is good; its hover gives SNR (1/CV), and its colour "
        "scale tops out at twice the <strong>{cv}</strong> line. SCI threshold: "
        "<strong>{sci}</strong>.",
    "quality.brain_views":
        "Red is rejected by screening, whatever its SCI; the rest are graded green to red by "
        "SCI against the run&rsquo;s line. Screening counts coupled windows, so a channel "
        "can be dropped at a high SCI.",
    "quality.brain_views_condition":
        "Both maps grade this condition&rsquo;s own SCI; only what the run rejected is drawn "
        "rejected.",
    "steps.motion_correction":
        "Method: <strong>{method}</strong>. Reported GVTD covers the "
        "<strong>{gvtd_set}</strong> channels, the set the analysis uses; the panel draws a "
        "row per channel set with its count. GVTD is an RMS across channels, so the set is "
        "part of the value.",
    "motion.roi_strip":
        "On the carpet the {gvtd_set} rows are grouped by ROI, coloured by the outer strip "
        "and named in the legend.",
    "motion_detail.picker":
        "Select a channel to compare its optical density before and after motion correction.",
    "motion_detail.condition":
        "Like the carpet, measured over the whole recording and narrowed to this condition: "
        "the GVTD row is filtered and thresholded run-wide.",
    "steps.beer_lambert": "Modified Beer-Lambert law applied with DPF = {dpf}.",
    "haemo.hbo_hbr_corr":
        "A cortical response drives HbO up and HbR down and pushes r negative, while "
        "shared systemic or motion signals push it positive. It moves with the passband, "
        "so compare runs at the same stage rather than against a fixed value.",
    "haemo.glm_residual":
        "Here the after stage is the GLM residual, with the task model removed along with "
        "the confounds. A weaker anticorrelation there can mean the model explained part of "
        "the response, not that the data got worse, so judge the run on the before stage.",
    "carpet.after_named":
        "Per-channel z-scored haemoglobin for <code>{stage}</code>, HbO above HbR, marked "
        "by the red and blue bar.",
    "carpet.roi_strip":
        "The strip outside it colours the rows by ROI, named in the legend.",
    "carpet.after":
        "Above it, a GVTD row per channel set before and after motion correction, on the "
        "same time axis, so a dark or light column can be read against the movement at that "
        "moment; thresholds and spike shading stay in the motion section. Each block is "
        "z-scored per channel, so it shows the structure left rather than amplitude; on the "
        "run page the title gives the median SD against <code>desc-preproc</code>.",
    "carpet.glm":
        "On a GLM run this is the residual, with the task model and confounds removed: what "
        "the model left unexplained, not the data the betas came from.",
    "carpet.condition":
        "The z-scoring is the whole recording&rsquo;s, so every condition&rsquo;s carpet "
        "shares one greyscale: dark is above the channel&rsquo;s run mean, light below, and "
        "higher contrast means noisier.",
    "psd.stages":
        "Long channels, HbO and HbR in each row, with single channels faint behind the mean. "
        "Shaded: Mayer wave (~0.1&thinsp;Hz){bands}. Dashed verticals mark the filter cutoffs.",
    "psd.band_resp": ", respiration ({lo}–{hi}&thinsp;Hz)",
    "psd.band_cardiac": ", cardiac ({lo}–{hi}&thinsp;Hz)",
    "psd.stage_list":
        "The top row is <code>desc-preproc</code>, before any filtering; the bottom row is "
        "{stages}, the signal the analysis reads.",
    "psd.errts":
        "<code>desc-errts</code> is the regression residual, so a dip at the Mayer or "
        "respiration frequencies is the short-channel and aux regressors at work.",
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
    "psd_detail.picker":
        "Select a channel. A short channel's page has no residual row: as a regressor, its "
        "residual is zero.",
    "epoch.timeline":
        "Every event on one axis, a row per condition plus one for all. It shows what the "
        "averages below cannot: a condition that stopped partway through, or a block started "
        "twice.",
    "epoch.timeline_condition":
        "<strong>This is the whole run, not just &ldquo;{label}&rdquo;</strong>: both "
        "failures show only against the conditions around it.",
    "epoch.grand_mean":
        "HbO and HbR averaged over the good long channels, baseline-corrected to {tmin} to "
        "0&thinsp;s, a panel per condition on one y scale with the task block shaded. Short "
        "channels are dotted: when the dotted line rises with the solid one, part of the "
        "response is scalp signal.",
    "epoch.channel_maps":
        "The condition-averaged response drawn along each channel&rsquo;s path, a head per "
        "condition; the slider steps through the response one second at a time. Nothing is "
        "interpolated between channels. Short channels get their own row on the same colour "
        "scale; a short row as strong as the long row above means part of the response is "
        "scalp signal.",
    "epoch.trial_image_roi":
        "Per ROI, HbO channels averaged first, then each trial a row, on the bandpassed "
        "signal before confound regression; runs with enough trials are smoothed across "
        "neighbouring trials. A panel per condition, the pooled one first. Select an ROI.",
    "epoch.trial_image_single":
        "Each trial is a row against time from onset, coloured by HbO: the response before "
        "averaging, on the bandpassed signal before confound regression. Runs with enough "
        "trials are smoothed across neighbouring trials. A panel per condition, the pooled "
        "one first. Select a channel.",
    "rest.alff":
        "Amplitude of low-frequency fluctuations (ALFF) and its fractional form (fALFF), "
        "from the denoised residual. The map shows where the amplitude is; the connectivity "
        "panel below gives each channel&rsquo;s value beside its correlations.",
    "rest.alff_layout":
        "One disc per channel at its source-detector midpoint, HbO on top and HbR below. The "
        "amplitude column draws mALFF, each channel over its own chromophore's mean, so HbO "
        "and HbR share one scale and the measured molar value is on the hover; fALFF is "
        "already a share. Grey channels carry no value, which is what a rejected one has. "
        "Short channels are not drawn.",
    "fc.channel_matrix":
        "Channel-by-channel Pearson r on top, HbO left and HbR right on one scale, on the "
        "signal the mode leaves: the confound residual in rest and denoise (the bandpassed "
        "data if denoise ran no regression), the task residual in glm. In rest mode each "
        "channel&rsquo;s ALFF and fALFF sit below in the same order. A channel with no "
        "correlation, as a rejected one has, is left off the matrix and the strips.",
    "fc.roi_matrix":
        "The same correlation between ROI-averaged signals, on the same scale. Channels are "
        "averaged before correlating, so this is not the mean of the cells above. Needs "
        "<code>--roi-mapping</code>.",
    "fc.seed_topography":
        "A flat map per seed ROI, each channel coloured by its correlation with the "
        "seed&rsquo;s mean signal, on the matrix&rsquo;s scale. Solid grey channels are "
        "inside the seed (undefined, not zero); faded grey ones were rejected. Short "
        "channels are not drawn.",
    "glm.activation":
        "HbO betas on the surface, three views per condition, on one colour scale shared by "
        "every condition so a weak condition reads as weak. One model over the whole "
        "recording, so every condition rests on the same channels. Select a condition.",
    "metrics.intro":
        "Hover a label for its definition. Bold labels are the ones that decide whether the run "
        "is usable. Where a cell holds two values, they are before and after a processing step.",
    "metrics.long_only":
        "Long channels only: the channels outside both separation ranges are left out of "
        "these averages and listed in their own block of the per-channel table below.",
    "metrics.set_help":
        "Judge the run on Long. Short is uncoloured: short channels couple well by "
        "construction and only feed the regressors.",
    "metrics.set_help_condition":
        "Values are this condition's; the channel sets are the run's.",
    "metrics.stages":
        "One panel per metric on the long channels' HbO (HbO&ndash;HbR r uses both), each on "
        "its own scale. Short channels are left out because they serve as regressors.",
    "metrics.stages_banded":
        "HbO&ndash;HbR, GCOR and CNR are recomputed in the analysis passband at every stage, so "
        "they will not match the per-stage values in the table above. Variance remaining is "
        "measured as stored and drops across the bandpass by design.",
    "metrics.stages_unbanded":
        "No passband was requested, so every panel is measured on each stage as stored.",
    "metrics.stages_no_kept_time":
        "No frame of {stages} lies outside a BAD_ span, and a spectrum is taken over kept "
        "frames only, so the respiration, cardiac and drift band panels are left out there.",
    "metrics.stages_greyed":
        "Grey panels show what left the recording, not quality, and direction is marked only "
        "across the whole chain.",
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
    "metrics.gvtd_sets_columns":
        "Each channel set is measured on its own, with its own GVTD cutoff and its own share of "
        "flagged channels, so <b>the three values in a row do not compare.</b> Corrected per "
        "channel is an average over the set.",
    "metrics.motion_arrow":
        "Two values are before and after motion correction.",
    "metrics.gvtd_set_help":
        "Judge the run on Long, the only coloured set. Each set is measured on its own, with its "
        "own GVTD cutoff and its own {pct:g}% bar.",
    "metrics.motion_extra_split":
        "A value in the All column with Long and Short blank belongs to no channel set: the "
        "motion sensor is not optical data, and censoring uses its own set "
        "(<code>--gvtd-censor</code>).",
    "metrics.motion_extra_long":
        "Counted on <b>long channels</b>: the frame rows count a timepoint when at least "
        "{pct:g}% of the set was flagged, while Spike count and Spike % count "
        "channel-samples.",
    "metrics.motion_extra_every":
        "Counted on every channel: the frame rows count a timepoint when at least {pct:g}% "
        "of the set was flagged, while Spike count and Spike % count channel-samples.",
    "metrics.haemo_timing_long":
        "In the Long column, two values are before and after denoising; All and Short show the "
        "value before denoising. The regression rows cover the confound regression alone, so "
        "their first value is the bandpassed signal and differs from the row above.",
    "metrics.haemo_timing_every":
        "Measured after Beer-Lambert. The global correlations start from the bandpassed signal, "
        "not the unfiltered one.",
    "metrics.no_drift":
        "Low-frequency drift is not among these: it grows with the span it is fitted over. The "
        "run's own page carries it.",
    "metrics.per_channel_stage":
        "Scores on the recording before any correction. HbO&ndash;HbR corr is after "
        "Beer-Lambert, before and after denoising where two values are shown.",
    "metrics.per_channel":
        "BAD means too few windows in which SCI and PSP both pass (<code>--min-good-frac</code>), "
        "so a channel can be BAD with a passing whole-run SCI; BAD (manual) was rejected by hand. "
        "A BAD short channel is left out of the regressors. Saturated % is the share of samples "
        "held at the detector&rsquo;s ceiling; those stretches carry no pulse, so they fail "
        "coupling.",
    "channel_summary":
        "Green pass, red fail, grey missing. Only Status rejects a channel: it fails when the "
        "share of windows where SCI (<code>--sci-threshold</code> {sci}) and PSP both pass is "
        "below <code>--min-good-frac</code>, or when the channel was rejected by hand. The other "
        "rows are for reference.",
    "condition_summary":
        "The condition pages&rsquo; channel grids in one figure, one metric&rsquo;s conditions on "
        "adjacent rows, same cutoffs (<code>--sci-threshold</code> {sci}). <b>In condition</b> "
        "says whether a channel passes within that condition alone; a channel rejected by hand "
        "fails everywhere. It rejects nothing, and Status on each condition page stays the run&rsquo;s.",
    "trial_qc":
        "Every trial window scored on its own, over {window}, on the intensity recording. "
        "Colour is relative within a row rather than a threshold: red marks the worse end of "
        "what this recording did, so a run with a fine mean SCI can still show the few trials "
        "where the cap moved. Whole-montage numbers, so read them against the metrics table's "
        "<b>All</b> row.",
    # ---- Subject index ----
    "subject_index.runs":
        "Read from each run&rsquo;s quality record (<code>desc-sqm</code>, or "
        "<code>desc-sqmraw</code> for a run only <code>nirspipe-qc prep-raw</code> "
        "measured). Open a run for its figures, provenance and full metric table.",
    "subject_index.outlier":
        "A marked cell sits at least {z} robust SDs (1.4826 &times; MAD) from this "
        "subject&rsquo;s median on that metric. It means the run differs from the others, "
        "not that it failed.",
    "subject_index.conditions":
        "Each condition is a window of the whole-run pass, so the rows compare with each "
        "other and with the whole-run row. Long channels throughout, except Channels "
        "passing, which counts every channel: on the whole-run row what the run kept, on a "
        "condition&rsquo;s row what passes on that stretch alone, a guide that rejects "
        "nothing. Runs cropped per condition before preprocessing are listed under "
        "<b>Runs</b> instead.",
    "subject_index.channel_condition":
        "Colour is the robust z inside one condition: how far a channel sat from the others "
        "in that block, not how the blocks rank; hover for the value. The panels are the "
        "profile&rsquo;s that have a per-channel form (GVTD, an RMS across channels, has "
        "none), plus <b>Coupled windows</b>, the line a channel is rejected on. One channel "
        "order throughout, sorted by how far each channel moves on the SCI panel.",
    "subject_index.timeline":
        "Smoothed over {smooth:g} s and sampled at that step. Long and short only, since "
        "<code>all</code> blends the two. Each band is coloured by its window in order; the "
        "colours need not match the run report&rsquo;s event colours.",
    "subject_index.rejections":
        "{n_clean} of {n_pairs} source-detector pairs survived every run. A pair is one row, "
        "rejected if either wavelength is. Read from each run&rsquo;s "
        "<code>_desc-channel_qc.tsv</code>, whose union <code>--bads-scope subject</code> "
        "takes, or <code>_desc-rawchannel_qc.tsv</code> for a run only prep-raw saw.",
    "subject_index.rebuilt":
        "Rebuilt from the records under <code>nirs/</code> each time either command runs, so "
        "it lists what is on disk, not what one command produced. The command below rebuilt "
        "it last.",

    # ---- Cohort report ----
    "group.small_cohort":
        "Only {n} runs. Boxes, robust z-scores and the outlier rule measure distance from "
        "the cohort's middle, which so few runs do not have; read the panels as the runs' "
        "own values side by side.",
    "group.outliers":
        "{count} of {n} runs sit more than {z} robust SDs from the cohort median on the "
        "average metric{worst}. They are listed under Outliers, and the furthest are drawn "
        "in colour in the over-time and by-condition panels.",
    "group.outliers_worst": "({runs} the furthest)",
    "group.no_outliers":
        "No run sits more than {z} robust SDs from the cohort median on the average metric.",
    "group.overview":
        "One row per metric and one dot per run at its robust z, coloured by channel set; the "
        "band marks |z| below 2.",
    "group.distributions":
        "Real values, one chart per group of metrics on the same scale, coloured by channel "
        "set. Click a point to open that run's report.",
    "group.conditions":
        "Each run's windows collapsed into its own blocks, one line per run, with the cohort "
        "median over them.",
    "group.condition_matrix":
        "Colour is the robust z inside each condition. Runs are sorted worst first, their "
        "names on the left edge.",
    "group.windowed":
        "Rows are metrics, columns channel sets: every run pale under the cohort's "
        "interquartile band and median, smoothed over {smooth} s. SCI here follows the QC "
        "window; the table's <code>sci_win_mean</code> is pinned to {sci_window} s and "
        "<code>sci_mean</code> covers the whole run.",
    "group.outlier_rule":
        "Runs whose mean |z| over every metric is past {z}: one test per run, not one per "
        "metric.",
    "group.table":
        "Every scalar the records carry, the check-only ones aside. Click a column header to "
        "sort; the first column stays in view.",
}
