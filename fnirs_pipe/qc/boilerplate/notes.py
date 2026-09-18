"""Section prose for the QC reports, so a paragraph is written once and read anywhere.

The per-metric hover text is next door in :mod:`~fnirs_pipe.qc.boilerplate.vocabulary`;
this is the layer above it, the paragraphs that introduce a section or say how to read the
panel under it.

**Half the report's prose is here and half is still in the template, and the split is not
arbitrary.** A paragraph lives here when it is the same sentence every time, value slots
included: ``{sci}`` and friends are filled by the caller. It stays in the template when it
*branches*, when a ``{% if %}`` decides which sentence to say. Moving one of those would
mean either embedding Jinja in a Python string, with no highlighting and no lint, or
assembling the sentence out of fragments in Python, which reads worse than the template
does. So: **if a paragraph can differ between two runs by more than a value, look in
subject_report.html.j2.**

The strings carry HTML (``<code>``, ``<b>``, ``&nbsp;``) and reach the page unescaped,
which is what ``render()`` does with every other variable; a report environment that turned
autoescaping on would have to wrap these.
"""

from __future__ import annotations

SECTION_NOTES = {
    "channel_detail.picker":
        "Select a channel to view its HbO/HbR timeseries and its PSD. This channel's epoch "
        "average is in the epoch section below, on the denoised signal: an epoch average on "
        "this unfiltered stage shows cardiac ripple rather than the shape of a response. Pairs "
        "marked <em>(short)</em> are short-separation: too shallow to reach cortex, so what "
        "they show is scalp haemodynamics.",
    "steps.raw_quality":
        "Scalp coupling index (SCI), peak spectral power (PSP) and coefficient of variation "
        "(CV) per channel and per window, all measured on <code>desc-sci</code>, the optical "
        "density before motion correction. SCI and PSP cross-correlate the two wavelengths, so "
        "they exist only in the optical density domain and have no counterpart after "
        "Beer-Lambert. CV is the row where low is good, and it carries SNR in its hover rather "
        "than in a row of its own, SNR being 1/CV exactly; its colour is pinned to twice the "
        "<strong>{cv}</strong> line rather than to the recording's worst window. SCI threshold: "
        "<strong>{sci}</strong>.",
    "steps.motion_correction":
        "Motion correction method: <strong>{method}</strong>. GVTD covers the "
        "<strong>{gvtd_set}</strong> channels, the set the analysis uses; the count is on the "
        "figure. It is an RMS across channels, so that set is part of the value.",
    "haemo.hbo_hbr_corr":
        "A genuine haemodynamic response drives HbO up and HbR down, so r near &minus;1 is the "
        "target and r near 0 or positive means a shared artifact.",
    "psd_detail.picker": "Select a channel to compare its PSD across the stages.",
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
        "Each stimulus repetition is a row (trial × time from onset), coloured by HbO — the "
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
    "fc.connectogram":
        "The same correlations as chords rather than as cells: one circle per chromophore, "
        "channels blocked by source with blank circle between blocks, and a chord coloured "
        "by r on the scale the matrices use. The nodes carry no colour of their own, the "
        "blocks being separated by position already. The title says which connections are "
        "drawn and how many. Rejected channels are off the circle: they carry no edge.",
    "fc.seed_topography":
        "One flat map per seed ROI, each channel coloured by its correlation with that seed's "
        "mean signal, on the same scale as the matrix above. Solid grey channels are inside the "
        "seed: their value is undefined, not zero. Faded grey channels were rejected, and are "
        "in no seed. Short channels are not drawn.",
    "glm.activation":
        "HbO betas projected onto the surface, three views per condition. The colour scale is "
        "shared across every condition, so this switch is a comparison rather than five "
        "separate pictures: a condition that barely activated reads as weak instead of being "
        "stretched to fill its own scale. One model over the whole recording, so every "
        "condition here rests on the same channel set. Select a condition.",
    "metrics.intro":
        'Hover a metric for what it means, which way is good and what it was measured on. <span '
        'class="ql-dot" aria-hidden="true">&#9679;</span> marks the ones that decide whether '
        'the run is usable. A pair written <b>a &rarr; b</b> is one metric measured either side '
        'of a processing step; a single number means that step did not run.',
    "metrics.long_only":
        "Long channels only. Short channels return far more light and a far stronger pulse, so "
        "averaging the two lifts SCI, PSP and SNR; they are judged on their own in the "
        "per-channel table below.",
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
    "metrics.no_drift":
        "Low-frequency drift is not among these: it grows with the span it is fitted over, so a "
        "per-condition value would compare the conditions' durations. The run's own page "
        "carries it.",
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
}


def section_note(key: str, **values: object) -> str:
    """One section's paragraph, with any value slots filled.

    ::

      section_note("trial_qc", window="-5 to 25 s")

    Returns '' for a key nothing is written for, the way
    :func:`~fnirs_pipe.qc.boilerplate.vocabulary.metric_summary` does: a renamed key leaves
    a missing paragraph rather than stopping the render half way down a report.
    """
    text = SECTION_NOTES.get(key, "")
    return text.format(**values) if (text and values) else text
