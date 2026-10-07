"""Section prose for the raw QC viewer, and for the per-condition pages cut from it."""

from __future__ import annotations

NOTES = {
    # ---- Summary ----
    "raw.screened_long":
        "Screened on every channel; the metrics below are read off the long channels.",
    "raw.screened_every":
        "Screened on every channel; no channel was identified as short.",
    "raw.condition_scope":
        "Condition “{label}” only, {t0:.0f} to {t1:.0f} s.",
    "raw.condition_sliced":
        "This view describes {label} only. Its numbers are read out of the quality record, "
        "sliced there out of the whole recording's windowed pass rather than measured on a cut "
        "of it, so they sit on the same window grid and the same filter as every other "
        "condition and as the run.",
    "raw.condition_verdict":
        "The verdict here is this condition's own, screened on its windows against the run's "
        "line. The recording was processed under the run's verdict, not this one; the run's "
        "page carries it.",
    "raw.condition_spectrum":
        "The spectrum, the grand mean and the HbO–HbR correlation are redrawn on a cut of the "
        "recording; the per-channel detail's spectrum and epoch average stay the whole run's.",

    # ---- Panels ----
    "raw.channel_detail":
        "Pick a channel here or click its trace above. The panel holds that channel's "
        "concentration after Beer-Lambert and its spectrum.",
    "raw.channel_detail_picked":
        "{pair}: concentration, spectrum and epoch average, after Beer-Lambert.",
    "raw.psd":
        "One mean per separation group (long, short, neither range), each channel's "
        "spectrum faint behind it, on optical density, before Beer-Lambert.",
    "raw.sci_psp":
        "Scalp coupling index, peak spectral power and coefficient of variation per channel, "
        "all on one window grid: SCI and PSP on the uncorrected optical density, CV on the raw "
        "intensity. In the SCI row red marks a channel screening rejected and amber one below the "
        "line; amber in PSP and red in CV mark the wrong side of the line.",
    "raw.motion":
        "Per-channel z-score carpet with the global variance of the temporal derivative "
        "above it. Measured over the whole recording; a per-condition page narrows the "
        "view rather than remeasuring, so its colour scale and its threshold are the "
        "run's.",
    "raw.motion_detail":
        "One channel's optical density either side of the correction, under its own "
        "separation class's GVTD. Measured over the whole recording like the carpet above, so "
        "a per-condition page narrows the view rather than remeasuring it.",
    "raw.hbo_hbr_corr":
        "The channel-by-channel correlation and each pair's HbO–HbR r, on the concentration "
        "after Beer-Lambert with no filtering, rejected channels included. A cortical response "
        "pushes r negative and shared systemic or motion signals push it positive; short "
        "channels see no cortex and are grouped apart. No line is drawn, since the value "
        "moves with the passband. Measured before motion correction, so it need not match "
        "the pipeline report, which measures after it.",
    "raw.hbo_hbr_corr_post":
        "Left matrix and open rings: before {method}. Right matrix and filled dots: after "
        "it. Pair by pair, the after side is what the pipeline report shows for "
        "<code>preproc</code> when it ran the same correction.",
    "raw.events":
        "Every event on one axis, one row per condition. This is what shows a condition that "
        "stopped being delivered partway through or a block started twice, which no average "
        "can.",
    "raw.epoch":
        "One row per condition, the good long channels averaged, with the good short ones "
        "dotted. The dotted trace is what this panel is read for: when it rises with the "
        "solid one, part of the response is scalp haemodynamics, which only short-channel "
        "regression (<code>--short-channel</code>) separates later. Measured before "
        "filtering and motion correction, so read the relation between the two traces "
        "rather than the shape of either.",
    "raw.trial_image":
        "Trials down the rows, time across, one panel per condition. A channel that was fine "
        "for the first half of a block and lost for the second reads as a band here and is "
        "averaged away everywhere else. A condition with fewer than two trials draws nothing, "
        "which is what a block design is.",

    # ---- Metrics ----
    "raw.scope_long":
        "Measured on <b>long channels only</b>. A short channel sits a few millimetres from "
        "its source, so it returns far more light and a far stronger pulse than any long "
        "channel; averaging the two together lifts SCI, PSP and SNR and can make a poorly "
        "coupled recording read as a good one.",
    "raw.scope_split":
        "Measured over <b>every channel</b>. The three channel sets are in the tables below; "
        "the verdict is read off the long ones.",
    "raw.scope_every":
        "Measured on the long channels where separations are known, on every channel where "
        "they are not; no channel was identified as short.",
    "raw.sets":
        "The same recording measured over three channel sets. <b>Long</b> is the set the "
        "verdict is read off; <b>Short</b> is left uncoloured, a short channel's coupling "
        "being high by construction, so that row answers whether the regressors those "
        "channels feed are trustworthy rather than whether the run is usable.",
    "raw.motion_post":
        "Each cell is <b>before &rarr; after {method}</b>. GVTD counts use the uncorrected "
        "side's cutoff on both sides; spike counts use each side's own per-channel cutoff, "
        "and the threshold's after value is not the one applied.",
    "raw.haemo_sets":
        "Mean HbO–HbR correlation over each channel set, on the unfiltered concentration with "
        "rejected channels included. Uncoloured: no line on it has a source. A short channel "
        "has no cortical anticorrelation, so its row is not read against the long one.",
    "raw.haemo_post":
        "Each cell is <b>before &rarr; after {method}</b>.",
    "raw.channel_summary":
        "Status / Coupled / SCI / CV / PSP / SNR per channel, green = pass and red = fail. "
        "Status is the screening verdict, read off the Coupled row; the others are "
        "reported only. Every row is coloured against this run's lines.",
    "raw.trial_qc":
        "Each event window scored on its own. Colour is relative within a metric row, not a "
        "threshold.",
    "raw.trial_qc_counted":
        "{n} trials • colour is relative within a metric row, not a threshold",
    "raw.channel_decisions":
        "One row per source-detector pair: a decision applies to the pair, and SCI is a "
        "property of the pair rather than of either wavelength. SNR, CV and spike show the "
        "first wavelength; both are in the channel TSV (<code>_desc-rawchannel_qc.tsv</code>) "
        "beside the quality record. HbO–HbR corr is measured before motion correction. "
        "Rows in red were rejected by screening, and Status "
        "names the criterion. Click the decision chip to cycle: &#8212; &#8594; good "
        "&#8594; bad. Saves automatically.",
    "raw.skipped":
        "Notes on what this page shows and leaves out. Failures are listed below them.",
}
