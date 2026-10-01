"""Section prose for the raw QC viewer, and for the per-condition pages cut from it."""

from __future__ import annotations

NOTES = {
    # ---- Summary ----
    "raw.screened_long": "Screened on the long channels.",
    "raw.screened_every":
        "Screened on every channel: this montage carries no short channels to judge "
        "separately.",
    "raw.condition_scope":
        "Condition “{label}” only, {t0:.0f} to {t1:.0f} s. The verdict here is this "
        "condition's own, screened on its windows against the run's line; the recording was "
        "processed under the run's, which the run's own page carries.",
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
        "The spectrum is the one panel measured on a cut of the recording rather than sliced "
        "out of the run's pass.",

    # ---- Panels ----
    "raw.channel_detail":
        "Pick a channel here or click its trace above. The panel holds that channel's "
        "concentration after Beer-Lambert and its spectrum.",
    "raw.channel_detail_picked":
        "{pair}: concentration, spectrum and epoch average, after Beer-Lambert.",
    "raw.psd": "Averaged over every channel, on optical density, before Beer-Lambert.",
    "raw.sci_psp":
        "Scalp coupling index, peak spectral power and coefficient of variation per channel, "
        "every row on one window grid and all three measured on the uncorrected optical "
        "density. Red marks a channel below the line.",
    "raw.motion":
        "Per-channel z-score carpet with the global variance of the temporal derivative under "
        "it. Measured over the whole recording; a per-condition page narrows the view rather "
        "than remeasuring, so its colour scale and its threshold are the run's.",
    "raw.motion_detail":
        "One channel's optical density either side of the correction, under its own "
        "separation class's GVTD. Measured over the whole recording like the carpet above, so "
        "a per-condition page narrows the view rather than remeasuring it.",
    "raw.events":
        "Every event on one axis, one row per condition. This is what shows a condition that "
        "stopped being delivered partway through or a block started twice, which no average "
        "can.",
    "raw.epoch":
        "One row per condition, every long channel averaged, with the short ones dotted. The "
        "dotted trace is what this panel is read for: when it rises with the solid one the "
        "response is scalp haemodynamics rather than activation, and no later step separates "
        "the two. Measured before filtering and motion correction, so read the relation "
        "between the two traces rather than the shape of either.",
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
        "Measured on every channel: this montage carries no short channels to judge "
        "separately.",
    "raw.sets":
        "The same recording measured over three channel sets. <b>Long</b> is the set the "
        "verdict is read off; <b>Short</b> is left uncoloured, a short channel's coupling "
        "being high by construction, so that row answers whether the regressors those "
        "channels feed are trustworthy rather than whether the run is usable.",
    "raw.motion_post":
        "Each cell is <b>before &rarr; after {method}</b>. Both sides are counted against the "
        "uncorrected side's cutoff.",
    "raw.channel_summary":
        "Status / SCI / CV / PSP / SNR per channel, green = pass and red = fail. Status "
        "screens on the coupled-window share; SCI, PSP, CV and SNR are reported only.",
    "raw.trial_qc":
        "Each event window scored on its own. Colour is relative within a metric row, not a "
        "threshold.",
    "raw.trial_qc_counted":
        "{n} trials • colour is relative within a metric row, not a threshold",
    "raw.channel_decisions":
        "One row per source-detector pair: a decision applies to the pair, and SCI is a "
        "property of the pair rather than of either wavelength. The per-wavelength numbers are "
        "in the channel metrics CSV written beside the quality record. Rows in red were "
        "rejected by screening, and Status names the criterion. Click the decision chip to "
        "cycle: &#8212; &#8594; good &#8594; bad. Saves automatically.",
    "raw.skipped":
        "Sections this run skipped because the recording does not carry what they need. Not "
        "failures.",
}
