"""Section prose for the raw QC viewer, and for the per-condition pages cut from it."""

from __future__ import annotations

NOTES = {
    # ---- Summary ----
    "raw.screened_long":
        "Screened on every channel; the metrics below are read off the long channels.",
    "raw.screened_every":
        "Screened on every channel; no channel was identified as short.",
    "raw.channels_tsv":
        "Marked bad by the recording's BIDS channels.tsv, whatever the screening found: {pairs}.",
    "raw.condition_scope":
        "Condition “{label}” only, {t0:.0f} to {t1:.0f} s.",
    "raw.condition_sliced":
        "This page covers {label} only, sliced out of the whole recording's windowed pass, so "
        "it shares the run's window grid and filter.",
    "raw.condition_verdict":
        "Rejected channels are the run's. The count above is a guide, the channels failing on "
        "this condition alone; it rejects nothing.",
    "raw.condition_spectrum":
        "The spectrum, grand mean and HbO–HbR correlation are redrawn on a cut of the "
        "recording; the per-channel spectrum stays the whole run's.",

    # ---- Panels ----
    "raw.signal":
        "Each pair's shorter wavelength in orange, its longer in blue; short channels dashed, "
        "unclassified ones dotted, rejected ones grey. Click a trace to select its channel and "
        "double-click to reset the view.",
    "raw.channel_detail":
        "Pick a channel by its trace above or in the list to see its concentration after "
        "Beer-Lambert and its spectrum.",
    "raw.channel_detail_picked":
        "{pair}: concentration and spectrum, after Beer-Lambert. Its epoch averages are under "
        "Trial Image.",
    "raw.psd":
        "Optical density before Beer-Lambert. One mean per separation group (long, short, "
        "neither), each channel faint behind it.",
    "raw.sci_psp":
        "SCI and PSP on the uncorrected optical density, CV on the raw intensity, on one window "
        "grid. Coloured cells are on the wrong side of the line; red in the SCI row is a "
        "rejected channel.",
    "raw.motion":
        "Per-channel z-scored carpet with GVTD above it, over the whole recording. A condition "
        "page only narrows the view; its scale and threshold are the run's.",
    "raw.motion_detail":
        "One channel's optical density before and after correction, under the GVTD of its "
        "separation class. Measured over the whole recording, like the carpet.",
    "raw.hbo_hbr_corr":
        "Channel-by-channel correlation and each pair's HbO–HbR r, unfiltered and before motion "
        "correction, rejected channels included. Negative r suggests cortical signal, positive "
        "systemic or motion; short channels are grouped apart.",
    "raw.hbo_hbr_corr_post":
        "Left matrix and open rings are before {method}, right matrix and filled dots after it.",
    "raw.events":
        "Every event on one axis, a row per condition. It shows what no average can: a "
        "condition that stopped partway through, or a block started twice.",
    "raw.epoch":
        "A row per condition: good long channels solid, good short ones dotted, uncorrected and "
        "low-passed for display below <code>--cardiac-l-freq</code>, so no heartbeat ripple. "
        "If the dotted trace rises with the solid one, part of the response is "
        "scalp signal, which <code>--short-channel</code> removes later.",
    "raw.trial_image":
        "Trials down, time across: all trials pooled, then the condition chosen on the head, "
        "each with its HbO and HbR average below. A channel lost partway through the run shows "
        "as a band in the pooled image. Low-passed as the grand mean is.",
    "raw.signal_by_position":
        "Each channel's HbO (solid) and HbR (dotted) average at its place on the head, one "
        "condition at a time (menu, top left), low-passed as the grand mean is. Click a channel "
        "for its trial images.",

    # ---- Metrics ----
    "raw.scope_long":
        "Measured on <b>long channels only</b>: short channels return more light and a stronger "
        "pulse and would flatter the run.",
    "raw.scope_split":
        "A value only in the All column is measured over <b>every channel</b>. The verdict is "
        "read off the Long column.",
    "raw.scope_every":
        "Measured on every channel; no channel was identified as short.",
    "raw.sets":
        "Each column is one channel set; judge the run on <b>Long</b>. <b>Short</b> is "
        "uncoloured: short channels couple well by construction and only feed the regressors.",
    "raw.motion_post":
        "Two values are before &rarr; after {method}. GVTD counts use the uncorrected cutoff on "
        "both sides; spike counts use each side's own.",
    "raw.haemo_sets":
        "Mean HbO–HbR r per channel set, unfiltered, rejected channels included. Uncoloured, as "
        "no published threshold exists; Short is not read against Long.",
    "raw.condition_summary":
        "The condition pages' channel grids in one figure, each metric's conditions on adjacent "
        "rows. <b>In condition</b> says whether a channel passes within that condition alone; "
        "it rejects nothing.",
    "raw.haemo_post":
        "Two values are before &rarr; after {method}.",
    "raw.channel_summary":
        "Green pass, red fail, against this run's lines. Only Status rejects a channel, read "
        "off the Passing row.",
    "raw.trial_qc":
        "Each event window scored on its own. Colour is relative within a metric row, not a "
        "threshold.",
    "raw.trial_qc_counted":
        "{n} trials, each scored on its own. Colour is relative within a metric row, not a "
        "threshold.",
    "raw.channel_decisions":
        "One row per source-detector pair; SNR, CV, Spike and Saturated print both wavelengths, "
        "and HbO–HbR corr is before motion correction. Red rows were rejected: BAD for too few "
        "passing windows, BAD (manual) by hand. Click a decision to cycle it; it saves "
        "automatically.",
    "raw.channel_table_condition":
        "One row per source-detector pair; SNR, CV, Spike and Saturated print both wavelengths, "
        "and HbO–HbR corr is before motion correction. Red rows are the run's rejections. A "
        "channel decision applies to the whole run, so it is made on the run page.",
    "raw.bad_spans":
        "Bad for the whole run. Switch marking on and drag along the raw signal; drag a "
        "band's edge to adjust it, or edit its times here. <code>nirspipe-rate export</code> "
        "writes them as a <code>--bad-spans</code> table.",
    "raw.rate_sci_line":
        "SCI (10 s) cells outlined in orange are under {line}, the line given to "
        "<code>nirspipe-rate raw</code>; red cells are under this run's screening line. The "
        "outline changes no verdict.",
    "raw.skipped":
        "What this page shows and leaves out. Failures are listed below.",
}
