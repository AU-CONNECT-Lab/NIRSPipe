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
        "This page covers {label} only. Its numbers are sliced out of the whole recording's "
        "windowed pass, not measured on a cut, so they share the run's window grid and filter.",
    "raw.condition_verdict":
        "Rejected channels are the run's. The count above is the channels failing on this "
        "condition's windows alone, against the run's line; it is a guide and rejects nothing.",
    "raw.condition_spectrum":
        "The spectrum, grand mean and HbO–HbR correlation are redrawn on a cut of the "
        "recording; the per-channel spectrum and epoch average stay the whole run's.",

    # ---- Panels ----
    "raw.layout":
        "Red squares are sources and blue dots detectors; a channel's colour is its SCI. "
        "Click a channel to highlight it.",
    "raw.signal":
        "Short channels in blue, long in orange. Click a trace to select its channel and "
        "double-click to reset the view.",
    "raw.channel_detail":
        "Pick a channel here or click its trace above to see its concentration after "
        "Beer-Lambert and its spectrum.",
    "raw.channel_detail_picked":
        "{pair}: concentration, spectrum and epoch average, after Beer-Lambert.",
    "raw.psd":
        "Optical density before Beer-Lambert. One mean per separation group (long, short, "
        "neither), each channel faint behind it.",
    "raw.sci_psp":
        "SCI and PSP on the uncorrected optical density, CV on the raw intensity, all on one "
        "window grid. In the SCI row red is a channel screening rejected and amber one below "
        "the line; in PSP amber, and in CV red, mark the wrong side of the line.",
    "raw.motion":
        "Per-channel z-scored carpet with GVTD above it, measured over the whole recording. A "
        "condition page narrows the view without remeasuring, so its colour scale and "
        "threshold are the run's.",
    "raw.motion_detail":
        "One channel's optical density before and after correction, under the GVTD of its "
        "separation class. Measured over the whole recording, like the carpet.",
    "raw.hbo_hbr_corr":
        "Channel-by-channel correlation and each pair's HbO–HbR r, on the unfiltered "
        "concentration before motion correction, rejected channels included. A cortical "
        "response pushes r negative and shared systemic or motion signal pushes it positive. "
        "Short channels see no cortex and are grouped apart. No line is drawn because r moves "
        "with the passband; the pipeline report measures after correction and need not match.",
    "raw.hbo_hbr_corr_post":
        "Left matrix and open rings are before {method}, right matrix and filled dots after "
        "it. The after side matches the pipeline report's <code>preproc</code> when it ran the "
        "same correction.",
    "raw.events":
        "Every event on one axis, a row per condition. It shows what no average can: a "
        "condition that stopped partway through, or a block started twice.",
    "raw.epoch":
        "A row per condition: good long channels averaged as the solid line, good short ones "
        "dotted. When the dotted trace rises with the solid one, part of the response is "
        "scalp signal, which short-channel regression (<code>--short-channel</code>) removes "
        "later. Unfiltered and before motion correction, so read how the two traces relate "
        "rather than the shape of either.",
    "raw.trial_image":
        "Trials down, time across, a panel per condition. A channel lost partway through a "
        "block shows as a band here and is averaged away everywhere else. A condition with "
        "fewer than two trials draws nothing.",

    # ---- Metrics ----
    "raw.scope_long":
        "Measured on <b>long channels only</b>. Short channels return far more light and a "
        "stronger pulse, so mixing them in would lift SCI, PSP and SNR and flatter a poorly "
        "coupled run.",
    "raw.scope_split":
        "A value only in the All column is measured over <b>every channel</b>. The verdict is "
        "read off the Long column.",
    "raw.scope_every":
        "Measured on every channel; no channel was identified as short.",
    "raw.sets":
        "Each column is one channel set, and the verdict is read off <b>Long</b>. "
        "<b>Short</b> is uncoloured: short channels couple well by construction, so that "
        "column says whether the regressors are trustworthy, not whether the run is usable.",
    "raw.motion_post":
        "Two values are before &rarr; after {method}. GVTD counts use the uncorrected cutoff "
        "on both sides, so the threshold's after value is not applied; spike counts use each "
        "side's own per-channel cutoff.",
    "raw.haemo_sets":
        "Mean HbO–HbR r per channel set, unfiltered, rejected channels included. Uncoloured: "
        "no threshold for it has a published source. Short channels carry no cortical "
        "anticorrelation, so their column is not read against Long.",
    "raw.condition_summary":
        "The condition pages' channel grids in one figure, each metric's conditions on "
        "adjacent rows, same cutoffs. <b>In condition</b> says whether a channel passes within "
        "that condition alone; it rejects nothing.",
    "raw.haemo_post":
        "Two values are before &rarr; after {method}.",
    "raw.channel_summary":
        "Green pass, red fail, against this run's lines. Only Status rejects a channel, and it "
        "is read off the Coupled row; the other rows are for reference.",
    "raw.trial_qc":
        "Each event window scored on its own. Colour is relative within a metric row, not a "
        "threshold.",
    "raw.trial_qc_counted":
        "{n} trials, each scored on its own. Colour is relative within a metric row, not a "
        "threshold.",
    "raw.channel_decisions":
        "One row per source-detector pair, since a decision and SCI belong to the pair. SNR, "
        "CV and Spike print both wavelengths. HbO–HbR corr is before motion correction. "
        "Red rows were rejected: BAD is too few coupled windows, BAD (manual) by hand. "
        "Click a decision to cycle it through unset, good and bad; it saves "
        "automatically.",
    "raw.rate_sci_line":
        "SCI (10 s) cells outlined in orange are under {line}, the line given to "
        "<code>nirspipe-rate raw</code>. A red cell is under the line this run was screened at; "
        "the outline changes no verdict.",
    "raw.skipped":
        "What this page shows and leaves out. Failures are listed below.",
}
