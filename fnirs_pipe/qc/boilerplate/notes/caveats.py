"""Caveats a run attaches to its pages: what the data could not support, and what follows."""

from __future__ import annotations

NOTES = {
    # ---- Montage ----
    "caveat.unregistered":
        "The optode positions are not registered to this recording's head coordinates: they "
        "sit a median of {reach:.0f} mm from the head centre while the fiducials put the scalp "
        "at {scalp:.0f} mm. Everything drawn from positions, the 3-D views, the flat maps and "
        "the topographies, is therefore not anatomical, and no channel can be placed on a "
        "brain region. Separations are measured between optodes rather than against the head, "
        "so the long / short split, the screening and every metric built on them are "
        "unaffected. Fixing it needs the digitised nasion and preauricular points the "
        "recording was taken with, or a standard montage put in their place; neither can be "
        "recovered from the file itself.",
    "caveat.no_separation":
        "No channel fell in either separation range, which is what a recording with no "
        "registered optode positions looks like. The quantitative metrics are over every "
        "channel rather than long channels only, and the per-channel table is not grouped. "
        "Anything that needs positions, including short-channel regression and the "
        "topographies, is unavailable for this run.",
    "caveat.unclassified":
        "{n} channel(s) sit at a separation the long and short ranges leave out ({ranges}). "
        "They were screened and their row in the per-channel table carries the whole-montage "
        "scores, but no split claims them, so they are in none of the long or short scalar "
        "metrics.{where}",
    "caveat.unclassified_where":
        "Theirs sit at {span}, so --short-max-dist {short_max} would make them short channels "
        "and --long-min-dist {long_min} would make them long; which is right depends on how "
        "deep this montage's short end reaches, which the separations alone do not settle.",
    "caveat.short_regression_none":
        "Short-channel regression was requested but this montage carries no short channel, so "
        "it did not run and no systemic signal was regressed out.",
    "caveat.short_regression_all_bad":
        "Short-channel regression was requested but all {n} short channels were rejected, so "
        "it did not run. Their scores are in the per-channel table.",

    # ---- ROI mapping ----
    "caveat.roi_overlap":
        "The ROI mapping lists {n} channel(s) in more than one ROI: {channels}. Each counts in "
        "every ROI that lists it, so those ROIs share part of what they average and the values "
        "between them are not independent.",

    # ---- Epochs and trials ----
    "caveat.chunked_trials":
        "Task annotations were cut into {chunk:g} s trials before epoching, so a trial in the "
        "epoch figures and the per-trial panel is one piece of a block rather than the whole "
        "block.",
    "caveat.epoch_window":
        "The epoch figures average a {tmin:g} to {tmax:g} s window while this run's events are "
        "{outruns:g} s long, so they describe the start of each block rather than the whole of "
        "it. Pass --epoch-tmin / --epoch-tmax to widen it. The per-trial panel below is "
        "unaffected: it scores each event over its own duration.",
    "caveat.epoch_skipped":
        "Grand mean, evoked channel map, trial images and per-trial quality were skipped "
        "because {reason}. The per-channel, carpet and layout figures show the continuous "
        "signal instead.",
    "caveat.block_design_trials":
        "the only event inside this window is the annotation that defines it, which is what a "
        "block design looks like",
    "caveat.one_trial":
        "this condition holds {n} trial inside its window, and one row is not a comparison",
    "caveat.few_trials":
        "this condition holds {n} trials inside its window, and one row is not a comparison",

    # ---- Filtering ----
    "caveat.filter_edge":
        "The first and last {edge_s:g} s of the filtered recording carry {ratio:.2f}x the RMS "
        "of everything between them. That is the {l_freq:g} Hz high-pass settling, not "
        "signal: a low cutoff needs a long filter. Every figure drawn on a filtered stage "
        "includes it, and a detrend does not remove it. A GLM run can avoid it by leaving "
        "--high-pass off and giving the low band to --drift-model cosine instead, which "
        "projects rather than filters.",

    # ---- Dyads ----
    "caveat.never_aligned":
        "these recordings were trimmed to a common length but never aligned on a shared "
        "trigger, so every number below assumes they already shared a clock",
    "caveat.cut_with_margin":
        "Input is a cut ({where}), made with a {margin:.1f} s margin on each side. "
        "--wtc-by-condition windows each block out of the transform, so the margin is what "
        "absorbs the cone and is not averaged.",
    "caveat.cut_no_margin":
        "Input is a cut ({where}), made with no margin. A wavelet transform of a segment has "
        "two edges of its own, so this run's band means are inflated by an amount that grows "
        "as the segment shortens; n_valid_frac reports the share of band cells that survived. "
        "Re-cut with `fnirs-prep crop --margin auto --band-fmin <f>` to avoid it.",
    "caveat.isc_unfiltered":
        "The files for {subjects} record no bandpass (stage {stages}). A whole-record "
        "correlation has no frequency axis, so drift and systemic physiology enter it "
        "directly, and two members recorded together drift alike. The wavelet coherence "
        "panels are unaffected. Point --desc at a filtered stage (filtered, errts) to read "
        "these numbers as neural.",

    # ---- Cohorts ----
    "caveat.bandpass_pairs":
        "{pre}_* was measured before the bandpass and {post}_* after it, so the difference "
        "between such a pair is mostly the filter and not what the step did. {n} metrics sit "
        "on both sides ({shown}). The subject report's stage comparison re-applies the "
        "passband before comparing; this table stores each stage as it stands and does not. "
        "*_band_frac carries the same trap with a denominator that moves, so read "
        "*_band_power instead.",
    "caveat.window_mismatch":
        "the table's sci_win_mean, psp_mean, cv_mean and snr_mean are measured over "
        "{pinned:g} s windows whatever --qc-window is set to, so they stay comparable across "
        "runs; this cohort binned its windowed panels at {windows} s, so a panel and its "
        "column do not describe the same stretch of recording.",

    # ---- Figure captions ----
    "figure.phase_arrows":
        "arrows: right = in phase, left = antiphase, up = {lead} leads by a quarter cycle, "
        "drawn only where coherence clears {clears}",
    "figure.coi_band": "washed-out band: outside the cone of influence",

    # ---- Every report's footer ----
    "footer.provenance_not_rendered":
        "Diagram not rendered: drawing it failed when this page was written. The reason is "
        "under Errors, or in the log on a page without that section; write the page again "
        "once it is fixed.",
}
