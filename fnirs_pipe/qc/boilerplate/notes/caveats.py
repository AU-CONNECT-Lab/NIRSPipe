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
        "optode positions looks like (every separation reads as zero), or one whose ranges "
        "exclude every channel. The quantitative metrics are over every channel rather "
        "than long channels only, the per-channel table is not grouped, and short-channel "
        "regression is unavailable for this run.",
    "caveat.unclassified":
        "{n} channel(s) sit at a separation the long and short ranges leave out ({ranges}). "
        "They were screened and their row in the per-channel table carries the whole-montage "
        "scores, but no split claims them, so they are in none of the long or short scalar "
        "metrics.{where}",
    "caveat.unclassified_where":
        "Theirs sit at {span}, so --short-max-dist {short_max} would make them short channels "
        "and --long-min-dist {long_min} would make them long; which is right depends on how "
        "deep this montage's short end reaches, which the separations alone do not settle.",
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
        "Annotations at least twice {chunk:g} s long were cut into {chunk:g} s trials "
        "before epoching, so such a trial in the epoch figures and the per-trial panel is "
        "one piece of a block rather than the whole block.",
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
        "no scored trial falls inside this window: the annotation that defines it is not "
        "one, and events with no duration are scored only with --epoch-tmin/--epoch-tmax",
    "caveat.one_trial":
        "this condition holds {n} trial inside its window, and one row is not a comparison",

    # ---- Filtering ----
    # TODO(review): attached at any ratio, including near or below 1; attach only above a stated ratio
    "caveat.filter_edge":
        "The first and last {edge_s:g} s of the filtered recording carry {ratio:.2f}x the "
        "RMS of everything between them. Well above 1 this is usually the {l_freq:g} Hz "
        "high-pass settling rather than signal: a low cutoff needs a long filter, every "
        "figure drawn on a filtered stage includes it, and a detrend does not remove it. A "
        "GLM run can avoid it by leaving --high-pass off and giving the low band to "
        "--drift-model cosine with --drift-high-pass, which projects rather than filters.",

    # ---- Dyads ----
    "caveat.never_aligned":
        "these recordings were trimmed to a common length but never aligned on a shared "
        "trigger, so every number below assumes they already shared a clock",
    "caveat.cut_with_margin":
        "Input is a cut ({where}), made with a {margin:.1f} s margin on each side. Each "
        "condition's row reads its block out of the transform, so there the margin absorbs "
        "the cone; the whole-run row still averages the margin unless --tstart/--tend "
        "restrict it to the cut's span.",
    "caveat.cut_no_margin":
        "Input is a cut ({where}), made with no margin. A wavelet transform of a segment "
        "has two edges of its own, so this run's band means are inflated by an amount that "
        "grows as the segment shortens; n_valid_frac reports the share of band cells that "
        "survived. Re-cut the stage this run reads with `fnirs-prep crop --input-desc "
        "&lt;desc&gt; --margin auto --band-fmin &lt;f&gt;`; the per-condition rows then "
        "exclude the margin.",
    # TODO(review): attached even when --isc-fmin/--isc-fmax band-limit the correlation; skip it then
    "caveat.isc_unfiltered":
        "The files for {subjects} record no bandpass (stage {stages}). A correlation has "
        "no frequency axis, so unless --isc-fmin/--isc-fmax limit its band, drift and "
        "systemic physiology enter it directly, and two members recorded together drift "
        "alike. The wavelet coherence panels are unaffected. Pass --isc-fmin/--isc-fmax, "
        "or point --desc at a filtered stage (filtered, errts).",

    # ---- Cohorts ----
    "caveat.bandpass_pairs":
        "{pre}_* was measured before the run's filter or model and {post}_* after it, so "
        "the difference between such a pair is mostly the frequencies that step removed "
        "and not what the step was for. {n} metrics sit on both sides ({shown}). The "
        "subject report's stage comparison re-applies the passband before comparing where "
        "there is one; this table stores each stage as it stands.",
    "caveat.window_mismatch":
        "the table's sci_win_mean, psp_mean, cv_mean and snr_mean are measured over "
        "{pinned:g} s windows whatever --window-length is set to, so they stay comparable "
        "across runs; this cohort binned its windowed panels at {windows} s, so a panel "
        "and its column are averaged over different window lengths and need not agree.",

    # ---- Figure captions ----
    "figure.phase_arrows":
        "arrows: right = in phase, left = antiphase, up = {lead} leads by a quarter cycle, "
        "drawn only where coherence clears {clears}",
    "figure.coi_band":
        "washed-out band: the cone of influence, where the record's edges reach",

    # ---- Every report's footer ----
    "footer.provenance_not_rendered":
        "Diagram not rendered: drawing it failed when this page was written. The reason is "
        "in the run's log (a dyad report also lists it under Errors); write the page again "
        "once it is fixed.",
}
