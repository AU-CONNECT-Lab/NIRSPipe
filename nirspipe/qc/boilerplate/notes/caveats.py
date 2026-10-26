"""Caveats a run attaches to its pages: what the data could not support, and what follows."""

from __future__ import annotations

NOTES = {
    # ---- Montage ----
    "caveat.unplaced":
        "The optodes could not be placed on the template head ({reason}). Nothing is drawn "
        "on the brain; the flat maps and topographies draw the coordinates as they stand and "
        "are not anatomical. Separations, screening and metrics are unaffected. Landmarks in "
        "the file or a standard montage would fix it.",
    "caveat.child_template":
        "--population child places the optodes on fsaverage, an adult head: MNE ships no "
        "template for 2 to 18 years. Pass --template to put the youngest on an infant one.",
    "caveat.no_separation":
        "No channel fell in either separation range: the recording has no optode positions "
        "(every separation reads as zero) or the ranges exclude every channel. Metrics "
        "cover every channel, the per-channel table is not grouped, and short-channel "
        "regression is unavailable.",
    "caveat.unclassified":
        "{n} channel(s) sit at a separation neither range covers ({ranges}). They were "
        "screened and keep their per-channel scores, but count in no long or short "
        "metric.{where}",
    "caveat.unclassified_where":
        "Theirs sit at {span}: --short-max-dist {short_max} would make them short and "
        "--long-min-dist {long_min} long, depending on how deep this montage's short end "
        "reaches.",
    "caveat.short_regression_all_bad":
        "Short-channel regression did not run: all {n} short channels were rejected. Their "
        "scores are in the per-channel table.",

    # ---- Channel screening ----
    "caveat.unscreened":
        "Channel screening could not judge this recording: {reason}. No channel was rejected "
        "by it, so OK here means unscreened; preprocessing refuses the recording.",

    # ---- Optics ----
    "caveat.dpf_age_range":
        "--dpf age used the Scholkmann and Wolf (2013) equation outside the 0 to 70 years and "
        "690 to 950 nm it is recommended for ({outside}), so its DPF there is an "
        "extrapolation. It is also only validated on the frontal head.",

    # ---- Motion correction ----
    "caveat.motion_overmarked":
        "Motion correction ({method}) marked {share:.0%} of the good channels' samples as "
        "motion. At {sfreq:g} Hz its threshold of {thresh:g} standard deviations may take "
        "physiology for motion; check the corrected signal and set --motion-std-thresh for "
        "this recording.",

    # ---- ROI mapping ----
    "caveat.roi_overlap":
        "The ROI mapping lists {n} channel(s) in more than one ROI: {channels}. Each counts "
        "in every ROI that lists it, so those ROIs are not independent.",

    # ---- Epochs and trials ----
    "caveat.chunked_trials":
        "Annotations at least twice {chunk:g} s long were cut into {chunk:g} s trials "
        "before epoching, so a trial in the epoch figures is one piece of a block.",
    "caveat.epoch_window":
        "The epoch figures average {tmin:g} to {tmax:g} s while this run's events are "
        "{outruns:g} s long, so they show only the start of each block. Pass --epoch-tmin / "
        "--epoch-tmax to widen it.",
    "caveat.epoch_skipped":
        "Grand mean, evoked channel map, trial images and per-trial quality were skipped "
        "because {reason}. The per-channel, carpet and layout figures show the continuous "
        "signal instead.",
    "caveat.short_trials":
        "{n} of {total} trial windows are shorter than {window:g} s, too short for an SCI, "
        "so they are left unscored. Pass --epoch-tmin / --epoch-tmax to score a fixed window "
        "from each onset.",
    "caveat.partial_trials":
        "{n} of {total} trial windows are {floor:g} to {window:g} s long, so they are scored "
        "on SCI and GVTD only: CV, SNR and PSP need {window:g} s.",
    "caveat.block_design_trials":
        "no scored trial falls inside this window: the annotation that defines it is not "
        "one, and events with no duration are scored only with --epoch-tmin/--epoch-tmax",
    "caveat.single_trial":
        "One trial in the epoch window, so its epoch panels show that trial rather than an "
        "average: {conds}.",
    "caveat.one_trial":
        "this condition holds {n} trial inside its window, and one row is not a comparison",

    # ---- Filtering ----
    "caveat.filter_edge":
        "The first and last {edge_s:g} s of the filtered recording, one period of the "
        "{l_freq:g} Hz high-pass, are partly filter settling. A GLM run avoids this with "
        "--drift-model cosine and --drift-high-pass in place of --high-pass.",
    "caveat.drift_auto_twice":
        "--drift-high-pass auto set the cutoff to {cutoff} Hz from {condition}, which occurs "
        "only twice, {gap} s apart. If it marks something other than a condition, leave it "
        "out with --events-path or nirspipe-prep edit-markers, or type a cutoff.",
    "caveat.drift_auto_once":
        "Seen once, so no interval for --drift-high-pass auto to read: {conditions}.",

    # ---- Dyads ----
    "caveat.never_aligned":
        "these recordings were trimmed to a common length but never aligned on a shared "
        "trigger, so every number below assumes they already shared a clock",
    "caveat.onsets_disagree":
        "The members&rsquo; markers disagree by more than {n} samples ({tol}&nbsp;s) after "
        "alignment: {listing}. Every window is cut from {ref}&rsquo;s markers, so a late "
        "trigger shifts that block and clock drift shifts everything after it.",
    "caveat.cut_with_margin":
        "Input is a cut ({where}) with a {margin:.1f} s margin on each side. Condition rows "
        "read their block out of the transform, so the margin absorbs the cone; the "
        "whole-run row still averages the margin unless --tstart/--tend restrict it.",
    "caveat.cut_no_margin":
        "Input is a cut ({where}) with no margin, so its own edges inflate the band means, "
        "more as the segment shortens (n_valid_frac gives the share of band cells kept). "
        "Re-cut with `nirspipe-prep crop --input-desc &lt;desc&gt; --margin auto "
        "--band-fmin &lt;f&gt;`.",
    "caveat.isc_unfiltered":
        "The files for {subjects} record no bandpass (stage {stages}), so drift and "
        "systemic physiology enter the correlation directly, and members recorded together "
        "drift alike. Pass --isc-fmin/--isc-fmax, or point --desc at a filtered stage "
        "(filtered, errts).",
    "caveat.isc_magnitude_phase":
        "|r| was tested against the phase-scrambled null on correlations that were not "
        "whitened, where |r| tends to sit above that null, so these p values run small. "
        "Whiten the correlation in nirspipe-hyper (--isc-whiten-s) or test it signed.",

    # ---- Cohorts ----
    "caveat.bandpass_pairs":
        "{pre}_* was measured before the run's filter or model and {post}_* after it, so "
        "the difference between such a pair is mostly what that step removed. {n} metrics "
        "sit on both sides ({shown}).",
    "caveat.window_mismatch":
        "the table's sci_win_mean, psp_mean, cv_mean and snr_mean use {pinned:g} s windows "
        "whatever --window-length is, while this cohort's windowed panels use {windows} s, "
        "so a panel and its column need not agree.",

    # ---- Figure captions ----
    "figure.phase_arrows":
        "arrows: right = in phase, left = antiphase, up = {lead} leads by a quarter cycle, "
        "drawn only where coherence clears {clears}",
    "figure.coi_band":
        "washed-out band: the cone of influence, where the record's edges reach",
    "figure.excluded_band":
        "washed-out columns: BAD_ time of either member, left out of every average",

    # ---- Every report's footer ----
    "footer.provenance_not_rendered":
        "Diagram not rendered: drawing it failed when this page was written. The reason is "
        "in the run's log (a dyad report also lists it under Errors); write the page again "
        "once it is fixed.",
}
