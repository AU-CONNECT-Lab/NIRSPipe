"""Section prose for the dyad reports: raw, post, the cohort page and a dyad's window index."""

from __future__ import annotations

NOTES = {
    # ---- Both dyad reports ----
    "hyper.clocks":
        "Offset at which each recording was cropped, at its first shared trigger (zero "
        "under <code>--no-align</code>, where recordings are only trimmed to the "
        "shortest). A dyad window is on the aligned clock and a member&rsquo;s own record "
        "is on its own, and the offsets are what convert between them.",

    # ---- Raw dyad report ----
    "hyper_raw.scope":
        "This page is the raw pass: every channel is screened on the recording as "
        "acquired, and the dyad numbers below are taken over the long channels. It is what "
        "says whether the recording is worth preprocessing. The synchrony below it is a "
        "screening flag read against its own null, not the analysis.",
    "hyper_raw.scope_post_link":
        'That is on the <a href="{href}">post report</a>.',
    "hyper_raw.blocks_verdict":
        "Times are on the shared clock, after the crop. A block that does not line up "
        "between the members&rsquo; rows of the timeline in section 1 below is a trigger "
        "that landed late in one member, or two clocks drifting apart.",
    "hyper_raw.alignment":
        "Alignment crops each recording at the first shared trigger, which removes a "
        "constant offset and nothing else (under <code>--no-align</code> the recordings "
        "are only trimmed to the shortest). What it cannot remove is a trigger that landed "
        "late in one member, or two device clocks drifting apart at the same nominal rate. "
        "Both show up as a block that does not line up below.",
    "hyper_raw.alignment_timeline":
        "One row per member. The top pair is what was recorded; the bottom pair is what every "
        "later panel reads.",
    "hyper_raw.usable":
        "A channel is usable by the dyad only while it is coupled in <em>both</em> members at "
        "the same moment, so the dyad&rsquo;s usable time is the intersection of the two and "
        "not the smaller of them: two members can each keep all but one pair and share one "
        "fewer than either, if the pair they each lose is a different one.",
    "hyper_raw.usable_panel":
        "The series over the carpet are each member&rsquo;s long-channel means of SCI and "
        "PSP, the two metrics the mask is made of, so a dip in either is the reason for "
        "the hole under it; CV is drawn as well but takes no part in the mask. Horizontal "
        "dotted rules are the screening lines and vertical ones are block edges.",
    "hyper_raw.motion":
        "SCI and PSP measure coupling rather than movement, so the carpet above can show a "
        "pair decoupled without saying whether the member moved. Each trace is that "
        "member&rsquo;s GVTD divided by its own median, so 1.0 is that member&rsquo;s "
        "usual level. GVTD is in each recording&rsquo;s own units, so the traces say when "
        "each member moved, not who moved more. The grey floor under the two traces is "
        "their pointwise minimum, which is high only where both are high: simultaneous "
        "movement.",
    "hyper_raw.motion_before":
        "The amber strip marks the spans where <em>every</em> member was spiking at once. "
        "A member spiking alone shows in that member&rsquo;s own trace, and on the "
        "usable-time carpet above only where it also broke SCI or PSP. The grey carpets "
        "are each member&rsquo;s per-channel optical density, z-scored the way the subject "
        "report draws it.",
    "hyper_raw.motion_after":
        "The same rows on the same axes, off each member&rsquo;s motion-corrected derivative. "
        "Both figures divide by the <em>uncorrected</em> median and both carpets are z-scored "
        "by the <em>uncorrected</em> mean and SD, so a fall here is movement removed rather "
        "than the yardstick following the data.",
    "hyper_raw.motion_together":
        "A member moving alone shows in that member&rsquo;s own trace above. <b>Both "
        "moving at once is the dyad&rsquo;s problem</b>: it raises any synchrony measure "
        "taken on the pair, and a shifted or scrambled copy of one member does not remove "
        "it. This is what the screening synchrony below has to be read against.",
    "hyper_raw.head":
        "The same screening, member by member and on the montage: panel a as each "
        "block&rsquo;s share of coupled windows, panel b as each window&rsquo;s SCI. A "
        "long channel is a bar of discs along its source-detector path, a short one a "
        "single larger disc in a grey ring; both share one colour scale, so the short "
        "channels read as a contamination check rather than a second map.",
    "hyper_raw.head_by_condition":
        "Colour is the share of that block&rsquo;s windows in which the channel was coupled, "
        "in that member.",
    "hyper_raw.head_slider":
        "Colour is that window&rsquo;s SCI. The block the slider sits in is named above the "
        "heads and moves with it.",
    "hyper_raw.screening":
        "Band coherence in {fmin}&ndash;{fmax}&thinsp;Hz, read as each window&rsquo;s rank "
        "inside a null drawn for <em>that</em> window by pairing one member against a "
        "phase-scrambled copy of the other. The estimator&rsquo;s floor moves with the window "
        "length, so two windows&rsquo; raw values are not on a common scale and neither is "
        "readable alone.",
    "hyper_raw.screening_scope":
        "This is a screening flag, not the analysis: the wavelet coherence is on the post "
        "report.",
    "hyper_raw.screening_scope_linked":
        "This is a screening flag, not the analysis: the wavelet coherence is on the post "
        'report, <a href="{href}">here</a>.',
    "hyper_raw.screening_strip":
        "One dot per channel, pale below the line; the diamond is the window&rsquo;s own "
        "rank with the channels pooled. Both are ranks over the same {n_iter} unseeded "
        "draws, so both move between runs. Read the diamond: it is one test per window "
        "rather than one per channel.",
    "hyper_raw.screening_flag":
        "<b>A flag, not a result.</b> Shared movement, shared task structure and shared "
        "physiology all raise this number and the surrogate removes none of them: it removes "
        "only that the two members were together <em>at that moment</em>.",
    "hyper_raw.comparable":
        "Everything downstream matches channels by S-D label and assumes one clock. An "
        "unmatched label is silent in every later panel, where it becomes a blank row. A "
        "nominal sampling-rate mismatch is refused by every inter-brain measure; a clock "
        "drift between devices at the same nominal rate is silent, and alignment cannot "
        "remove it.",
    "hyper_raw.member_metrics":
        "Hover a column header for which end is better; SCI is coloured against this "
        "run&rsquo;s threshold. One table: each member&rsquo;s long-channel values where "
        "the montage has short channels, every channel where it has none.",
    "hyper_raw.group_sqm":
        "The dyad&rsquo;s own numbers rather than either member&rsquo;s: how many channel pairs "
        "survived in both, and where the screening synchrony sits against its own null.",
    "hyper_raw.channel_summary":
        "Green = kept by every member, yellow = mixed, red = kept by none.",
    "hyper_raw.channel_decisions":
        "One row per source-detector pair, both members side by side, with the columns the "
        "subject report prints and each member&rsquo;s decision at the end of its own "
        "block. Rows tinted red were rejected in the first member; each member&rsquo;s "
        "Status cell says whether that member rejected it and on which criterion. Click a "
        "chip to cycle: &mdash; &rarr; good &rarr; bad. Saves to each member&rsquo;s JSON, "
        "and is only active when the page is served by <code>fnirs-rate hyper</code>.",
    "hyper_raw.per_channel":
        "The one view an individual report cannot give: two members on the same axes.",

    # ---- Post dyad report ----
    "hyper_post.scope_condition":
        "Every coherence panel below is this window read out of the whole-run transform, "
        "so it carries the recording&rsquo;s cone of influence rather than two edges of "
        "its own (under <code>--wtc-cond-transform</code> the window has a padded "
        "transform of its own instead). The correlation panels are computed on the cut "
        "window.",
    "hyper_post.scope_restricted":
        "Restricted to {window} by <code>--tstart</code>/<code>--tend</code>, taken out of the "
        "transform of the whole recording. The coherence is averaged across every condition "
        "inside that window.",
    "hyper_post.scope_whole":
        "The coherence over the whole recording, averaged across every condition in it. The "
        "per-condition pages are the ones to read for a block design.",
    "hyper_post.too_short":
        "This window holds only {cycles} cycles of <code>--wtc-band-fmin</code>, the slowest "
        "frequency being averaged. A coherence there is a statement about a phase relationship "
        "this window has seen about that many times, so the low-frequency end of every number "
        "below rests on very few independent looks at it. Raise <code>--wtc-band-fmin</code>, "
        "or read that end as indicative. Nothing is masked or dropped on account of this.",
    "hyper_post.roi_grouping":
        "The channel-to-region mapping from <code>--roi-mapping</code>, which is what the "
        "per-ROI panels below are averaged over.",
    "hyper_post.chromophores":
        "The maps below show {stacked}, stacked and labelled, and the matrices put the "
        "chromophores side by side on one colour scale; the tables carry {listed} in a "
        "<code>chromophore</code> column, and the correlation panels always show both. "
        "<b>HbO and HbR are two parallel passes, never mixed and never averaged.</b> Read "
        "them as a consistency check rather than two results: HbR is less contaminated by "
        "scalp and systemic circulation, so a coupling in HbO but not in HbR is a caution "
        "flag, read as present or absent rather than compared by value.",
    "hyper_post.wtc_channel":
        "Morlet wavelet. Arrows are relative phase, right in phase and up {lead} leading. "
        "The washed-out band above the white dash (low frequencies are at the top) is the "
        "cone of influence, where the record&rsquo;s edges reach.",
    "hyper_post.selector_order":
        "The left selector is {left} and the right {right}, the order the map title reads in.",
    "hyper_post.wtc_roi":
        "The member channels&rsquo; maps averaged cell by cell, bad channels excluded; arrows "
        "and cone as above. <b>These ones are live</b>: hover for the time, frequency and "
        "coherence under the pointer, and drag to zoom. The per-channel maps above are "
        "stills.",
    "hyper_post.wtc_roi_matrix":
        "Band mean per ROI pair, the cone of influence excluded by default: one number per "
        "map the selectors above reach, and the rows of the "
        "<code>agg-roi_stat-wtc_relmat.tsv</code> table.",
    "hyper_post.wtc_roi_matrix_chroma":
        "{sides} right, on one colour scale: reading them side by side is the consistency "
        "check the two chromophores are run for.",
    "hyper_post.wtc_roi_matrix_live":
        "Live, so hover names the pairing and prints its value.",
    "hyper_post.wtc_chan_matrix":
        "Band mean per channel pair, row = {row} and column = {col}: one number per map "
        "the selectors above reach, and the rows of the <code>stat-wtc_relmat.tsv</code> "
        "table. There is no within-brain cell anywhere in it. Hover a cell to see which "
        "pairing it is.",
    "hyper_post.isc_window":
        "Pearson r between the two members' time courses, every site of one against every site "
        "of the other, over this window only, z-scored inside it. The channel panels pair "
        "their matrix with a connectogram ({arc_rule}); the ROI matrix has no null to rank "
        "pairings by and draws none.",
    "hyper_post.isc_run":
        "Pearson r between the two members&rsquo; time courses, every site of one against "
        "every site of the other, over the whole recording, or over the "
        "<code>--tstart</code>/<code>--tend</code> window when one was given. The channel "
        "panels pair their matrix with a connectogram ({arc_rule}); the ROI matrix has no "
        "null to rank pairings by and draws none.",
    "hyper_post.isc_roi_matrix":
        "The channel correlations averaged in Fisher z inside each ROI pair, so this rests "
        "on the same channels the ROI coherence above does. Row = {row} and column = "
        "{col}, diagonal = the homologous region.",
    "hyper_post.isc_roi_matrix_chroma":
        "{sides} right, on one colour scale.",
    "hyper_post.hover_pairing":
        "Hover a cell to see which pairing it is.",
    "hyper_post.numbers_window":
        "Every value the panels above were drawn from, one row per pairing over this window.",
    "hyper_post.numbers_run":
        "Every value the panels above were drawn from, one row per pairing, each condition "
        "beside the whole run, which is the comparison no single panel above can show.",
    "hyper_post.numbers":
        "Rounded here only: the <code>stat-wtc_relmat.tsv</code>, "
        "<code>agg-roi_stat-wtc_relmat.tsv</code> and <code>stat-isc_relmat.tsv</code> "
        "tables under the group&rsquo;s <code>nirs/</code> carry the same numbers at full "
        "precision. <b>% outside COI</b> beside a column group is the share of that "
        "window&rsquo;s band cells clear of the record&rsquo;s edges, the one thing behind "
        "a WTC mean no figure here shows. It is one number per window, set by where the "
        "window sits in the recording and by the band rather than by the channels: a "
        "coherence resting on a third of its cells is not the same measurement as one that "
        "kept all of them.",
    "hyper_post.roi_value":
        "An ROI value averages channel values rather than analysing an averaged signal: "
        "the WTC as a plain mean, the ISC in Fisher z, a correlation being signed where a "
        "coherence is not. In the homologous table the WTC averages only the same-channel "
        "pairs, while the ISC beside it averages every pairing inside the region.",
    "hyper_post.member_metrics":
        "One table per channel set. Hover a column header for which end is better; SCI is "
        "coloured against <code>--sci-threshold</code>, which should be the line the "
        "members were screened with.",
    "hyper_post.member_metrics_sets":
        "Each table is its own measurement, not a subset of the one above it: GVTD is an RMS "
        "across the channels of its set and a retention rate is a fraction of them. The "
        "coherence is computed on the long channels.",
    "hyper_post.member_metrics_condition":
        "Over this window only, read off each member&rsquo;s own record, so these are the "
        "numbers that member&rsquo;s per-condition page prints. A blank row is one the record "
        "has no per-condition value for. This reports, it does not re-decide: the channel set "
        "is the whole recording&rsquo;s, so a channel that came loose here is still in. "
        '<a href="{href}">Whole-run page</a>.',
    "hyper_post.member_metrics_missing":
        'No member&rsquo;s record holds this window (no <code>by_condition</code> section, '
        'a split window from <code>--wtc-window-s</code>, or bounds that do not match), so '
        'there are no per-condition numbers for it. The whole recording&rsquo;s are on the '
        '<a href="{href}">whole-run page</a>.',

    # ---- Cohort hyperscanning report ----
    "hyper_group.row_order":
        "Every panel keeps one row order, the dyads sorted by their shared usable time, so a "
        "dyad is read across the page rather than looked up in each panel.",
    "hyper_group.small_cohort":
        "Only {n} dyads: the cohort median measures how far a dyad sits from the middle of "
        "its cohort, and this cohort has no middle to measure from. Read the panels as the "
        "dyads' own values side by side.",
    "hyper_group.usable":
        "The intersection, not the smaller of the two members: two members can each keep all "
        "but one pair and share one fewer than either, if the pair they each lose is a "
        "different one. The middle band is an asymmetric loss, one cap or one electrode, and "
        "the right band a shared one, both members out at the same moment. Two dyads with the "
        "same usable share and different middles are two different problems.",
    "hyper_group.pair_field":
        "A column pale down the whole cohort is the cap or the optode rather than the dyad, "
        "which is the one finding no single dyad page can carry.",
    "hyper_group.condition_dials":
        "The same share cut by condition on the left, comparable across dyads in a way absolute "
        "seconds are not; on the right the {n} dyads that lost the most, a ring per condition "
        "with the dyad's overall share innermost. The field says which dyads are worth "
        "opening, the dials what happened inside them.",
    "hyper_group.no_shared_pairs":
        "No channel pair has a usable share in every dyad (a dyad without a usable-time "
        "table has none), so a column of this panel would not be the same pair down the "
        "page.",
    "hyper_group.null":
        "Raw coherence cannot share an axis across windows: the estimator's floor moves with "
        "the window length, so one value can be unremarkable in one window and out of reach in "
        "another. The rank inside that window's own surrogate null is what is comparable, and "
        "it is the channel mean's rank, the channels pooled before the comparison.",

    # ---- A dyad's window index ----
    "hyper_index.columns":
        "Coherence: the mean band WTC over the homologous pairs; a crossed run&rsquo;s "
        "other pairings are on each window&rsquo;s page. <b>outside COI</b>: the share of "
        "band cells clear of the recording&rsquo;s edges, smaller for a window near either "
        "end. <b>ISC</b>: the mean same-channel correlation, computed on the cut window "
        "and z-scored inside it. Open a window for its maps, matrices and phase arrows.",
    "hyper_index.past_null":
        "<b>past null</b> counts the channel pairings the null was drawn for (every "
        "crossed pairing when the null is crossed, the default) whose coherence beat the "
        "{pct}th percentile of their own surrogate draws, out of those measured. Read it "
        "before the coherence: a coherence on its own has no line to clear, and a short "
        "window&rsquo;s mean scatters more than a long one&rsquo;s, while a rank inside a "
        "null drawn for that window is readable on its own.",
    "hyper_index.outlier":
        "A marked cell sits at least {z} robust SDs (1.4826 &times; MAD) from the median "
        "of this dyad&rsquo;s rows on that chromophore. It says the window differs from "
        "its neighbours, not that it failed.",
    "hyper_index.whole_run":
        "The whole-run row averages across every condition under it, so for a block design "
        "it is a summary and not a result. The rows below it are windows read out of the "
        "whole-run wavelet transform (unless <code>--wtc-cond-transform</code> was given), "
        "which keeps a short condition from being inflated by edges of its own; a tree "
        "whose conditions were cropped to separate files before the analysis shows them as "
        "separate tasks instead, each with a whole-run row of its own.",
    "hyper_index.rebuilt":
        "This page is assembled from the tables under <code>nirs/</code> each time the dyad is "
        "analysed, so it lists whatever is on disk rather than what one command produced. The "
        "command below is the one that last rebuilt it.",
}
