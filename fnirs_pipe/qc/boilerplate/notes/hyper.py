"""Section prose for the dyad reports: raw, post, the cohort page and a dyad's window index."""

from __future__ import annotations

NOTES = {
    # ---- Raw dyad report ----
    "hyper_raw.scope":
        "This page is the raw pass: every channel is measured, nothing is split by separation "
        "and nothing has been corrected. It is what says whether the recording is worth "
        "preprocessing. The synchrony below it is a screening flag read against its own null, "
        "not the analysis.",
    "hyper_raw.scope_post_link":
        'That is on the <a href="{href}">post report</a>.',
    "hyper_raw.clocks":
        "Crop offset from the first shared trigger; all recordings are then trimmed to the "
        "shortest. A dyad window is on the aligned clock and a member&rsquo;s own record is on "
        "its own, and these are what convert between them.",
    "hyper_raw.blocks_verdict":
        "Times are on the shared clock, after the crop. A block that does not line up between "
        "the two rows of the figure above is a trigger that landed late in one member, or two "
        "clocks drifting apart.",
    "hyper_raw.alignment":
        "Alignment crops each recording at the first shared trigger, which removes a constant "
        "offset and nothing else. What it cannot remove is a trigger that landed late in one "
        "member, or a sampling-rate difference that pulls the two apart across the run. Both "
        "show up as a block that does not line up below.",
    "hyper_raw.alignment_timeline":
        "One row per member. The top pair is what was recorded; the bottom pair is what every "
        "later panel reads.",
    "hyper_raw.usable":
        "A channel is usable by the dyad only while it is coupled in <em>both</em> members at "
        "the same moment, so the dyad&rsquo;s usable time is the intersection of the two and "
        "not the smaller of them: two members can each keep all but one pair and share one "
        "fewer than either, if the pair they each lose is a different one.",
    "hyper_raw.usable_panel":
        "The series over the carpet are each member&rsquo;s long-channel means for the metrics "
        "the mask is made of, off the same matrices, so a dip in a line is the reason for the "
        "hole under it. Dotted rules are the screening lines.",
    "hyper_raw.motion":
        "SCI and PSP are blind to movement by construction, so the carpet above can say a pair "
        "decoupled but never that the member moved. Each trace is that member&rsquo;s GVTD "
        "divided by its own median, so 1.0 is that member&rsquo;s usual level. GVTD is in each "
        "recording&rsquo;s own units, so the traces say when each member moved, not who moved "
        "more. The grey floor under the two traces is their pointwise minimum, which is high "
        "only where both are high: simultaneous movement.",
    "hyper_raw.motion_before":
        "The amber strip marks the spans where <em>every</em> member was spiking at once; a "
        "member spiking alone costs that member&rsquo;s channels, which the usable-time carpet "
        "above already shows. The grey carpets are each member&rsquo;s per-channel optical "
        "density, z-scored the way the subject report draws it.",
    "hyper_raw.motion_after":
        "The same rows on the same axes, off each member&rsquo;s motion-corrected derivative. "
        "Both figures divide by the <em>uncorrected</em> median and both carpets are z-scored "
        "by the <em>uncorrected</em> mean and SD, so a fall here is movement removed rather "
        "than the yardstick following the data.",
    "hyper_raw.motion_together":
        "A member moving alone costs that member&rsquo;s channels, which the panels above "
        "already show. <b>Both moving at once is the dyad&rsquo;s problem</b>: it raises any "
        "synchrony measure taken on the pair, and a shifted or scrambled copy of one member "
        "does not remove it. This is what the screening synchrony below has to be read "
        "against.",
    "hyper_raw.head":
        "The same screening verdict as the carpet above, on the montage instead of on a "
        "channel axis. A long channel is a bar of discs along its source-detector path, a "
        "short one an outlined square; both share one colour scale, which is what makes the "
        "short channels readable as a contamination check rather than a second map.",
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
        "readable alone. This is a screening flag, not the analysis: the wavelet coherence is "
        "on the post report{post_link}.",
    "hyper_raw.screening_post_link":
        ', <a href="{href}">here</a>',
    "hyper_raw.screening_strip":
        "One pale dot per channel, the diamond is the window&rsquo;s own rank with the channels "
        "pooled; read the diamond. A per-channel rank over a hundred draws moves with the draw; "
        "the window&rsquo;s does not.",
    "hyper_raw.screening_flag":
        "<b>A flag, not a result.</b> Shared movement, shared task structure and shared "
        "physiology all raise this number and the surrogate removes none of them: it removes "
        "only that the two members were together <em>at that moment</em>.",
    "hyper_raw.comparable":
        "Everything downstream matches channels by S-D label and assumes one clock. A mismatch "
        "here is silent in every later panel: an unmatched label becomes a blank row, and a "
        "sampling-rate difference becomes drift that alignment cannot remove.",
    "hyper_raw.member_metrics":
        "&uarr; / &darr; marks the better direction; SCI is coloured against this run&rsquo;s "
        "own threshold and HbO&ndash;HbR by sign. One unheaded table: the raw pass measures "
        "every channel and does not split by separation.",
    "hyper_raw.group_sqm":
        "The dyad&rsquo;s own numbers rather than either member&rsquo;s: how many channel pairs "
        "survived in both, and where the screening synchrony sits against its own null.",
    "hyper_raw.channel_summary":
        "Green = kept by every member, yellow = mixed, red = kept by none.",
    "hyper_raw.channel_decisions":
        "One row per source-detector pair, both members side by side, with the columns the "
        "subject report prints and each member&rsquo;s decision at the end of its own block. "
        "Rows in red were rejected by screening and Status names the criterion. Click a chip "
        "to cycle: &mdash; &rarr; good &rarr; bad. Saves to each member&rsquo;s JSON, and is "
        "only active when the page is served by <code>fnirs-rate hyper</code>.",
    "hyper_raw.per_channel":
        "The one view an individual report cannot give: two members on the same axes.",

    # ---- Post dyad report ----
    "hyper_post.scope_condition":
        "Every panel below is this window read out of the whole-run transform, so it carries "
        "the recording&rsquo;s cone of influence and not two edges of its own.",
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
    "hyper_post.clocks":
        "Crop offset from the first shared trigger; all recordings are then trimmed to the "
        "shortest. A dyad window is on the aligned clock and a member&rsquo;s own record is on "
        "its own, and this table is what converts between them.",
    "hyper_post.roi_grouping":
        "The channel-to-region mapping from <code>--roi-mapping</code>, which is what the "
        "per-ROI panels below are averaged over.",
    "hyper_post.chromophores":
        "Every panel below shows {stacked}, stacked and labelled, and "
        "<code>hyper-wtc.tsv</code> carries {listed} in a <code>chromophore</code> column. "
        "<b>HbO and HbR are two parallel passes, never mixed and never averaged.</b> Read them "
        "as a consistency check, not two results: HbR is the less contaminated by scalp and "
        "systemic circulation, so a coupling that shows in HbO and not in HbR is a caution "
        "flag. It is not a quantitative test: coherence is unsigned and bounded, so there is "
        "no expected relationship between an HbO value and an HbR one. Read it as present or "
        "absent. The maps stack; the two matrix panels put the chromophores side by side on "
        "one colour scale, which is that check in one figure, and the inter-brain correlation "
        "panels below do the same.",
    "hyper_post.wtc_channel":
        "Morlet wavelet. Arrows are relative phase, right in phase and up {lead} leading; the "
        "washed-out band below the white dash sits outside the cone of influence.",
    "hyper_post.selector_order":
        "The left selector is {left} and the right {right}, the order the map title reads in.",
    "hyper_post.wtc_roi":
        "The member channels&rsquo; maps averaged cell by cell, bad channels excluded; arrows "
        "and cone as above. <b>These ones are live</b>: hover for the time, frequency and "
        "coherence under the pointer, and drag to zoom. The per-channel maps above are "
        "stills.",
    "hyper_post.wtc_roi_matrix":
        "Band mean per ROI pair, inside the COI: one number per map the selectors above reach, "
        "and the rows of <code>hyper-wtc-roichan.tsv</code>.",
    "hyper_post.wtc_roi_matrix_chroma":
        "{sides} right, on one colour scale: reading them side by side is the consistency "
        "check the two chromophores are run for.",
    "hyper_post.wtc_roi_matrix_live":
        "Live, so hover names the pairing and prints its value.",
    "hyper_post.wtc_chan_matrix":
        "Band mean per channel pair, row = {row} and column = {col}: one number per map the "
        "selectors above reach, and the rows of <code>hyper-wtc.tsv</code>. There is no "
        "within-brain cell anywhere in it. Hover a cell to see which pairing it is.",
    "hyper_post.isc":
        "Pearson r between the two members' time courses, every site of one against every site "
        "of the other, {scope}. The channel panels pair their matrix with a connectogram "
        "({arc_rule}); the ROI matrix has no null to rank pairings by and draws none.",
    "hyper_post.isc_scope_window":
        "over this window only, z-scored inside it",
    "hyper_post.isc_scope_run":
        "over the whole recording",
    "hyper_post.isc_roi_matrix":
        "The channel values of <code>hyper-isc-*.tsv</code> averaged in Fisher z inside each "
        "ROI pair, so this rests on the same channels the ROI coherence above does. Row = "
        "{row} and column = {col}, diagonal = the homologous region.",
    "hyper_post.isc_roi_matrix_chroma":
        "{sides} right, on one colour scale.",
    "hyper_post.hover_pairing":
        "Hover a cell to see which pairing it is.",
    "hyper_post.numbers":
        "Every value the panels above were drawn from, one row per pairing{scope}. Not rounded "
        "anywhere but here: <code>hyper-wtc.tsv</code>, <code>hyper-wtc-roichan.tsv</code> and "
        "<code>hyper-isc-*.tsv</code> under the group&rsquo;s <code>nirs/</code> carry the "
        "same numbers at full precision. <b>% in COI</b> beside a column group is the share of "
        "that window&rsquo;s band cells that survived the cone of influence, which is the one "
        "thing behind a WTC mean that no figure here shows. It is one number per window rather "
        "than per pairing, the cone depending on the window length and the band and not on the "
        "channels: a coherence resting on a third of its cells is not the same measurement as "
        "one that kept all of them.",
    "hyper_post.numbers_scope_window":
        " over this window",
    "hyper_post.numbers_scope_run":
        ", each condition beside the whole run, which is the comparison no single panel above "
        "can show",
    "hyper_post.roi_value":
        "An ROI value is the mean of the channel pairs inside the two regions, not a second "
        "analysis of an averaged signal. The ROI WTC is the plain mean of those coherences and "
        "the ROI ISC the Fisher z mean of those correlations, a correlation being signed where "
        "a coherence is not.",
    "hyper_post.member_metrics":
        "One table per channel set. &uarr; / &darr; marks the better direction; SCI is "
        "coloured against this run&rsquo;s own threshold and HbO&ndash;HbR by sign.",
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
        "Neither member&rsquo;s record carries a <code>by_condition</code> section, so there "
        "are no per-condition numbers for this window. The whole recording&rsquo;s are on the "
        '<a href="{href}">whole-run page</a>; re-running <code>fnirs-pipe</code> on these '
        "subjects writes the section.",

    # ---- Cohort hyperscanning report ----
    "hyper_group.row_order":
        "Every panel keeps one row order, the dyads sorted by their shared usable time, so a "
        "dyad is read across the page rather than looked up in each panel.",
    "hyper_group.small_cohort":
        "Only {n} dyads: the cohort median and the outlier rule each measure how far a dyad "
        "sits from the middle of its cohort, and this cohort has no middle to measure from. "
        "Read the panels as the dyads' own values side by side.",
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
        "seconds are not; on the right the six dyads that lost the most, a ring per condition "
        "with the dyad's overall share innermost. The field says which dyads are worth "
        "opening, the dials what happened inside them.",
    "hyper_group.no_shared_pairs":
        "The dyads share no channel pair, so a column of this panel would not be the same pair "
        "down the page.",
    "hyper_group.null":
        "Raw coherence cannot share an axis across windows: the estimator's floor moves with "
        "the window length, so one value can be unremarkable in one window and out of reach in "
        "another. The rank inside that window's own surrogate null is what is comparable, and "
        "it is the channel mean's rank, the channels pooled before the comparison.",

    # ---- A dyad's window index ----
    "hyper_index.columns":
        "Coherence is the mean over the homologous channel pairs, the band mean each window's "
        "own table holds; a crossed run's off-diagonal pairings are in the tables and on each "
        "page, not averaged into this column. <b>inside COI</b> is the share of band cells "
        "that fell inside the cone of influence, so a window near either end of the recording "
        "keeps fewer. <b>ISC</b> is the mean same-channel inter-brain correlation. Unlike the "
        "coherence it is cut from the recording rather than read out of a whole-record "
        "transform; a correlation has no frequency axis, so a cut window carries no edge of "
        "its own. Each window is z-scored inside itself, the correlation over a stretch being "
        "against that stretch's mean. Open a window for its coherence maps, its matrices and "
        "the phase arrows.",
    "hyper_index.past_null":
        "<b>past null</b> counts the channel pairs whose coherence beat the {pct}th percentile "
        "of their own surrogate draws, out of the pairs that were measured. Read it before the "
        "coherence: the coherence has a floor that moves with the window length, so two "
        "windows' values do not compare and neither is readable alone, while a rank inside a "
        "null drawn for that window is both.",
    "hyper_index.outlier":
        "A marked cell sits more than {z} median absolute deviations from this dyad's other "
        "windows on that chromophore. It says the window differs from its neighbours, not that "
        "it failed.",
    "hyper_index.whole_run":
        "The whole-run row averages across every condition under it, so for a block design it "
        "is a summary and not a result. Rows marked as windows were read out of the whole-run "
        "wavelet transform, which is what keeps a short condition from being inflated by edges "
        "of its own; a tree whose conditions were cropped to separate files before the "
        "analysis shows them as separate tasks instead, each with a whole-run row of its own.",
    "hyper_index.rebuilt":
        "This page is assembled from the tables under <code>nirs/</code> each time the dyad is "
        "analysed, so it lists whatever is on disk rather than what one command produced. The "
        "command below is the one that last rebuilt it.",
}
