"""Section prose for the dyad reports: raw, post, the cohort page and a dyad's window index."""

from __future__ import annotations

NOTES = {
    # ---- Both dyad reports ----
    "hyper.clocks":
        "Each recording was cropped at its first shared trigger, by the offset shown (zero "
        "under <code>--no-align</code>, which only trims to the shortest). Dyad windows are "
        "on the aligned clock and each member&rsquo;s record on its own; the offset converts "
        "between them.",

    # ---- Raw dyad report ----
    "hyper_raw.scope":
        "The raw pass: every channel is screened on the recording as acquired, and the dyad "
        "numbers are taken over the long channels. It says whether the recording is worth "
        "preprocessing; synchrony between the members is the analysis, run on the "
        "preprocessed signal.",
    "hyper_raw.scope_post_link":
        'That is on the <a href="{href}">post report</a>.',
    "hyper_raw.blocks_verdict":
        "Times are on the shared clock, after the crop. A block whose rows do not line up in "
        "the timeline above is a trigger that landed late in one member, or two clocks "
        "drifting apart.",
    "hyper_raw.alignment":
        "Cropping at the first shared trigger removes a constant offset and nothing else "
        "(<code>--no-align</code> only trims to the shortest). A trigger that landed late in "
        "one member, or two device clocks drifting apart at the same nominal rate, stays and "
        "shows as a block that does not line up below.",
    "hyper_raw.alignment_timeline":
        "One row per member: the top pair as recorded, the bottom pair as every later panel "
        "reads it. The last row is what alignment left between the members, too small to see "
        "in the bars.",
    "hyper_raw.onset_residuals":
        "Each block&rsquo;s onset on the shared clock minus {ref}&rsquo;s. A difference "
        "beyond {tol}&nbsp;s ({n} samples) is a trigger that landed late in one member, or "
        "two clocks drifting apart.",
    "hyper_raw.usable":
        "A channel is usable by the dyad only while it is coupled in <em>both</em> members "
        "at once. Usable time is the intersection, not the smaller of the two: if each "
        "member loses a different pair, they share one fewer than either keeps.",
    "hyper_raw.usable_panel":
        "Above the carpet are each member&rsquo;s long-channel mean SCI and PSP, the two "
        "metrics the mask is built from, so a dip explains the hole under it. CV is drawn "
        "but takes no part in the mask. Dotted horizontal rules are the screening lines, "
        "vertical ones block edges.",
    "hyper_raw.motion":
        "SCI and PSP measure coupling, not movement. Each trace is a member&rsquo;s GVTD "
        "over its own median, so 1.0 is that member&rsquo;s usual level, and the traces say "
        "when each moved, not who moved more. The grey floor is the pointwise minimum of the "
        "two, high only where both moved at once.",
    "hyper_raw.motion_before":
        "The amber strip marks spans where <em>every</em> member was spiking at once; a "
        "member spiking alone shows in its own trace. The grey carpets are each "
        "member&rsquo;s per-channel optical density, z-scored as in the subject report.",
    "hyper_raw.motion_roi":
        "On each carpet the long rows are grouped by ROI, coloured by the outer strip and "
        "named in the legend.",
    "hyper_raw.motion_after":
        "The same rows after motion correction. Both figures divide by the "
        "<em>uncorrected</em> median and z-score the carpets by the uncorrected mean and SD, "
        "so a drop here is movement removed, not a moved yardstick.",
    "hyper_raw.motion_together":
        "The two motion figures in numbers, each before &rarr; after motion correction. A "
        "member&rsquo;s row is its mean GVTD over its own median. <b>Both at once</b> is the "
        "mean of the two traces&rsquo; pointwise minimum, high only where both moved "
        "together; the row under it is how much the correction lowered it, and the last row "
        "counts the amber strip&rsquo;s spans. Both moving at once raises any synchrony "
        "measured on the pair, and shifting or scrambling one member does not remove it.",
    "hyper_raw.head":
        "The same screening on the montage, member by member: the first map shows each "
        "block&rsquo;s share of coupled windows, the second each window&rsquo;s SCI. A long "
        "channel is a bar of discs along its path, a short one a larger disc in a grey ring, "
        "both on one colour scale. Light grey is a channel with no value.",
    "hyper_raw.head_by_condition":
        "Colour is the share of the block&rsquo;s windows in which the channel was coupled "
        "in that member.",
    "hyper_raw.head_slider":
        "One head per member, moved through the recording by the slider. Colour is the "
        "window&rsquo;s SCI; the block the slider sits in is named above the heads.",
    "hyper_raw.comparable":
        "Everything downstream matches channels by S-D label and assumes one clock. An "
        "unmatched label becomes a blank row without warning. Different nominal sampling "
        "rates are refused by every inter-brain measure; drift between devices at the same "
        "nominal rate passes silently, and alignment cannot remove it.",
    "hyper_raw.member_metrics":
        "A column per member: long channels where the montage has short ones, every channel "
        "otherwise. Hover a metric for which end is better; windowed SCI is coloured against "
        "this run&rsquo;s threshold.",
    "hyper_raw.group_sqm":
        "The dyad&rsquo;s own numbers: how many channel pairs survived in both members, and "
        "the aligned span they share.",
    "hyper_raw.channel_summary":
        "Green is kept by every member, yellow mixed, red kept by none.",
    "hyper_raw.channel_decisions":
        "A row per channel pair, each member&rsquo;s columns side by side under its name, "
        "long and short channels in separate tables. A member&rsquo;s columns are tinted red "
        "where that member rejected the pair: BAD is too few coupled windows, BAD (manual) "
        "by hand. Click a decision to cycle it through unset, good and bad; it saves to each "
        "member&rsquo;s JSON when the page is served by <code>nirspipe-rate hyper</code>.",
    "hyper_raw.rate_sci_line":
        "SCI (10 s) cells outlined in orange are under {line}, the line given to "
        "<code>nirspipe-rate hyper</code>. A red cell is under the line that member was "
        "screened at; the outline changes no verdict.",
    "hyper_raw.per_channel":
        "Both members&rsquo; signals and PSD on the same axes, the one view a single-member "
        "report cannot give.",

    # ---- Post dyad report ----
    "hyper_post.scope_condition":
        "The coherence panels read this window out of the whole-run transform, so it has the "
        "recording&rsquo;s cone of influence rather than edges of its own "
        "(<code>--wtc-cond-transform</code> gives it a padded transform instead). The "
        "correlation panels use the cut window.",
    "hyper_post.scope_restricted":
        "Restricted to {window} by <code>--tstart</code>/<code>--tend</code>, read out of "
        "the whole recording&rsquo;s transform and averaged across every condition inside it.",
    "hyper_post.scope_whole":
        "Coherence over the whole recording, averaged across every condition. For a block "
        "design, read the per-condition pages.",
    "hyper_post.too_short":
        "This window holds only {cycles} cycles of <code>--wtc-band-fmin</code>, the slowest "
        "frequency averaged, so the low-frequency end of every number below rests on very "
        "few independent looks. Raise <code>--wtc-band-fmin</code> or read that end as "
        "indicative; nothing is masked or dropped for it.",
    "hyper_post.roi_grouping":
        "The channel-to-region mapping from <code>--roi-mapping</code>, which the per-ROI "
        "panels average over.",
    "hyper_post.chromophores":
        "The maps show {stacked}, stacked; the matrices put the chromophores side by side on "
        "one colour scale; the tables carry {listed} in a <code>chromophore</code> column. "
        "<b>HbO and HbR are parallel passes, never "
        "mixed or averaged.</b> Read them as a consistency check: HbR is less affected by "
        "scalp and systemic circulation, so coupling in HbO but not in HbR is a caution "
        "flag, read as present or absent rather than by value.",
    "hyper_post.chromophore_one":
        "Only {chroma} was run (<code>--wtc-chroma {flag}</code>), for the coherence and the "
        "correlation alike: every map, matrix, panel and table carries it alone. "
        "<code>--wtc-chroma both</code> adds the other chromophore as a consistency check.",
    "hyper_post.wtc_channel":
        "Morlet wavelet. Arrows are relative phase: right is in phase, up is {lead} leading. "
        "The washed-out band above the white dash (low frequencies at the top) is the cone "
        "of influence, where the record&rsquo;s edges reach.",
    "hyper_post.selector_order":
        "The left selector is {left} and the right {right}, the order the map title reads in.",
    "hyper_post.wtc_roi":
        "The member channels&rsquo; maps averaged cell by cell, bad channels left out; cone "
        "as above. Arrows mark where the averaged coherence reaches {arrow_min}, since no "
        "null was drawn for these maps. Unlike the stills above, these are live: hover for "
        "time, frequency and coherence, and drag to zoom.",
    "hyper_post.wtc_roi_null":
        "The member channels&rsquo; maps averaged cell by cell, bad channels left out; cone "
        "as above. Arrows mark where the averaged map clears its own null: the "
        "phase-scrambled surrogates averaged the same way, at each frequency&rsquo;s 95th "
        "percentile. Unlike the stills above, these are live: hover for time, frequency and "
        "coherence, and drag to zoom.",
    "hyper_post.wtc_roi_matrix":
        "Band mean per ROI pair, the cone of influence excluded by default: one number per "
        "map the selectors above reach, as in <code>agg-roi_stat-wtc_relmat.tsv</code>.",
    "hyper_post.wtc_roi_matrix_chroma":
        "{sides} right, on one colour scale, so the two can be checked against each other.",
    "hyper_post.wtc_roi_matrix_live":
        "Hover a cell for its pairing and value.",
    "hyper_post.wtc_chan_matrix":
        "Band mean per channel pair, rows {row} and columns {col}: one number per map the "
        "selectors above reach, as in <code>stat-wtc_relmat.tsv</code>. No cell lies within "
        "one brain. Hover a cell for its pairing.",
    "hyper_post.isc_window":
        "Pearson r between the members&rsquo; time courses, every site of one against every "
        "site of the other, over this window only and z-scored inside it. The channel panels "
        "pair each matrix with a connectogram ({arc_rule}); the ROI matrix has no null to "
        "rank pairings by, so it draws none.",
    "hyper_post.isc_run":
        "Pearson r between the members&rsquo; time courses, every site of one against every "
        "site of the other, over the whole recording or the "
        "<code>--tstart</code>/<code>--tend</code> window. The channel panels pair each "
        "matrix with a connectogram ({arc_rule}); the ROI matrix has no null to rank "
        "pairings by, so it draws none.",
    "hyper_post.isc_roi_matrix":
        "Channel correlations averaged in Fisher z inside each ROI pair, on the same "
        "channels as the ROI coherence above. Rows {row}, columns {col}; the diagonal is the "
        "homologous region.",
    "hyper_post.isc_roi_matrix_chroma":
        "{sides} right, on one colour scale.",
    "hyper_post.hover_pairing":
        "Hover a cell for its pairing.",
    "hyper_post.numbers_window":
        "Every value the panels above were drawn from, one row per pairing, over this window.",
    "hyper_post.numbers_run":
        "Every value the panels above were drawn from, one row per pairing, each condition "
        "beside the whole run, a comparison no panel shows.",
    "hyper_post.numbers":
        "Rounded here; <code>stat-wtc_relmat.tsv</code>, "
        "<code>agg-roi_stat-wtc_relmat.tsv</code> and <code>stat-isc_relmat.tsv</code> under "
        "the group&rsquo;s <code>nirs/</code> hold full precision. <b>% outside COI</b> is "
        "the share of the window&rsquo;s band cells clear of the record&rsquo;s edges. It is "
        "one number per window, set by where the window sits and by the band, not by the "
        "channels; a coherence resting on a third of its cells is not the same measurement "
        "as one that kept them all.",
    "hyper_post.roi_value":
        "An ROI value averages channel values, not an averaged signal: the WTC as a plain "
        "mean, the ISC in Fisher z, since a correlation is signed. In the homologous table "
        "both average only the same-channel pairs.",
    "hyper_post.member_metrics":
        "A block per channel set, a column per member. Hover a metric for which end is "
        "better. Windowed SCI is coloured against each member&rsquo;s own "
        "<code>--sci-threshold</code> from its screening sidecar, and left uncoloured where "
        "none was recorded.",
    "hyper_post.member_metrics_sets":
        "Each block is its own measurement, not a subset of the one above: GVTD is an RMS "
        "over its set&rsquo;s channels and a retention rate a share of them. The coherence "
        "uses the long channels.",
    "hyper_post.member_metrics_condition":
        'This window only, from each member&rsquo;s own record, as that member&rsquo;s '
        'condition page prints it; a blank column has no per-condition value. Nothing is '
        'decided again: the channel set is the whole recording&rsquo;s, so a channel that '
        'came loose here is still in. <a href="{href}">Whole-run page</a>.',
    "hyper_post.condition_grid":
        "Each member&rsquo;s channel pairs over this window, from its own record. <b>Run</b> "
        "is the member&rsquo;s whole-run screening, the set every coherence here used, so "
        "<i>rejected</i> is always the run&rsquo;s verdict. <b>In condition</b> says whether "
        "a pair passes on this window alone, against that member&rsquo;s own line; a pair "
        "rejected by hand fails everywhere. It rejects nothing. Hover a cell for its "
        "coupled-window share.",
    "hyper_post.condition_summary":
        "Every condition page&rsquo;s channel grid in one figure, a column per channel pair, "
        "so a pair failing in one condition shows as a lone red cell. <b>Run</b> is each "
        "member&rsquo;s whole-run screening, the set every coherence used. Each condition "
        "row says whether a pair passes on that condition alone, against that member&rsquo;s "
        "own line; a pair rejected by hand fails everywhere. It rejects nothing.",
    "hyper_post.member_metrics_missing":
        'No member&rsquo;s record holds this window (no <code>by_condition</code> section, a '
        'split window from <code>--wtc-window-s</code>, or bounds that do not match), so it '
        'has no per-condition numbers. The whole recording&rsquo;s are on the <a '
        'href="{href}">whole-run page</a>.',

    # ---- Cohort hyperscanning report ----
    "hyper_group.row_order":
        "Every panel keeps one row order, the dyads sorted by shared usable time, so a dyad "
        "reads across the page.",
    "hyper_group.small_cohort":
        "Only {n} dyads. The cohort median measures distance from the cohort's middle, which "
        "so few dyads do not have; read the panels as the dyads' own values side by side.",
    "hyper_group.usable_read":
        "One bar per dyad. A cell is one channel pair in one window, usable when it was "
        "coupled in both members at the same moment.",
    "hyper_group.pair_field_read":
        "One dot per cell, dark usable and pale lost, with the pair columns ordered worst "
        "first.",
    "hyper_group.table":
        "Click a column header to sort. The first column stays in view, and a dyad's name "
        "opens its own report.",
    "hyper_group.usable":
        "The intersection, not the smaller of the two members: if each loses a different "
        "pair, they share one fewer than either keeps. The middle band is a one-sided loss "
        "(one cap or one optode), the right band a shared one, both members out at once. Two "
        "dyads with the same usable share and different middles have different problems.",
    "hyper_group.pair_field":
        "A column pale down the whole cohort points at the cap or the optode rather than the "
        "dyad, which no single dyad page can show.",
    "hyper_group.condition_dials":
        "Left, the same share by condition, which compares across dyads where seconds do "
        "not. Right, the {n} dyads that lost the most, a ring per condition with the overall "
        "share innermost. The field says which dyads to open, the dials what happened in "
        "them.",
    "hyper_group.no_shared_pairs":
        "No channel pair has a usable share in every dyad (a dyad without a usable-time "
        "table has none), so a column of this panel would not be the same pair down the "
        "page.",

    # ---- The cohort test page (nirspipe-hyper-groupnull) ----
    "groupnull.section": "Whole brain over {pairings} pairings; the marks read {p_col}.",
    "groupnull.page":
        "Each section is one statistic, one null and one chromophore, from the tables "
        "<code>nirspipe-hyper-groupnull</code> wrote beside this page. Marks read the "
        "<b>paired</b> test: each occasion against the mean of its own null draws, so its "
        "spread is between occasions, the unit the left panels draw. The <b>resample</b> "
        "test ranks the cohort mean in a null redrawn from each occasion&rsquo;s draws, so "
        "its spread is within occasions. Both are in each section&rsquo;s table.",
    "groupnull.occasions":
        "A row per condition over the whole brain. Left, each occasion&rsquo;s real value "
        "against the mean of its own null draws, hollow where it is not above; right, the "
        "same pairs side by side with both means and their 95% intervals. The caption is the "
        "paired test, absent with fewer than three occasions.",
    "groupnull.regions":
        "Each region&rsquo;s lift, the real mean minus the null mean, on a scale centred on "
        "zero. A dot marks the ones whose paired test is below {alpha} ({p_col}).",

    # ---- A dyad's window index ----
    "hyper_index.columns":
        "<b>WTC</b>: mean band coherence over the same-channel pairs, and under "
        "<b>crossed</b> over every pairing (a dash if the run was not crossed). <b>outside "
        "COI</b>: the share of band cells clear of the recording&rsquo;s edges, smaller near "
        "either end. <b>ISC</b>: mean correlation in Fisher z, same-channel and crossed "
        "alike, on the cut window z-scored inside it. Open a window for its maps, matrices "
        "and phase arrows.",
    "hyper_index.past_null":
        "<b>past null</b> counts the pairs (same-channel, or under <b>crossed</b> every "
        "pairing with a null) whose coherence beat the {pct}th percentile of their own "
        "surrogate draws, out of those measured. Read it before the coherence: a coherence "
        "has no line to clear and a short window&rsquo;s mean scatters more, while a rank "
        "inside that window&rsquo;s own null reads on its own.",
    "hyper_index.outlier":
        "A marked cell sits at least {z} robust SDs (1.4826 &times; MAD) from this "
        "dyad&rsquo;s median on that chromophore. It means the window differs from the "
        "others, not that it failed.",
    "hyper_index.whole_run":
        "The whole-run row averages every condition, so for a block design it is a summary, "
        "not a result. The rows below are windows read out of the whole-run transform "
        "(unless <code>--wtc-cond-transform</code> was given), so a short condition is not "
        "inflated by edges of its own. Conditions cropped to separate files before the "
        "analysis show as separate tasks, each with its own whole-run row.",
    "hyper_index.rebuilt":
        "Rebuilt from the tables under <code>nirs/</code> each time the dyad is analysed, so "
        "it lists what is on disk, not what one command produced. The command below rebuilt "
        "it last.",
}
