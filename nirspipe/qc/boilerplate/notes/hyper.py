"""Section prose for the dyad reports: raw, post, the cohort page and a dyad's window index."""

from __future__ import annotations

NOTES = {
    # ---- Both dyad reports ----
    "hyper.clocks":
        "Each recording was cropped at its first shared trigger by the offset shown (zero "
        "under <code>--no-align</code>). Dyad windows use the aligned clock; the offset "
        "converts to each member&rsquo;s own.",

    # ---- Raw dyad report ----
    "hyper_raw.scope":
        "The raw pass screens every channel on the recording as acquired, to say whether it "
        "is worth preprocessing. Synchrony is measured on the preprocessed signal.",
    "hyper_raw.scope_post_link":
        'That is on the <a href="{href}">post report</a>.',
    "hyper_raw.blocks_verdict":
        "Times are on the shared clock, after the crop. A block that does not line up above "
        "means a late trigger or drifting clocks.",
    "hyper_raw.alignment":
        "Cropping at the first shared trigger removes a constant offset only "
        "(<code>--no-align</code> just trims to the shortest). A late trigger or drifting "
        "clocks remain and show as a block that does not line up below.",
    "hyper_raw.alignment_timeline":
        "One row per member: top as recorded, bottom as every later panel reads it. The last "
        "row is the offset left between the members.",
    "hyper_raw.onset_residuals":
        "Each block&rsquo;s onset on the shared clock minus {ref}&rsquo;s. Beyond "
        "{tol}&nbsp;s ({n} samples) means a late trigger or drifting clocks.",
    "hyper_raw.usable":
        "A channel is usable by the dyad only while it is passing in <em>both</em> members "
        "at once, so the dyad can keep less time than either member. A pair a member "
        "rejected is greyed and counted as lost, whatever its windows show.",
    "hyper_raw.usable_panel":
        "Above the carpet: each member&rsquo;s long-channel mean SCI and PSP, which build the "
        "mask (PSP only with <code>--psp-threshold</code>), so a dip explains the hole "
        "beneath. CV takes no part. Dotted lines are the "
        "screening thresholds and block edges.",
    "hyper_raw.motion":
        "Each trace is a member&rsquo;s GVTD over its own median, so 1.0 is that "
        "member&rsquo;s usual level. The grey floor is the lower of the two, high only where "
        "both moved at once.",
    "hyper_raw.motion_before":
        "The amber strip marks spans where <em>every</em> member spiked at once. The grey "
        "carpets are each member&rsquo;s z-scored optical density, as in the subject report.",
    "hyper_raw.motion_roi":
        "On each carpet the long rows are grouped by ROI, coloured by the outer strip and "
        "named in the legend.",
    "hyper_raw.motion_after":
        "The GVTD rows and carpets after motion correction, scaled by the <em>uncorrected</em> "
        "median, mean and SD, so a drop is movement removed. Each GVTD row has its own y "
        "range, so compare the tick values with the figure above, not the heights.",
    "hyper_raw.head":
        "The same screening on the montage, member by member: first each block&rsquo;s share "
        "of passing windows, then window by window. Bars are long channels, ringed discs "
        "short ones; light grey has no value.",
    "hyper_raw.head_by_condition":
        "Colour is the share of the block&rsquo;s windows in which the channel passed "
        "in that member.",
    "hyper_raw.head_slider":
        "A row per member, moved through the recording by the slider: left to right, whether "
        "each window passed, then its SCI and PSP, each centred on its line where it has "
        "one. The block is named above the heads.",
    "hyper_raw.comparable":
        "Downstream steps match channels by S-D label and assume one clock: an unmatched "
        "label becomes a blank row, different sampling rates are refused, and drift at the "
        "same nominal rate passes unnoticed.",
    "hyper_raw.member_metrics":
        "A channel set over each member&rsquo;s column; hover a label for its meaning. "
        "<b>Mean GVTD</b> is the member&rsquo;s GVTD over its own median, before &rarr; "
        "after correction.",
    "hyper_raw.group_sqm":
        "The dyad&rsquo;s own numbers, a column per channel set; a pair is good when every "
        "member kept it. <b>Both at once</b> is GVTD where both members moved together, "
        "before &rarr; after correction; the spans row counts the amber strip.",
    "hyper_raw.channel_summary":
        "Green is kept by every member, yellow mixed, red kept by none.",
    "hyper_raw.channel_decisions":
        "A row per channel pair, members side by side, long and short channels in separate "
        "tables. Red marks a pair that member rejected: BAD for too few passing windows, BAD "
        "(manual) by hand. Click a decision to cycle it; it saves when the page is served by "
        "<code>nirspipe-rate hyper</code>.",
    "hyper_raw.channels_tsv":
        "{sid}: marked bad by the recording&rsquo;s BIDS channels.tsv, whatever the screening "
        "found: {pairs}.",
    "hyper_raw.rate_sci_line":
        "SCI (10 s) cells outlined in orange are under {line}, the line given to "
        "<code>nirspipe-rate hyper</code>; red cells are under the screening line. The "
        "outline changes no verdict.",
    "hyper_raw.per_channel":
        "Both members&rsquo; signals and PSD on the same axes, the one view a single-member "
        "report cannot give.",

    # ---- Post dyad report ----
    "hyper_post.scope_condition":
        "Coherence is read out of the whole-run transform, so this window has no edges of "
        "its own (<code>--wtc-cond-transform</code> pads it instead). Correlation uses the "
        "cut window.",
    "hyper_post.scope_restricted":
        "Restricted to {window} by <code>--tstart</code>/<code>--tend</code>, read out of "
        "the whole recording&rsquo;s transform and averaged across every condition inside it.",
    "hyper_post.scope_whole":
        "Coherence over the whole recording, averaged across every condition. For a block "
        "design, read the per-condition pages.",
    "hyper_post.too_short":
        "This window holds only {cycles} cycles of <code>--wtc-band-fmin</code>, so the low "
        "end of the band rests on very few looks. Read it as indicative, or raise "
        "<code>--wtc-band-fmin</code>.",
    "hyper_post.roi_grouping":
        "The channel-to-region mapping from <code>--roi-mapping</code>, which the per-ROI "
        "panels average over.",
    "hyper_post.chromophores":
        "Maps stack {stacked}; matrices and tables carry them side by side. <b>Each "
        "chromophore is a separate pass, never mixed with another.</b> Coupling in HbO but not "
        "HbR is a caution flag.",
    "hyper_post.chromophore_one":
        "Only {chroma} was run (<code>--wtc-chroma {flag}</code>), for coherence and "
        "correlation alike; HbO and HbR together (<code>--wtc-chroma hbo hbr</code>) check "
        "each other.",
    "hyper_post.wtc_channel":
        "Morlet wavelet. Arrows are relative phase: right is in phase, up is {lead} leading. "
        "Above the white dash (low frequencies) is the cone of influence, where the edges "
        "reach.",
    "hyper_post.selector_order":
        "The left selector is {left} and the right {right}, the order the map title reads in.",
    "hyper_post.wtc_roi":
        "Member channel maps averaged cell by cell, bad channels left out. Arrows mark where "
        "the averaged map reaches {arrow_min}, since no null was drawn. Hover for values, drag "
        "to zoom.",
    "hyper_post.wtc_roi_null":
        "Member channel maps averaged cell by cell, bad channels left out. Arrows mark where "
        "the averaged map clears its own null (phase-scrambled, 95th percentile). Hover for "
        "values, drag to zoom.",
    "hyper_post.wtc_roi_matrix":
        "Band mean per ROI pair, outside the cone of influence by default, as in "
        "<code>agg-roi_stat-wtc_relmat.tsv</code>.",
    "hyper_post.wtc_roi_matrix_chroma":
        "{sides} from left to right, on one colour scale, so they can be checked against "
        "each other.",
    "hyper_post.wtc_roi_matrix_live":
        "Hover a cell for its pairing and value.",
    "hyper_post.wtc_chan_matrix":
        "Band mean per channel pair, rows {row} and columns {col}, as in "
        "<code>stat-wtc_relmat.tsv</code>. Hover a cell for its pairing.",
    "hyper_post.isc_window":
        "Pearson r between the members&rsquo; time courses, every site against every site, "
        "over this window, z-scored inside it. Channel matrices come with a connectogram "
        "({arc_rule}); the ROI matrix has none.",
    "hyper_post.isc_run":
        "Pearson r between the members&rsquo; time courses, every site against every site, "
        "over the whole recording or the <code>--tstart</code>/<code>--tend</code> window. "
        "Channel matrices come with a connectogram ({arc_rule}); the ROI matrix has none.",
    "hyper_post.isc_roi_matrix":
        "Channel correlations averaged in Fisher z within each ROI pair. Rows {row}, columns "
        "{col}; the diagonal is the homologous region.",
    "hyper_post.isc_roi_matrix_chroma":
        "{sides} from left to right, on one colour scale.",
    "hyper_post.hover_pairing":
        "Hover a cell for its pairing.",
    "hyper_post.numbers_window":
        "Every value the panels above were drawn from, one row per pairing, over this window.",
    "hyper_post.numbers_run":
        "Every value the panels above were drawn from, one row per pairing, each condition "
        "beside the whole run.",
    "hyper_post.numbers_files":
        "The channel table is folded. Every pairing at full precision, every condition "
        "included:",
    "hyper_post.numbers":
        "Rounded here; full precision is in the tables under the group&rsquo;s "
        "<code>nirs/</code>. <b>% outside COI</b> is the share of the window&rsquo;s band "
        "cells clear of the record&rsquo;s edges; a coherence resting on few cells is less "
        "reliable.",
    "hyper_post.roi_value":
        "An ROI value averages channel values: WTC as a plain mean, ISC in Fisher z. The "
        "homologous table averages same-channel pairs only.",
    "hyper_post.member_metrics":
        "A channel set over each member&rsquo;s column; hover a label for its meaning. "
        "Windowed SCI is coloured against each member&rsquo;s own "
        "<code>--sci-threshold</code>.",
    "hyper_post.member_metrics_sets":
        "Each set is measured on its own, not as a subset of the next. The coherence uses "
        "the long channels.",
    "hyper_post.member_metrics_condition":
        'This window only, from each member&rsquo;s own record; a blank column has no '
        'per-condition value. The channel set stays the whole run&rsquo;s. '
        '<a href="{href}">Whole-run page</a>.',
    "hyper_post.condition_grid":
        "Each member&rsquo;s channel pairs over this window. <b>Run</b> is the whole-run "
        "screening every coherence used; <b>In condition</b> says whether a pair passes on "
        "this window alone, and rejects nothing. Hover a cell for its passing-window share.",
    "hyper_post.condition_summary":
        "Every condition&rsquo;s channel grid in one figure, a column per pair: a pair "
        "failing in one condition shows as a lone red cell. <b>Run</b> is the screening "
        "every coherence used; condition rows reject nothing.",
    "hyper_post.member_metrics_missing":
        'No member&rsquo;s record holds this window (no <code>by_condition</code> section, a '
        '<code>--wtc-window-s</code> split, or mismatched bounds), so it has no '
        'per-condition numbers. See the <a href="{href}">whole-run page</a>.',
    "hyper_post.member_metrics_refused":
        "{sid}&rsquo;s record puts this condition at {record}&nbsp;s on the aligned clock and "
        "the dyad at {dyad}&nbsp;s, over {n} samples ({tol}&nbsp;s) apart, so {sid} has no "
        "per-condition numbers here. See the alignment check under Clocks and ROIs.",

    # ---- Cohort hyperscanning report ----
    "hyper_group.row_order":
        "Every panel keeps one row order, the dyads sorted by shared usable time, so a dyad "
        "reads across the page.",
    "hyper_group.small_cohort":
        "Only {n} dyads, too few for a cohort median to mean much; read the panels as each "
        "dyad's own values side by side.",
    "hyper_group.usable_read":
        "One bar per dyad. A cell is one channel pair in one window, usable when it was "
        "passing in both members at the same moment.",
    "hyper_group.pair_field_read":
        "One dot per cell, dark usable and pale lost, with the pair columns ordered worst "
        "first.",
    "hyper_group.table":
        "Click a column header to sort. The first column stays in view, and a dyad's name "
        "opens its own report.",
    "hyper_group.usable":
        "The middle band is a loss in one member (one cap or optode), the right band a loss "
        "in both at once.",
    "hyper_group.pair_field":
        "A column pale down the whole cohort points at the cap or the optode rather than the "
        "dyad, which no single dyad page can show.",
    "hyper_group.condition_dials":
        "Left, the same share by condition, comparable across dyads. Right, the {n} dyads "
        "that lost the most, a ring per condition with the overall share innermost.",
    "hyper_group.no_shared_pairs":
        "No channel pair has a usable share in every dyad, so the columns could not line up "
        "and the panel is left out.",

    # ---- The cohort test page (nirspipe-hyper-groupnull) ----
    "groupnull.section": "Whole brain over {pairings} pairings; the marks read {p_col}.",
    "groupnull.page":
        "Each section is one statistic, one null and one chromophore, from the tables "
        "<code>nirspipe-hyper-groupnull</code> wrote beside this page. Marks read the "
        "<b>paired</b> test, each occasion against its own null mean; the <b>resample</b> "
        "test is in each section&rsquo;s table.",
    "groupnull.occasions":
        "Left, each occasion&rsquo;s real value against its own null mean, hollow where not "
        "above; right, both means with 95% intervals. The caption is the paired test, given "
        "from three occasions.",
    "groupnull.rolepairs":
        "By role pair: each group&rsquo;s pairings between those two roles, averaged within "
        "the group, against the same draws.",
    "groupnull.regions":
        "Each region&rsquo;s lift, the real mean minus the null mean, printed in its cell on a "
        "scale centred on zero; bold with a star where its paired test is below {alpha} ({p_col}).",

    # ---- A dyad's window index ----
    "hyper_index.columns":
        "<b>WTC</b>: mean band coherence over same-channel pairs, and over every pairing "
        "under <b>crossed</b>. <b>outside COI</b>: share of band cells clear of the edges. "
        "<b>ISC</b>: mean correlation in Fisher z. Open a window for its maps and matrices.",
    "hyper_index.past_null":
        "<b>past null</b> counts the pairs whose coherence beat the {pct}th percentile of "
        "their own surrogate draws. Read it before the coherence, which has no line to clear.",
    "hyper_index.outlier":
        "A marked cell sits at least {z} robust SDs from this dyad&rsquo;s median on that "
        "chromophore: the window differs from the others, it did not fail.",
    "hyper_index.whole_run":
        "The whole-run row averages every condition, so for a block design it is a summary, "
        "not a result. Condition rows are read out of the whole-run transform unless "
        "<code>--wtc-cond-transform</code> was given.",
    "hyper_index.rebuilt":
        "Rebuilt from the tables under <code>nirs/</code> each time the dyad is analysed; "
        "the command below rebuilt it last.",
}
