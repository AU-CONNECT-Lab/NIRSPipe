"""The dyad raw page (fnirs-qc hyper-raw): each figure against its table and against the planted truth."""

from itertools import combinations

import numpy as np
import pytest
from mne.preprocessing.nirs import beer_lambert_law

from tests._dyad_fingerprint import DPF, OWN_SPIKE, SHARED_SPIKE, SYNC, own_freq
from tests.figure_accuracy._dyad import js_var
from tests.figure_accuracy._payload import one_figure, plotly_figures
from tests.figure_accuracy._read import share_at, traces, xy

GROUPS = ("G01", "G02")


def _page(groups, group):
    return groups.page(group, raw=True).read_text(encoding="utf-8")


def _aligned_haemo(groups, member):
    """What the dyad page claims to draw: Beer-Lambert on the member's recording, from its trigger."""
    raw = beer_lambert_law(groups.read(member.sid, "sci").copy(), ppf=list(DPF))
    return raw.crop(tmin=member.offset)


# ---- alignment ----

@pytest.mark.parametrize("group", GROUPS)
def test_the_alignment_table_carries_each_member_s_planted_offset(groups, group):
    rows = {r["subject_id"]: r["offset_s"] for r in js_var(_page(groups, group), "_ALIGN")}
    assert rows == {m.sid: pytest.approx(m.offset, abs=0.1) for m in groups.truth.group(group)}


@pytest.mark.parametrize("group", GROUPS)
def test_the_block_table_is_the_planted_blocks_on_the_aligned_clock(groups, group):
    blocks = js_var(_page(groups, group), "_BLOCKS")
    for name in ("ca", "cb"):
        assert blocks[name] == pytest.approx(list(groups.truth.block(name)), abs=0.1)


@pytest.mark.parametrize("group", GROUPS)
def test_the_timeline_draws_each_member_s_blocks_on_its_own_clock_then_on_the_shared_one(
        groups, group):
    fig = one_figure(groups.figure(group, "desc-rawalignment_nirs.html"))
    members = groups.truth.group(group)
    for axis, own in (("y", True), ("y2", False)):
        ticks = dict(zip(fig["layout"][axis.replace("y", "yaxis")]["tickvals"],
                         fig["layout"][axis.replace("y", "yaxis")]["ticktext"]))
        for bar in (t for t in fig["data"] if t["yaxis"] == axis and t["name"] != SYNC):
            member = groups.truth.member(ticks[bar["y"][0]])
            onset, stop = groups.truth.block(bar["name"])
            assert bar["base"][0] == pytest.approx(onset + (member.offset if own else 0.0),
                                                   abs=0.1), (axis, member.sid, bar["name"])
            assert bar["x"][0] == pytest.approx(stop - onset, abs=0.1)
    assert {t["y"][0] for t in fig["data"]} == set(range(len(members)))


# ---- screening coherence ----

@pytest.mark.parametrize("group", GROUPS)
def test_the_screening_strip_draws_each_channel_s_percentile_from_the_table(groups, group):
    fig = one_figure(groups.figure(group, "desc-rawscreening_nirs.html"))
    table = groups.table(group, "cond-all_stat-coherence_relmat.tsv")
    for window, rows in table.groupby("window", sort=False):
        trace = traces(fig, window)[0]
        drawn = sorted(zip(trace["customdata"], np.asarray(trace["x"], float)))
        assert drawn == sorted(zip(rows.ch_name, rows.percentile)), window


def test_the_screening_strip_ranks_the_planted_block_coupling_first(groups):
    """K1 couples S2_D2 to S2_D2 inside ca only: its channel tops ca, and is no standout in cb."""
    fig = one_figure(groups.figure("G01", "desc-rawscreening_nirs.html"))
    table = groups.table("G01", "cond-all_stat-coherence_relmat.tsv")
    ca = table[table.window == "ca"].set_index("ch_name").coherence
    assert ca.idxmax() == "S2_D2" and ca["S2_D2"] > 2 * ca.drop("S2_D2").max()
    trace = traces(fig, "ca")[0]
    assert dict(zip(trace["customdata"], trace["x"]))["S2_D2"] == 100.0


@pytest.mark.xfail(strict=True, reason="D3: a window too short for one Welch bin in the band "
                   "has NaN coherence and is ranked as the 0th percentile")
def test_a_window_with_no_measurable_coherence_is_not_drawn_at_the_0th_percentile(groups):
    fig = one_figure(groups.figure("G01", "desc-rawscreening_nirs.html"))
    table = groups.table("G01", "cond-all_stat-coherence_relmat.tsv")
    unmeasured = table[table.coherence.isna()]
    assert len(unmeasured)                      # the 30 s trigger span: no bin in 0.01-0.10 Hz
    for window in unmeasured.window.unique():
        assert not any(np.asarray(traces(fig, window)[0]["x"], float) == 0.0), window


@pytest.mark.xfail(strict=True, reason="D4: a group of three draws three pairings' points "
                   "under channel names alone, and its window summary is the first pairing's")
def test_a_triad_s_screening_strip_names_the_pairing_of_every_point(groups):
    fig = one_figure(groups.figure("G02", "desc-rawscreening_nirs.html"))
    table = groups.table("G02", "cond-all_stat-coherence_relmat.tsv")
    sids = [m.sid for m in groups.truth.group("G02")]
    for window in table.window.unique():
        trace = traces(fig, window)[0]
        labels = [str(c) for c in trace["customdata"]]
        for a, b in combinations(sids, 2):
            assert any(a in s and b in s for s in labels), (window, a, b)


@pytest.mark.xfail(strict=True, reason="D4: the record's window percentile is the first "
                   "pairing's while its coherence is the mean over every pairing")
def test_a_triad_s_window_percentile_is_not_one_pairing_s(groups):
    table = groups.table("G02", "cond-all_stat-coherence_relmat.tsv")
    windows = groups.record("G02")["screening"]["windows"]
    first = table.groupby(["window", "sub1", "sub2"], sort=False).window_percentile.first()
    for window, by_pair in first.groupby(level=0):
        if by_pair.nunique() > 1:
            assert windows[window]["percentile"] != by_pair.iloc[0], window


# ---- motion ----

def _peak_near(x, y, t, half=3.0):
    near = (x > t - half) & (x < t + half)
    return float(np.nanmax(y[near]))


@pytest.mark.parametrize("stage", ["before", "after"])
def test_each_member_s_motion_row_carries_its_own_movement_and_not_the_other_s(groups, stage):
    fig = one_figure(groups.figure("G01", f"desc-rawmotion{stage}_nirs.html"))
    members = groups.truth.group("G01")
    for m in members:
        x, y = xy(traces(fig, m.sid)[0])              # the long-channel GVTD row comes first
        base = float(np.nanmedian(y))
        assert _peak_near(x, y, OWN_SPIKE[m.subject]) > 3 * base, m.sid
        assert _peak_near(x, y, SHARED_SPIKE) > 3 * base, m.sid
        other = next(o for o in members if o is not m)
        assert _peak_near(x, y, OWN_SPIKE[other.subject]) < 2 * base, (m.sid, other.sid)


def test_both_at_once_is_high_only_where_both_members_moved(groups):
    fig = one_figure(groups.figure("G01", "desc-rawmotionbefore_nirs.html"))
    x, y = xy(traces(fig, "both at once")[0])
    shared = _peak_near(x, y, SHARED_SPIKE)
    assert shared > 3 * float(np.nanmedian(y))
    for m in groups.truth.group("G01"):
        assert _peak_near(x, y, OWN_SPIKE[m.subject]) < shared / 3, m.sid


@pytest.mark.xfail(strict=True, reason="D5: the filter's edge at the start of the shared "
                   "clock is flagged as a moment every member spiked")
def test_both_spiking_spans_cover_only_the_moment_every_member_moved(groups):
    fig = one_figure(groups.figure("G01", "desc-rawmotionbefore_nirs.html"))
    xs = [v for v in traces(fig, "both spiking")[0]["x"] if v is not None]
    spans = list(zip(xs[0::4], xs[2::4]))
    assert spans and all(a - 5 < SHARED_SPIKE < b + 5 for a, b in spans), spans


def test_each_carpet_is_titled_with_the_member_whose_movement_it_shows(groups):
    fig = one_figure(groups.figure("G01", "desc-rawmotionbefore_nirs.html"))
    titles = [a["text"] for a in fig["layout"]["annotations"] if "carpet" in a.get("text", "")]
    carpets = [t for t in fig["data"] if t["type"] == "heatmap"]
    assert len(titles) == len(carpets) == 2
    for title, carpet in zip(titles, carpets):
        member = groups.truth.member(title.split()[0])
        other = next(o for o in groups.truth.group("G01") if o is not member)
        x = np.asarray(carpet["x"], float)
        # the planted movement is on every channel, so the median row shows it and noise does not
        col = np.median(np.abs(np.asarray(carpet["z"], float)), axis=0)
        own, others = (_peak_near(x, col, OWN_SPIKE[s.subject]) for s in (member, other))
        assert own > 2 * others, (title, own, others)


# ---- the channel detail ----

@pytest.mark.parametrize("group", GROUPS)
def test_each_member_s_detail_trace_is_its_own_haemoglobin_on_the_shared_clock(groups, group):
    for m in groups.truth.group(group):
        haemo = _aligned_haemo(groups, m)
        for pair in groups.truth.long_pairs:
            fig = plotly_figures(groups.figure(
                group, f"chan-{pair.replace('_', '')}_desc-rawdetail_nirs.html"))[0]
            for chroma, axis in (("hbo", "x"), ("hbr", "x2")):
                x, y = xy(traces(fig, m.sid, axis=axis)[0])
                truth = haemo.get_data(picks=[f"{pair} {chroma}"])[0] * 1e6
                idx = np.round(x * haemo.info["sfreq"]).astype(int)
                np.testing.assert_allclose(y, truth[idx], rtol=1e-5, atol=1e-6,
                                           err_msg=f"{m.sid} {pair} {chroma}")


@pytest.mark.parametrize("group", GROUPS)
def test_each_member_s_detail_trace_carries_that_member_s_own_frequency(groups, group):
    members = groups.truth.group(group)
    for k, pair in enumerate(groups.truth.long_pairs):
        fig = plotly_figures(groups.figure(
            group, f"chan-{pair.replace('_', '')}_desc-rawdetail_nirs.html"))[0]
        for m in members:
            if pair == m.rejected:
                continue
            x, y = xy(traces(fig, m.sid, axis="x")[0])
            scores = {o.sid: share_at(x, y, own_freq(o.subject, k)) for o in members}
            assert max(scores, key=scores.get) == m.sid, (pair, m.sid, scores)


# ---- channels ----

@pytest.mark.parametrize("group", GROUPS)
def test_the_decision_table_marks_each_member_s_own_rejection_only(groups, group):
    rows = js_var(_page(groups, group), "_DECISION_ROWS")
    for m in groups.truth.group(group):
        bad = {r["pair"] for r in rows if r["by_sub"][m.sid]["is_bad"]}
        assert bad == {m.rejected}, m.sid


@pytest.mark.parametrize("group", GROUPS)
def test_the_decision_table_s_sci_is_each_member_s_own_windowed_sci(groups, group):
    rows = js_var(_page(groups, group), "_DECISION_ROWS")
    for m in groups.truth.group(group):
        own = groups.member_record(m.sid)["per_channel"]["raw"]["sci_win_per_channel"]
        for r in rows:
            assert r["by_sub"][m.sid]["sci_win_value"] == pytest.approx(
                own[f"{r['pair']} 760"], abs=1e-6), (m.sid, r["pair"])


@pytest.mark.xfail(strict=True, reason="D1: group_quality reads sci_win_per_channel off the "
                   "scalar sections, where it never is, so the channel summary is never drawn")
@pytest.mark.parametrize("group", GROUPS)
def test_the_channel_summary_is_drawn(groups, group):
    assert groups.figure(group, "desc-rawchsummary_nirs.html").exists()


@pytest.mark.xfail(strict=True, reason="D1: the dyad channel table's sci_win is empty")
@pytest.mark.parametrize("group", GROUPS)
def test_the_dyad_channel_table_carries_each_member_s_windowed_sci(groups, group):
    table = groups.table(group, "desc-channel_qc.tsv")
    for m in groups.truth.group(group):
        own = groups.member_record(m.sid)["per_channel"]["raw"]["sci_win_per_channel"]
        rows = table[table.subject_id == m.sid].set_index("channel").sci_win
        for ch, value in own.items():
            assert rows[ch] == pytest.approx(value, abs=1e-6), (m.sid, ch)


@pytest.mark.xfail(strict=True, reason="D1: with no windowed SCI the group record counts no "
                   "channel as all good, mixed or all bad")
@pytest.mark.parametrize("group", GROUPS)
def test_the_group_record_counts_every_pair(groups, group):
    record = groups.record(group)
    n = len(groups.truth.long_pairs) + len(groups.truth.short_pairs)
    assert record["n_total"] == n
    rejected = {m.rejected for m in groups.truth.group(group)}
    assert record["n_mixed"] == len(rejected)


# ---- the panels that read the shared screening grid ----

@pytest.mark.xfail(strict=True, reason="S1: coupled_grid gives up when the members' offsets "
                   "differ, so the usable-time panel and both head figures are never drawn")
@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("desc", ["rawusable", "rawheadcond", "rawheadslider"])
def test_the_panels_on_the_shared_screening_grid_are_drawn(groups, group, desc):
    assert groups.figure(group, f"desc-{desc}_nirs.html").exists()


@pytest.mark.xfail(strict=True, reason="S1: with no shared grid the comparability table "
                   "counts no long pair for either member")
@pytest.mark.parametrize("group", GROUPS)
def test_the_comparability_table_counts_each_member_s_long_pairs(groups, group):
    members = js_var(_page(groups, group), "_MEMBERS")
    assert {r["subject_id"]: r["n_long"] for r in members} == \
        {m.sid: len(groups.truth.long_pairs) for m in groups.truth.group(group)}
