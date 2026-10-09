"""The dyad raw page (nirspipe-qc hyper-raw): each figure against its table and against the planted truth."""

import re

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
    assert {t["y"][0] for t in fig["data"] if t["type"] == "bar"} == set(range(len(members)))


def _residual_table(page: str) -> list[tuple[str, str, float, str]]:
    body = re.search(r'id="onset-residual-table">.*?<tbody>(.*?)</tbody>', page, re.S).group(1)
    return [(c, m, float(d), w) for c, m, d, w in
            re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td></tr>", body)]


@pytest.mark.parametrize("group", GROUPS)
def test_the_onset_residuals_are_the_table_s_and_zero_for_planted_shared_blocks(groups, group):
    fig = one_figure(groups.figure(group, "desc-rawalignment_nirs.html"))
    rows = _residual_table(_page(groups, group))
    first, *others = [m.sid for m in groups.truth.group(group)]
    drawn = {(x, t["name"]): y for t in fig["data"] if t["yaxis"] == "y3"
             for x, y in zip(t["x"], t["y"])}
    # value: the panel draws the page's table, every other member against the first
    assert drawn == {(c, m): pytest.approx(d, abs=5e-4) for c, m, d, _w in rows}
    assert {m for _c, m, _d, _w in rows} == set(others)
    # truth: every block was planted at one moment on the shared clock, so at most the crop's
    # rounding to a sample is left
    assert {"ca", "cb"} <= {c for c, *_ in rows}
    assert all(abs(d) <= 0.051 and w == "yes" for _c, _m, d, w in rows), rows


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


@pytest.mark.parametrize("group", GROUPS)
def test_the_channel_summary_is_drawn(groups, group):
    assert groups.figure(group, "desc-rawchsummary_nirs.html").exists()


@pytest.mark.parametrize("group", GROUPS)
def test_the_dyad_channel_table_carries_each_member_s_windowed_sci(groups, group):
    table = groups.table(group, "desc-channel_qc.tsv")
    for m in groups.truth.group(group):
        own = groups.member_record(m.sid)["per_channel"]["raw"]["sci_win_per_channel"]
        rows = table[table.subject_id == m.sid].set_index("channel").sci_win
        for ch, value in own.items():
            assert rows[ch] == pytest.approx(value, abs=1e-6), (m.sid, ch)


@pytest.mark.parametrize("group", GROUPS)
def test_the_group_record_counts_every_pair(groups, group):
    record = groups.record(group)
    n = len(groups.truth.long_pairs) + len(groups.truth.short_pairs)
    assert record["n_total"] == n
    rejected = {m.rejected for m in groups.truth.group(group)}
    assert record["n_mixed"] == len(rejected)


# ---- the panels that read the shared screening grid ----

@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("desc", ["rawusable", "rawheadcond", "rawheadslider"])
def test_the_panels_on_the_shared_screening_grid_are_drawn(groups, group, desc):
    assert groups.figure(group, f"desc-{desc}_nirs.html").exists()


def _carpet(groups, group):
    """{pair: (window centres, status)} off the usable-time carpet; 2 both, 1 one, 0 neither."""
    fig = one_figure(groups.figure(group, "desc-rawusable_nirs.html"))
    heat = next(t for t in fig["data"] if t["type"] == "heatmap")
    x, z = np.asarray(heat["x"], float), np.asarray(heat["z"], float)
    return {label.split()[0]: (x, z[i]) for i, label in enumerate(heat["y"])}


@pytest.mark.parametrize("group", GROUPS)
def test_the_usable_carpet_sits_on_the_shared_clock(groups, group):
    for x, _ in _carpet(groups, group).values():
        # windows from 0 on the shared clock, the last one cut short by the span's end
        steps = np.diff(x)
        assert x[0] == pytest.approx(steps[0] / 2) and np.allclose(steps[:-1], steps[0]), x[:3]
        assert x[-1] <= groups.truth.block("cb")[1] + 40


@pytest.mark.parametrize("group", GROUPS)
def test_the_usable_carpet_loses_each_pair_to_the_members_that_fail_it(groups, group):
    from tests._dyad_fingerprint import CONDITION_FAILING
    members = groups.truth.group(group)
    rejected = {m.rejected for m in members}
    spans: dict = {}
    for m in members:
        name, onset, span = CONDITION_FAILING[m.subject]
        spans.setdefault(name, []).append((onset, onset + span))
    for pair, (x, status) in _carpet(groups, group).items():
        half = np.diff(x)[0] / 2
        inside = np.zeros(len(x), bool)
        touched = np.zeros(len(x), bool)
        for t0, t1 in spans.get(pair, []):
            inside |= (x - half >= t0) & (x + half <= t1)
            touched |= (x + half > t0) & (x - half < t1)
        # a member's planted span costs the pair every window inside it
        assert inside.any() == (pair in spans), pair
        assert np.all(status[inside] < 2), pair
        rest = status[~touched]
        if pair in rejected:
            assert np.mean(rest == 1) > 0.85, (pair, np.mean(rest == 1))
        else:
            assert np.mean(rest == 2) > 0.85, (pair, np.mean(rest == 2))


@pytest.mark.parametrize("group", GROUPS)
def test_the_usable_table_is_the_carpet_cut_by_condition(groups, group):
    table = groups.table(group, "desc-usable_qc.tsv")
    carpet = _carpet(groups, group)
    blocks = js_var(_page(groups, group), "_BLOCKS")
    for r in table.itertuples():
        x, status = carpet[r.pair]
        t0, t1 = blocks[r.condition]
        inside = status[(x >= t0) & (x <= t1)]
        assert len(inside) == r.n_windows, (r.pair, r.condition)
        # the table keeps four decimals
        assert np.mean(inside == 2) == pytest.approx(r.usable_frac, abs=5e-5), (r.pair, r.condition)


@pytest.mark.parametrize("group", GROUPS)
def test_each_head_by_block_colours_its_own_member_s_rejection(groups, group):
    fig = one_figure(groups.figure(group, "desc-rawheadcond_nirs.html"))
    members = groups.truth.group(group)
    n_blocks = len([a for a in fig["layout"]["annotations"] if a.get("text")])
    for trace in fig["data"]:
        text, colour = trace.get("text"), (trace.get("marker") or {}).get("color")
        if text is None or colour is None or isinstance(colour, str):
            continue
        n = int((trace.get("xaxis") or "x")[1:] or 1) - 1
        member = members[n // n_blocks]
        share = {}
        for pair, value in zip(text, np.asarray(colour, float)):
            share.setdefault(pair, []).append(value)
        for pair, values in share.items():
            if pair in groups.truth.long_pairs:
                low = np.mean(values) < 0.3
                assert low == (pair == member.rejected), (member.sid, pair, np.mean(values))


@pytest.mark.parametrize("group", GROUPS)
def test_the_channel_summary_marks_a_pair_one_member_rejected_as_mixed(groups, group):
    fig = one_figure(groups.figure(group, "desc-rawchsummary_nirs.html"))
    dots = fig["data"][0]
    legend = {t["name"]: t["marker"]["color"] for t in fig["data"][1:]}
    rejected = {m.rejected for m in groups.truth.group(group)}
    for pair, colour in zip(dots["x"], dots["marker"]["color"]):
        want = "Mixed" if pair in rejected else "All good"
        assert colour == legend[want], (pair, colour)


@pytest.mark.parametrize("group", GROUPS)
def test_the_comparability_table_counts_each_member_s_long_pairs(groups, group):
    members = js_var(_page(groups, group), "_MEMBERS")
    assert {r["subject_id"]: r["n_long"] for r in members} == \
        {m.sid: len(groups.truth.long_pairs) for m in groups.truth.group(group)}
