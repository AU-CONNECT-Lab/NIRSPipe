"""The dyad channel grids: each member's run verdict and each condition's own result, by pair."""

import re
from itertools import combinations

import pytest

from fnirs_pipe.qc.figures.subject.sci_psp_panel import _BAD_COLOR, _GOOD_COLOR
from fnirs_pipe.utils import pair_of
from tests._dyad_fingerprint import CONDITION_FAILING
from tests.figure_accuracy._payload import one_figure

GROUPS = ("G01", "G02")
BLOCKS = ("ca", "cb")


def _pairings(groups, group):
    return list(combinations([m.sid for m in groups.truth.group(group)], 2))


def _page(groups, group, pair, cond=None):
    return groups.page(group, pair=pair if len(groups.truth.group(group)) > 2 else None,
                       cond=cond)


def _figure(page):
    html = page.read_text(encoding="utf-8")
    found = re.search(r'id="ConditionChannels".*?<iframe[^>]*src="([^"]+)"', html, re.S)
    assert found, f"{page.name}: no channel grid"
    return one_figure(page.parent / found.group(1))


def _cells(page) -> dict:
    """``{(member, row, pair): colour}``, each read off where the cell is drawn, not its hover."""
    fig = _figure(page)
    dots, layout = fig["data"][0], fig["layout"]
    pairs = dict(zip(layout["xaxis"]["tickvals"], layout["xaxis"]["ticktext"]))
    rows = dict(zip(layout["yaxis"]["tickvals"], layout["yaxis"]["ticktext"]))
    heads = sorted((a["y"], re.sub(r"</?b>", "", a["text"])) for a in layout["annotations"]
                   if a.get("yref") == "y")
    cells = {}
    for x, y, colour in zip(dots["x"], dots["y"], dots["marker"]["color"]):
        member = [sid for at, sid in heads if at <= y][-1]
        cells[(member, rows[y], pairs[x])] = colour
    return cells


def _failing(groups, sid, cond) -> set:
    entry = groups.member_record(sid)["by_condition"][cond]
    return {pair_of(c) for c in entry["bad_channels"]}


def _run_rejected(groups, sid) -> set:
    return {pair_of(c) for c in groups.sidecar(sid, "sci")["bad_channels"]}


def _planted_failing(groups, sid, cond) -> set:
    """The truth: the run's rejection fails every condition, a planted pair its own block only."""
    out = {groups.truth.member(sid).rejected}
    planted = CONDITION_FAILING.get(sid.removeprefix("sub-"))
    if planted:
        name, onset, span = planted
        t0, t1 = groups.truth.block(cond)
        if t0 <= onset and onset + span <= t1:
            out.add(name)
    return out


def _check(cells, sid, row, pairs, failing):
    for p in pairs:
        colour = cells[(sid, row, p)]
        assert colour in (_GOOD_COLOR, _BAD_COLOR), (sid, row, p, colour)
        assert (colour == _BAD_COLOR) == (p in failing), (sid, row, p)


# ---- value checks: each cell is the record it says it reads ----

@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("cond", BLOCKS)
def test_a_condition_page_grid_is_each_member_s_run_verdict_and_condition_record(
        groups, group, cond):
    pairs = groups.truth.long_pairs + groups.truth.short_pairs
    for a, b in _pairings(groups, group):
        cells = _cells(_page(groups, group, (a, b), cond))
        assert {k[1] for k in cells} == {"Run", "In condition"}
        for sid in (a, b):
            _check(cells, sid, "Run", pairs, _run_rejected(groups, sid))
            _check(cells, sid, "In condition", pairs, _failing(groups, sid, cond))


@pytest.mark.parametrize("group", GROUPS)
def test_the_run_page_grid_carries_every_condition_s_record(groups, group):
    pairs = groups.truth.long_pairs + groups.truth.short_pairs
    for a, b in _pairings(groups, group):
        cells = _cells(_page(groups, group, (a, b)))
        assert {k[1] for k in cells} == {"Run", *BLOCKS}
        for sid in (a, b):
            _check(cells, sid, "Run", pairs, _run_rejected(groups, sid))
            for cond in BLOCKS:
                _check(cells, sid, cond, pairs, _failing(groups, sid, cond))


@pytest.mark.parametrize("group", GROUPS)
def test_the_run_page_grid_and_each_condition_page_grid_agree(groups, group):
    for a, b in _pairings(groups, group):
        run = _cells(_page(groups, group, (a, b)))
        for cond in BLOCKS:
            page = _cells(_page(groups, group, (a, b), cond))
            for (sid, row, p), colour in page.items():
                assert run[(sid, cond if row == "In condition" else row, p)] == colour, \
                    (cond, sid, row, p)


def test_the_hover_names_each_cell_s_coupled_share_and_its_member_s_own_line(groups):
    for cond in BLOCKS:
        dots = _figure(_page(groups, "G01", None, cond))["data"][0]
        for m in groups.truth.group("G01"):
            entry = groups.member_record(m.sid)["by_condition"][cond]
            frac = entry["per_channel"]["good_frac_per_channel"]
            line = groups.sidecar(m.sid, "sci")["parameters"]["min_good_frac"]
            for p in groups.truth.long_pairs:
                share = min(v for c, v in frac.items() if pair_of(c) == p)
                text = next(t for t in dots["text"]
                            if t.startswith(f"{m.sid} · {cond} · {p} ·"))
                assert f"coupled {share:.2f} against {m.sid}'s line {line:.2f}" in text, text


# ---- truth checks: the planted verdicts ----

@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("cond", BLOCKS)
def test_a_pair_failing_one_block_only_is_kept_by_the_run_and_failed_in_that_block(
        groups, group, cond):
    pairs = groups.truth.long_pairs
    for a, b in _pairings(groups, group):
        cells = _cells(_page(groups, group, (a, b), cond))
        for sid in (a, b):
            _check(cells, sid, "Run", pairs, {groups.truth.member(sid).rejected})
            _check(cells, sid, "In condition", pairs, _planted_failing(groups, sid, cond))


def test_the_fingerprint_plants_a_condition_only_failure_on_every_member(groups):
    grouped = [m for m in groups.truth.members if m.group]
    assert set(CONDITION_FAILING) == {m.subject for m in grouped}
    for m in grouped:
        name, _, _ = CONDITION_FAILING[m.subject]
        coupled = {c.a_pair if c.a == m.subject else c.b_pair
                   for c in groups.truth.couplings if m.subject in (c.a, c.b)}
        assert name != m.rejected and name not in coupled, m.subject


@pytest.mark.parametrize("group", GROUPS)
def test_the_grid_columns_are_the_long_pairs_then_the_short(groups, group):
    for a, b in _pairings(groups, group):
        layout = _figure(_page(groups, group, (a, b), "ca"))["layout"]
        assert layout["xaxis"]["ticktext"] == (groups.truth.long_pairs
                                               + groups.truth.short_pairs)
        names = {a["text"] for a in layout["annotations"] if a.get("yref") == "paper"}
        assert {"long", "short"} <= names


def test_a_pairing_page_grid_lists_only_its_own_two_members(groups):
    for a, b in _pairings(groups, "G02"):
        for cond in (None, *BLOCKS):
            assert {k[0] for k in _cells(_page(groups, "G02", (a, b), cond))} == {a, b}
