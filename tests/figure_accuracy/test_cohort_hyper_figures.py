"""The cohort page over groups (fnirs-qc cohort-hyper): each figure against each group's record."""

import numpy as np
import pytest

from tests.figure_accuracy._dyad import html_tables
from tests.figure_accuracy._payload import one_figure

GROUPS = ("G01", "G02")


def _label(group):
    return f"group-{group}_task-main"


def _table(groups):
    rows = next(rows for _, rows in html_tables(groups.hyper / "desc-groups_report.html")
                if rows and rows[0][0].rstrip("⇅") == "Dyad")
    head = [h.rstrip("⇅") for h in rows[0]]
    return {r[0]: dict(zip(head, r)) for r in rows[1:]}


def test_every_null_strip_marker_is_its_group_s_window_percentile(groups):
    fig = one_figure(groups.hyper / "figures" / "desc-groupsnullstrip_nirs.html")
    ticks = dict(zip(fig["layout"]["yaxis"]["tickvals"], fig["layout"]["yaxis"]["ticktext"]))
    markers = [t for t in fig["data"] if t.get("mode") == "markers"]
    assert {t["name"] for t in markers} >= {"whole run", "ca", "cb"}
    for trace in markers:
        for x, y, label in zip(trace["x"], trace["y"], trace["customdata"]):
            assert ticks[y] == label
            group = label.split("_")[0].removeprefix("group-")
            want = groups.record(group)["screening"]["windows"][trace["name"]]["percentile"]
            assert x == pytest.approx(want), (label, trace["name"])


def test_the_table_carries_each_group_s_alignment(groups):
    table = _table(groups)
    for group in GROUPS:
        row = table[_label(group)]
        offsets = [m.offset for m in groups.truth.group(group)]
        assert float(row["Max offset"].split()[0]) == pytest.approx(max(offsets), abs=0.01)
        assert row["Offsets equal"] == "no"
        windows = groups.record(group)["screening"]["windows"]
        assert int(row["Above null"]) == sum(w["percentile"] >= 95 for w in windows.values())


@pytest.mark.xfail(strict=True, reason="D3: the window too short for a Welch bin counts as the "
                   "0th percentile in the median the table prints")
def test_the_median_percentile_leaves_out_a_window_with_no_measurable_coherence(groups):
    table = _table(groups)
    for group in GROUPS:
        windows = groups.record(group)["screening"]["windows"].values()
        measured = [w["percentile"] for w in windows if np.isfinite(w["coherence"])]
        assert float(table[_label(group)]["Median pct"]) == pytest.approx(np.median(measured))


@pytest.mark.xfail(strict=True, reason="S1: no shared screening grid, so no group has a "
                   "usable-time table and the panels that split it are never drawn")
@pytest.mark.parametrize("desc", ["groupsusablebars", "groupspairfield", "groupsconditiondials"])
def test_the_usable_time_panels_are_drawn(groups, desc):
    assert (groups.hyper / "figures" / f"desc-{desc}_nirs.html").exists()


@pytest.mark.xfail(strict=True, reason="D1: no windowed SCI in the group record, so no share "
                   "of channels good in both members")
def test_the_table_prints_the_share_of_channels_good_in_every_member(groups):
    table = _table(groups)
    for group in GROUPS:
        assert table[_label(group)]["Good in both"] != "—", group
