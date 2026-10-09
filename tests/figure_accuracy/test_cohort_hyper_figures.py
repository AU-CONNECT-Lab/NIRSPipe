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


def _measured(groups, group):
    """{strip row: {window: percentile}} off the group record, unmeasured windows left out."""
    pairings = groups.record(group)["screening"]["pairings"]
    return {(_label(group) if len(pairings) == 1
             else f"{_label(group)} · {p['sub1']} × {p['sub2']}"):
            {w: v["percentile"] for w, v in p["windows"].items() if np.isfinite(v["percentile"])}
            for p in pairings}


def test_the_null_strip_has_one_row_per_pairing_each_its_own_percentiles(groups):
    fig = one_figure(groups.hyper / "figures" / "desc-groupsnullstrip_nirs.html")
    ticks = dict(zip(fig["layout"]["yaxis"]["tickvals"], fig["layout"]["yaxis"]["ticktext"]))
    want = {row: pct for group in GROUPS for row, pct in _measured(groups, group).items()}
    assert set(ticks.values()) == set(want)       # a dyad one row, the triad three
    drawn: dict = {}
    for trace in (t for t in fig["data"] if t.get("mode") == "markers"):
        for x, y, label in zip(trace["x"], trace["y"], trace["customdata"]):
            assert ticks[y] == label
            drawn.setdefault(label, {})[trace["name"]] = x
    assert set(drawn) == set(want)
    for row, pct in want.items():
        assert drawn[row] == pytest.approx(pct), row


def test_the_table_carries_each_group_s_alignment(groups):
    table = _table(groups)
    for group in GROUPS:
        row = table[_label(group)]
        offsets = [m.offset for m in groups.truth.group(group)]
        assert float(row["Max offset"].split()[0]) == pytest.approx(max(offsets), abs=0.01)
        assert row["Offsets equal"] == "no"
        values = [v for pct in _measured(groups, group).values() for v in pct.values()]
        assert int(row["Above null"]) == sum(v >= 95 for v in values)


def test_the_median_percentile_leaves_out_a_window_with_no_measurable_coherence(groups):
    table = _table(groups)
    for group in GROUPS:
        records = groups.record(group)["screening"]["pairings"]
        assert any(not np.isfinite(v["coherence"]) for p in records for v in p["windows"].values())
        values = [v for pct in _measured(groups, group).values() for v in pct.values()]
        assert float(table[_label(group)]["Median pct"]) == pytest.approx(np.median(values),
                                                                          abs=0.5)


@pytest.mark.parametrize("desc", ["groupsusablebars", "groupspairfield", "groupsconditiondials"])
def test_the_usable_time_panels_are_drawn(groups, desc):
    assert (groups.hyper / "figures" / f"desc-{desc}_nirs.html").exists()


def test_the_table_prints_the_share_of_channels_good_in_every_member(groups):
    table = _table(groups)
    for group in GROUPS:
        assert table[_label(group)]["Good in both"] != "—", group
