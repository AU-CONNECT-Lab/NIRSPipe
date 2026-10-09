"""The cohort page over groups (fnirs-qc cohort-hyper): each figure against each group's record."""

import pytest

from tests.figure_accuracy._dyad import html_tables

GROUPS = ("G01", "G02")


def _label(group):
    return f"group-{group}_task-main"


def _table(groups):
    rows = next(rows for _, rows in html_tables(groups.hyper / "desc-groups_report.html")
                if rows and rows[0][0].rstrip("⇅") == "Dyad")
    head = [h.rstrip("⇅") for h in rows[0]]
    return {r[0]: dict(zip(head, r)) for r in rows[1:]}


def test_the_table_carries_each_group_s_alignment(groups):
    table = _table(groups)
    for group in GROUPS:
        row = table[_label(group)]
        offsets = [m.offset for m in groups.truth.group(group)]
        assert float(row["Max offset"].split()[0]) == pytest.approx(max(offsets), abs=0.01)
        assert row["Offsets equal"] == "no"


@pytest.mark.parametrize("desc", ["groupsusablebars", "groupspairfield", "groupsconditiondials"])
def test_the_usable_time_panels_are_drawn(groups, desc):
    assert (groups.hyper / "figures" / f"desc-{desc}_nirs.html").exists()


def test_the_table_prints_the_share_of_channels_good_in_every_member(groups):
    table = _table(groups)
    for group in GROUPS:
        assert table[_label(group)]["Good in both"] != "—", group
