"""A dyad page judges each member's SCI against the line that member was screened at."""

from nirspipe.qc.hyper.hyper_report import (
    _members_only, member_sci_lines, sci_lines_text, subject_metric_tables,
)


def _sci_class(tables, member):
    table = tables[0]
    metric = next(m for m in table["metrics"] if m["key"] == "sci_win_mean")
    return metric["cells"][table["members"].index(member)]["cls"]


def test_each_member_s_sci_is_judged_against_its_own_line():
    by_set = {"all": {"A": {"sci_win_mean": 0.75}, "B": {"sci_win_mean": 0.75}}}
    tables = subject_metric_tables(by_set, ["A", "B"], {"A": 0.7, "B": 0.8})
    assert (_sci_class(tables, "A"), _sci_class(tables, "B")) == ("qm-ok", "qm-bad")


def test_a_member_with_no_recorded_line_is_left_uncoloured():
    tables = subject_metric_tables({"all": {"A": {"sci_win_mean": 0.5}}}, ["A"], {"A": None})
    assert _sci_class(tables, "A") == ""


def test_a_pairing_page_keeps_only_its_own_members_columns():
    by_set = {"all": {"A": {"sci_win_mean": 0.9}, "B": {"sci_win_mean": 0.6},
                      "C": {"sci_win_mean": 0.7}}}
    whole = subject_metric_tables(by_set, ["A", "B", "C"], {})[0]
    pair = _members_only(whole, ["A", "C"])
    assert pair["members"] == ["A", "C"]
    cells = whole["metrics"][0]["cells"]
    assert pair["metrics"][0]["cells"] == [cells[0], cells[2]]


def test_the_lines_are_read_off_each_member_s_screening():
    sqm = {"A": {"screen_cutoffs": {"sci": 0.7, "psp": 0.1}}, "B": {}}
    lines = member_sci_lines(sqm, ["A", "B"])
    assert lines == {"A": 0.7, "B": None}
    assert sci_lines_text({"A": 0.7, "B": 0.7}) == "0.70"
    assert sci_lines_text(lines) == "A 0.70, B not recorded"


def test_a_record_with_no_psp_line_is_masked_on_sci_alone():
    from nirspipe.qc.metrics.windowed import passing_mask_from_matrices

    sci, psp = [[0.9, 0.9, 0.5]], [[0.01, 0.01, 0.01]]
    assert passing_mask_from_matrices(sci, psp, 0.8, None).tolist() == [[True, True, False]]
    assert not passing_mask_from_matrices(sci, psp, 0.8, 0.1).any()
