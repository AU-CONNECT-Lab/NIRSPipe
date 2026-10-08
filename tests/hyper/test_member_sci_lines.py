"""A dyad page judges each member's SCI against the line that member was screened at."""

import numpy as np

from fnirs_pipe.pipeline.hyper.group_quality import _screen_windows
from fnirs_pipe.qc.hyper.hyper_report import member_sci_lines, sci_lines_text, subject_metric_tables


def _sci_class(tables, member):
    table = tables[0]
    column = [c["key"] for c in table["columns"]].index("sci_win_mean")
    return next(row["cells"][column]["cls"] for row in table["rows"] if row["member"] == member)


def test_each_member_s_sci_is_judged_against_its_own_line():
    by_set = {"all": {"A": {"sci_win_mean": 0.75}, "B": {"sci_win_mean": 0.75}}}
    tables = subject_metric_tables(by_set, ["A", "B"], {"A": 0.7, "B": 0.8})
    assert (_sci_class(tables, "A"), _sci_class(tables, "B")) == ("qm-ok", "qm-bad")


def test_a_member_with_no_recorded_line_is_left_uncoloured():
    tables = subject_metric_tables({"all": {"A": {"sci_win_mean": 0.5}}}, ["A"], {"A": None})
    assert _sci_class(tables, "A") == ""


def test_the_lines_are_read_off_each_member_s_screening():
    sqm = {"A": {"screen_cutoffs": {"sci": 0.7, "psp": 0.1}}, "B": {}}
    lines = member_sci_lines(sqm, ["A", "B"])
    assert lines == {"A": 0.7, "B": None}
    assert sci_lines_text({"A": 0.7, "B": 0.7}) == "0.70"
    assert sci_lines_text(lines) == "A 0.70, B not recorded"


def test_a_record_with_no_line_is_not_re_masked_at_a_guessed_one():
    record = {"windowed": {"sci_matrix": np.full((2, 3), 0.9).tolist(),
                           "psp_matrix": np.full((2, 3), 0.2).tolist(),
                           "sci_times": [[0, 10], [10, 20], [20, 30]]}}
    assert _screen_windows(record, {"psp": 0.1}) == {}
    assert _screen_windows(record, {"sci": 0.8, "psp": 0.1})["mask"].all()
