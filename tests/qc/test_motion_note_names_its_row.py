"""A page note that points at a row of the motion table names the row as the table heads it."""

from nirspipe.cli.qc import _build_parser
from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.boilerplate.vocabulary import metric_label
from nirspipe.qc.common.channel_table import MOTION_SPLIT_COLUMNS


def test_the_one_comparable_row_is_named_by_its_heading():
    heading = metric_label("motion_corrected_frac_mean")
    assert ("motion_corrected_frac_mean", heading) in MOTION_SPLIT_COLUMNS
    assert heading in section_note("metrics.gvtd_sets_columns")


def test_raw_qc_s_correction_help_says_where_the_corrected_file_goes():
    (action,) = [a for a in _build_parser()._actions if a.dest == "motion_correction"]
    assert "Nothing is written back" not in action.help
    assert "stages/" in action.help
