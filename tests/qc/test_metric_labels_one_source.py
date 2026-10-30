"""Every view heads a metric with the registry's label and prints it in the registry's format."""

import pytest

from nirspipe.qc.boilerplate.vocabulary import metric_label
from nirspipe.qc.common.channel_table import MOTION_SPLIT_COLUMNS, OD_SPLIT_COLUMNS
from nirspipe.qc.subject.subject_index import _COLUMNS, _COND_COLUMNS, collect_runs

from tests.qc.test_subject_index import _run


@pytest.mark.parametrize("columns", [OD_SPLIT_COLUMNS, MOTION_SPLIT_COLUMNS])
def test_the_split_tables_head_each_metric_with_its_registry_label(columns):
    assert [text for _, text in columns] == [metric_label(key) for key, _ in columns]


def test_the_short_forms_are_gone():
    heads = {text for _, text in (*OD_SPLIT_COLUMNS, *MOTION_SPLIT_COLUMNS)}
    assert not heads & {"Spike share", "Corrected per channel", "Corrected % frames",
                        "Corrected frames", "Corrected segments"}


def test_the_condition_columns_carry_the_registry_labels():
    assert [head for head, _, _ in _COND_COLUMNS] == [
        metric_label(key) for _, key, _ in _COND_COLUMNS]


def test_the_index_prints_a_run_as_its_report_does(tmp_path):
    _run(tmp_path, "a", sci=0.8137)
    (row,) = collect_runs(tmp_path)
    cells = {head: cell["value"] for (head, _, _), cell in zip(_COLUMNS, row["metrics"])}
    assert cells["Mean SCI (whole run)"] == "0.814"
    assert cells["Channel retention"] == "100.0%"
    assert cells["Motion corrected % (exp.)"] == "10.0%"
    assert cells["HbO-HbR corr"] == "0.200"
