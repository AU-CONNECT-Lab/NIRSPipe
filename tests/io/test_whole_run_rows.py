"""A table holding the whole run beside its conditions names the whole run's rows, never n/a."""

import pandas as pd

from nirspipe.io.tables import LABEL_DTYPES, join_whole_run, split_whole_run, write_tsv


def _halves():
    whole = pd.DataFrame({"label": ["a", "b"], "coherence": [0.1, 0.2]})
    by_condition = pd.DataFrame({"condition": ["game", "rest"], "label": ["a", "a"],
                                 "coherence": [0.3, 0.4]})
    return whole, by_condition


def test_the_whole_run_rows_come_first_under_their_own_condition():
    whole, by_condition = _halves()
    table = join_whole_run([whole], [by_condition])
    assert table["condition"].tolist() == ["whole_run", "whole_run", "game", "rest"]
    assert list(table.columns) == ["condition", "label", "coherence"]


def test_a_table_with_no_conditions_keeps_no_condition_column():
    whole, _ = _halves()
    assert list(join_whole_run([whole], []).columns) == ["label", "coherence"]


def test_read_back_the_halves_split_apart_and_a_groupby_keeps_the_whole_run(tmp_path):
    whole, by_condition = _halves()
    path = write_tsv(join_whole_run([whole], [by_condition]), tmp_path / "t.tsv")
    table = pd.read_csv(path, sep="\t", dtype=LABEL_DTYPES)
    assert table["condition"].notna().all()
    # pandas drops n/a keys from a groupby by default, which is what lost the whole run
    assert table.groupby("condition").size().to_dict() == {"game": 1, "rest": 1, "whole_run": 2}
    got_whole, got_conditions = split_whole_run(table)
    pd.testing.assert_frame_equal(got_whole.reset_index(drop=True), whole)
    assert got_conditions["condition"].tolist() == ["game", "rest"]
