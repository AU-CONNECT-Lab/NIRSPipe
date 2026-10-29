"""A dyad with no blocks files its usable time under the whole run's own label, and the dials say it in words."""

import numpy as np

from nirspipe.qc.figures.hyper.group_hyper_figures import build_condition_dials
from nirspipe.qc.hyper.hyper_usable import usable_table


def _grid():
    member_a = np.array([[True, True, False, True], [True, False, False, True]])
    return {"t": [0.0, 10.0, 20.0, 30.0], "pairs": ["S1_D1", "S2_D2"],
            "ok": {"a": member_a, "b": np.ones_like(member_a)}}


def test_a_dyad_with_no_blocks_reads_whole_run_in_its_condition_column():
    table = usable_table(_grid(), ["a", "b"]).set_index("pair")
    assert table["condition"].tolist() == ["whole_run", "whole_run"]
    # passing in both members in three and in two of the four windows
    assert table["usable_frac"].tolist() == [0.75, 0.5]


def test_the_dials_print_the_whole_run_in_words():
    rows = [{"label": "group-G1", "usable": {"usable_window_frac": 0.625},
             "by_cond": {"whole_run": 0.625}}]
    text = build_condition_dials(rows, ["group-G1"]).to_json()
    assert "whole run" in text and "whole_run" not in text
