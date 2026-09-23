"""What the dyad index adds on top of the links: the scale the coherence is read against.

The Windows table listed a coherence per window and nothing to judge it by. Coherence has a
floor that moves with the window length, so those values do not compare down the column, and
the phase-scrambled null's tables carrying each pair's own rank sat unread beside the page. These cover
the column that reads them, and the one thing that goes wrong quietly: a tree with no null
must keep the column off rather than print a zero that reads as a result.
"""

import pandas as pd

from fnirs_pipe.io.naming import report_name
from fnirs_pipe.qc.hyper.hyper_index import NULL_PERCENTILE, _links, _past_null
from tests.hyper._names import name


def _null_table(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_the_count_is_the_pairs_that_beat_their_own_draws():
    df = _null_table(chromophore=["hbo"] * 4 + ["hbr"] * 4,
                 percentile=[99.0, 96.0, 50.0, 3.0, 12.0, 40.0, 61.0, 88.0])
    assert _past_null(df) == {"hbo": (2, 4), "hbr": (0, 4)}


def test_the_threshold_is_the_one_the_tables_column_is_drawn_at():
    df = _null_table(chromophore=["hbo", "hbo"],
                 percentile=[float(NULL_PERCENTILE), NULL_PERCENTILE - 0.1])
    assert _past_null(df) == {"hbo": (1, 2)}


def test_a_pair_with_no_rank_is_left_out_of_the_denominator():
    # a channel blank in every iteration was ranked against nothing; counting it as a miss
    # would report a dyad as less coupled the more of it went missing
    df = _null_table(chromophore=["hbo"] * 3, percentile=[99.0, float("nan"), 10.0])
    assert _past_null(df) == {"hbo": (1, 2)}


def test_one_condition_is_counted_over_its_own_rows():
    df = _null_table(chromophore=["hbo"] * 4,
                 condition=["game1", "game1", "rest", "rest"],
                 percentile=[99.0, 98.0, 1.0, 2.0])
    assert _past_null(df, ("condition", "game1")) == {"hbo": (2, 2)}
    assert _past_null(df, ("condition", "rest")) == {"hbo": (0, 2)}


def test_one_pairing_is_counted_over_its_own_rows():
    df = _null_table(chromophore=["hbo"] * 4,
                 sub1=["a", "a", "a", "a"], sub2=["b", "b", "c", "c"],
                 percentile=[99.0, 98.0, 1.0, 2.0])
    assert _past_null(df, None, ("a", "b")) == {"hbo": (2, 2)}


def test_a_tree_with_no_null_gets_no_column():
    # the column is hidden on an empty dict; a 0/0 would read as "nothing was coupled"
    assert _past_null(None) == {}
    assert _past_null(_null_table(chromophore=["hbo"], coherence=[0.4])) == {}


def test_only_the_artefacts_on_disk_are_linked(tmp_path):
    stem = "group-d01_task-full"
    (tmp_path / "nirs").mkdir()
    (tmp_path / report_name(stem, desc="raw")).write_text("x", encoding="utf-8")
    (tmp_path / "nirs" / name("d01", "full")).write_text("x", encoding="utf-8")

    links = _links(tmp_path, stem)
    assert [link["text"] for link in links] == ["raw QC", "coherence"]
    assert links[0]["href"] == report_name(stem, desc="raw")
