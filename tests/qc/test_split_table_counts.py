"""A share and its count print in one cell, and only when the caller asks for it."""

from nirspipe.qc.common.channel_table import COUNT_BESIDE, MOTION_SPLIT_COLUMNS, split_table

_LONG = {"spike_pct_frames": 25.4, "spike_num_frames": 2158,
         "spike_pct_frames_post": 0.5, "spike_num_frames_post": 42}


def test_folding_puts_the_count_beside_its_share_either_side_of_the_step():
    table = split_table([("Long", 22, _LONG, True)], MOTION_SPLIT_COLUMNS, fold_counts=True)
    keys = [c["key"] for c in table["columns"]]
    assert keys == ["spike_pct_frames"], "the count's own column is gone"
    cell = table["rows"][0]["cells"][0]
    assert cell["value"].endswith("(2158)") and cell["post"].endswith("(42)")


def test_without_folding_the_count_keeps_its_column():
    table = split_table([("Long", 22, _LONG, True)], MOTION_SPLIT_COLUMNS)
    assert [c["key"] for c in table["columns"]] == ["spike_pct_frames", "spike_num_frames"]
    assert "(" not in table["rows"][0]["cells"][0]["value"]


def test_a_count_without_its_share_is_left_alone():
    table = split_table([("Long", 22, {"spike_num_frames": 2158}, True)], MOTION_SPLIT_COLUMNS,
                        fold_counts=True)
    assert [c["key"] for c in table["columns"]] == ["spike_num_frames"]


def test_every_pair_names_a_share_and_a_count_the_tables_print():
    printed = {key for key, _ in MOTION_SPLIT_COLUMNS} | {"spike_pct", "spike_count"}
    assert set(COUNT_BESIDE) | set(COUNT_BESIDE.values()) <= printed
