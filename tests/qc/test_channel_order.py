"""Channel tables and the channel grid list pairs by source then detector number, a pair's wavelengths side by side."""

from nirspipe.qc.common.channel_table import channel_rows, heatmap_args, pair_rows

# as a device may write them: one wavelength block, then the other, S10 before S2
ACQUISITION = ["S10_D1 760", "S2_D10 760", "S2_D9 760", "S1_D2 760",
               "S10_D1 850", "S2_D10 850", "S2_D9 850", "S1_D2 850"]


def _record(short=()):
    long_ = [ch for ch in ACQUISITION if ch.split()[0] not in short]
    shorts = [ch for ch in ACQUISITION if ch.split()[0] in short]
    per_channel = {"raw_long": {"sci_per_channel": {ch: 0.9 for ch in long_}}}
    if shorts:
        per_channel["raw_short"] = {"sci_per_channel": {ch: 0.9 for ch in shorts}}
    return {"per_channel": per_channel}


def test_rows_go_by_source_then_detector_with_both_wavelengths_together():
    rows = channel_rows(_record(), {ch: 0.9 for ch in ACQUISITION}, [])
    assert [r["name"] for r in rows] == [
        "S1_D2 760", "S1_D2 850", "S2_D9 760", "S2_D9 850",
        "S2_D10 760", "S2_D10 850", "S10_D1 760", "S10_D1 850"]


def test_the_grid_keeps_long_first_and_pair_order_inside_each_block():
    rows = channel_rows(_record(short=("S2_D9",)), {ch: 0.9 for ch in ACQUISITION}, [])
    args = heatmap_args(rows)
    assert args["ch_names"] == [
        "S1_D2 760", "S1_D2 850", "S2_D10 760", "S2_D10 850", "S10_D1 760", "S10_D1 850",
        "S2_D9 760", "S2_D9 850"]
    assert args["split_at"] == 6


def test_a_pair_list_from_the_haemoglobin_file_sets_membership_not_order():
    rows = channel_rows(_record(), {ch: 0.9 for ch in ACQUISITION}, [])
    pairs = pair_rows(rows, ["S10_D1", "S2_D10", "S2_D9", "S1_D2", "S3_D3"])
    assert [r["pair"] for r in pairs] == ["S1_D2", "S2_D9", "S2_D10", "S3_D3", "S10_D1"]
    assert pairs[3]["sci_win"] is None, "a listed pair with no row is still there, empty"
