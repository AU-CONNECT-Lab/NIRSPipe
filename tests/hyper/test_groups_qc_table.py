"""The dyad cohort table: shares at full precision, the kept share as a share, a dictionary of its columns."""

import json

import numpy as np
import pandas as pd

from nirspipe.qc.hyper.group_hyper_writer import build_group_hyper_report
from nirspipe.qc.hyper.hyper_usable import usable_scalars
from nirspipe.qc.metrics.hyper import compute_hyper_sqm


def test_the_share_kept_in_every_member_and_the_offset_are_unrounded():
    channels = {f"S{i}_D{i} 760": 0.9 for i in (1, 2, 3)}
    sqm_data = {"A": {"sci_win_per_channel": channels, "bad_channels": ["S1_D1 760"]},
                "B": {"sci_win_per_channel": channels, "bad_channels": []}}
    sqm = compute_hyper_sqm(sqm_data, {}, {"A": 0.1234567, "B": 0.0}, ["A", "B"], 0.8)
    # S2_D2 and S3_D3 kept by both, S1_D1 by B alone
    assert sqm["pct_all_good"] == 2 / 3
    assert sqm["max_offset_s"] == 0.1234567


def test_the_usable_shares_are_the_cell_counts_unrounded():
    member_a = np.array([[True] * 6 + [False], [True, False] * 3 + [True], [False] * 7])
    grid = {"t": np.arange(7) * 10.0, "pairs": ["S1_D1", "S2_D2", "S3_D3"],
            "ok": {"A": member_a, "B": np.ones_like(member_a)}}
    out = usable_scalars(grid, ["A", "B"])
    # 6 + 4 + 0 of 21 cells pass in both, the other 11 in B alone
    assert out["usable_window_frac"] == 10 / 21
    assert out["one_member_frac"] == 11 / 21
    assert out["neither_frac"] == 0.0
    assert out["usable_pairs_mean"] == 10 / 7


def test_the_table_keys_its_rows_by_bids_name_and_describes_every_column(tmp_path):
    nirs = tmp_path / "group-G1" / "nirs"
    nirs.mkdir(parents=True)
    (nirs / "group-G1_task-rest_desc-sqm_qc.json").write_text(json.dumps({
        "step": "hyper_sqm", "usable_window_frac": 10 / 21, "one_member_frac": 11 / 21,
        "neither_frac": 0.0, "n_long_pairs": 3, "n_windows": 7, "usable_pairs_mean": 10 / 7,
        "usable_stretch_s": 60.0, "aligned_duration_s": 70.0, "max_offset_s": 0.1234567,
        "pct_all_good": 2 / 3, "alignment": {"align_offset_s": {"A": 0.1234567, "B": 0.0}}}))
    build_group_hyper_report(tmp_path)

    table = pd.read_csv(tmp_path / "desc-groups_qc.tsv", sep="\t", float_precision="round_trip")
    sidecar = json.loads((tmp_path / "desc-groups_qc.json").read_text(encoding="utf-8"))
    assert table.columns[0] == "bids_name"
    assert table["bids_name"].tolist() == ["group-G1_task-rest"]
    assert table["usable_window_frac"].iloc[0] == 10 / 21
    assert list(sidecar) == list(table.columns)
    assert all(entry["Description"] for entry in sidecar.values())
