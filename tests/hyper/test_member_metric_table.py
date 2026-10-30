"""The dyad pages' quality tables: a channel set over each member, and the dyad's own numbers."""

import numpy as np

from nirspipe.qc.boilerplate.vocabulary import MISSING_VALUE
from nirspipe.qc.hyper.hyper_report import (
    dyad_metric_table, member_metric_table, raw_member_rows, subject_metric_tables,
)
from nirspipe.qc.metrics.hyper import compute_hyper_sqm, motion_summary


def _by_set(short=True):
    return {
        "all": {"A": {"channel_retention_rate": 0.5}, "B": {"channel_retention_rate": 0.6}},
        "long": {"A": {"channel_retention_rate": 0.7, "hbo_hbr_corr_mean": -0.4},
                 "B": {"channel_retention_rate": 0.8, "hbo_hbr_corr_mean": -0.3}},
        "short": ({"A": {"channel_retention_rate": 0.9}, "B": {"channel_retention_rate": 1.0}}
                  if short else {}),
    }


def _row(table, key):
    return next(r for g in table["groups"] for r in g["rows"] if r["key"] == key)


def test_each_set_sits_over_its_members_columns():
    table = member_metric_table(subject_metric_tables(_by_set(), ["A", "B"], {}))
    assert table["sets"] == ["All", "Long", "Short"]
    assert table["members"] == ["A", "B"]
    cells = [c["text"] for c in _row(table, "channel_retention_rate")["cells"]]
    assert cells == ["50.0%", "60.0%", "70.0%", "80.0%", "90.0%", "100.0%"]


def test_the_heart_rate_is_an_optical_density_row_with_each_member_s_value():
    by_set = {"all": {"A": {"heart_rate": 1.3777}, "B": {"heart_rate": 1.0861}}}
    table = member_metric_table(subject_metric_tables(by_set, ["A", "B"], {}))
    row = _row(table, "heart_rate")
    assert [c["text"] for c in row["cells"]] == ["1.38", "1.09"]
    assert next(g["title"] for g in table["groups"] if row in g["rows"]) == "Optical density"


def test_a_metric_one_set_lacks_prints_a_dash_in_that_set_only():
    table = member_metric_table(subject_metric_tables(_by_set(), ["A", "B"], {}))
    cells = [c["text"] for c in _row(table, "hbo_hbr_corr_mean")["cells"]]
    assert cells[:2] == [MISSING_VALUE] * 2 and cells[4:] == [MISSING_VALUE] * 2
    assert MISSING_VALUE not in cells[2:4]


def test_groups_come_in_the_subject_report_s_order_with_the_page_rows_first():
    extra = {"Recording": [{"key": "", "label": "S-D pairs", "summary": "",
                            "key_metric": False, "cells": []}]}
    table = member_metric_table(subject_metric_tables(_by_set(), ["A", "B"], {}), extra)
    assert [g["title"] for g in table["groups"]] == ["Recording", "Optical density",
                                                     "Haemoglobin"]


def test_an_unsplit_montage_prints_one_header_row():
    table = member_metric_table(subject_metric_tables(_by_set(short=False), ["A", "B"], {}))
    assert table["sets"] == []
    assert len(_row(table, "channel_retention_rate")["cells"]) == 2


def test_the_raw_page_rows_put_a_member_value_under_the_first_set_and_count_pairs_per_set():
    info = [{"subject_id": s, "sfreq": 5.0, "duration_s": 600.0, "n_pairs": 26, "n_long": 22,
             "n_short": 4} for s in ("A", "B")]
    motion = {"canonical": "long",
              **{k: {"before": {"A": 1.5, "B": 1.2}, "after": {"A": 1.1, "B": 1.0}}
                 for k in ("all", "long", "short")}}
    rows = raw_member_rows(info, motion, ["A", "B"], ["All", "Long", "Short"])
    rate, _duration, pairs = rows["Recording"]
    assert [c["text"] for c in rate["cells"]] == ["5.0000", "5.0000", "", "", "", ""]
    assert [c["text"] for c in pairs["cells"]] == ["26", "26", "22", "22", "4", "4"]
    gvtd = [c["text"] for c in rows["Motion"][0]["cells"]]
    assert gvtd == ["1.50 → 1.10", "1.20 → 1.00"] * 3


def test_the_dyad_table_counts_pairs_per_set_and_puts_the_spans_under_the_canonical_set():
    sqm = {"n_all_good": 20, "n_mixed": 4, "n_all_bad": 2, "n_total": 26, "max_offset_s": 1.25,
           "pairs_by_set": {"long": {"n_all_good": 17, "n_mixed": 3, "n_all_bad": 2,
                                     "n_total": 22},
                            "short": {"n_all_good": 3, "n_mixed": 1, "n_all_bad": 0,
                                      "n_total": 4}},
           "motion": {"canonical": "long", "spike_spans_both": {"before": 5, "after": 1},
                      **{k: {"before": {"together": 0.9}, "after": {"together": 0.6},
                             "reduction": 0.33} for k in ("all", "long", "short")}}}
    table = dyad_metric_table(sqm, ["All", "Long", "Short"])
    rows = {r["label"]: [c["text"] for c in r["cells"]] for r in table["rows"]}
    assert table["columns"] == ["All", "Long", "Short"]
    assert rows["Pairs good in every member"] == ["20 / 26", "17 / 22", "3 / 4"]
    assert rows["Pairs bad in every member"] == ["2", "2", "0"]
    assert rows["Both at once (x own median)"] == ["0.90 → 0.60"] * 3
    assert rows["Spans where both spiked"] == [MISSING_VALUE, "5 → 1", MISSING_VALUE]
    assert rows["Max alignment offset (s)"] == ["1.250", "", ""]


def test_pair_counts_split_by_set_add_up_to_the_whole():
    sqm_data = {sid: {"sci_win_per_channel": {p: 0.9 for p in ("S1_D1", "S2_D2", "S3_D9")},
                      "bad_channels": bad}
                for sid, bad in (("A", ["S1_D1 760"]), ("B", ["S1_D1 850", "S3_D9 760"]))}
    sqm = compute_hyper_sqm(sqm_data, {}, {}, ["A", "B"], 0.8,
                            pair_sets={"long": {"S1_D1", "S2_D2"}, "short": {"S3_D9"}})
    assert (sqm["n_all_good"], sqm["n_mixed"], sqm["n_all_bad"]) == (1, 1, 1)
    assert sqm["pairs_by_set"]["long"] == {"n_all_good": 1, "n_mixed": 0, "n_all_bad": 1,
                                           "n_unknown": 0, "n_total": 2}
    assert sqm["pairs_by_set"]["short"]["n_mixed"] == 1


def test_the_motion_summary_carries_the_all_trace_and_names_the_canonical_set():
    y = np.array([1.0, 2.0, 3.0])
    motion = {"sets": ["long", "short"], "stages": ["before"], "spikes_both": {},
              "series": {"before": {name: [("A", y), ("B", y * 2)]
                                    for name in ("long", "short", "all")}}}
    out = motion_summary(motion)
    assert out["canonical"] == "long"
    assert out["all"]["before"] == {"A": 2.0, "B": 4.0, "together": 2.0}


def test_only_the_long_set_is_coloured_where_the_montage_splits():
    # the cutoffs are drawn for long channels, as the member's own table colours them
    table = member_metric_table(subject_metric_tables(_by_set(), ["A", "B"], {}))
    classes = [c["cls"] for c in _row(table, "channel_retention_rate")["cells"]]
    assert classes[:2] == ["", ""] and classes[4:] == ["", ""]
    assert all(classes[2:4])


def test_a_montage_with_one_set_is_coloured():
    table = member_metric_table(subject_metric_tables(_by_set(short=False), ["A", "B"], {}))
    assert all(c["cls"] for c in _row(table, "channel_retention_rate")["cells"])
