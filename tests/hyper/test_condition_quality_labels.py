"""A dyad's per-condition quality table names a condition's channel share for what it is."""

from fnirs_pipe.qc.hyper.hyper_report import condition_subject_metrics, subject_metric_tables


def _labels(tables):
    return {c["key"]: c["label"] for table in tables for c in table["columns"]}


def _entry(share):
    return {"window_s": [10.0, 50.0],
            "od_by_set": {"all": {"channel_retention_rate": share, "sci_win_mean": 0.9}}}


def test_a_condition_table_calls_the_share_passing_in_condition():
    sqm = {"A": {"by_condition": {"talk": _entry(0.75)}},
           "B": {"by_condition": {"talk": _entry(1.0)}}}
    tables = condition_subject_metrics(sqm, ["A", "B"], [("talk", 10.0, 50.0)], 0.8)
    labels = _labels(tables["talk"])
    assert labels["channel_retention_rate"] == "Passing in condition"
    assert "Channel retention" not in labels.values()


def test_the_whole_run_table_keeps_channel_retention():
    tables = subject_metric_tables({"all": {"A": {"channel_retention_rate": 0.75}}}, ["A"], 0.8)
    assert _labels(tables)["channel_retention_rate"] == "Channel retention"
