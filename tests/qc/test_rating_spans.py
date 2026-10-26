"""Stretches marked bad on the raw viewer's run page reach the pipeline as a --bad-spans table."""

import json

import pandas as pd
import pytest

from nirspipe.cli.rate import _build_parser, main as rate_main
from nirspipe.cli.workflows import _spans_for
from nirspipe.io.naming import bad_spans_path, report_name
from nirspipe.qc.rating.app import RawRatingApp
from nirspipe.qc.rating.export import SPANS_NAME, export_spans

RUN = "sub-01_ses-02_task-rest_run-01"


def _viewer(tmp_path):
    html = tmp_path / "sub-01" / report_name(RUN, desc="raw")
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text("<html><head></head></html>", encoding="utf-8")
    return RawRatingApp(html, tmp_path).app.test_client()


def test_marked_stretches_are_saved_in_time_order_and_read_back(tmp_path):
    client = _viewer(tmp_path)
    assert client.get("/rate_config").json["mark_spans"] is True

    r = client.post("/save_bad_spans", json={RUN: [[300.12345, 312.5], [12.0, 15.5]]})
    assert r.status_code == 200, r.json
    saved = {RUN: [[12.0, 15.5], [300.123, 312.5]]}
    assert json.loads(bad_spans_path(tmp_path, "01", task="rest", session="02")
                      .read_text(encoding="utf-8")) == saved
    assert client.get("/load_bad_spans").json == saved


@pytest.mark.parametrize("payload", [
    {RUN: [[15.5, 12.0]]},                   # stop before start
    {RUN: [[-1.0, 5.0]]},                    # before the recording
    {RUN: [[1.0, float("inf")]]},
    {RUN: [["a", 5.0]]},
    {RUN: [[1.0, 2.0, 3.0]]},
    {RUN + "_cond-alone": [[1.0, 2.0]]},     # a condition page marks nothing
    [[1.0, 2.0]],
])
def test_a_stretch_that_is_not_a_span_of_the_run_is_refused(tmp_path, payload):
    client = _viewer(tmp_path)
    assert client.post("/save_bad_spans", json=payload).status_code == 400
    assert not bad_spans_path(tmp_path, "01", task="rest", session="02").exists()


def test_the_old_switch_is_refused():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["raw", "out", "--participant-label", "01", "--mark-spans"])


def test_the_exported_table_is_what_bad_spans_reads(tmp_path):
    _viewer(tmp_path).post("/save_bad_spans", json={RUN: [[12.0, 15.5], [300.0, 312.5]]})
    table = export_spans([tmp_path], tmp_path / "out")

    assert table == tmp_path / "out" / SPANS_NAME
    assert list(pd.read_csv(table, sep="\t").columns) == [
        "participant_id", "session", "task", "acq", "run", "onset", "duration"]
    entities = {"session": "02", "task": "rest", "run": 1}
    assert _spans_for(str(table), "01", entities, flag="--bad-spans") == [(12.0, 3.5), (300.0, 12.5)]
    assert _spans_for(str(table), "01", {**entities, "run": 2}, flag="--bad-spans") == []


def test_no_span_file_writes_no_table_and_the_export_says_nothing_of_spans(tmp_path, capsys):
    assert export_spans([tmp_path], tmp_path) is None
    rate_main(["export", str(tmp_path)])
    assert not (tmp_path / SPANS_NAME).exists()
    assert "--bad-spans" not in capsys.readouterr().out


def test_the_export_command_names_the_table_for_bad_spans(tmp_path, capsys):
    _viewer(tmp_path).post("/save_bad_spans", json={RUN: [[12.0, 15.5]]})
    rate_main(["export", str(tmp_path)])
    assert f"--bad-spans {tmp_path / SPANS_NAME}" in capsys.readouterr().out
