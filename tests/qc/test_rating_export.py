"""The rating viewers' decisions reach the pipeline as one --bad-channels table."""

import json

import pandas as pd
import pytest

from nirspipe.cli.rate import main as rate_main
from nirspipe.cli.workflows import _bad_channels_for, _keep_channels_for
from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import channel_decisions_path
from nirspipe.io.naming import rating_path
from nirspipe.pipeline.prep_pipeline import release_kept
from nirspipe.qc.boilerplate.vocabulary import template_slots
from nirspipe.qc.rating.export import channel_rows, export_ratings, verdict_rows


def _decisions(tree, subject, task, data, session=None):
    path = channel_decisions_path(tree, subject, task=task, session=session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_run_s_decisions_become_one_row_and_the_dyad_s_a_row_for_every_run(tmp_path):
    _decisions(tmp_path, "01", "rest", {
        "sub-01_task-rest_run-01": {"S1_D1 hbo": "bad", "S1_D1 hbr": "bad",
                                    "S2_D2 hbo": "good", "S4_D4 hbo": "unrated"},
        "_hyper": {"S3_D3 hbo": "bad"}}, session="02")

    rows, conflicts = channel_rows([tmp_path])

    assert conflicts == []
    assert rows == [
        {"participant_id": "sub-01", "session": "02", "task": "rest", "acq": "", "run": "01",
         "rater": "", "bad_channels": "S1_D1", "keep_channels": "S2_D2"},
        {"participant_id": "sub-01", "session": "02", "task": "rest", "acq": "", "run": "",
         "rater": "", "bad_channels": "S3_D3", "keep_channels": ""},
    ]


def test_a_pair_rated_both_ways_for_one_recording_is_reported(tmp_path):
    _decisions(tmp_path, "01", "rest", {"sub-01_task-rest_run-01": {"S1_D1 hbo": "good"},
                                        "_hyper": {"S1_D1 hbo": "bad"}})
    _, conflicts = channel_rows([tmp_path])
    assert conflicts == ["sub-01 sub-01_task-rest_run-01: S1_D1"]


def test_the_exported_table_is_what_bad_channels_reads(tmp_path):
    _decisions(tmp_path, "01", "rest", {"sub-01_task-rest_run-01": {"S1_D1 hbo": "bad",
                                                                    "S2_D2 hbo": "good"}})
    channels, _, _ = export_ratings([tmp_path], tmp_path / "out")
    entities = {"task": "rest", "run": 1}

    assert _bad_channels_for(str(channels), "01", entities) == ["S1_D1"]
    assert _keep_channels_for(str(channels), "01", entities) == ["S2_D2"]
    assert _keep_channels_for(str(channels), "01", {"task": "rest", "run": 2}) == []
    # a bare list on the command line keeps nothing
    assert _keep_channels_for("S1_D1,S2_D2", "01", entities) == []


def test_a_decision_on_one_acq_reaches_that_acq_only(tmp_path):
    _decisions(tmp_path, "01", "rest", {"sub-01_task-rest_acq-a_run-01": {"S1_D1 hbo": "bad"}})
    channels, _, _ = export_ratings([tmp_path], tmp_path / "out")
    assert pd.read_csv(channels, sep="\t", dtype=str)["acq"].tolist() == ["a"]
    assert _bad_channels_for(str(channels), "01",
                             {"task": "rest", "acquisition": "a", "run": 1}) == ["S1_D1"]
    assert _bad_channels_for(str(channels), "01",
                             {"task": "rest", "acquisition": "b", "run": 1}) == []


def test_final_verdicts_are_listed_for_reading(tmp_path):
    stem = "sub-01_task-rest_desc-raw_report"
    path = rating_path(tmp_path, stem)
    path.write_text(json.dumps({"stem": stem, "ratings": {"sub-01_task-rest_Final": "bad",
                                                          "Signal": "good"},
                                "notes": {"sub-01_task-rest_Final": "cap slipped"}}),
                    encoding="utf-8")

    assert verdict_rows([tmp_path]) == [{"page": stem, "rater": "", "key": "sub-01_task-rest_Final",
                                         "verdict": "bad", "note": "cap slipped"}]


def test_an_empty_tree_still_writes_both_tables_with_their_headers(tmp_path):
    channels, verdicts, _ = export_ratings([tmp_path], tmp_path)
    assert list(pd.read_csv(channels, sep="\t").columns) == [
        "participant_id", "session", "task", "acq", "run", "rater", "bad_channels",
        "keep_channels"]
    assert list(pd.read_csv(verdicts, sep="\t").columns) == ["page", "rater", "key", "verdict",
                                                             "note"]


def test_a_kept_pair_comes_out_of_screening_s_rejections():
    screened = ["S1_D1 760", "S1_D1 850", "S2_D2 760", "S2_D2 850"]
    assert release_kept(screened, ["S2_D2"]) == (["S1_D1 760", "S1_D1 850"],
                                                 ["S2_D2 760", "S2_D2 850"])
    assert release_kept(screened, ["S9_D9"]) == (screened, [])


def test_the_methods_name_the_kept_channels():
    slots = template_slots("sci_marking", {"sci_threshold": 0.8,
                                           "kept_channels": ["S2_D2 760", "S2_D2 850"]})
    assert slots["kept"] == (" Channel S2_D2 was kept after visual review although the "
                             "screening rejected it.")
    assert template_slots("sci_marking", {"sci_threshold": 0.8})["kept"] == ""


def test_the_cli_writes_both_tables(tmp_path, capsys):
    from nirspipe.cli import rate

    _decisions(tmp_path, "01", "rest", {"sub-01_task-rest": {"S1_D1 hbo": "bad"}})
    rate.main(["export", str(tmp_path)])
    assert "--bad-channels" in capsys.readouterr().out
    assert (tmp_path / "desc-ratedchannels_qc.tsv").exists()
    assert (tmp_path / "desc-ratedpages_qc.tsv").exists()


def test_a_decision_saved_under_a_condition_page_s_key_is_refused(tmp_path, capsys):
    key = "sub-01_task-rest_run-01_cond-alone"
    _decisions(tmp_path, "01", "rest", {"sub-01_task-rest_run-01": {"S2_D2 hbo": "bad"},
                                        key: {"S1_D1 hbo": "good"}})
    with pytest.raises(StageError, match=key):
        channel_rows([tmp_path])
    with pytest.raises(SystemExit) as stop:
        rate_main(["export", str(tmp_path)])
    assert stop.value.code == 1
    assert key in capsys.readouterr().err
