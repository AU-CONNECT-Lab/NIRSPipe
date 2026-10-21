"""Two recordings of one subject told apart only by acq keep apart in every derivative name."""

import json

import pytest

from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _bad_channels_for, _spans_for
from nirspipe.exceptions import MissingDerivativesError
from nirspipe.io.bids import bids_label
from nirspipe.io.derivatives import carry_entities, select_one_run
from nirspipe.io.naming import figure_name, record_table_name, report_name
from nirspipe.io.optode_space import channel_coords_path
from nirspipe.pipeline.glm import _glm_name

from tests._fingerprint import CLI_ARGS, fingerprint_raw
from tests._synth import _write_dataset_root, _write_subject

ENTITIES = {"subject": "01", "session": "02", "task": "rest", "acquisition": "a", "run": 1,
            "suffix": "nirs", "datatype": "nirs"}


def test_acq_is_carried_into_names_between_task_and_run():
    assert carry_entities(ENTITIES) == {"task": "rest", "acquisition": "a", "run": 1}
    label = bids_label("01", ENTITIES)
    assert label == "sub-01_ses-02_task-rest_acq-a_run-1"
    assert report_name(label) == "sub-01_ses-02_task-rest_acq-a_run-1_report.html"
    assert (figure_name(label, "carpet")
            == "sub-01_ses-02_task-rest_acq-a_run-1_desc-carpet_nirs.html")
    assert (record_table_name(label, "sqm", "timeseries", statistic="sci")
            == "sub-01_ses-02_task-rest_acq-a_run-1_stat-sci_desc-sqm_timeseries.tsv")
    snirf = f"{label}_desc-errts_nirs.snirf"
    assert _glm_name(snirf, "design") == "sub-01_ses-02_task-rest_acq-a_run-1_design.tsv"
    assert (channel_coords_path(snirf).name
            == "sub-01_ses-02_task-rest_acq-a_run-1_space-fsaverage_channels.tsv")


def test_a_table_row_with_acq_matches_that_acq_and_a_blank_one_every_acq(tmp_path):
    table = tmp_path / "bads.tsv"
    table.write_text("acq\tbad_channels\na\tS1_D1\n\tS2_D2\n", encoding="utf-8")
    assert _bad_channels_for(str(table), "01", {"task": "rest", "acquisition": "a"}) == [
        "S1_D1", "S2_D2"]
    assert _bad_channels_for(str(table), "01", {"task": "rest", "acquisition": "b"}) == ["S2_D2"]
    # a recording without acq is not the one the row names
    assert _bad_channels_for(str(table), "01", {"task": "rest"}) == ["S2_D2"]
    spans = tmp_path / "spans.tsv"
    spans.write_text("acq\tonset\tduration\nacq-b\t10\t5\n", encoding="utf-8")
    assert _spans_for(str(spans), "01", {"acquisition": "b"}, flag="--bad-spans") == [(10.0, 5.0)]
    assert _spans_for(str(spans), "01", {"acquisition": "a"}, flag="--bad-spans") == []


def test_a_member_found_twice_under_two_acqs_is_refused_naming_them(tmp_path):
    files = [tmp_path / f"sub-01_task-rest_acq-{a}_run-01_desc-preproc_nirs.snirf" for a in "ab"]
    with pytest.raises(MissingDerivativesError, match="differ by acq: a, b"):
        select_one_run(files, what="desc-preproc", subject_id="sub-01", task="rest")


def test_two_inputs_told_apart_only_by_acq_write_two_output_sets(tmp_path):
    bids = tmp_path / "bids"
    _write_dataset_root(bids, ["01"])
    for acq, seed in (("a", 1), ("b", 2)):
        raw, _ = fingerprint_raw("01", "tapping", seed=seed, blocks=True)
        _write_subject(bids, "01", f"tapping_acq-{acq}_run-01", raw)
    table = tmp_path / "bads.tsv"
    table.write_text("acq\tbad_channels\na\tS1_D1\n", encoding="utf-8")
    run_cli.main([str(bids), str(tmp_path / "out"), "participant", *CLI_ARGS, "--mode",
                  "denoise", "--skip-bids-validation", "--bad-channels", str(table),
                  "--no-report"])
    nirs = tmp_path / "out" / "sub-01" / "nirs"
    names = sorted(p.name for p in nirs.iterdir())
    assert not [n for n in names if "_acq-" not in n]
    with_a = [n.replace("_acq-a_", "_acq-b_") for n in names if "_acq-a_" in n]
    assert with_a and with_a == [n for n in names if "_acq-b_" in n]
    assert sorted(p.name for p in nirs.glob("*desc-sqm_qc.json")) == [
        "sub-01_task-tapping_acq-a_run-01_desc-sqm_qc.json",
        "sub-01_task-tapping_acq-b_run-01_desc-sqm_qc.json"]

    def bads(acq):
        side = nirs / f"sub-01_task-tapping_acq-{acq}_run-01_desc-sci_nirs.json"
        return set(json.loads(side.read_text(encoding="utf-8"))["bad_channels"])

    assert {"S1_D1 760", "S1_D1 850"} <= bads("a") and not {"S1_D1 760"} & bads("b")
