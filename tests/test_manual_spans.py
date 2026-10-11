"""--bad-spans: stretches the user names as corrupted are BAD_manual, left out of screening and filled."""

import json
import logging
from datetime import datetime, timezone

import mne
import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _spans_for, _refuse_bad_span_rows
from nirspipe.io.bids import get_layout
from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.censor_fill import corrupted_mask
from nirspipe.pipeline.prep_pipeline import intensity_to_od, mark_bad_channels
from nirspipe.qc.boilerplate.vocabulary import span_keys
from nirspipe.qc.subject.record_io import read_record
from nirspipe.utils.spans import (MANUAL, UNSELECTED, bad_spans, excluded_spans, mark_manual,
                                  mark_unselected)

from tests._fingerprint import CLI_ARGS, fingerprint_raw, make_fingerprint_dataset

SFREQ = 10.0


def _raw(seconds: float):
    raw = mne.io.RawArray(np.zeros((2, int(SFREQ * seconds))),
                          mne.create_info(["S1_D1 760", "S1_D1 850"], SFREQ, "fnirs_cw_amplitude"),
                          verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    return raw


def _table(tmp_path, text: str, name: str = "bad.tsv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---- marking ----

def test_overlapping_stretches_become_one_manual_span():
    raw = mark_manual(_raw(600), [(100, 20), (110, 30), (400, 10)])
    assert bad_spans(raw) == [(100.0, 140.0, MANUAL), (400.0, 410.0, MANUAL)]


def test_a_stretch_past_the_end_is_clipped_and_one_starting_outside_it_is_refused(caplog):
    with caplog.at_level(logging.INFO, logger="nirspipe"):
        raw = mark_manual(_raw(100), [(90, 50)])
    assert bad_spans(raw) == [(90.0, 100.0, MANUAL)]
    assert "--bad-spans stretch 90-140 s clipped to the recording's end at 100 s" in caplog.text
    with pytest.raises(ValueError, match="--bad-spans onset 100 s lies outside the recording"):
        mark_manual(_raw(100), [(100, 5)])


def test_manual_spans_are_corrupted_and_credited_to_the_flag():
    raw = mark_manual(mark_unselected(_raw(400), [(0, 300)]), [(50, 10)])
    got = {(s["description"], s["kind"], s["source"]) for s in excluded_spans(raw)}
    assert got == {(UNSELECTED, "unselected", "--keep-spans"),
                   (MANUAL, "corrupted", "--bad-spans")}
    # so the fill takes them, as it takes BAD_gvtd
    assert np.flatnonzero(corrupted_mask(raw)).tolist() == list(range(500, 600))


def test_the_methods_name_manual_spans_on_the_screening_step():
    assert span_keys("sci_pruning", {"bad_spans": [[50.0, 10.0]]}, []) == ["excluded_manual"]
    assert span_keys("sci_pruning", {"bad_spans": []}, []) == []


# ---- table ----

def test_the_table_reads_and_refuses_as_the_keep_spans_table_does(mini_bids, tmp_path):
    path = _table(tmp_path, "participant_id\ttask\tonset\tduration\n"
                            "\ttask-rest\t10\t5\n"
                            "sub-01\trest\t200\t50\n")
    rest = {"subject": "01", "task": "rest"}
    assert _spans_for(str(path), "01", rest, flag="--bad-spans") == [(10.0, 5.0),
                                                                     (200.0, 50.0)]
    with pytest.raises(ValueError, match=r"--bad-spans table .* line 2: duration '0'"):
        _spans_for(str(_table(tmp_path, "onset\tduration\n5\t0\n")), "01", rest,
                   flag="--bad-spans")
    layout = get_layout(mini_bids)
    with pytest.raises(SystemExit, match=r"--bad-spans table .* line 2 \(task-motor\) matches "
                                         r"no recording"):
        _refuse_bad_span_rows(str(_table(tmp_path, "task\tonset\tduration\nmotor\t0\t10\n")),
                              layout, flag="--bad-spans")
    with pytest.raises(SystemExit, match=r"--bad-spans table .* line 2: onset 100000 s lies "
                                         r"past the end of"):
        _refuse_bad_span_rows(str(_table(tmp_path, "task\tonset\tduration\n"
                                         "tapping\t100000\t10\n")),
                              layout, flag="--bad-spans")


# ---- screening ----

@pytest.mark.parametrize("scope", ["run", "task"])
def test_manual_time_does_not_decide_the_channel_set(scope):
    raw, truth = fingerprint_raw("01", "tapping", blocks=True)
    decoupled = truth.block_bad[0]        # decoupled 290-370 s, inside the second block

    def rejected(spans):
        od = intensity_to_od(mark_manual(raw.copy(), spans))
        _, bads, *_ = mark_bad_channels(od, 0.75, 0.8, 1.9, psp_threshold=0.12,
                                        min_good_frac=0.85, screen_scope=scope)
        return {b.split()[0] for b in bads}

    assert decoupled in rejected([])
    assert decoupled not in rejected([(285.0, 90.0)])


# ---- the whole run ----

SYNTH_ARGS = ("--dpf", "6", "6", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
              "--cardiac-h-freq", "1.5", "--resp-l-freq", "0.2", "--resp-h-freq", "0.5",
              "--participant-label", "01", "--task-label", "tapping", "--no-report",
              "--skip-bids-validation")


def test_a_recording_no_row_names_comes_out_bit_identical(mini_bids, tmp_path):
    plain, tabled = tmp_path / "plain", tmp_path / "tabled"
    table = _table(tmp_path, "participant_id\tonset\tduration\nsub-02\t10\t20\n")
    run_cli.main([str(mini_bids), str(plain), "participant", *SYNTH_ARGS])
    run_cli.main([str(mini_bids), str(tabled), "participant", *SYNTH_ARGS,
                  "--bad-spans", str(table)])

    files = sorted(p.relative_to(plain) for p in plain.rglob("*_nirs.snirf"))
    assert files and files == sorted(p.relative_to(tabled) for p in tabled.rglob("*_nirs.snirf"))
    for rel in files:
        a, b = read_snirf(plain / rel), read_snirf(tabled / rel)
        assert np.array_equal(a.get_data(), b.get_data()), rel
        assert a.info["bads"] == b.info["bads"]
        assert a.annotations.description.tolist() == b.annotations.description.tolist()
        sidecar = lambda root: json.loads((root / rel).with_suffix(".json").read_text(encoding="utf-8"))  # noqa: E731
        assert sidecar(plain)["excluded_spans"] == sidecar(tabled)["excluded_spans"] == []
        assert sidecar(plain)["parameters"] == sidecar(tabled)["parameters"]


def test_a_run_with_bad_spans_through_the_cli(tmp_path):
    bids, truth = make_fingerprint_dataset(tmp_path, blocks=True)
    table = _table(tmp_path, "task\tonset\tduration\ntask-tapping\t285\t90\n")
    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--min-good-frac", "0.85",
                  "--bad-spans", str(table), "--no-report", "--skip-bids-validation"])

    (od,) = out.rglob("*desc-od_nirs.json")
    assert json.loads(od.read_text(encoding="utf-8"))["excluded_spans"] == [
        {"onset": 285.0, "duration": 90.0, "description": MANUAL, "kind": "corrupted",
         "source": "--bad-spans"}]
    (sci,) = out.rglob("*desc-sci_nirs.json")
    sci_sidecar = json.loads(sci.read_text(encoding="utf-8"))
    assert sci_sidecar["parameters"]["bad_spans"] == [[285.0, 90.0]]
    assert truth.block_bad[0] not in {b.split()[0] for b in sci_sidecar["bad_channels"]}
    (record,) = out.rglob("*desc-sqm_qc.json")
    excluded = read_record(record)["excluded"]
    assert excluded["kept_s"] == 310.0
    assert excluded["corrupted_frac"] == pytest.approx(0.225)
    (cli_sci,) = out.rglob("*desc-sci_nirs.snirf")
    assert bad_spans(read_snirf(cli_sci)) == [(285.0, 375.0, MANUAL)]


# ---- the report ----

def test_the_report_is_handed_the_spans_the_run_added(tmp_path, monkeypatch):
    from nirspipe.qc.subject import report

    seen = {}
    monkeypatch.setattr(report, "build_subject_report",
                        lambda **kw: seen.update(kw) or [])
    bids, _ = make_fingerprint_dataset(tmp_path, blocks=True)
    keep = _table(tmp_path, "onset\tduration\n0\t360\n", "keep.tsv")
    bad = _table(tmp_path, "onset\tduration\n100\t20\n")
    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--keep-spans", str(keep),
                  "--bad-spans", str(bad), "--gvtd-censor", "long", "--skip-bids-validation"])

    segments = seen["segments"]
    assert segments["BAD_unselected"] == [(360.0, 40.0)]
    assert segments["BAD_manual"] == [(100.0, 20.0)]
    (sci,) = out.rglob("*desc-sci_nirs.json")
    gvtd = [(s["onset"], s["duration"])
            for s in json.loads(sci.read_text(encoding="utf-8"))["excluded_spans"]
            if s["description"] == "BAD_gvtd"]
    assert gvtd and segments["BAD_gvtd"] == gvtd
