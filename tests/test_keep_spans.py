"""--keep-spans: time outside the named stretches is BAD_unselected, and screening leaves it out."""

import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone

import mne
import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _keep_spans_for, _refuse_bad_keep_span_rows
from nirspipe.exceptions import StageError
from nirspipe.io.bids import get_layout
from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.prep_pipeline import PrepConfig, intensity_to_od, mark_bad_channels, run_prep
from nirspipe.qc.metrics.windowed import _counted, counted_screen_windows, coupled_windows
from nirspipe.qc.subject.record_io import read_record
from nirspipe.utils.spans import (UNSELECTED, add_bad_spans, bad_spans, excluded_spans,
                                  excluded_time, mark_unselected)

from tests._fingerprint import CLI_ARGS, DPF, fingerprint_raw, make_fingerprint_dataset

SFREQ = 10.0


def _raw(seconds: float, onsets=(), durations=(), descs=()):
    raw = mne.io.RawArray(np.zeros((2, int(SFREQ * seconds))),
                          mne.create_info(["S1_D1 760", "S1_D1 850"], SFREQ, "fnirs_cw_amplitude"),
                          verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations(list(onsets), list(durations), list(descs)))
    return raw


def _table(tmp_path, text: str):
    path = tmp_path / "keep.tsv"
    path.write_text(text, encoding="utf-8")
    return path


# ---- marking ----

def test_the_time_outside_the_stretches_is_marked_unselected():
    raw = mark_unselected(_raw(600), [(100, 100), (250, 100), (150, 120)])
    assert bad_spans(raw) == [(0.0, 100.0, UNSELECTED), (350.0, 600.0, UNSELECTED)]


def test_a_stretch_past_the_end_is_clipped_and_one_starting_outside_it_is_refused(caplog):
    with caplog.at_level(logging.INFO, logger="nirspipe"):
        raw = mark_unselected(_raw(100), [(40, 500)])
    assert bad_spans(raw) == [(0.0, 40.0, UNSELECTED)]
    assert "clipped to the recording's end at 100 s" in caplog.text
    with pytest.raises(ValueError, match="outside the recording"):
        mark_unselected(_raw(100), [(100, 5)])


def test_no_stretches_leave_the_recording_whole():
    raw = _raw(100, [10.0], [0.0], ["tap"])
    assert mark_unselected(raw, []).annotations.description.tolist() == ["tap"]


# ---- sidecar and record ----

def test_each_span_names_its_kind_and_where_it_came_from():
    raw = _raw(400, [10.0], [5.0], ["BAD_motion"])
    given = bad_spans(raw)
    mark_unselected(raw, [(0, 300)])
    add_bad_spans(raw, [(50.0, 4.0)], "BAD_gvtd")

    got = {(s["description"], s["kind"], s["source"]) for s in excluded_spans(raw, given)}
    assert got == {("BAD_motion", "corrupted", "input file"),
                   (UNSELECTED, "unselected", "--keep-spans"),
                   ("BAD_gvtd", "corrupted", "--gvtd-censor")}


def test_the_record_counts_each_instant_once():
    raw = mark_unselected(_raw(600), [(0, 300)])
    add_bad_spans(raw, [(290.0, 30.0)], "BAD_gvtd")
    assert excluded_time(raw) == {
        "total_s": 600.0, "kept_s": 290.0,
        "unselected_frac": 0.5, "corrupted_frac": 0.05,
        "n_spans": 2}


# ---- table ----

@pytest.mark.parametrize("text, message", [
    ("onset\tduration\nabc\t10\n", r"line 2: onset 'abc' is not a number"),
    ("onset\tduration\n5\t0\n", r"line 2: duration '0' is not a positive number"),
    ("onset\tduration\n-1\t10\n", r"line 2: onset -1 s lies before the recording starts"),
    ("subject\tonset\tduration\n01\t0\t10\n", r"unknown columns \['subject'\]"),
    ("onset\n0\n", r"needs a duration column"),
])
def test_a_malformed_row_is_refused(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        _keep_spans_for(str(_table(tmp_path, text)), "01", {"subject": "01"})


def test_rows_select_recordings_as_the_bad_channel_table_does(tmp_path):
    path = _table(tmp_path, "participant_id\ttask\tonset\tduration\n"
                            "\ttask-rest\t0\t100\n"
                            "sub-01\trest\t200\t50\n"
                            "02\t\t10\t20\n")
    rest = {"subject": "01", "task": "rest"}
    assert _keep_spans_for(str(path), "01", rest) == [(0.0, 100.0), (200.0, 50.0)]
    assert _keep_spans_for(str(path), "03", {**rest, "subject": "03"}) == [(0.0, 100.0)]
    assert _keep_spans_for(str(path), "01", {**rest, "task": "tapping"}) == []
    assert _keep_spans_for(None, "01", rest) == []


def test_a_row_starting_past_a_recording_or_matching_none_stops_the_run(mini_bids, tmp_path):
    layout = get_layout(mini_bids)
    with pytest.raises(SystemExit, match=r"line 3: onset 100000 s lies past the end of"):
        _refuse_bad_keep_span_rows(str(_table(tmp_path, "task\tonset\tduration\nrest\t0\t10\n"
                                              "tapping\t100000\t10\n")), layout)
    with pytest.raises(SystemExit, match=r"line 2 \(task-motor\) matches no recording"):
        _refuse_bad_keep_span_rows(str(_table(tmp_path, "task\tonset\tduration\n"
                                              "motor\t0\t10\n")), layout)
    with pytest.raises(SystemExit, match="not found"):
        _refuse_bad_keep_span_rows(str(tmp_path / "missing.tsv"), layout)


# ---- screening ----

def test_a_window_touching_a_span_is_not_counted():
    assert _counted([5.0, 15.0, 25.0], 5.0, None, [(18.0, 19.0)]).tolist() == [True, False, True]
    # a scope is still decided on the centre
    assert _counted([5.0, 15.0, 25.0], 5.0, [("a", 0.0, 12.0)], []).tolist() == [True, False,
                                                                                    False]


def test_the_pre_check_and_the_screening_count_the_same_windows():
    od = intensity_to_od(mark_unselected(fingerprint_raw("01", "tapping")[0], [(33.0, 200.0)]))
    got = coupled_windows(od, 0.8, 1.9, 0.75, 0.12)
    centers, counted = counted_screen_windows(od)
    assert np.array_equal(centers, got["centers"])
    expected = got["mask"][:, counted].mean(axis=1)
    assert [got["fractions"][ch] for ch in od.ch_names] == expected.tolist()
    assert counted.sum() == 19        # 33-233 s holds windows 40-230 s whole


def test_without_spans_the_share_is_the_plain_window_mean():
    od = intensity_to_od(fingerprint_raw("01", "tapping")[0])
    got = coupled_windows(od, 0.8, 1.9, 0.75, 0.12)
    assert [got["fractions"][ch] for ch in od.ch_names] == got["mask"].mean(axis=1).tolist()


@pytest.mark.parametrize("scope", ["run", "task"])
def test_unselected_time_does_not_decide_the_channel_set(scope):
    raw, truth = fingerprint_raw("01", "tapping", blocks=True)
    decoupled = truth.block_bad[0]        # decoupled 290-370 s, inside the second block

    def rejected(keep):
        od = intensity_to_od(mark_unselected(raw.copy(), keep))
        _, bads, *_ = mark_bad_channels(od, 0.75, 0.8, 1.9, psp_threshold=0.12,
                                        min_good_frac=0.85, screen_scope=scope)
        return {b.split()[0] for b in bads}

    assert decoupled in rejected([])
    assert decoupled not in rejected([(0.0, 280.0)])


def test_fewer_than_two_windows_left_stops_the_run():
    od = intensity_to_od(mark_unselected(fingerprint_raw("01", "tapping")[0], [(100.0, 15.0)]))
    # 100-115 s holds one whole window, 100-110 s
    with pytest.raises(StageError, match=r"has 1 window\(s\) of 10 s to count and needs 2 "
                                         r"\(15 s of the recording"):
        mark_bad_channels(od, 0.75, 0.8, 1.9)


# ---- the whole run ----

def _prep(raw, out_dir, keep):
    config = PrepConfig(subject="01", dpf=list(DPF), sci_threshold=0.75, cardiac_l_freq=0.8,
                        cardiac_h_freq=1.9, resp_l_freq=0.15, resp_h_freq=0.45,
                        motion_correction="tddr", gvtd_censor="long", keep_spans=keep)
    run_prep(raw.copy(), config, out_dir, source_entities={"task": "tapping"})
    return {p.name.split("_desc-")[1]: p for p in out_dir.rglob("*_nirs.snirf")}


def test_a_stretch_covering_the_whole_recording_changes_nothing(tmp_path):
    raw = fingerprint_raw("01", "tapping")[0]
    whole = _prep(raw, tmp_path / "whole", [])
    covered = _prep(raw, tmp_path / "covered", [(0.0, 1000.0)])
    assert whole.keys() == covered.keys()
    for desc, path in whole.items():
        a, b = read_snirf(path), read_snirf(covered[desc])
        assert np.array_equal(a.get_data(), b.get_data()), desc
        assert a.info["bads"] == b.info["bads"]
        assert a.annotations.description.tolist() == b.annotations.description.tolist()
        sidecar = lambda p: json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))  # noqa: E731
        assert sidecar(path)["excluded_spans"] == sidecar(covered[desc])["excluded_spans"]


def test_a_run_with_keep_spans_through_the_cli_and_its_script(tmp_path):
    bids, truth = make_fingerprint_dataset(tmp_path, blocks=True)
    table = _table(tmp_path, "task\tonset\tduration\ntask-tapping\t0\t280\n")
    out, script_out = tmp_path / "out", tmp_path / "script"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--min-good-frac", "0.85",
                  "--keep-spans", str(table), "--no-report", "--skip-bids-validation"])

    (od,) = out.rglob("*desc-od_nirs.json")
    assert json.loads(od.read_text(encoding="utf-8"))["excluded_spans"] == [
        {"onset": 280.0, "duration": 120.0, "description": UNSELECTED, "kind": "unselected",
         "source": "--keep-spans"}]
    (sci,) = out.rglob("*desc-sci_nirs.json")
    bads = json.loads(sci.read_text(encoding="utf-8"))["bad_channels"]
    assert truth.block_bad[0] not in {b.split()[0] for b in bads}
    (record,) = out.rglob("*desc-sqm_qc.json")
    excluded = read_record(record)["excluded"]
    assert excluded["kept_s"] == 280.0
    assert excluded["unselected_frac"] == pytest.approx(0.3)

    script = (out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({script_out.as_posix()!r})",
                    script, count=1, flags=re.M)
    (tmp_path / "script.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(tmp_path / "script.py")], check=True, cwd=tmp_path)
    (script_sci,) = script_out.rglob("*desc-sci_nirs.snirf")
    (cli_sci,) = out.rglob("*desc-sci_nirs.snirf")
    for path in (script_sci, cli_sci):
        assert bad_spans(read_snirf(path)) == [(280.0, 400.0, UNSELECTED)]
    assert read_snirf(script_sci).info["bads"] == read_snirf(cli_sci).info["bads"]
