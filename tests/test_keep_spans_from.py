"""The --keep-spans table from condition names (nirspipe-prep keep-spans) and --keep-spans-from MARKER."""

import json
import logging
import shutil
from datetime import datetime, timezone

import mne
import numpy as np
import pytest

from nirspipe.cli import prep as prep_cli
from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _span_rows, _spans_for, _refuse_bad_span_rows
from nirspipe.io.bids import get_layout
from nirspipe.io.snirf import read_snirf
from nirspipe.utils.spans import UNSELECTED, bad_spans, marker_onset, measured_from

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset
from tests._synth import _write_dataset_root, _write_subject, make_bids_dataset, synth_raw

SFREQ = 10.0


def _raw(seconds: float, onsets=(), durations=(), descs=()):
    raw = mne.io.RawArray(np.zeros((2, int(SFREQ * seconds))),
                          mne.create_info(["S1_D1 760", "S1_D1 850"], SFREQ, "fnirs_cw_amplitude"),
                          verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations(list(onsets), list(durations), list(descs)))
    return raw


def _table(tmp_path, text: str, name: str = "keep.tsv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def blocked(tmp_path_factory):
    """sub-01 and sub-02: task-mixed ("talk" 20-170 s, "listen" 190-340 s) and task-rest (none)."""
    return make_bids_dataset(tmp_path_factory.mktemp("blocked"), tasks=("mixed", "rest"))


def _helper(bids, out, *extra):
    prep_cli.main(["keep-spans", str(bids), str(out), "--participant-label", "01", "02",
                   "--skip-bids-validation", *extra])


# ---- the helper ----

def test_the_helper_writes_one_row_per_occurrence_and_the_table_reads_back(blocked, tmp_path):
    out = tmp_path / "keep.tsv"
    _helper(blocked, out, "--task-label", "mixed", "--conditions", "listen", "talk")

    assert out.read_text(encoding="utf-8").splitlines()[0].split("\t") == [
        "participant_id", "task", "onset", "duration"]
    got = [(r["participant_id"], r["session"], r["task"], r["run"], r["onset"], r["duration"])
           for r in _span_rows(out)]
    assert got == [(sub, "", "mixed", "", onset, 150.0)
                   for sub in ("01", "02") for onset in (20.0, 190.0)]
    assert _spans_for(str(out), "02", {"subject": "02", "task": "mixed"}) == [
        (20.0, 150.0), (190.0, 150.0)]


def test_duration_sets_every_occurrence(blocked, tmp_path):
    out = tmp_path / "keep.tsv"
    _helper(blocked, out, "--task-label", "mixed", "--conditions", "talk", "--duration", "60")
    assert [(r["onset"], r["duration"]) for r in _span_rows(out)] == [(20.0, 60.0)] * 2


def test_a_recording_without_any_named_condition_stops_the_helper(blocked, tmp_path, capsys):
    out = tmp_path / "keep.tsv"
    with pytest.raises(SystemExit):
        _helper(blocked, out, "--conditions", "talk")
    assert "sub-01_task-rest_nirs.snirf holds none of the conditions talk" in capsys.readouterr().err
    assert not out.exists()


def test_a_zero_length_occurrence_needs_a_duration(tmp_path, capsys):
    bids = tmp_path / "bids"
    _write_dataset_root(bids, ["01"])
    raw = synth_raw("01", "rest")
    raw.set_annotations(mne.Annotations([5.0, 250.0], [0.0, 0.0], ["start", "start"]))
    _write_subject(bids, "01", "rest", raw)
    out = tmp_path / "keep.tsv"
    argv = ["keep-spans", str(bids), str(out), "--participant-label", "01",
            "--conditions", "start", "--skip-bids-validation"]

    with pytest.raises(SystemExit):
        prep_cli.main(argv)
    assert "condition 'start' at 5 s has no duration; give --duration" in capsys.readouterr().err
    assert not out.exists()
    prep_cli.main([*argv, "--duration", "100"])
    assert [(r["onset"], r["duration"]) for r in _span_rows(out)] == [(5.0, 100.0),
                                                                      (250.0, 100.0)]


def test_two_recordings_the_table_cannot_tell_apart_are_warned_about(tmp_path, caplog):
    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("mixed",))
    nirs = bids / "sub-01" / "nirs"
    for suffix in (".snirf", ".json"):
        shutil.copy(nirs / f"sub-01_task-mixed_nirs{suffix}",
                    nirs / f"sub-01_task-mixed_acq-b_nirs{suffix}")
    with caplog.at_level(logging.WARNING, logger="nirspipe"):
        prep_cli.main(["keep-spans", str(bids), str(tmp_path / "keep.tsv"),
                       "--participant-label", "01", "--conditions", "talk",
                       "--skip-bids-validation"])
    assert "share every label the table has a column for" in caplog.text


# ---- the marker origin ----

def test_stretches_are_measured_from_the_first_marker():
    raw = _raw(600, [300.0, 12.5], [0.0, 0.0], ["start", "start"])
    assert marker_onset(raw, "start") == 12.5
    assert measured_from(raw, [(-10, 70), (100, 5)], "start") == [(2.5, 70.0), (112.5, 5.0)]
    # a recording no row names needs no marker
    assert measured_from(_raw(600), [], "start") == []
    with pytest.raises(ValueError, match="marker 'start' is not among the recording's"):
        measured_from(_raw(600), [(0, 10)], "start")


def test_a_negative_onset_is_read_only_when_measured_from_a_marker(tmp_path):
    path = _table(tmp_path, "onset\tduration\n-10\t70\n")
    assert _spans_for(str(path), "01", relative=True) == [(-10.0, 70.0)]
    with pytest.raises(ValueError, match="onset -10 s lies before the recording starts"):
        _spans_for(str(path), "01")


def test_the_check_before_any_subject_runs(blocked, tmp_path):
    layout = get_layout(blocked)

    def check(text):
        _refuse_bad_span_rows(str(_table(tmp_path, text)), layout, marker="talk")

    check("task\tonset\tduration\nmixed\t-10\t100\n")
    with pytest.raises(SystemExit, match=r"marker 'talk' is not among the annotations of "
                                         r"sub-0\d_task-rest_nirs.snirf, which --keep-spans "
                                         r"table .* line 2 names"):
        check("onset\tduration\n-10\t100\n")
    with pytest.raises(SystemExit, match=r"line 2: onset -30 s \(from 'talk' at 20 s\) lies "
                                         r"before the start of"):
        check("task\tonset\tduration\nmixed\t-30\t100\n")
    with pytest.raises(SystemExit, match=r"line 2: onset 380 s \(from 'talk' at 20 s\) lies "
                                         r"past the end of .* \(400 s\)"):
        check("task\tonset\tduration\nmixed\t380\t10\n")


def test_the_marker_needs_the_table(tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_cli.main([str(tmp_path / "bids"), str(tmp_path / "out"), "participant",
                      *CLI_ARGS, "--keep-spans-from", "talk"])
    assert "--keep-spans-from needs --keep-spans" in capsys.readouterr().err


# ---- the whole run ----

def test_a_run_measured_from_a_marker_through_the_cli(tmp_path):
    bids, _ = make_fingerprint_dataset(tmp_path, blocks=True)   # "cb" from 210 s
    table = _table(tmp_path, "task\tonset\tduration\ntapping\t-10\t100\n")
    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--keep-spans", str(table),
                  "--keep-spans-from", "cb", "--no-report", "--skip-bids-validation"])

    (od,) = out.rglob("*desc-od_nirs.json")
    sidecar = json.loads(od.read_text(encoding="utf-8"))
    assert [(s["onset"], s["duration"], s["source"]) for s in sidecar["excluded_spans"]] == [
        (0.0, 200.0, "--keep-spans"), (300.0, 100.0, "--keep-spans")]
    (sci,) = out.rglob("*desc-sci_nirs.json")
    params = json.loads(sci.read_text(encoding="utf-8"))["parameters"]
    assert params["keep_spans"] == [[200.0, 100.0]]
    assert params["keep_spans_from"] == "cb"
    (cli_sci,) = out.rglob("*desc-sci_nirs.snirf")
    assert bad_spans(read_snirf(cli_sci)) == [(0.0, 200.0, UNSELECTED),
                                              (300.0, 400.0, UNSELECTED)]


SYNTH_ARGS = ("--dpf", "6", "6", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
              "--cardiac-h-freq", "1.5", "--resp-l-freq", "0.2", "--resp-h-freq", "0.5",
              "--participant-label", "01", "--task-label", "tapping", "--no-report",
              "--skip-bids-validation")


def test_a_recording_no_row_names_comes_out_bit_identical(mini_bids, tmp_path):
    plain, tabled = tmp_path / "plain", tmp_path / "tabled"
    table = _table(tmp_path, "participant_id\ttask\tonset\tduration\nsub-02\ttapping\t-5\t20\n")
    run_cli.main([str(mini_bids), str(plain), "participant", *SYNTH_ARGS])
    run_cli.main([str(mini_bids), str(tabled), "participant", *SYNTH_ARGS,
                  "--keep-spans", str(table), "--keep-spans-from", "tapping"])

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
        assert sidecar(tabled)["parameters"].get("keep_spans_from") is None
