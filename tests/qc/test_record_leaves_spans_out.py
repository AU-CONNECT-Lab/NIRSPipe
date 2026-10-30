"""A disturbance inside a stretch the user marked moves no number of the pipeline's record or per-trial table."""

import json
from datetime import datetime, timezone

import mne
import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.pipeline.prep_pipeline import PrepConfig
from nirspipe.qc.subject.report import _section_trial_qc

from tests._synth import _channel_layout, _write_dataset_root, _write_subject

MEAS_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
SFREQ = 10.0
DURATION = 300.0
PULSE_HZ = 1.2
STRETCH = (120.0, 60.0)
FLAGS = ("--dpf", "6", "--sci-threshold", "0.5", "--cardiac-l-freq", "0.7",
         "--cardiac-h-freq", "2.0", "--resp-l-freq", "0.1", "--resp-h-freq", "0.5",
         "--no-report", "--skip-bids-validation")
RAW_KEYS = ("heart_rate", "cp_mean", "cardiac_ratio")


def _recording(disturbed):
    """A shared pulse at both wavelengths, and with ``disturbed`` a louder 1.7 Hz one inside STRETCH."""
    names, locs, _ = _channel_layout(4, short=False)
    rng = np.random.default_rng(3)
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    onset, length = STRETCH
    inside = (t >= onset) & (t < onset + length)
    data = np.empty((len(names), t.size))
    for i in range(len(names)):
        signal = 0.02 * np.sin(2 * np.pi * PULSE_HZ * t) + 0.002 * rng.normal(size=t.size)
        if disturbed:
            signal = signal + inside * (0.15 * np.sin(2 * np.pi * 1.7 * t))
        data[i] = 0.1 * (1.0 + signal)
    info = mne.create_info(names, SFREQ, ["fnirs_cw_amplitude"] * len(names))
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(MEAS_DATE)
    raw.info["subject_info"] = {"his_id": "01"}
    raw.set_annotations(mne.Annotations([20.0, 200.0], [40.0, 40.0], ["task", "task"]))
    return raw


def _spans_table(path, rows):
    path.write_text("onset\tduration\n" + "".join(f"{o:g}\t{d:g}\n" for o, d in rows),
                    encoding="utf-8")
    return path


def _raw_section(root, name, disturbed, *flags):
    bids = root / f"bids_{name}"
    _write_dataset_root(bids, ["01"])
    _write_subject(bids, "01", "tapping", _recording(disturbed))
    out = root / f"out_{name}"
    run_cli.main([str(bids), str(out), "participant", *FLAGS, *flags])
    (path,) = out.rglob("*_desc-sqm_qc.json")
    return json.loads(path.read_text(encoding="utf-8"))["raw"]


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    root = tmp_path_factory.mktemp("spans")
    bad = _spans_table(root / "bad.tsv", [STRETCH])
    keep = _spans_table(root / "keep.tsv", [(0.0, STRETCH[0]),
                                            (sum(STRETCH), DURATION - sum(STRETCH))])
    return {"bad_clean": _raw_section(root, "bad_clean", False, "--bad-spans", str(bad)),
            "bad_dirty": _raw_section(root, "bad_dirty", True, "--bad-spans", str(bad)),
            "keep_dirty": _raw_section(root, "keep_dirty", True, "--keep-spans", str(keep))}


@pytest.mark.parametrize("key", RAW_KEYS)
def test_a_disturbance_inside_bad_spans_moves_no_raw_number(records, key):
    assert records["bad_dirty"][key] == pytest.approx(records["bad_clean"][key], rel=0.02)


@pytest.mark.parametrize("key", RAW_KEYS)
def test_a_disturbance_outside_keep_spans_moves_no_raw_number(records, key):
    assert records["keep_dirty"][key] == pytest.approx(records["bad_clean"][key], rel=0.02)


def test_the_heart_rate_stays_the_recording_s_own_pulse(records):
    assert records["bad_dirty"]["heart_rate"] == pytest.approx(PULSE_HZ, abs=0.05)


def _trial_table(tmp_path, raw, name):
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.5, cardiac_l_freq=0.7,
                        cardiac_h_freq=2.0, resp_l_freq=0.1, resp_h_freq=0.5,
                        epoch_tmin=-30.0, epoch_tmax=120.0, bad_spans=[STRETCH])
    out = _section_trial_qc(raw, config, "01", [], tmp_path, lambda key, **_: f"{name}{key}.html",
                            [])
    return out["trial_qc_rows"]


def test_a_trial_over_a_bad_span_scores_as_if_the_disturbance_never_happened(tmp_path):
    clean = _trial_table(tmp_path, _recording(False), "clean")
    dirty = _trial_table(tmp_path, _recording(True), "dirty")
    assert len(clean) == len(dirty) == 2
    for (_, _, want), (_, _, got) in zip(clean, dirty):
        assert got["cp_mean"] == pytest.approx(want["cp_mean"], rel=0.02)
