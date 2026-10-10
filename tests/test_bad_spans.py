"""`BAD_` spans are censoring marks, never conditions or task time, and no stage may lose them."""

import re
import subprocess
import sys
from datetime import datetime, timezone

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.cli.workflows import run_participant_level
from nirspipe.exceptions import AlignmentError
from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.glm import run_glm_pipeline
from nirspipe.pipeline.censor_fill import corrupted_mask
from nirspipe.pipeline.glm_censored import censor_summary, kept_frames
from nirspipe.pipeline.hyper.alignment import align_recordings, onset_residuals
from nirspipe.pipeline.prep_pipeline import intensity_to_od, od_to_haemo
from nirspipe.pipeline.restingstate import compute_alff
from nirspipe.qc.common.figure_io import extract_markers
from nirspipe.qc.common.windows import condition_windows
from nirspipe.qc.figures.subject.raw_figures import _topo_layers, build_channel_figure
from nirspipe.qc.metrics import gvtd_censor_spans
from nirspipe.qc.metrics._helpers import epochable_events, long_short_channels
from nirspipe.qc.metrics.aggregate import comparable_stage_metrics
from nirspipe.qc.metrics.haemo import _band_mean
from nirspipe.qc.metrics.windowed import task_scope_windows
from nirspipe.qc.subject.sqm_record import _condition_cnr
from nirspipe.utils.lineage import lineage_of
from nirspipe.utils.run_script import write_run_script

from tests._synth import synth_raw

SFREQ = 10.0


def _hbo_raw(data, onsets=(), durations=(), descs=()):
    data = np.atleast_2d(data)
    names = [f"S{i}_D{i} hbo" for i in range(1, data.shape[0] + 1)]
    raw = mne.io.RawArray(data, mne.create_info(names, SFREQ, "hbo"), verbose="error")
    # without a date mne reads a cropped copy's annotations from the crop, not the recording
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations(list(onsets), list(durations), list(descs)))
    return raw


# ---- GLM events table ----

def test_a_bad_row_in_an_events_table_is_not_a_condition(tmp_path):
    raw = synth_raw("01", "tapping", duration=120.0, bad_pair=None, motion_onset=None)
    haemo = od_to_haemo(intensity_to_od(raw), [6.0])
    path = tmp_path / "events.tsv"
    pd.DataFrame({"onset": [30.0, 60.0], "duration": [5.0, 40.0],
                  "trial_type": ["tap", "BAD_gvtd"]}).to_csv(path, sep="\t", index=False)
    *_, dm, resid = run_glm_pipeline(
        haemo, stim_dur=None, hrf_model="spm", noise_model="ols", drift_model="polynomial",
        high_pass=None, drift_order=1, fir_delays=None, events_path=str(path))
    assert "BAD_gvtd" not in dm.columns
    assert lineage_of(resid).params["conditions"] == ["tap"]


# ---- screening scope ----

def test_task_scope_leaves_out_a_long_bad_span():
    raw = _hbo_raw(np.zeros(int(SFREQ * 600)), [20.0, 400.0], [300.0, 60.0], ["rest", "BAD_gvtd"])
    assert task_scope_windows(raw) == [("rest", 20.0, 320.0)]


# ---- per-condition CNR ----

def test_a_bad_span_reaching_into_a_condition_still_rejects_its_epochs():
    rng = np.random.default_rng(0)
    # epochs run 5 s before to 15 s after each tap, so the span covers the first two
    haemo = _hbo_raw(rng.standard_normal((2, int(SFREQ * 200))) * 1e-6,
                     [50.0, 65.0, 85.0, 105.0, 125.0], [45.0, 0.0, 0.0, 0.0, 0.0],
                     ["BAD_gvtd", "tap", "tap", "tap", "tap"])
    assert _condition_cnr(haemo, 65.0, 125.0)["cnr_n_epochs"] == 2


# ---- ALFF ----

def test_an_all_nan_channel_has_no_falff():
    t = np.arange(int(SFREQ * 100)) / SFREQ
    data = np.vstack([np.sin(2 * np.pi * 0.05 * t), np.full(t.size, np.nan)])
    out = compute_alff(_hbo_raw(data), low_pass=0.08, high_pass=0.01)
    assert np.isnan(out["falff"].iloc[1])


# ---- trials ----

def test_a_run_marked_only_with_bad_spans_has_nothing_to_epoch():
    raw = _hbo_raw(np.zeros(int(SFREQ * 100)), [10.0], [20.0], ["BAD_gvtd"])
    events, _ = epochable_events(raw, -5.0, 25.0)
    assert len(events) == 0


# ---- report stage metrics ----

def _noise_haemo(spans=()):
    data = np.random.default_rng(0).standard_normal((2, int(SFREQ * 300))) * 1e-6
    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], SFREQ, ["hbo", "hbr"])
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations([s[0] for s in spans], [s[1] for s in spans],
                                        ["BAD_unselected"] * len(spans)))
    return raw


def _stage_metrics(raw):
    return comparable_stage_metrics([("a", raw), ("b", raw.copy())], 0.01, None,
                                    0.7, 2.0, 0.1, 0.5)


@pytest.mark.parametrize("onset", [0.0, 1e-12])      # a cropped cut lands a hair past 0
def test_stage_metrics_of_a_cut_wholly_under_a_bad_span_leave_out_the_spectra(onset):
    metrics = _stage_metrics(_noise_haemo([(onset, 300.0)]))
    assert metrics["no_kept_time"] == ["a", "b"]
    assert "resp_band_power_hbo" not in metrics["quality"]
    assert not metrics["removed"]
    assert "hbo_hbr_corr_mean" in metrics["quality"]


def test_stage_spectra_without_a_span_are_unchanged():
    raw = _noise_haemo()
    metrics = _stage_metrics(raw)
    assert metrics["no_kept_time"] == []
    psd = raw.compute_psd(verbose=False)
    freqs, hbo = psd.freqs, psd.get_data(picks="hbo")
    assert metrics["quality"]["resp_band_power_hbo"][0] == _band_mean(freqs, hbo, 0.1, 0.5)
    assert metrics["removed"]["drift_band_power_hbo"][0] == _band_mean(freqs, hbo, 0.0, 0.01)



def _mne_kept(raw):
    _, times = raw.get_data(reject_by_annotation="omit", return_times=True, verbose=False)
    kept = np.zeros(raw.n_times, dtype=bool)
    kept[np.round(times * raw.info["sfreq"]).astype(int)] = True
    return kept


def test_kept_frames_round_span_edges_as_mne_does():
    raw = _hbo_raw(np.zeros(int(SFREQ * 100)), [20.04], [10.0], ["BAD_x"])
    np.testing.assert_array_equal(kept_frames(raw), _mne_kept(raw))


def test_every_span_mask_takes_mne_frames_on_a_cropped_odd_rate_recording():
    sfreq = 4.975124378109454
    info = mne.create_info(["S1_D1 hbo"], sfreq, "hbo")
    raw = mne.io.RawArray(np.zeros((1, 5000)), info, verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations([100.13, 300.0, 512.77], [50.31, 400.0, 33.33],
                                        ["BAD_gvtd", "BAD_unselected", "BAD_gvtd"]))
    for cut in (raw, raw.copy().crop(120.0, 900.0)):
        kept = _mne_kept(cut)
        np.testing.assert_array_equal(kept_frames(cut), kept)
        corrupted = corrupted_mask(cut)
        assert not (corrupted & kept).any()
        summary = censor_summary(cut, kept_frames(cut))
        assert summary["kept_frames"] == kept.sum()
        assert summary["censored_corrupted_frac"] == corrupted.mean()


# ---- condition windows ----

def _joined(descs):
    raw = _hbo_raw(np.zeros(int(SFREQ * 600)), [50.0, 400.0], [100.0, 100.0], descs)
    return mne.concatenate_raws([raw.copy().crop(40, 160), raw.copy().crop(390, 510)],
                                verbose="error")


def test_a_join_between_segments_is_not_a_condition():
    joined = _joined(["rest", "task"])
    assert [label for label, *_ in condition_windows(joined, min_duration=0.0)] == ["rest", "task"]


def test_two_joined_members_without_a_shared_trigger_are_not_aligned():
    raws = {"01": _joined(["a1", "a2"]), "02": _joined(["b1", "b2"])}
    with pytest.raises(AlignmentError):
        align_recordings(raws, task="hold")


def test_a_join_has_no_row_in_the_dyad_onset_table():
    raws = {"sub-01": _joined(["rest", "task"]), "sub-02": _joined(["rest", "task"])}
    rows = onset_residuals(raws, ["sub-01", "sub-02"])
    assert [r["condition"] for r in rows] == ["rest", "task"]


# ---- exported run script ----

def _gvtd_spans(out_dir):
    (path,) = out_dir.rglob("*desc-sci*_nirs.snirf")
    ann = read_snirf(path).annotations
    return [(round(o, 3), round(d, 3)) for o, d, desc
            in zip(ann.onset, ann.duration, ann.description) if desc == "BAD_gvtd"]


def test_the_exported_script_censors_what_the_run_censored(mini_bids, tmp_path):
    cli_out, script_out = tmp_path / "cli", tmp_path / "script"
    args = dict(
        analysis_level="participant", session_label=None, task_label=["tapping"],
        bids_filter_file=None, work_dir=None, verbose=False, skip_bids_validation=True,
        ignore=None, n_jobs=1, no_report=True, mode=None,
        dpf=[6.0, 6.0], sci_threshold=0.8, motion_correction="tddr",
        cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5,
        # a line low enough that the synthetic run has a few short spans to lose
        gvtd_censor="long", gvtd_censor_n_std=5.0, gvtd_min_epoch_s=30.0,
        bids_dir=mini_bids, output_dir=cli_out, participant_label=["01"])
    original = sys.argv
    sys.argv = ["nirspipe", str(mini_bids), str(cli_out), "participant"]
    try:
        run_participant_level(args)
    finally:
        sys.argv = original

    script = (cli_out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({script_out.as_posix()!r})",
                    script, count=1, flags=re.M)
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=tmp_path)

    run_spans = _gvtd_spans(cli_out)
    assert run_spans
    assert _gvtd_spans(script_out) == run_spans


def test_the_exported_script_screens_as_the_run_screened(tmp_path):
    # the synthetic channels are bimodal, so no setting of these moves the rejected set and
    # only the script text can show whether they reach the call
    args = {"bids_dir": str(tmp_path), "dpf": [6.0], "sci_threshold": 0.8,
            "motion_correction": "tddr", "cardiac_l_freq": 0.7, "cardiac_h_freq": 1.5,
            "resp_l_freq": 0.2, "resp_h_freq": 0.5,
            "psp_threshold": 0.05, "min_good_frac": 0.6, "screen_scope": "task"}
    write_run_script(args, "01", "20261009_120000", tmp_path)
    script = (tmp_path / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    for constant in ("PSP_THRESHOLD  = 0.05", "MIN_GOOD_FRAC  = 0.6", "SCREEN_SCOPE   = 'task'"):
        assert constant in script
    call = script[script.index("mark_bad_channels(\n"):script.index("save_step(raw_od, \"sci\"")]
    for key in ("PSP_THRESHOLD", "MIN_GOOD_FRAC", "SCREEN_SCOPE"):
        assert f"={key}," in call


# ---- GVTD censoring ----

def test_a_rejected_channel_does_not_censor_the_run():
    od = intensity_to_od(synth_raw("01", "rest", bad_pair=None, motion_onset=None))
    long_names, _ = long_short_channels(od, None)
    noisy = long_names[:2]
    rows = [od.ch_names.index(c) for c in noisy]
    for onset in (60, 140, 220):
        start = int(onset * od.info["sfreq"])
        od._data[rows, start:start + int(2 * od.info["sfreq"])] += 0.5
    with_noisy = gvtd_censor_spans(od, n_std=10.0)[0]
    od.info["bads"] = noisy
    without = gvtd_censor_spans(od.copy().drop_channels(noisy), n_std=10.0)[0]

    assert with_noisy != without          # the spikes would censor the run if counted
    assert gvtd_censor_spans(od, n_std=10.0)[0] == without


# ---- evoked figures ----

def _taps_under_a_bad_span():
    # four taps; the third one's epoch (-5 to 25 s) overlaps the BAD_ span and holds a bump
    data = np.zeros((2, int(SFREQ * 200)))
    data[:, int(SFREQ * 100):int(SFREQ * 110)] = 1e-3
    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], SFREQ, ["hbo", "hbr"])
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations([20.0, 50.0, 100.0, 140.0, 95.0], [0.0] * 4 + [20.0],
                                        ["tap"] * 4 + ["BAD_gvtd"]))
    return raw


def test_the_channel_epoch_panel_leaves_out_a_trial_under_a_bad_span():
    raw = _taps_under_a_bad_span()
    *_, epoch_fig = build_channel_figure(raw, extract_markers(raw), "S1_D1", max_ts_pts=4000)
    assert epoch_fig.layout.annotations[0].text == "tap (n=3)"


def test_the_evoked_topo_layer_leaves_out_a_trial_under_a_bad_span():
    raw = _taps_under_a_bad_span()
    layers = _topo_layers(raw, extract_markers(raw), [0, 1], 4000, -5.0, 25.0)
    assert layers[0]["n"] == 3
