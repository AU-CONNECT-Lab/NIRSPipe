"""External confound regressors read out of a SNIRF file's aux group.

Three contracts live here, and each exists because getting it wrong is silent.

Reading: MNE does not touch the aux group and the helper in mne-nirs raises on two shapes
the format permits, a scalar channel name and a (T, 1) data array. Both appear in files from
real devices, so both are built here rather than assumed away.

Rate: aux is sampled above the optical rate, so reaching the data's time axis is decimation.
Interpolating without a low-pass first does not fail, it quietly folds high-frequency motion
into the band a confound regression then works in. The test is a tone above the target
Nyquist: naive interpolation returns it at full amplitude wearing a different frequency.

Band: a regressor carrying variance the data no longer has inflates the denominator of its
own beta and puts that variance back into the residual. Short channels avoid this by riding
through the same filter as the data. Aux comes from outside and has to be filtered on
purpose, which is what `_aux_regressors` does.

Crop: every SNIRF the package writes goes through a Raw, and a Raw cannot hold an aux
channel, so a cropped recording arrives at postprocessing with nothing to regress unless the
aux group is put back explicitly. The last section is what keeps that from regressing.
"""

import gzip
import logging
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.io.auxiliary import (
    TIME_COLUMN,
    aux_table_path,
    find_aux_table,
    read_aux_snirf,
    read_aux_table,
    resample_to_grid,
    write_aux_table,
    write_aux_window,
)
from fnirs_pipe.io.snirf import write_snirf
from fnirs_pipe.pipeline.crop import crop_snirf_from_path
from fnirs_pipe.pipeline.glm import _aux_regressors
from fnirs_pipe.pipeline.post_pipeline import (
    PostConfig,
    _has_confounds,
    _warn_unmatched_design_band,
)
from tests._synth import synth_raw

AUX_FS = 98.0        # deliberately not a whole multiple of the optical rate
DATA_FS = 10.0


def _haemo(duration: float = 120.0) -> mne.io.Raw:
    raw = synth_raw("01", "hold", duration=duration, motion_onset=None, bad_pair=None)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    return mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)


def _snirf_with_aux(tmp_path: Path, channels: dict[str, np.ndarray], fs: float = AUX_FS,
                    scalar_name: bool = True, column_vector: bool = True) -> Path:
    """A real snirf with an aux group bolted on, shaped the way devices actually write it.

    `scalar_name` writes the channel name as an HDF5 scalar rather than a 1-element array,
    and `column_vector` writes the samples as (T, 1) rather than (T,). Both are legal and
    both are what the reader in mne-nirs cannot open.
    """
    path = tmp_path / "sub-01_task-hold_nirs.snirf"
    write_snirf(_haemo(20.0), path)

    n = max(len(v) for v in channels.values())
    times = np.arange(n) / fs
    with h5py.File(path, "a") as handle:
        for index, (name, values) in enumerate(channels.items(), start=1):
            group = handle["nirs"].create_group(f"aux{index}")
            group.create_dataset("name", data=name.encode() if scalar_name else [name.encode()])
            group.create_dataset(
                "dataTimeSeries", data=values[:, None] if column_vector else values)
            group.create_dataset("time", data=times[:len(values)])
            group.create_dataset("dataUnit", data=b"m/s^2")
    return path


# ---- reading ----

def test_a_scalar_name_and_a_column_vector_are_both_read(tmp_path):
    values = np.linspace(0.0, 1.0, 400)
    path = _snirf_with_aux(tmp_path, {"ACC_X": values})

    times, data, units = read_aux_snirf(path)
    assert list(data) == ["ACC_X"]
    assert data["ACC_X"].shape == (400,)          # the trailing axis is squeezed away
    assert units["ACC_X"] == "m/s^2"
    assert times["ACC_X"][1] == pytest.approx(1 / AUX_FS)


def test_a_one_dimensional_series_and_an_array_name_read_the_same(tmp_path):
    values = np.linspace(0.0, 1.0, 400)
    path = _snirf_with_aux(tmp_path, {"ACC_X": values}, scalar_name=False, column_vector=False)
    _, data, _ = read_aux_snirf(path)
    assert data["ACC_X"] == pytest.approx(values)


def test_a_file_with_no_aux_group_reads_as_empty(tmp_path):
    path = tmp_path / "sub-01_task-hold_nirs.snirf"
    write_snirf(_haemo(20.0), path)
    assert read_aux_snirf(path) == ({}, {}, {})


def test_a_channel_whose_timestamps_do_not_match_its_samples_is_dropped(tmp_path):
    path = _snirf_with_aux(tmp_path, {"GOOD": np.zeros(400)})
    with h5py.File(path, "a") as handle:
        del handle["nirs"]["aux1"]["time"]
        handle["nirs"]["aux1"].create_dataset("time", data=np.arange(10) / AUX_FS)

    _, data, _ = read_aux_snirf(path)
    assert data == {}


# ---- rate ----

def test_interpolating_without_a_low_pass_would_alias_and_this_does_not():
    """A tone above the target Nyquist has to be removed, not folded into the passband."""
    t_src = np.arange(0, 60.0, 1 / AUX_FS)
    t_dst = np.arange(0, 60.0, 1 / DATA_FS)
    # 7 Hz against a 5 Hz target Nyquist, so naive interpolation returns it as 3 Hz:
    # inside the band, at full amplitude, and indistinguishable from real slow motion
    tone = np.sin(2 * np.pi * 7.0 * t_src)

    naive = np.interp(t_dst, t_src, tone)
    clean = resample_to_grid(t_src, tone, t_dst)

    assert naive.std() > 0.5                          # arrives at nearly full amplitude
    assert clean.std() < 0.05                         # and is gone once filtered first


def test_a_tone_inside_the_target_band_survives():
    t_src = np.arange(0, 60.0, 1 / AUX_FS)
    t_dst = np.arange(0, 60.0, 1 / DATA_FS)
    tone = np.sin(2 * np.pi * 0.5 * t_src)

    kept = resample_to_grid(t_src, tone, t_dst)
    assert kept.std() == pytest.approx(np.sqrt(0.5), abs=0.05)


def test_going_up_in_rate_is_interpolation_alone():
    t_src = np.arange(0, 10.0, 1 / DATA_FS)
    # inside the source span: past its last stamp np.interp holds rather than extrapolates
    t_dst = np.arange(0, t_src[-1], 1 / AUX_FS)
    ramp = t_src.copy()
    assert resample_to_grid(t_src, ramp, t_dst) == pytest.approx(t_dst, abs=1e-6)


def test_the_recorded_timestamps_are_used_rather_than_sample_counts():
    """A clock that drifts places the last sample somewhere sample counting would not.

    The source grid runs 2% slow, so its last stamp sits well past where a constant rate
    would put it. Reading a ramp back at the true stamps is what proves the interpolation
    went through `t_src` instead of assuming it was uniform.
    """
    n = 2000
    t_true = np.arange(n) / AUX_FS * 1.02
    ramp = t_true.copy()
    t_dst = np.linspace(0, t_true[-1], 200)

    out = resample_to_grid(t_true, ramp, t_dst)
    # loose enough for the anti-alias filter's own edge behaviour on a ramp, tight enough
    # that the 2% a constant-rate assumption would be out by could not pass
    assert out == pytest.approx(t_dst, abs=0.02)


# ---- the table on disk ----

def test_the_table_is_named_after_the_run_and_not_the_stage(tmp_path):
    preproc = tmp_path / "sub-01_ses-a_task-hold_run-2_desc-preproc_nirs.snirf"
    assert aux_table_path(preproc).name == "sub-01_ses-a_task-hold_run-2_desc-aux_timeseries.tsv.gz"
    # every stage of the run finds the same table, which is what lets post read what prep wrote
    errts = tmp_path / "sub-01_ses-a_task-hold_run-2_desc-errts_nirs.snirf"
    assert aux_table_path(errts) == aux_table_path(preproc)


def test_writing_then_reading_the_table_returns_the_samples(tmp_path):
    values = {"ACC_X": np.sin(np.linspace(0, 20, 500)), "ACC_Y": np.cos(np.linspace(0, 20, 500))}
    source = _snirf_with_aux(tmp_path, values)
    out = aux_table_path(tmp_path / "sub-01_task-hold_desc-preproc_nirs.snirf")

    table, facts = write_aux_table(source, out)
    assert list(table.columns) == [TIME_COLUMN, "ACC_X", "ACC_Y"]
    assert facts["SamplingFrequency"] == pytest.approx(AUX_FS, abs=0.01)
    assert facts["Units"] == {"ACC_X": "m/s^2", "ACC_Y": "m/s^2"}
    assert facts["timestamp_column"] == TIME_COLUMN

    back = read_aux_table(out)
    assert back["ACC_X"].to_numpy() == pytest.approx(values["ACC_X"], abs=1e-5)
    assert find_aux_table(tmp_path / "sub-01_task-hold_desc-preproc_nirs.snirf") == out


def test_a_recording_with_no_aux_writes_no_table(tmp_path):
    source = tmp_path / "sub-01_task-hold_nirs.snirf"
    write_snirf(_haemo(20.0), source)
    out = aux_table_path(tmp_path / "sub-01_task-hold_desc-preproc_nirs.snirf")

    assert write_aux_table(source, out) is None
    assert not out.exists()
    assert find_aux_table(tmp_path / "sub-01_task-hold_desc-preproc_nirs.snirf") is None


# ---- regressors ----

def _write_table(path: Path, columns: dict[str, np.ndarray], t: np.ndarray) -> Path:
    frame = pd.DataFrame({TIME_COLUMN: t, **columns})
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        frame.to_csv(handle, sep="\t", index=False, float_format="%.6g")
    return path


@pytest.fixture
def haemo_and_table(tmp_path):
    """A recording and an aux table over the same span, one slow column and one fast one."""
    haemo = _haemo(120.0)
    t = np.arange(0, haemo.times[-1], 1 / (haemo.info["sfreq"] * 9.7))
    path = _write_table(
        tmp_path / "aux.tsv.gz",
        {"SLOW": np.sin(2 * np.pi * 0.05 * t), "FAST": np.sin(2 * np.pi * 2.0 * t)},
        t,
    )
    return haemo, path


def test_every_regressor_lands_on_the_data_length_and_is_standardised(haemo_and_table):
    haemo, path = haemo_and_table
    cols = _aux_regressors(haemo, path, None, (0.01, 0.2))

    assert sorted(cols) == ["aux_FAST", "aux_SLOW"]      # prefixed, so no design matrix clash
    for values in cols.values():
        assert len(values) == haemo.n_times
        assert values.mean() == pytest.approx(0.0, abs=1e-9)
        assert values.std() == pytest.approx(1.0)


def test_naming_channels_selects_them_and_an_unknown_name_is_reported(haemo_and_table, caplog):
    haemo, path = haemo_and_table
    assert sorted(_aux_regressors(haemo, path, ["SLOW"], (0.01, 0.2))) == ["aux_SLOW"]

    with caplog.at_level(logging.WARNING):
        assert _aux_regressors(haemo, path, ["NOT_A_CHANNEL"], (0.01, 0.2)) == {}
    assert "NOT_A_CHANNEL" in caplog.text


def test_a_column_with_nothing_left_in_the_band_says_so(haemo_and_table, caplog):
    """It is not dropped: the z-score would otherwise hand back unit-variance filter residue."""
    haemo, path = haemo_and_table
    with caplog.at_level(logging.WARNING):
        cols = _aux_regressors(haemo, path, ["FAST"], (0.01, 0.2))

    assert "FAST" in caplog.text and "variance" in caplog.text
    assert cols["aux_FAST"].std() == pytest.approx(1.0)


def test_no_band_leaves_the_regressors_unfiltered(haemo_and_table):
    """rest mode's broadband rerun regresses un-bandpassed data and must match it."""
    haemo, path = haemo_and_table
    with_band = _aux_regressors(haemo, path, ["FAST"], (0.01, 0.2))["aux_FAST"]
    without   = _aux_regressors(haemo, path, ["FAST"], None)["aux_FAST"]
    # both are z-scored, so they differ in shape rather than in scale
    assert np.corrcoef(with_band, without)[0, 1] < 0.5


def test_a_short_aux_table_is_flagged_rather_than_extrapolated(tmp_path, caplog):
    haemo = _haemo(120.0)
    t = np.arange(0, haemo.times[-1] / 2, 1 / (haemo.info["sfreq"] * 9.7))
    path = _write_table(tmp_path / "aux.tsv.gz", {"ACC": np.sin(2 * np.pi * 0.05 * t)}, t)

    with caplog.at_level(logging.WARNING):
        cols = _aux_regressors(haemo, path, None, (0.01, 0.2))
    assert "held constant" in caplog.text
    assert len(cols["aux_ACC"]) == haemo.n_times


# ---- wiring ----

def _config(**kwargs) -> PostConfig:
    base = dict(subject="01", cardiac_l_freq=0.7, cardiac_h_freq=1.5,
                resp_l_freq=0.1, resp_h_freq=0.5)
    return PostConfig(**{**base, **kwargs})


def test_aux_alone_is_enough_to_make_denoise_regress():
    assert _has_confounds(_config(aux=True))
    assert not _has_confounds(_config())


def test_the_design_band_warning_fires_only_when_the_drift_model_cannot_cover_it(caplog):
    matched = _config(high_pass=0.01, drift_model="cosine", drift_high_pass=0.01)
    with caplog.at_level(logging.WARNING):
        _warn_unmatched_design_band(matched)
    assert not caplog.text

    for config in (_config(high_pass=0.01, drift_model="polynomial", drift_order=1),
                   _config(high_pass=0.01, drift_model="none"),
                   # a cosine basis that stops below the filter leaves the gap between them
                   _config(high_pass=0.01, drift_model="cosine", drift_high_pass=0.005)):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _warn_unmatched_design_band(config)
        assert "underestimated" in caplog.text, config.drift_model


def test_no_bandpass_means_nothing_to_match(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_unmatched_design_band(_config(drift_model="none"))
    assert not caplog.text


# ---- surviving a crop ----

def _raw_snirf_with_aux(tmp_path: Path, duration: float, fs: float = AUX_FS) -> Path:
    """A raw-intensity recording whose one aux channel records its own timestamp.

    Making the samples equal to the time they were taken at is what lets a test say where in
    the original recording a cropped window came from: the value is the answer.
    """
    path = tmp_path / "sub-01_task-main_nirs.snirf"
    write_snirf(synth_raw("01", "rest", duration=duration, motion_onset=None, bad_pair=None),
                path)

    stamps = np.arange(0, duration, 1 / fs)
    with h5py.File(path, "a") as handle:
        group = handle["nirs"].create_group("aux1")
        group.create_dataset("name", data=b"ACCEL_X")
        group.create_dataset("dataTimeSeries", data=stamps.copy())
        group.create_dataset("time", data=stamps)
        group.create_dataset("dataUnit", data=b"m/s^2")
    return path


def test_a_cropped_recording_keeps_its_aux(tmp_path):
    """Without this the aux group is gone by the time postprocessing looks for it."""
    source = _raw_snirf_with_aux(tmp_path, 200.0)
    out = crop_snirf_from_path(source, tmp_path / "deriv", "01", tmin=50.0, tmax=150.0)[0]

    times, data, units = read_aux_snirf(out)
    assert list(data) == ["ACCEL_X"]
    assert units["ACCEL_X"] == "m/s^2"
    # rebased to the segment, like the markers and the optical data
    assert times["ACCEL_X"][0] == pytest.approx(0.0, abs=1 / AUX_FS)
    assert times["ACCEL_X"][-1] == pytest.approx(100.0, abs=1 / AUX_FS)
    # the samples still say which part of the original recording they are
    assert data["ACCEL_X"] == pytest.approx(times["ACCEL_X"] + 50.0, abs=1e-6)


def test_the_aux_rate_is_not_touched_by_the_crop(tmp_path):
    """Resampling is postprocessing's decision; baking it in here fixes one downstream rate."""
    source = _raw_snirf_with_aux(tmp_path, 200.0)
    out = crop_snirf_from_path(source, tmp_path / "deriv", "01", tmin=50.0, tmax=150.0)[0]

    times, _, _ = read_aux_snirf(out)
    assert 1 / np.median(np.diff(times["ACCEL_X"])) == pytest.approx(AUX_FS, abs=0.01)


def test_each_segment_of_a_multi_segment_crop_gets_its_own_window(tmp_path):
    source = _raw_snirf_with_aux(tmp_path, 300.0)
    segments = pd.DataFrame({"onset": [10.0, 200.0], "duration": [60.0, 60.0],
                             "task": ["early", "late"]})
    outs = crop_snirf_from_path(source, tmp_path / "deriv", "01", segments_df=segments)

    assert [p.name for p in outs] == ["sub-01_task-early_nirs.snirf",
                                      "sub-01_task-late_nirs.snirf"]
    for path, onset in zip(outs, [10.0, 200.0]):
        times, data, _ = read_aux_snirf(path)
        assert data["ACCEL_X"] == pytest.approx(times["ACCEL_X"] + onset, abs=1e-6)


def test_a_combined_crop_lays_the_windows_end_to_end(tmp_path):
    """The optical data is concatenated, so the aux has to be too or the two disagree."""
    source = _raw_snirf_with_aux(tmp_path, 300.0)
    segments = pd.DataFrame({"onset": [10.0, 200.0], "duration": [60.0, 60.0]})
    out = crop_snirf_from_path(source, tmp_path / "deriv", "01",
                               segments_df=segments, combine=True)[0]

    times, data, _ = read_aux_snirf(out)
    stamps = times["ACCEL_X"]
    assert stamps[0] == pytest.approx(0.0, abs=1 / AUX_FS)
    assert stamps[-1] == pytest.approx(120.0, abs=1 / AUX_FS)
    assert np.all(np.diff(stamps) > 0)                # one axis, not two overlapping ones
    # the join is visible in the values: the second window comes from 190 s later
    first, second = data["ACCEL_X"][stamps < 60.0], data["ACCEL_X"][stamps > 60.0]
    assert first.max() < 71.0 and second.min() > 199.0


def test_cropping_a_recording_with_no_aux_writes_no_aux(tmp_path):
    source = tmp_path / "sub-01_task-main_nirs.snirf"
    write_snirf(synth_raw("01", "rest", duration=200.0, motion_onset=None, bad_pair=None),
                source)
    out = crop_snirf_from_path(source, tmp_path / "deriv", "01", tmin=50.0, tmax=150.0)[0]
    assert read_aux_snirf(out) == ({}, {}, {})


def test_a_window_holding_no_aux_sample_is_reported_rather_than_written(tmp_path, caplog):
    source = _raw_snirf_with_aux(tmp_path, 200.0)
    out = tmp_path / "empty.snirf"
    write_snirf(synth_raw("01", "rest", duration=20.0, motion_onset=None, bad_pair=None), out)

    with caplog.at_level(logging.WARNING):
        assert write_aux_window(source, out, [(500.0, 600.0)]) == []
    assert "ACCEL_X" in caplog.text
    assert read_aux_snirf(out) == ({}, {}, {})
