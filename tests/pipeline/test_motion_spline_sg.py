"""Spline-SG motion correction: the authors' detection, their SNR rule, the filtering, the wiring.

The package follows the authors' implementation, which departs from the paper's text, so
several tests pin the implementation on purpose: the gradient fence counts the signal's own
level at the start, the baseline-shift threshold is a median with a zero added, runs of 0.5
to 1 s are kept by amplitude alone, the SNR leaves out the first and last stretch, a ratio of
exactly 3 keeps a channel's own marks, the shorter wavelength decides whether a pair is
corrected, and every channel, bad ones included, leaves low-passed near 0.1 Hz.
"""

import json

import mne
import numpy as np
import pytest

from nirspipe.pipeline import motion
from nirspipe.pipeline.motion import (
    _spline_sg_candidates,
    _spline_sg_keep,
    _spline_sg_motion_correct,
    _spline_sg_motion_mask,
    _spline_sg_shift_threshold,
    _spline_sg_snr,
    correct_motion,
    spline_sg_cutoff,
)
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep
from nirspipe.utils.lineage import lineage_of, stage_of

from tests._synth import synth_raw

SFREQ = 10.0
N_T = 6000


def _names(n_pairs: int) -> list[str]:
    return [f"S{p + 1}_D1 {wl}" for p in range(n_pairs) for wl in (760, 850)]


def _raw(d: np.ndarray, bads: tuple[str, ...] = (), sfreq: float = SFREQ) -> mne.io.Raw:
    names = _names(d.shape[1] // 2)
    raw = mne.io.RawArray(d.T.copy(), mne.create_info(names, sfreq, "fnirs_od"), verbose="error")
    raw.info["bads"] = list(bads)
    return raw


def _with_step(seed: int = 0) -> np.ndarray:
    """Two pairs sitting at 0.3 OD with a heartbeat and noise; S1_D1 steps by 0.2 at 300 s."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_T) / SFREQ
    d = 0.3 + 0.01 * np.sin(2 * np.pi * 1.1 * t)[:, None] + 0.002 * rng.normal(size=(N_T, 4))
    d[3000:, :2] += 0.2
    return d


# ---- Detection ----

def test_the_gradient_fence_counts_the_signals_own_level_at_the_start():
    # the full convolution's first two values are -s1[0] and -s1[1]: a record sitting away
    # from zero flags its first two samples, the same record at zero does not
    noise = 0.001 * np.random.default_rng(0).normal(size=2000)
    assert {0, 1} <= set(_spline_sg_candidates(5.0 + noise, SFREQ))
    assert not {0, 1} & set(_spline_sg_candidates(noise, SFREQ))


def test_the_shift_threshold_is_the_median_with_a_zero_added():
    # the paper takes the largest 0.5 s change; the implementation the median, a zero joining
    # the pool. At 10 Hz the lag is 5 samples; the last stretch is too short to count
    x = np.r_[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 9, 9].astype(float)
    runs = [slice(0, 6), slice(6, 12), slice(12, 18), slice(18, 20)]
    assert _spline_sg_shift_threshold(x, runs, SFREQ) == 1.5


def test_runs_of_half_a_second_to_a_second_are_kept_by_amplitude_alone():
    # the paper sends spike-like runs over 0.5 s to the spline; the implementation keeps
    # spans over 0.1 s and under 0.5 s, or over 1 s
    span = np.arange(13)
    assert np.flatnonzero(_spline_sg_keep(np.zeros(13), span, 1.0, SFREQ)).tolist() == [2, 3, 4,
                                                                                          11, 12]
    assert _spline_sg_keep(np.array([2.0]), np.array([7]), 1.0, SFREQ).all()


def test_the_snr_leaves_out_the_first_and_last_long_stretch():
    alternate = np.tile([-1.0, 1.0], 20)
    a, b, c = 10 + alternate, 4 + alternate, 0.5 + alternate
    x = np.r_[a, np.zeros(30), b, c]
    runs = [slice(0, 40), slice(40, 70), slice(70, 110), slice(110, 150)]   # 30 samples is 3 s
    # b alone: mean 4 over its sample SD sqrt(40 / 39)
    assert _spline_sg_snr(x, runs, SFREQ) == pytest.approx(3.9496835316262997)
    assert np.isnan(_spline_sg_snr(x, runs[:2], SFREQ))


def test_the_snr_rule_per_pair(monkeypatch):
    n = 50
    marks = [np.arange(n) == c for c in range(10)]
    # pairs: both below 3 | second below | first below | first exactly 3 | first unusable
    snrs = [1, 2, 5, 1, 1, 5, 3, 1, np.nan, 1]
    calls = iter(zip(marks, snrs))
    monkeypatch.setattr(motion, "_spline_sg_channel_mask", lambda y, sfreq: next(calls))
    d = np.zeros((n, 10))
    d[:, 8] = 100 + np.tile([-1.0, 1.0], n // 2)          # scored on the whole record: ~100
    mask, n_below = _spline_sg_motion_mask(d, SFREQ, _names(5))
    assert n_below == 1
    assert not mask[:, :2].any()
    assert (mask[:, 3] == marks[2]).all() and (mask[:, 4] == marks[5]).all()
    assert (mask[:, 6] == marks[6]).all() and (mask[:, 7] == marks[7]).all()
    assert (mask[:, 9] == marks[8]).all()


# ---- The correction ----

def test_a_baseline_shift_is_rejoined():
    out = _spline_sg_motion_correct(_raw(_with_step()))[0].get_data().T
    for ch in (0, 1):
        assert abs(out[3200:4000, ch].mean() - out[2000:2800, ch].mean()) < 0.02
    assert abs(out[3200:4000, 2].mean() - out[2000:2800, 2].mean()) < 0.02


def _filtered_only(raw: mne.io.Raw, monkeypatch) -> np.ndarray:
    with monkeypatch.context() as m:
        m.setattr(motion, "_spline_sg_motion_mask",
                  lambda d, sfreq, ch_names: (np.zeros(d.shape, bool), 0))
        return _spline_sg_motion_correct(raw)[0].get_data().T


def test_the_shorter_wavelength_decides_whether_a_pair_is_corrected(monkeypatch):
    d = _with_step()
    plain = _filtered_only(_raw(d), monkeypatch)
    second_bad = _spline_sg_motion_correct(_raw(d, bads=("S1_D1 850",)))[0].get_data().T
    first_bad = _spline_sg_motion_correct(_raw(d, bads=("S1_D1 760",)))[0].get_data().T
    assert not np.allclose(second_bad[:, :2], plain[:, :2])
    assert (first_bad[:, :2] == plain[:, :2]).all()


def test_every_channel_leaves_low_passed_near_a_tenth_of_a_hertz():
    # zero-mean sines fail the SNR rule, so only the filtering acts; the bad channel too
    t = np.arange(N_T) / SFREQ
    d = np.column_stack([0.01 * np.sin(2 * np.pi * f * t) for f in (0.03, 0.03, 0.25, 0.25)])
    out = _spline_sg_motion_correct(_raw(d, bads=("S2_D1 760",)))[0].get_data().T
    gain = np.ptp(out[1000:5000], axis=0) / np.ptp(d[1000:5000], axis=0)
    assert (gain[:2] > 0.95).all() and (gain[2:] < 0.2).all()


@pytest.mark.parametrize("sfreq", [5.0, 7.8125, 10.0, 50.0])
def test_the_filter_cutoff_is_near_a_tenth_of_a_hertz_at_any_rate(sfreq):
    assert 0.1 < spline_sg_cutoff(sfreq) < 0.11


def test_at_or_below_four_hz_it_refuses():
    quiet = 0.01 * np.random.default_rng(0).normal(size=(600, 4))
    with pytest.raises(ValueError, match="more than 4 Hz"):
        _spline_sg_motion_correct(_raw(quiet, sfreq=4.0))
    _spline_sg_motion_correct(_raw(quiet, sfreq=4.5))


# ---- Wiring ----

def test_correct_motion_stamps_spline_sg_and_what_it_marked():
    out = correct_motion(_raw(_with_step()), method="spline_sg")
    params = lineage_of(out).params
    assert stage_of(out) == "motcorrected" and params["method"] == "spline_sg"
    assert set(params["spline_sg"]) == {"motion_share", "segments", "pairs_below_snr"}
    assert params["spline_sg"]["segments"] > 0


def test_prep_writes_the_spline_sg_summary_into_the_motcorrected_sidecar(tmp_path):
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.8, motion_correction="spline_sg",
                        cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)
    run_prep(synth_raw("01", "rest"), config, tmp_path, source_entities={"task": "rest"})
    sidecar = json.loads(next(tmp_path.rglob("*desc-motcorrected_nirs.json")).read_text())
    assert sidecar["parameters"]["motion_correction"] == "spline_sg"
    assert set(sidecar["spline_sg"]) == {"motion_share", "segments", "pairs_below_snr"}
