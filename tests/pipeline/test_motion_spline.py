"""Spline motion correction: per-pair detection, the smoothing spline, the bookkeeping, the wiring.

The steps follow the published procedure exactly, quirks included, so several tests pin a
quirk on purpose: a pair is detected on both wavelengths even when one is bad, a segment runs
one sample early, the last clean stretch starts one sample earlier still, a record that ends
in motion keeps its last sample raw, and a record that starts in motion is matched to the
stretch after it through a window that stops one sample short of the segment's end.
"""

import json

import mne
import numpy as np
import pytest
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.optimize import minimize

from nirspipe.pipeline.motion import (
    SPLINE_P,
    _smoothing_spline,
    _spline_correct,
    _spline_motion_correct,
    _spline_motion_mask,
    _spline_window,
    _tpca_motion_mask,
    correct_motion,
)
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep
from nirspipe.utils.lineage import lineage_of, stage_of

from tests._synth import synth_raw

SFREQ = 10.0
N_T = 6000               # long enough that one step does not inflate its own SD past 50 SDs
N_CH = 6
NAMES = [f"S{i // 2 + 1}_D1 {760 if i % 2 == 0 else 850}" for i in range(N_CH)]
STEP_AT = 3000


def _quiet(n_t: int = N_T, seed: int = 0) -> np.ndarray:
    """Time x channels white noise far below any detection threshold."""
    return 0.01 * np.random.default_rng(seed).normal(size=(n_t, N_CH))


def _stepped(amp: float, cols: tuple[int, ...] = (0,)) -> np.ndarray:
    d = _quiet()
    d[STEP_AT:, list(cols)] += amp
    return d


def _raw(d: np.ndarray, bads: tuple[int, ...] = (), sfreq: float = SFREQ) -> mne.io.Raw:
    raw = mne.io.RawArray(d.T.copy(), mne.create_info(NAMES, sfreq, "fnirs_od"), verbose="error")
    raw.info["bads"] = [NAMES[i] for i in bads]
    return raw


# ---- Detection ----

def test_a_step_marks_its_pair_with_the_published_window():
    # 5-sample change window, 10 samples either side: rows 2995..2999 see the step, widened
    # to 2985..3009, and a row marks the sample after it
    mask = _spline_motion_mask(_stepped(2.0), SFREQ, NAMES, np.ones(N_CH, bool))
    assert (np.flatnonzero(mask[:, 0]) == np.arange(2986, 3011)).all()
    assert (mask[:, 1] == mask[:, 0]).all()
    assert not mask[:, 2:].any()


def test_a_bad_wavelength_still_feeds_its_pairs_detection():
    good = np.ones(N_CH, bool)
    good[0] = False
    mask = _spline_motion_mask(_stepped(2.0), SFREQ, NAMES, good)
    assert not mask[:, 0].any()
    assert mask[:, 1].any()


def test_detection_takes_fifty_sds_where_tpca_takes_twenty():
    # a 0.7 step is about 42 SDs of this channel's first difference
    d = _stepped(0.7)
    good = np.ones(N_CH, bool)
    assert _tpca_motion_mask(d, SFREQ, good).any()
    assert not _spline_motion_mask(d, SFREQ, NAMES, good).any()


def test_a_record_never_starts_in_motion():
    # a row marks the sample after it, so the first sample cannot be marked
    d = _quiet()
    d[1:, 0] += 10.0
    mask = _spline_motion_mask(d, SFREQ, NAMES, np.ones(N_CH, bool))
    assert mask[1:, 0].any() and not mask[0].any()


# ---- The smoothing spline ----

@pytest.mark.parametrize("n", [5, 40, 1000])
def test_the_smoothing_spline_is_the_penalised_fit_with_lambda_from_p(n):
    # p * fit + (1 - p) * roughness has the minimiser of fit + (1 - p) / p * roughness
    rng = np.random.default_rng(n)
    t = 3.0 + np.arange(n) / 7.8125
    y = np.cumsum(rng.normal(size=n))
    expected = make_smoothing_spline(t, y, lam=(1 - SPLINE_P) / SPLINE_P)(t)
    assert np.abs(_smoothing_spline(t, y, SPLINE_P) - expected).max() < 1e-10 * np.abs(y).max()


@pytest.mark.parametrize("n", [3, 4])
def test_a_short_segment_minimises_the_same_objective(n):
    # the fit above needs five points: minimise directly over the natural cubic spline
    # through candidate values, whose second derivative is piecewise linear
    rng = np.random.default_rng(n)
    t = np.arange(n) / SFREQ
    y = rng.normal(size=n)
    h = np.diff(t)

    def objective(g):
        m = CubicSpline(t, g, bc_type="natural")(t, 2)
        roughness = np.sum(h / 3 * (m[:-1] ** 2 + m[:-1] * m[1:] + m[1:] ** 2))
        return SPLINE_P * np.sum((y - g) ** 2) + (1 - SPLINE_P) * roughness

    best = minimize(objective, y, method="BFGS", options={"gtol": 1e-10}).x
    assert np.allclose(_smoothing_spline(t, y, SPLINE_P), best, atol=1e-6)


def test_two_points_come_back_as_they_are():
    y = np.array([1.0, 3.0])
    assert (_smoothing_spline(np.array([0.0, 0.1]), y, SPLINE_P) == y).all()


# ---- The published bookkeeping ----
# at 5 Hz a rejoining window is one sample for any stretch of 2 to 14 samples

SLOW = 5.0
T12 = np.arange(12) / SLOW


def test_the_rejoining_window():
    # whole stretch below 1.5 samples, floor(1.5) = 1 below 15, a tenth from there on
    assert [_spline_window(k, SLOW) for k in (1, 2, 14, 15, 29, 40)] == [1, 1, 1, 1, 2, 4]


def test_a_segment_runs_one_sample_early():
    # motion on samples 5-8 (1-based) makes the segment 4-7: sample 4 loses the line through
    # 4-7 although it was clean, and sample 8 rides with the clean stretch after it
    y = np.array([0, 0, 0, 10, 11, 12, 13, 20, 20, 20, 20, 20], float)
    t_inc = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    out = _spline_correct(y, t_inc, T12, SLOW)
    assert np.allclose(out, [0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7], atol=1e-9)


def test_the_last_clean_stretch_starts_on_the_segments_last_sample():
    # its one-sample window there makes sample 7 take sample 6's value exactly, which a
    # spline residual would not
    y = np.random.default_rng(1).normal(size=12)
    t_inc = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    out = _spline_correct(y, t_inc, T12, SLOW)
    assert out[6] == out[5]


def test_a_record_that_ends_in_motion_keeps_its_last_sample_raw():
    y = np.random.default_rng(2).normal(size=12)
    t_inc = np.r_[np.ones(8), np.zeros(4)].astype(int)
    out = _spline_correct(y, t_inc, T12, SLOW)
    assert out[-1] == y[-1]
    assert not np.allclose(out[7:11], y[7:11])


def test_clean_stretches_keep_their_raw_shape():
    y = np.random.default_rng(3).normal(size=40)
    t_inc = np.ones(40, int)
    t_inc[10:15] = 0
    t_inc[25:30] = 0
    out = _spline_correct(y, t_inc, np.arange(40) / SLOW, SLOW)
    assert (out[:9] == y[:9]).all()
    # the stretch between the segments is samples 15-24 (1-based), shifted as one piece
    assert np.ptp(out[14:24] - y[14:24]) < 1e-12


def test_a_record_that_starts_in_motion_meets_the_stretch_after_it():
    # not reachable from the detector. The segment is samples 1-3; its window stops at
    # sample 2 rather than 3 and the next window is sample 4, so sample 2 lands on sample 4
    y = np.random.default_rng(4).normal(size=12)
    t_inc = np.r_[np.zeros(4), np.ones(8)].astype(int)
    out = _spline_correct(y, t_inc, T12, SLOW)
    assert out[1] == y[3]


# ---- The whole correction ----

def test_a_baseline_shift_is_rejoined():
    d = _stepped(2.0, cols=(0, 1))
    out = _spline_motion_correct(_raw(d))[0].get_data().T
    for col in (0, 1):
        assert abs(out[3100:, col].mean() - out[:2900, col].mean()) < 0.05
    assert (out[:, 2:] == d[:, 2:]).all()


def test_bad_channels_come_back_unchanged():
    d = _stepped(2.0, cols=(0, 1, 2))
    out, info = _spline_motion_correct(_raw(d, bads=(0, 1)))
    assert (out.get_data()[:2] == d.T[:2]).all()
    assert info["segments"] == 2                      # one per wavelength of S2_D1


def test_quiet_data_passes_through_untouched():
    d = _quiet()
    out, info = _spline_motion_correct(_raw(d))
    assert (out.get_data() == d.T).all()
    assert info == {"motion_share": 0.0, "segments": 0}


def test_below_ten_thirds_hz_it_refuses_rather_than_average_nothing():
    with pytest.raises(ValueError, match="at least 3.333 Hz"):
        _spline_motion_correct(_raw(_quiet(n_t=600), sfreq=3.3))
    _spline_motion_correct(_raw(_quiet(n_t=600), sfreq=3.4))


# ---- Wiring ----

def test_correct_motion_stamps_spline_and_what_it_marked():
    out = correct_motion(_raw(_stepped(2.0)), method="spline")
    params = lineage_of(out).params
    assert stage_of(out) == "motcorrected" and params["method"] == "spline"
    assert params["spline"]["segments"] == 2 and params["spline"]["motion_share"] > 0


def test_prep_writes_the_spline_summary_into_the_motcorrected_sidecar(tmp_path):
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.8, motion_correction="spline",
                        cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)
    run_prep(synth_raw("01", "rest"), config, tmp_path, source_entities={"task": "rest"})
    sidecar = json.loads(next(tmp_path.rglob("*desc-motcorrected_nirs.json")).read_text())
    assert sidecar["parameters"]["motion_correction"] == "spline"
    assert set(sidecar["spline"]) == {"motion_share", "segments"}
