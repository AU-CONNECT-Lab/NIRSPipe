"""Wavelet motion correction: each published step pinned, then what the whole achieves.

The shift-invariant transform keeps every phase of a level as its own block and inverts
exactly; each block takes its own fence from its unpadded head, with Hazen quartiles, and
clips its padding too; the coarsest detail level is never cleaned; a record padded to 2**N
samples gets N - WAVELET_LOWEST_SCALE levels.
"""

import mne
import numpy as np
import pytest

from nirspipe.pipeline import motion
from nirspipe.pipeline.motion import (
    WAVELET_LOWEST_SCALE,
    _wavelet_motion_correct,
    _wl_clip_iqr,
    _wl_filter_coeffs,
    _wl_inverse,
    _wl_transform,
    correct_motion,
)
from nirspipe.utils.lineage import stage_of

try:
    import pywt
except ImportError:
    pywt = None

requires_pywt = pytest.mark.skipif(pywt is None, reason="PyWavelets is not installed")

SFREQ = 10.0
DURATION = 400.0
SPIKE_T = 120.0
SPIKE_AMP = 2.0          # about 7x the physiological standard deviation below
STEP_T = 260.0
STEP_AMP = 1.0


# ---- Synthetic OD trace ----

def _od_trace(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One OD-space trace with artifacts, and the same trace without them.

    Cardiac, respiration and a Mayer-band oscillation, plus a 0.3 s spike at
    SPIKE_T and a permanent baseline step at STEP_T. Both artifacts sit far from
    the ends, where the power-of-2 padding discontinuity distorts reconstruction.
    """
    clean = _physiology(seed)
    dirty = clean.copy()
    dirty[int(SPIKE_T * SFREQ):int(SPIKE_T * SFREQ) + 3] += SPIKE_AMP
    dirty[int(STEP_T * SFREQ):] += STEP_AMP
    return dirty, clean


def _physiology(seed: int) -> np.ndarray:
    """Cardiac, respiration, a Mayer-band oscillation and noise. Standard deviation 0.28."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    return (0.30 * np.sin(2 * np.pi * 1.0 * t)
            + 0.15 * np.sin(2 * np.pi * 0.25 * t)
            + 0.20 * np.sin(2 * np.pi * 0.05 * t)
            + 0.05 * rng.normal(size=t.size))


def _od_raw(trace: np.ndarray) -> mne.io.Raw:
    info = mne.create_info(["S1_D1 760"], SFREQ, "fnirs_od")
    return mne.io.RawArray(trace[np.newaxis, :], info, verbose="error")


def _corrected(trace: np.ndarray, **kwargs) -> np.ndarray:
    return _wavelet_motion_correct(_od_raw(trace), **kwargs).get_data()[0]


# ---- The transform ----

@requires_pywt
def test_the_transform_and_its_inverse_round_trip():
    x = np.random.default_rng(0).normal(size=256)
    assert np.allclose(_wl_inverse(_wl_transform(x, 4, "db2"), "db2"), x, atol=1e-12)


@requires_pywt
def test_each_level_keeps_every_phase_as_its_own_block():
    x = np.random.default_rng(1).normal(size=16)

    def dwt(s):
        return pywt.dwt(s, "db2", mode="periodization")

    wp = _wl_transform(x, 2, "db2")
    (cA, cD), (cA_sh, cD_sh) = dwt(x), dwt(np.roll(x, 1))
    assert np.allclose(wp[1], np.r_[cD, cD_sh])
    level2 = [dwt(block)[1] for a in (cA, cA_sh) for block in (a, np.roll(a, 1))]
    assert np.allclose(wp[2], np.concatenate(level2))


@requires_pywt
def test_a_threshold_nothing_can_exceed_round_trips_exactly():
    """With no coefficient clipped, padding and centring must cancel."""
    dirty, _ = _od_trace()
    assert np.allclose(_corrected(dirty, iqr_factor=1e9), dirty, atol=1e-9)


@requires_pywt
def test_output_scales_with_its_input():
    """The IQR fence scales with the data, which is why no noise normalisation is needed."""
    dirty, _ = _od_trace()
    assert np.allclose(_corrected(dirty * 1000.0), _corrected(dirty) * 1000.0, rtol=1e-8)


@requires_pywt
def test_a_record_gets_n_minus_the_lowest_scale_levels(monkeypatch):
    seen = []
    real = motion._wl_transform

    def spy(x, n_levels, wavelet):
        seen.append(n_levels)
        return real(x, n_levels, wavelet)

    monkeypatch.setattr(motion, "_wl_transform", spy)
    _corrected(_od_trace()[0])                 # 4000 samples, padded to 2**12
    assert seen == [12 - WAVELET_LOWEST_SCALE]


# ---- The coefficient filter ----

# Hazen quartiles of nine values: Q1 sits at position 2.75 and Q3 at 7.25 of the sorted
# values, so with 1..7 in the middle Q1 = 1.75, Q3 = 6.25, IQR = 4.5 whatever the two ends
# hold, and the 1.5 fence is [1.75 - 6.75, 6.25 + 6.75] = [-5, 13].
def test_clip_iqr_zeroes_strictly_beyond_the_hand_computed_fence():
    on_fence = np.array([[-5.0, 1, 2, 3, 4, 5, 6, 7, 13.0]])
    assert (_wl_clip_iqr(on_fence, 9, 1.5) == on_fence).all()

    past_fence = np.array([[-5.01, 1, 2, 3, 4, 5, 6, 7, 13.01]])
    out = _wl_clip_iqr(past_fence, 9, 1.5)[0]
    assert out[0] == 0.0 and out[-1] == 0.0
    assert (out[1:-1] == np.arange(1.0, 8.0)).all()


def test_clip_iqr_fence_is_proportional_to_the_factor():
    # factor 0.5: fence [1.75 - 2.25, 6.25 + 2.25] = [-0.5, 8.5]
    block = np.array([[-0.6, 1, 2, 3, 4, 5, 6, 7, 8.6]])
    assert (_wl_clip_iqr(block, 9, 0.5)[0][[0, -1]] == 0.0).all()
    assert (_wl_clip_iqr(block, 9, 1.5) == block).all()


def test_clip_iqr_is_not_widened_by_the_outlier_it_is_clipping():
    # control: mean + 3 SD is dragged out by the 1000 and lets 30 through
    block = np.array([[1.0, 2, 3, 4, 5, 6, 7, 30, 1000]])
    naive_kept = abs(30 - block.mean()) <= 3 * block.std()
    out = _wl_clip_iqr(block, 9, 1.5)[0]          # Q1 2.75, Q3 12.75, fence [-12.25, 27.75]
    assert naive_kept
    assert out[-2] == 0.0 and out[-1] == 0.0


def test_padding_stays_out_of_the_quartiles_but_is_clipped():
    head = [-5.0, 1, 2, 3, 4, 5, 6, 7, 13.0]
    row = np.array([head + [1000.0, 1000.0, 2.0]])
    out = _wl_clip_iqr(row, 9, 1.5)[0]
    assert (out[:9] == head).all()                # the fence is still [-5, 13]
    assert (out[9:] == [0.0, 0.0, 2.0]).all()


def test_each_block_takes_its_own_fence():
    rows = np.array([[-5.0, 1, 2, 3, 4, 5, 6, 7, 13.01],
                     [-50.0, 10, 20, 30, 40, 50, 60, 70, 130.0]])
    out = _wl_clip_iqr(rows, 9, 1.5)
    assert out[0, -1] == 0.0                      # beyond its own fence of 13
    assert out[1, -1] == 130.0                    # on its own fence of 130


def test_the_coarsest_detail_level_is_never_cleaned():
    wp = np.random.default_rng(2).normal(size=(4, 64))
    wp[:, ::16] = 100.0                           # an outlier in every block of every row
    out = _wl_filter_coeffs(wp, 1.5, 64)
    assert (out[0] == wp[0]).all()                # the approximation
    assert (out[3] == wp[3]).all()                # the coarsest detail level
    assert not (out[1] == wp[1]).all() and not (out[2] == wp[2]).all()


# ---- What the correction achieves ----

@requires_pywt
def test_the_spike_is_at_least_halved():
    dirty, clean = _od_trace()
    win = slice(int((SPIKE_T - 2) * SFREQ), int((SPIKE_T + 2) * SFREQ))
    before = np.abs(dirty[win] - clean[win]).max()
    after = np.abs(_corrected(dirty)[win] - clean[win]).max()
    assert after < 0.5 * before


@requires_pywt
def test_an_artifact_free_stretch_is_not_rewritten():
    dirty, clean = _od_trace()
    quiet = slice(int(40 * SFREQ), int(100 * SFREQ))
    residual = np.std(_corrected(dirty)[quiet] - clean[quiet])
    assert residual < 0.1 * np.std(clean[quiet])


# ---- Dispatch ----

def test_none_passes_the_signal_through_and_still_stamps():
    raw = _od_raw(_od_trace()[0])
    out = correct_motion(raw, method="none")
    assert out is raw
    assert stage_of(out) == "motcorrected"


def test_a_missing_method_is_an_error_not_a_default():
    with pytest.raises(ValueError):
        correct_motion(_od_raw(_od_trace()[0]), method=None)


@requires_pywt
def test_wavelet_tddr_runs_the_wavelet_then_tddr():
    # the order matters: the reverse gives a different signal
    info = mne.create_info(["S1_D1 760", "S1_D1 850"], SFREQ, "fnirs_od")
    raw = mne.io.RawArray(np.vstack([_od_trace(0)[0], _od_trace(1)[0]]), info, verbose="error")
    for ch, wavelength in zip(raw.info["chs"], (760.0, 850.0)):
        ch["loc"][9] = wavelength
    tddr = mne.preprocessing.nirs.temporal_derivative_distribution_repair
    out = correct_motion(raw, method="wavelet_tddr")
    assert stage_of(out) == "motcorrected"
    assert (out.get_data() == tddr(_wavelet_motion_correct(raw)).get_data()).all()
    assert not np.allclose(out.get_data(), _wavelet_motion_correct(tddr(raw)).get_data())


def test_every_cli_offers_every_method():
    import argparse

    from nirspipe.cli.qc import _build_parser as qc_parser
    from nirspipe.cli.run import _build_parser as run_parser

    def motion_choices(parser):
        for action in parser._actions:
            if "--motion-correction" in action.option_strings:
                yield action.choices
            elif isinstance(action, argparse._SubParsersAction):
                for sub in action.choices.values():
                    yield from motion_choices(sub)

    found = [c for build in (qc_parser, run_parser) for c in motion_choices(build())]
    assert found, "no --motion-correction argument reached"
    for choices in found:
        assert set(choices) == {"tddr", "wavelet", "tpca", "spline", "spline_sg", "wavelet_tddr",
                                "spline_wavelet", "none"}


def _gui_motion_dropdown() -> str:
    # read rather than import: pages/analysis.py calls dash.register_page() at module level,
    # which raises outside a running app
    from pathlib import Path

    import nirspipe
    src = (Path(nirspipe.__file__).parent / "interface" / "pages" / "analysis.py"
           ).read_text(encoding="utf-8")
    assert '"an-motion-correction"' in src
    # the motion-correction dropdown alone: the censor fill offers a spline of its own
    dropdown = src[src.index('"an-motion-correction"'):]
    return dropdown[:dropdown.index("clearable")]


def test_gui_offers_exactly_the_cli_choices():
    import re

    from nirspipe.cli.run import _MOTION_CHOICES

    assert re.findall(r'"value": "(\w+)"', _gui_motion_dropdown()) == _MOTION_CHOICES


@requires_pywt
def test_wavelet_output_is_stamped_motcorrected():
    out = correct_motion(_od_raw(_od_trace()[0]), method="wavelet")
    assert stage_of(out) == "motcorrected"
