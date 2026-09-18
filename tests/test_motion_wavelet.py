"""Wavelet motion correction: the invariants, and a bench for two inherited choices.

The invariants hold whatever the tuning. A large iqr_factor clips nothing, so the
whole pad / centre / normalise / SWT / iSWT / trim round trip must return the input
unchanged; the padding region must stay out of the statistics; the output must scale
with its input, which is what makes the noise normalisation a no-op; and correct_motion
must stamp what it returns.

The bench exists because two choices inside _wl_filter_coeffs were carried over from a
reference implementation rather than decided here, and either one changes results:

  blocks   Each detail level used to be split into 2**(n_levels - i - 1) chunks with the
           IQR taken per chunk. The reference that came from stored its coefficients as
           shift branches, so a chunk there was the whole recording seen through one
           phase; SWT arrays are time-ordered, so a chunk here was a time window instead.
           The method is defined over one distribution per level across the entire time
           span, so the split is gone. _filter_blocked keeps the old behaviour so the bench
           can measure what the change cost or bought.
  level    Used to be fixed at 4, reaching scales of roughly 0.2 to 1.6 s at 10 Hz, so a
           baseline shift was never reached. The depth now follows the recording length and
           reaches about 25 s. The reference also picks which of those levels to clean by
           contamination; _filter_level_pick shows why that rule does not survive the move
           to SWT, and the bench keeps it so the decision is not re-litigated from scratch.

test_bench prints the metrics for both choices on two traces. It asserts nothing about
which wins: that call needs real recordings. Run it with

    pytest tests/test_motion_wavelet.py -k bench -s

The two traces are not interchangeable. With n_levels = 4 the block counts run 8 / 4 / 2
/ 1 from the coarsest level to the finest, so the finest level is not split at all and a
short spike, whose energy sits there, never meets the block logic. Splitting can only
change anything when an artifact is wide enough to reach a blocked level and clustered
enough to dominate one 51 s block, which is what _burst_trace builds and _od_trace does
not.
"""

import mne
import numpy as np
import pytest

from fnirs_pipe.pipeline import motion
from fnirs_pipe.pipeline.motion import (
    _wavelet_motion_correct,
    _wl_clip_iqr,
    _wl_filter_coeffs,
    correct_motion,
)
from fnirs_pipe.utils.lineage import stage_of

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

BURST_START, BURST_END = 150.0, 210.0   # spans one whole 51.2 s block at level 4
BURST_PERIOD = 4.0
ISOLATED_T = 300.0
TRANSIENT_S = 1.5                       # wide enough to reach the blocked coarse levels
TRANSIENT_AMP = 1.5


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


def _bump(width: int, amplitude: float) -> np.ndarray:
    return amplitude * np.sin(np.pi * np.arange(width) / width) ** 2


def _burst_trace(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Physiology, a 60 s stretch packed with 1.5 s transients, one isolated transient.

    The burst is the only thing that can separate blocked from whole-level. A block at
    level 4 is 51.2 s, so the block landing inside the burst is roughly a third artifact
    by sample count, which widens its IQR fence until the transients stop reading as
    outliers. The transient at ISOLATED_T is the control: same shape, clean block.
    """
    clean = _physiology(seed)
    dirty = clean.copy()
    width = int(TRANSIENT_S * SFREQ)
    for k, onset in enumerate(np.arange(BURST_START, BURST_END, BURST_PERIOD)):
        i = int(onset * SFREQ)
        dirty[i:i + width] += _bump(width, TRANSIENT_AMP * (-1) ** k)
    i = int(ISOLATED_T * SFREQ)
    dirty[i:i + width] += _bump(width, TRANSIENT_AMP)
    return dirty, clean


def _od_raw(trace: np.ndarray) -> mne.io.Raw:
    info = mne.create_info(["S1_D1 760"], SFREQ, "fnirs_od")
    return mne.io.RawArray(trace[np.newaxis, :], info, verbose="error")


def _corrected(trace: np.ndarray, **kwargs) -> np.ndarray:
    return _wavelet_motion_correct(_od_raw(trace), **kwargs).get_data()[0]


# ---- Invariants ----

@requires_pywt
def test_a_threshold_nothing_can_exceed_round_trips_exactly():
    """With no coefficient clipped, padding, centring and normalisation must cancel."""
    dirty, _ = _od_trace()
    assert np.allclose(_corrected(dirty, iqr_factor=1e9), dirty, atol=1e-9)


@requires_pywt
def test_iswt_reads_only_the_coarsest_approximation():
    """Why _wl_filter_coeffs may put the same cA in every returned tuple."""
    x = np.random.default_rng(0).normal(size=1024)
    coeffs = pywt.swt(x, "db2", level=4)
    same_cA = [(coeffs[0][0], cD) for _, cD in coeffs]
    assert np.allclose(pywt.iswt(same_cA, "db2"), x)


@requires_pywt
def test_padding_coefficients_are_never_clipped():
    n, signal_length = 256, 200
    x = np.random.default_rng(1).normal(size=n)
    coeffs = pywt.swt(x, "db2", level=3)
    out = _wl_filter_coeffs(coeffs, 1.5, signal_length)
    for (_, cD_in), (_, cD_out) in zip(coeffs, out):
        assert np.array_equal(cD_in[signal_length:], cD_out[signal_length:])


@requires_pywt
def test_output_scales_with_its_input():
    """The IQR fence scales with the data, so the noise normalisation cancels out."""
    dirty, _ = _od_trace()
    assert np.allclose(_corrected(dirty * 1000.0), _corrected(dirty) * 1000.0, rtol=1e-8)


def test_clip_iqr_writes_through_the_callers_view():
    a = np.ones(20)
    a[10] = 1000.0
    _wl_clip_iqr(a[5:15], 1.5)
    assert a[10] == 0.0        # the outlier, zeroed inside the view
    assert a[3] == 1.0         # outside the view, untouched


# ---- What the shipped configuration achieves ----

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


def test_spline_says_it_has_no_backend():
    with pytest.raises(NotImplementedError):
        correct_motion(_od_raw(_od_trace()[0]), method="spline")


def test_spline_is_not_offered_by_the_clis():
    # it used to pass argparse and fail at the correction step, the worst moment to find out
    import argparse

    from fnirs_pipe.cli.qc import _build_parser as qc_parser
    from fnirs_pipe.cli.run import _build_parser as run_parser

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
        assert "spline" not in choices
        assert {"tddr", "wavelet", "none"} <= set(choices)


def test_gui_does_not_offer_spline():
    # read rather than import: pages/analysis.py calls dash.register_page() at module level,
    # which raises outside a running app
    from pathlib import Path

    import fnirs_pipe
    src = (Path(fnirs_pipe.__file__).parent / "interface" / "pages" / "analysis.py"
           ).read_text(encoding="utf-8")
    assert '"an-motion-correction"' in src
    assert '"value": "spline"' not in src


@requires_pywt
def test_wavelet_output_is_stamped_motcorrected():
    out = correct_motion(_od_raw(_od_trace()[0]), method="wavelet")
    assert stage_of(out) == "motcorrected"


# ---- Bench for the two inherited choices ----

def _filter_level_pick(coeffs, iqr_factor: float, signal_length: int):
    """The inherited level selection, kept as the counterexample that it does not port.

    It cleans only the levels whose contamination count is at or above the 90th percentile.
    Under a decimated transform that count is largely a proxy for how many coefficients a
    level holds, which halves as levels coarsen. SWT gives every level the same length, so
    the count instead ranks levels by how heavy-tailed the physiology is there, and it picks
    the coarsest level, where no artifact lives at all.
    """
    cAf = coeffs[0][0].copy()
    cleaned, eta = [], []
    for _, cD in coeffs:
        cDf = cD.copy()
        block = cDf[:signal_length]
        q25, q75 = np.percentile(block, [25, 75])
        fence = iqr_factor * (q75 - q25)
        outliers = (block > q75 + fence) | (block < q25 - fence)
        block[outliers] = 0.0
        cleaned.append(cDf)
        eta.append(int(np.count_nonzero(outliers)))
    cut = np.percentile(eta, 90.0)
    return [(cAf, cleaned[i] if eta[i] >= cut else cD.copy())
            for i, (_, cD) in enumerate(coeffs)]


def _filter_blocked(coeffs, iqr_factor: float, signal_length: int):
    """What _wl_filter_coeffs did before: one IQR per time window per level."""
    n = len(coeffs[0][0])
    n_levels = len(coeffs)
    cAf = coeffs[0][0].copy()
    out = []
    for i, (_, cD) in enumerate(coeffs):
        n_blocks = 2 ** (n_levels - i - 1)
        block_length = n // n_blocks
        cDf = cD.copy()
        for b in range(n_blocks):
            start, end = b * block_length, min(signal_length, (b + 1) * block_length)
            if end > start:
                _wl_clip_iqr(cDf[start:end], iqr_factor)
        out.append((cAf, cDf))
    return out


def _metrics(corrected: np.ndarray, dirty: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    """Spike and step are 1.0 for an uncorrected trace; lower is better for all three."""
    spike = slice(int((SPIKE_T - 2) * SFREQ), int((SPIKE_T + 2) * SFREQ))
    quiet = slice(int(40 * SFREQ), int(100 * SFREQ))
    edge = slice(int((STEP_T - 2) * SFREQ), int((STEP_T + 2) * SFREQ))
    return {
        "spike_residual": (np.abs(corrected[spike] - clean[spike]).max()
                           / np.abs(dirty[spike] - clean[spike]).max()),
        "clean_distortion": (np.std(corrected[quiet] - clean[quiet])
                             / np.std(clean[quiet])),
        "step_edge": (np.abs(np.diff(corrected[edge])).max()
                      / np.abs(np.diff(dirty[edge])).max()),
    }


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2)))


def _burst_metrics(corrected: np.ndarray, dirty: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    """Residual inside the burst, at the isolated control transient, and in the clear."""
    burst = slice(int(BURST_START * SFREQ), int(BURST_END * SFREQ))
    control = slice(int((ISOLATED_T - 2) * SFREQ), int((ISOLATED_T + TRANSIENT_S + 2) * SFREQ))
    quiet = slice(int(40 * SFREQ), int(100 * SFREQ))
    return {
        "burst_residual": _rms(corrected[burst] - clean[burst]) / _rms(dirty[burst] - clean[burst]),
        "control_residual": (np.abs(corrected[control] - clean[control]).max()
                             / np.abs(dirty[control] - clean[control]).max()),
        "clean_distortion": np.std(corrected[quiet] - clean[quiet]) / np.std(clean[quiet]),
    }


@requires_pywt
def test_bench_reports_both_choices_on_both_traces(monkeypatch, capsys):
    variants = (("blocked (before)", _filter_blocked),
                ("per level (now)", motion._wl_filter_coeffs),
                ("+ level pick (no)", _filter_level_pick))

    def _sweep(dirty, clean, metrics):
        rows = []
        for name, filt in variants:
            for level in (4, 8, None):
                monkeypatch.setattr(motion, "_wl_filter_coeffs", filt)
                rows.append((f"{name} level={level or 'auto'}",
                             metrics(_corrected(dirty, level=level), dirty, clean)))
        return rows

    tables = [
        ("isolated spike + step", ["spike_residual", "clean_distortion", "step_edge"],
         _sweep(*_od_trace(), _metrics)),
        ("clustered transients", ["burst_residual", "control_residual", "clean_distortion"],
         _sweep(*_burst_trace(), _burst_metrics)),
    ]

    with capsys.disabled():
        for title, keys, rows in tables:
            print()
            print(title)
            print(f"{'config':<22}" + "".join(f"{k:>18}" for k in keys))
            for name, m in rows:
                print(f"{name:<22}" + "".join(f"{m[k]:>18.3f}" for k in keys))

    # the bench only reports; the one thing every configuration must still do is
    # leave the artifact-free stretch recognisable
    for _, _, rows in tables:
        for name, m in rows:
            assert m["clean_distortion"] < 1.0, name
