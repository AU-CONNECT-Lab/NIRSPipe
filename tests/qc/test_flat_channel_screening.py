"""A channel that does not move cannot be coupled, however mne-nirs scores it.

A detector stuck at its ceiling gives both wavelengths the same constant. mne-nirs filters
that to identical rounding residue and scales it up, so the windowed SCI reads about 1 and the
peak power is large: the pair passed the coupled-window screening and later broke the AR GLM.
"""

import mne
import numpy as np
import pytest

from fnirs_pipe.qc.metrics.coupling import blank_flat_windows, compute_psp_scores
from fnirs_pipe.qc.metrics.screening import resolve_cutoffs, screen_channels, screening_scores

SFREQ, SECONDS = 5.0, 600
CARDIAC = (0.7, 2.0)


def _od(pairs: dict[int, np.ndarray]) -> mne.io.RawArray:
    """Optical density, one source per pair, both wavelengths given the same pulse plus noise."""
    rng = np.random.default_rng(0)
    names, rows, chs = [], [], []
    for source, pulse in pairs.items():
        for wavelength in (760, 850):
            names.append(f"S{source}_D1 {wavelength}")
            rows.append(pulse + (0.3 * rng.standard_normal(pulse.size) if pulse.any() else 0.0))
            chs.append((source, wavelength))
    info = mne.create_info(names, SFREQ, "fnirs_od")
    for ch, (source, wavelength) in zip(info["chs"], chs):
        ch["loc"][3:6] = [0.01 * source, 0.0, 0.0]
        ch["loc"][6:9] = [0.01 * source + 0.03, 0.0, 0.0]
        ch["loc"][9] = wavelength
    return mne.io.RawArray(np.vstack(rows), info, verbose="error")


@pytest.fixture
def pulse():
    t = np.arange(int(SFREQ * SECONDS)) / SFREQ
    return t, np.sin(2 * np.pi * 1.1 * t)


def _screen(raw):
    cutoffs = resolve_cutoffs()
    scores = screening_scores(raw, *CARDIAC, cutoffs=cutoffs)
    return scores, screen_channels(scores, cutoffs)[0]


def test_a_flat_pair_is_rejected_and_a_pulsing_one_kept(pulse):
    _, heart = pulse
    raw = _od({1: heart, 2: np.zeros_like(heart)})
    scores, bad = _screen(raw)
    assert bad == ["S2_D1 760", "S2_D1 850"]
    assert scores["good_frac"]["S2_D1 760"] == 0.0
    assert scores["good_frac"]["S1_D1 760"] == 1.0


def test_a_pair_that_goes_flat_halfway_counts_only_its_moving_windows(pulse):
    t, heart = pulse
    raw = _od({1: heart, 3: np.where(t < SECONDS / 2, heart, 0.0)})
    scores, bad = _screen(raw)
    assert scores["good_frac"]["S3_D1 850"] == pytest.approx(0.5, abs=0.02)
    assert "S3_D1 850" in bad          # below the default 0.75


def test_one_flat_wavelength_blanks_its_partner_too(pulse):
    _, heart = pulse
    raw = _od({1: heart})
    raw._data[1] = 0.0
    blanked = blank_flat_windows(raw, np.ones((2, 3)), 10.0)
    assert np.isnan(blanked).all()


def test_a_flat_pair_has_no_peak_power_to_report(pulse):
    _, heart = pulse
    psp = compute_psp_scores(_od({1: heart, 2: np.zeros_like(heart)}), *CARDIAC)
    assert set(psp) == {"S1_D1 760", "S1_D1 850"}
