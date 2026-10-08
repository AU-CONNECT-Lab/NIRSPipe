"""What a long recording's figures keep when they are thinned for display.

The figure-accuracy recordings are too short to be thinned at all, so these are built at the
length of a real session, where the channel detail keeps every few samples and the carpet
one column in about twenty.
"""

import mne
import numpy as np
import pytest

from fnirs_pipe.qc.figures.common.motion_panel import CARPET_Z, carpet_z
from fnirs_pipe.qc.figures.subject.raw_figures import build_channel_figure

SFREQ = 10.0
N = 40000
PULSE_HZ = 1.2
PULSE_M = 1e-7


def _amplitude_at(x, y, freq):
    return 2 * abs(np.mean((y - y.mean()) * np.exp(-2j * np.pi * freq * x)))


@pytest.mark.xfail(strict=True, reason="the detail keeps every step-th sample with no low-pass, "
                                       "so the pulse folds below the drawn rate's Nyquist")
def test_the_channel_detail_draws_no_slow_wave_the_signal_does_not_have():
    t = np.arange(N) / SFREQ
    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], SFREQ, ["hbo", "hbr"])
    raw = mne.io.RawArray(np.vstack([PULSE_M * np.sin(2 * np.pi * PULSE_HZ * t)] * 2), info,
                          verbose="error")
    detail, _, _ = build_channel_figure(raw, [], "S1_D1", max_ts_pts=4000, epoch=False)
    x, y = np.asarray(detail.data[0].x, float), np.asarray(detail.data[0].y, float)
    drawn_rate = 1.0 / np.median(np.diff(x))
    folded = abs(PULSE_HZ - round(PULSE_HZ / drawn_rate) * drawn_rate)
    assert _amplitude_at(x, y, folded) < 0.1 * PULSE_M * 1e6


@pytest.mark.xfail(strict=True, reason="the carpet keeps every step-th sample, so a burst "
                                       "shorter than one column can fall between them")
def test_a_burst_shorter_than_one_column_still_shows_on_the_carpet():
    t = np.arange(N) / SFREQ
    data = 0.01 * np.random.default_rng(0).standard_normal((1, N))
    data[0, 20001:20004] += 1.0
    z, t_ds, _ = carpet_z(data, t)
    covering = np.searchsorted(t_ds, t[20001], side="right") - 1
    assert abs(z[0, covering]) == CARPET_Z
