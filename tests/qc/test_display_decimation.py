"""What a long recording's figures keep when they are thinned for display.

The figure-accuracy recordings are too short to be thinned at all, so these are built at the
length of a real session, where the channel detail keeps two samples in twenty and the
carpet draws one column for twenty.
"""

import mne
import numpy as np

from fnirs_pipe.qc.figures.common.motion_panel import CARPET_Z, carpet_z
from fnirs_pipe.qc.figures.subject.raw_figures import build_channel_figure

SFREQ = 10.0
N = 40000
# a stride of ten would fold this to 0.02 Hz, a 50 s wave
PULSE_HZ = 1.02
PULSE_M = 1e-7


def test_the_channel_detail_draws_no_slow_wave_the_signal_does_not_have():
    t = np.arange(N) / SFREQ
    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], SFREQ, ["hbo", "hbr"])
    raw = mne.io.RawArray(np.vstack([PULSE_M * np.sin(2 * np.pi * PULSE_HZ * t)] * 2), info,
                          verbose="error")
    detail, _, _ = build_channel_figure(raw, [], "S1_D1", max_ts_pts=4000, epoch=False)
    x, y = np.asarray(detail.data[0].x, float), np.asarray(detail.data[0].y, float)
    assert len(y) <= 4000
    # the pulse averages out over 10 s; a folded copy of it does not
    local = [y[(x >= a) & (x < a + 10)].mean() for a in np.arange(0, x[-1] - 10, 10)]
    assert np.abs(local).max() < 0.1 * PULSE_M * 1e6


def test_a_burst_shorter_than_one_column_still_shows_on_the_carpet():
    t = np.arange(N) / SFREQ
    data = 0.01 * np.random.default_rng(0).standard_normal((1, N))
    data[0, 20001:20004] += 1.0
    z, t_ds, _ = carpet_z(data, t)
    nearest = np.argmin(np.abs(t_ds - t[20002]))
    assert abs(z[0, nearest]) == CARPET_Z
