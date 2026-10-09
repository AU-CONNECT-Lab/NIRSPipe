"""What a long recording's figures keep when they are thinned for display.

The figure-accuracy recordings are too short to be thinned at all, so these are built at the
length of a real session, where the channel detail keeps two samples in twenty and the
carpet draws one column for twenty.
"""

import mne
import numpy as np

from nirspipe.qc.figures.common.motion_panel import CARPET_Z, carpet_z
from nirspipe.qc.figures.subject.raw_figures import (build_channel_figure,
                                                        build_evoked_topo_figure)

SFREQ = 10.0
N = 40000
PULSE_HZ = 1.02
PULSE_M = 1e-7
# where a stride of ten or twenty folds the pulse: a 50 s wave the signal does not have
FOLDED_HZ = 0.02


def _pulse_only():
    t = np.arange(N) / SFREQ
    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], SFREQ, ["hbo", "hbr"])
    return mne.io.RawArray(np.vstack([PULSE_M * np.sin(2 * np.pi * PULSE_HZ * t)] * 2), info,
                           verbose="error")


def _no_slow_wave(x, y, max_pts):
    amplitude = PULSE_M * 1e6
    assert len(y) <= max_pts
    folded = 2 * abs(np.mean((y - y.mean()) * np.exp(-2j * np.pi * FOLDED_HZ * x)))
    assert folded < 0.2 * amplitude
    # the pulse is drawn as the band it spans, not smoothed away
    assert y.max() > 0.9 * amplitude and y.min() < -0.9 * amplitude


def test_the_channel_detail_draws_no_slow_wave_the_signal_does_not_have():
    detail, _, _ = build_channel_figure(_pulse_only(), [], "S1_D1", max_ts_pts=4000, epoch=False)
    _no_slow_wave(np.asarray(detail.data[0].x, float), np.asarray(detail.data[0].y, float), 4000)


def test_the_channel_map_s_continuous_view_draws_no_slow_wave_either():
    """A run with no events, a resting-state one, gets the continuous signal in every cell."""
    raw = _pulse_only()
    for ch in raw.info["chs"]:
        ch["loc"][:3] = (0.01, 0.02, 0.05)
    fig = build_evoked_topo_figure(raw, [], max_ts_pts=2000)
    hbo = next(t for t in fig.data if t.type == "scatter" and (t.name or "").startswith("HBO"))
    _no_slow_wave(np.asarray(hbo.x, float), np.asarray(hbo.y, float), 2000)


def test_a_burst_shorter_than_one_column_still_shows_on_the_carpet():
    t = np.arange(N) / SFREQ
    data = 0.01 * np.random.default_rng(0).standard_normal((1, N))
    data[0, 20001:20004] += 1.0
    z, t_ds, _ = carpet_z(data, t)
    nearest = np.argmin(np.abs(t_ds - t[20002]))
    assert abs(z[0, nearest]) == CARPET_Z
