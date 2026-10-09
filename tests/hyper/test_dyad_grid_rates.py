"""The dyad screening grid pairs members' windows by time, so members recorded at different rates still share one."""

import numpy as np

from fnirs_pipe.qc.metrics.hyper import coupled_grid, member_series

PAIRS = ["S1_D1 760", "S1_D1 850", "S2_D1 760", "S2_D1 850"]


def _member(centers, mask_pair_row=None):
    n = len(centers)
    mask = np.ones((len(PAIRS), n), dtype=bool)
    if mask_pair_row is not None:
        mask[:2, mask_pair_row] = False
    sci = np.tile(np.arange(n, dtype=float), (len(PAIRS), 1))
    return {"screen_windows": {"mask": mask, "centers": np.asarray(centers, dtype=float),
                               "sci": sci, "psp": sci, "channel_order": PAIRS},
            "per_channel_long": {"sci_per_channel": {p: 1.0 for p in PAIRS}}}


def test_members_on_one_grid_pair_window_for_window():
    centers = 5.0 + 10.0 * np.arange(6)
    sqm = {"A": _member(centers), "B": _member(centers, mask_pair_row=3)}
    grid = coupled_grid(sqm, ["A", "B"], {})
    assert np.array_equal(grid["index"]["B"], np.arange(6))
    assert grid["ok"]["B"][0].tolist() == [True, True, True, False, True, True]
    assert np.array_equal(member_series(sqm, "B", grid)["sci"], np.arange(6.0))


def test_a_member_whose_windows_drift_is_matched_by_time():
    # 10.05 s windows, as 50 samples at 4.975 Hz, against 10 s ones: 0.05 s drift a window
    a = 5.0 + 10.0 * np.arange(120)
    b = 5.025 + 10.05 * np.arange(119)
    grid = coupled_grid({"A": _member(a), "B": _member(b)}, ["A", "B"], {})
    assert grid is not None
    idx = grid["index"]["B"]
    matched = idx >= 0
    assert np.abs(b[idx[matched]] - a[matched]).max() <= 5.0
    # one window of A falls where the drift reaches half a window, between two of B's
    gap = np.flatnonzero(~matched)
    assert len(gap) == 1 and not grid["ok"]["B"][:, gap].any()
    series = member_series({"A": _member(a), "B": _member(b)}, "B", grid)
    assert np.flatnonzero(np.isnan(series["sci"])).tolist() == gap.tolist()


def test_members_with_no_window_in_common_have_no_grid():
    sqm = {"A": _member(5.0 + 10.0 * np.arange(5)), "B": _member(500.0 + 10.0 * np.arange(5))}
    assert coupled_grid(sqm, ["A", "B"], {}) is None
