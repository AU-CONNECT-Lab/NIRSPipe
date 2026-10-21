"""Joining a task's runs for `--combine-runs`: the per-run z-score, the join, and the FC it gives.

The FC check rests on an identity that needs no package code: once every segment has mean 0
and variance 1 (ddof 0), the Pearson r of the joined series is the length-weighted mean of
the segments' own r, so no run outweighs another for being larger.
"""

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nirspipe.exceptions import StageError
from nirspipe.pipeline.combine_runs import join_runs, zscore_kept
from nirspipe.pipeline.restingstate import compute_fc

SFREQ = 10.0
NAMES = ["S1_D1 hbo", "S2_D2 hbo", "S1_D1 hbr", "S2_D2 hbr"]


def _run(data, bads=(), annotations=()):
    info = mne.create_info(NAMES, SFREQ, ch_types=["hbo", "hbo", "hbr", "hbr"])
    raw = mne.io.RawArray(np.asarray(data, float), info, verbose="error")
    raw.info["bads"] = list(bads)
    if annotations:
        onset, duration, description = zip(*annotations)
        raw.set_annotations(mne.Annotations(onset, duration, description))
    return raw


def test_a_row_is_scaled_by_its_kept_frames_and_a_flat_row_is_zero():
    keep = np.array([True, True, True, False])
    got = zscore_kept(np.array([[1.0, 2, 3, 100], [5, 5, 5, 9]]), keep)
    # kept mean 2, SD sqrt(2/3): (x - 2) * sqrt(3/2)
    assert_allclose(got[0], [-1.224745, 0.0, 1.224745, 120.024997], atol=1e-6)
    assert_allclose(got[1], 0.0)


def test_the_runs_join_in_order_with_their_marks_and_every_runs_bads():
    rng = np.random.default_rng(0)
    first = _run(rng.standard_normal((4, 50)), bads=["S1_D1 hbo"],
                 annotations=[(1.0, 2.0, "BAD_manual")])
    second = _run(rng.standard_normal((4, 30)), bads=["S2_D2 hbr"],
                  annotations=[(2.0, 0.0, "tap")])
    joined, onsets = join_runs([first, second])
    assert joined.n_times == 80 and onsets == [0.0, 5.0]
    assert joined.info["bads"] == ["S1_D1 hbo", "S2_D2 hbr"]
    assert_allclose(joined.annotations.onset, [1.0, 7.0])
    assert list(joined.annotations.description) == ["BAD_manual", "tap"]


@pytest.mark.parametrize("change, match", [
    (lambda raw: raw.rename_channels({"S2_D2 hbr": "S3_D3 hbr"}), "channel list"),
    (lambda raw: raw.resample(5.0), "resample-sfreq"),
])
def test_runs_that_differ_in_channels_or_rate_are_refused(change, match):
    rng = np.random.default_rng(1)
    first, second = (_run(rng.standard_normal((4, 40))) for _ in range(2))
    change(second)
    with pytest.raises(StageError, match=match):
        join_runs([first, second])


def test_the_joined_correlation_is_the_length_weighted_mean_of_the_runs():
    rng = np.random.default_rng(2)

    def pair(n, rho, scale, offset):
        a = rng.standard_normal(n)
        b = rho * a + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        return np.vstack([a, b, rng.standard_normal((2, n))]) * scale + offset

    first, second = pair(120, 0.8, 1.0, 0.0), pair(80, -0.3, 10.0, 5.0)
    joined, _ = join_runs([_run(first), _run(second)])
    r1, r2 = (np.corrcoef(x[0], x[1])[0, 1] for x in (first, second))
    fc = compute_fc(joined, "hbo")
    assert fc.loc["S1_D1 hbo", "S2_D2 hbo"] == pytest.approx((120 * r1 + 80 * r2) / 200,
                                                             abs=1e-12)
    # unscaled, the second run's tenfold amplitude and its offset would run the joint r
    raw_r = np.corrcoef(np.hstack([first[0], second[0]]), np.hstack([first[1], second[1]]))[0, 1]
    assert abs(raw_r - (120 * r1 + 80 * r2) / 200) > 0.1


def test_a_bad_span_neither_scales_its_run_nor_enters_the_correlation():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((4, 100))
    data[:, 20:40] += 50.0
    joined, _ = join_runs([_run(data, annotations=[(2.0, 2.0, "BAD_manual")]),
                           _run(rng.standard_normal((4, 60)))])
    keep = np.r_[np.ones(20, bool), np.zeros(20, bool), np.ones(120, bool)]
    kept = joined.get_data()[:, keep]
    assert_allclose(kept[:, :80].mean(axis=1), 0.0, atol=1e-12)
    assert_allclose(kept[:, :80].std(axis=1), 1.0, atol=1e-12)
    expected = np.corrcoef(kept[0], kept[1])[0, 1]
    assert compute_fc(joined, "hbo").loc["S1_D1 hbo", "S2_D2 hbo"] == pytest.approx(expected,
                                                                                    abs=1e-12)
