"""Joining a task's runs for `--combine-runs`: the per-run z-score, the join, and the FC it gives.

The FC check rests on an identity that needs no package code: once every segment has mean 0
and variance 1 (ddof 0), the Pearson r of the joined series is the length-weighted mean of
the segments' own r, so no run outweighs another for being larger. A channel missing from a
run is held to a second identity: Pearson r is unchanged by the per-run z-score, so its joined
r over the runs it shares with another channel is that pair's r in those runs alone.
"""

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nirspipe.exceptions import StageError
from nirspipe.pipeline.combine_runs import join_roi_signals, join_runs, zscore_kept
from nirspipe.pipeline.restingstate import compute_fc, compute_fc_roi, compute_fc_seed

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


def test_the_runs_join_in_order_with_their_marks_and_a_channel_only_where_it_is_good():
    rng = np.random.default_rng(0)
    first = _run(rng.standard_normal((4, 50)), bads=["S1_D1 hbo", "S1_D1 hbr"],
                 annotations=[(1.0, 2.0, "BAD_manual")])
    second = _run(rng.standard_normal((4, 30)), bads=["S2_D2 hbr", "S1_D1 hbr"],
                  annotations=[(2.0, 0.0, "tap")])
    joined, onsets = join_runs([first, second])
    assert joined.n_times == 80 and onsets == [0.0, 5.0]
    # bad only where every run marked it
    assert joined.info["bads"] == ["S1_D1 hbr"]
    data = joined.get_data()
    assert np.isnan(data[0, :50]).all() and not np.isnan(data[0, 50:]).any()
    assert np.isnan(data[3, 50:]).all() and not np.isnan(data[3, :50]).any()
    assert np.isnan(data[2]).all() and not np.isnan(data[1]).any()
    assert_allclose(joined.annotations.onset, [1.0, 7.0])
    assert list(joined.annotations.description) == ["BAD_manual", "tap"]


def test_an_absent_channel_other_than_a_bad_one_is_left_out_of_its_run():
    rng = np.random.default_rng(4)
    runs = [_run(rng.standard_normal((4, 20))), _run(rng.standard_normal((4, 20)))]
    joined, _ = join_runs(runs, absent=[{"S2_D2 hbo"}, set()])
    assert np.isnan(joined.get_data()[1, :20]).all() and joined.info["bads"] == []


def test_a_channel_missing_from_a_run_is_correlated_over_the_runs_it_shares():
    rng = np.random.default_rng(5)
    first, second, third = (rng.standard_normal((4, n)) * 3 + 1 for n in (60, 40, 50))
    runs = [_run(first, bads=["S1_D1 hbo"]), _run(second), _run(third)]
    fc = compute_fc(join_runs(runs)[0], "hbo")
    # S1_D1 is missing from the first run: its r is the length-weighted mean of the other two
    r2, r3 = (np.corrcoef(x[0], x[1])[0, 1] for x in (second, third))
    assert fc.loc["S1_D1 hbo", "S2_D2 hbo"] == pytest.approx((40 * r2 + 50 * r3) / 90, abs=1e-12)
    assert fc.loc["S1_D1 hbo", "S1_D1 hbo"] == pytest.approx(1.0)


def test_a_bad_span_and_a_missing_run_both_stay_out_of_a_pair():
    rng = np.random.default_rng(6)
    first, second = rng.standard_normal((4, 60)), rng.standard_normal((4, 40))
    second[:, 10:20] += 30.0
    runs = [_run(first, bads=["S2_D2 hbo"]),
            _run(second, annotations=[(1.0, 1.0, "BAD_manual")])]
    fc = compute_fc(join_runs(runs)[0], "hbo")
    shared = np.r_[np.arange(10), np.arange(20, 40)]
    expected = np.corrcoef(second[0, shared], second[1, shared])[0, 1]
    assert fc.loc["S1_D1 hbo", "S2_D2 hbo"] == pytest.approx(expected, abs=1e-12)


def test_an_roi_is_averaged_over_each_runs_own_good_members():
    rng = np.random.default_rng(7)
    first, second = rng.standard_normal((4, 50)), rng.standard_normal((4, 30)) * 5
    runs = [_run(first, bads=["S1_D1 hbo"]), _run(second)]
    signals, members = join_roi_signals(runs, ["run-01", "run-02"],
                                        {"A": ["S1_D1", "S2_D2"]}, "hbo")
    assert members == {"run-01": {"A": ["S2_D2 hbo"]}, "run-02": {"A": ["S1_D1 hbo", "S2_D2 hbo"]}}

    def z(x):
        return (x - x.mean()) / x.std()

    assert_allclose(signals["A"], np.r_[z(first[1]), z(second[:2].mean(axis=0))], atol=1e-12)


def test_an_roi_with_no_good_member_in_a_run_is_nan_there_and_paired_over_the_rest():
    rng = np.random.default_rng(8)
    first, second = rng.standard_normal((4, 50)), rng.standard_normal((4, 30))
    runs = [_run(first, bads=["S1_D1 hbo"]), _run(second)]
    roi_map = {"A": ["S1_D1"], "B": ["S2_D2"]}
    signals, members = join_roi_signals(runs, ["run-01", "run-02"], roi_map, "hbo")
    assert np.isnan(signals["A"][:50]).all() and "A" not in members["run-01"]
    joined, _ = join_runs(runs)
    fc = compute_fc_roi(joined, roi_map, "hbo", signals=signals)
    assert fc.loc["A", "B"] == pytest.approx(np.corrcoef(second[0], second[1])[0, 1], abs=1e-12)
    seed = compute_fc_seed(joined, roi_map, "hbo", signals=signals,
                           members={"A": ["S1_D1 hbo"], "B": ["S2_D2 hbo"]})
    assert np.isnan(seed.loc["A", "S1_D1 hbo"])
    assert seed.loc["A", "S2_D2 hbo"] == pytest.approx(fc.loc["A", "B"], abs=1e-12)


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
