"""What `--isc-whiten` and `--isc-phase-null` add to the inter-brain correlation.

A haemoglobin trace is strongly autocorrelated, so a Pearson r between two of them rests on
far fewer independent observations than it has samples and the value it reaches with no
coupling at all is large. Two answers to that, and the tests here hold both: whitening moves
the estimate onto the scale its sample count implies, and the phase-scrambled null measures
that scale directly. Scrambling preserves each signal's own spectrum, so the null carries the
autocorrelation whether or not the whitening ran, which is why the two compose.

The statistical tests run on the private row-level helpers rather than through Raw objects:
what is under test is the estimator, and building a montage around it would only make the
signals harder to control.
"""

import mne
import numpy as np
import pytest

from fnirs_pipe.pipeline.hyper.isc import (
    _ar_whiten, _isc_from_rows, _isc_matrix, compute_isc, compute_isc_pairs,
)
from tests._synth import synth_raw

N = 4000


def _ar(rng, coefs, n=N):
    """One realisation of the autoregressive process with these coefficients."""
    p = len(coefs)
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(p, n):
        x[t] = coefs @ x[t - p:t][::-1] + e[t]
    return x


def _haemo(subject: str) -> mne.io.Raw:
    raw = synth_raw(subject, "hold", duration=40.0, motion_onset=None)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    return mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)


# ---- the autoregressive fit ----

def test_the_order_chosen_is_the_order_the_process_has():
    """BIC has to find the generating order, or whitening either leaves autocorrelation
    behind or spends coefficients removing signal."""
    rng = np.random.default_rng(0)
    assert _ar_whiten(_ar(rng, np.array([0.9])))[1] == 1
    order = _ar_whiten(_ar(rng, np.array([0.4, -0.3, 0.2, -0.1, 0.25])))[1]
    assert 4 <= order <= 7


def test_whitening_leaves_a_series_without_its_autocorrelation():
    rng = np.random.default_rng(1)
    x = _ar(rng, np.array([0.97]))
    lag1 = lambda s: float(np.corrcoef(s[:-1], s[1:])[0, 1])

    assert lag1(x) > 0.9
    assert abs(lag1(_ar_whiten(x)[0][40:])) < 0.1


@pytest.mark.parametrize("bad", ["nan", "constant", "short"])
def test_a_row_that_cannot_be_fitted_comes_back_untouched(bad):
    """A rejected channel arrives as NaN and has to survive as NaN: order 0 and the input
    back, not an exception and not zeros."""
    x = {"nan": np.full(N, np.nan), "constant": np.ones(N), "short": np.arange(4.0)}[bad]
    out, order = _ar_whiten(x)

    assert order == 0
    assert out.shape == x.shape
    np.testing.assert_array_equal(np.isnan(out), np.isnan(x))


# ---- what whitening does to the correlation ----

def test_whitening_puts_r_back_on_the_scale_its_sample_count_implies():
    """The claim the whole option rests on. Independent pairs, so every r is null: without
    whitening their spread is far wider than 1/sqrt(n), with it, close to it."""
    rng = np.random.default_rng(2)
    a = np.array([_ar(rng, np.array([0.97])) for _ in range(24)])
    b = np.array([_ar(rng, np.array([0.97])) for _ in range(24)])

    raw_r = _isc_matrix(a, b, 0)[0].ravel()
    wht_r = _isc_matrix(a, b, 32)[0].ravel()
    expected = 1.0 / np.sqrt(N)

    assert raw_r.std() > 4 * expected
    assert wht_r.std() == pytest.approx(expected, rel=0.25)


def test_whitening_keeps_a_shared_driver_visible_against_its_own_null():
    """Shrinking every r would be no use if it shrank a real coupling into the null with
    everything else. The pair sharing a driver has to stand out after whitening too."""
    rng = np.random.default_rng(3)
    driver = _ar(rng, np.array([0.97]))
    a = np.array([driver + 2.0 * _ar(rng, np.array([0.97])) for _ in range(6)])
    b = np.array([driver + 2.0 * _ar(rng, np.array([0.97])) for _ in range(6)])
    independent = np.array([_ar(rng, np.array([0.97])) for _ in range(6)])

    coupled = np.abs(_isc_matrix(a, b, 32)[0]).mean()
    uncoupled = np.abs(_isc_matrix(a, independent, 32)[0]).mean()
    assert coupled > 5 * uncoupled


def test_whiten_zero_is_the_unwhitened_correlation():
    """The off switch has to be a true one: the matrix a 0 gives is the plain Pearson r."""
    rng = np.random.default_rng(4)
    a = np.array([_ar(rng, np.array([0.9])) for _ in range(3)])
    b = np.array([_ar(rng, np.array([0.9])) for _ in range(3)])

    mat, orders1, orders2, lags = _isc_matrix(a, b, 0)
    assert orders1 is None and orders2 is None
    assert not lags.any(), "no search was asked for, so every cell won at lag 0"
    assert mat == pytest.approx(_isc_from_rows(a, b)[0])
    assert mat[0, 0] == pytest.approx(np.corrcoef(a[0], b[0])[0, 1], abs=1e-9)


# ---- the pair table ----

@pytest.fixture(scope="module")
def dyad():
    return {"11": _haemo("11"), "12": _haemo("12")}


def test_the_pair_table_carries_the_z_and_the_order_each_channel_used(dyad):
    mat, names, frame, level = compute_isc_pairs(dyad, ["11", "12"], "hbo", whiten=16)

    assert len(frame) == len(names) ** 2
    assert list(frame.columns[:6]) == ["sub1", "sub2", "label", "label2", "r", "r_z"]
    assert frame["r_z"].to_numpy() == pytest.approx(
        np.arctanh(frame["r"].clip(-0.999999, 0.999999).to_numpy()), nan_ok=True)
    assert {"ar_order", "ar_order2"} <= set(frame.columns)
    assert (frame["ar_order"] >= 0).all()
    # the frame is the matrix, read the other way round
    index = {n: i for i, n in enumerate(names)}
    row = frame.iloc[7]
    assert row["r"] == pytest.approx(mat[index[row["label"]], index[row["label2"]]])


def test_the_order_columns_are_absent_when_nothing_was_whitened(dyad):
    _, _, frame, _ = compute_isc_pairs(dyad, ["11", "12"], "hbo", whiten=0)
    assert "ar_order" not in frame.columns
    assert "percentile" not in frame.columns


def test_the_null_columns_rank_the_magnitude_not_the_sign(dyad):
    """A correlation is signed and a surrogate is as likely to land either side of zero, so
    the question the null answers is whether the pair is coupled, not which way. A strongly
    negative r has to score high, not low."""
    ids = ["11", "12"]
    raws = {k: v.copy() for k, v in dyad.items()}
    picks = mne.pick_types(raws["11"].info, fnirs="hbo")
    rng = np.random.default_rng(5)
    shared = _ar(rng, np.array([0.9]), n=raws["11"].n_times)
    raws["11"]._data[picks[0]] = 1e-6 * shared
    raws["12"]._data[picks[0]] = -1e-6 * shared

    _, names, frame, level = compute_isc_pairs(raws, ids, "hbo", whiten=0, n_null=20, seed=1)
    assert {"null_abs_mean", "null_abs_sd", "null_abs_p95", "percentile"} <= set(frame.columns)

    label = names[0]
    cell = frame[(frame["label"] == label) & (frame["label2"] == label)].iloc[0]
    assert cell["r"] < -0.9
    assert cell["percentile"] == 100.0
    assert (frame["null_abs_mean"] >= 0).all()


def test_a_rejected_channel_is_blank_in_the_table_and_is_not_ranked(dyad):
    """Blank rather than dropped, the convention every per-channel product here follows, and
    a NaN r cannot be ranked against draws that are NaN too."""
    ids = ["11", "12"]
    raws = {k: v.copy() for k, v in dyad.items()}
    rejected = "S2_D2"
    raws["11"].info["bads"] = [c for c in raws["11"].ch_names
                                  if c.startswith(rejected)]

    mat, names, frame, level = compute_isc_pairs(raws, ids, "hbo", whiten=16, n_null=5, seed=1)
    assert rejected in names                       # the axis is the montage, not the survivors
    blanked = frame[frame["label"] == rejected]
    assert blanked["r"].isna().all()
    assert blanked["percentile"].isna().all()
    # only the rejecting member's side goes blank
    assert frame[frame["label2"] == rejected]["r"].notna().any()


def test_the_matrix_entry_point_agrees_with_the_table(dyad):
    """`compute_isc` and `compute_isc_pairs` are two doors onto one computation; the report
    draws one and writes the other, so a divergence would be invisible."""
    ids = ["11", "12"]
    direct, names_a = compute_isc(dyad, ids, "hbo", whiten=16)
    via_table, names_b, _, _ = compute_isc_pairs(dyad, ids, "hbo", whiten=16)

    assert names_a == names_b
    assert direct == pytest.approx(via_table, nan_ok=True)


# ---- the lag search ----

def test_a_shifted_copy_is_found_at_the_shift_it_was_made_with():
    """The point of the search: two haemodynamic responses do not peak together, and a
    same-sample correlation reads a coupling a second apart as no coupling."""
    rng = np.random.default_rng(6)
    source = _ar(rng, np.array([0.9]))
    shift = 7
    a = np.vstack([source])
    b = np.vstack([np.r_[np.zeros(shift), source[:-shift]]])

    at_zero, lag_zero = _isc_from_rows(a, b, max_lag=0)
    best, lag = _isc_from_rows(a, b, max_lag=20)

    assert abs(at_zero[0, 0]) < 0.8
    assert best[0, 0] > 0.95
    assert lag[0, 0] == shift
    assert lag_zero[0, 0] == 0


def test_the_search_keeps_the_sign_of_an_anticorrelated_pairing():
    """Largest in magnitude, not largest signed: the published form suits a metric built for
    positive coupling, and this matrix carries both signs."""
    rng = np.random.default_rng(7)
    source = _ar(rng, np.array([0.9]))
    a = np.vstack([source])
    b = np.vstack([-source])

    best, _ = _isc_from_rows(a, b, max_lag=10)
    assert best[0, 0] < -0.95


def test_searching_raises_the_value_under_no_coupling():
    """A maximum over many shifts is larger than any one of them, which is why a lagged run
    belongs with a null searched the same way."""
    rng = np.random.default_rng(8)
    a = np.array([_ar(rng, np.array([0.5])) for _ in range(8)])
    b = np.array([_ar(rng, np.array([0.5])) for _ in range(8)])

    at_zero = np.abs(_isc_from_rows(a, b, max_lag=0)[0]).mean()
    searched = np.abs(_isc_from_rows(a, b, max_lag=25)[0]).mean()
    assert searched > at_zero


def test_the_null_is_searched_the_same_way_as_the_value(dyad):
    """So the inflation the search adds is in both and the percentile stays readable."""
    ids = ["11", "12"]
    _, _, plain, _ = compute_isc_pairs(dyad, ids, "hbo", n_null=20, seed=2)
    _, _, lagged, _ = compute_isc_pairs(dyad, ids, "hbo", max_lag_s=2.0, n_null=20, seed=2)

    assert "lag_s" not in plain.columns
    assert "lag_s" in lagged.columns
    assert lagged["lag_s"].abs().max() <= 2.0 + 1e-9
    # both sides rose, so the ranks do not saturate as a searched value against an unsearched
    # null would
    assert lagged["null_abs_mean"].mean() > plain["null_abs_mean"].mean()
    assert lagged["percentile"].mean() < 90
    assert (lagged["percentile"] < 100).any()


# ---- which pairings get a chord ----

def test_the_arc_rule_prefers_a_null_to_a_number_and_a_number_to_a_quantile():
    """Three tiers in the order they deserve to be believed in: a per-cell surrogate level is
    a test, a fixed cut is a number somebody chose on a scale that moves with the
    preprocessing, and a quantile of the matrix keeps the same share whatever the data did."""
    from fnirs_pipe.qc.figures.hyper.hyper_post_figures import _arc_rule

    z = np.array([[0.10, 0.50, np.nan],
                  [0.90, -0.70, 0.20],
                  [0.05, 0.30, 0.40]])
    level = np.full_like(z, 0.45)

    kept, rule = _arc_rule(z, "Pearson r", 0.3, level, 0.9)
    assert kept.sum() == 5 and "0.3" in rule            # an explicit number wins outright
    kept, rule = _arc_rule(z, "Pearson r", None, level, 0.9)
    assert kept.sum() == 3 and "own null" in rule
    kept, rule = _arc_rule(z, "Pearson r", None, None, 0.9)
    assert "display cut" in rule and "not a test" in rule
    kept, rule = _arc_rule(z, "Pearson r", None, None, None)
    assert kept.sum() == 8 and "all 8" in rule          # every finite pairing, the NaN aside


def test_a_cell_with_no_level_measured_gets_no_chord():
    """A NaN level is a cell the surrogates never reached; drawing it would read as a pairing
    that beat a null that was never taken."""
    from fnirs_pipe.qc.figures.hyper.hyper_post_figures import _arc_rule

    z = np.array([[0.9, 0.9]])
    level = np.array([[0.1, np.nan]])
    kept, _ = _arc_rule(z, "Pearson r", None, level, None)
    assert kept.tolist() == [[True, False]]


def test_the_null_level_comes_back_as_a_matrix_shaped_like_the_correlations(dyad):
    """The panel thresholds cell by cell, so the level has to leave `compute_isc_pairs` as a
    matrix and not only as a column of the table."""
    mat, names, frame, level = compute_isc_pairs(dyad, ["11", "12"], "hbo",
                                                 n_null=10, seed=3)
    assert level.shape == mat.shape == (len(names), len(names))
    index = {n: i for i, n in enumerate(names)}
    row = frame.iloc[5]
    assert level[index[row["label"]], index[row["label2"]]] == pytest.approx(row["null_abs_p95"])


def test_no_null_means_no_level(dyad):
    _, _, _, level = compute_isc_pairs(dyad, ["11", "12"], "hbo")
    assert level is None
