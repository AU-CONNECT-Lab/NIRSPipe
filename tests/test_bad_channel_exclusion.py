"""Rejected channels must not reach an aggregate that every good channel is scaled by.

Three places had this wrong. The worst was the short-channel regressor: it sits in the
design matrix, so one bad short channel shifted every channel's fit. None of the three
raises, none of them shows up on clean data, and all three change numbers silently,
which is why each is tested the same way — mark a channel bad, then make that channel
absurd, and assert the aggregate did not move. Each pairing has a companion test that
the same distortion *does* move the aggregate while the channel is good, so a function
that stopped reading the channel at all cannot pass by accident.
"""

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from fnirs_pipe.pipeline.glm import _short_channel_regressors
from fnirs_pipe.pipeline.restingstate import compute_alff, compute_fc_roi

SFREQ = 10.0
DURATION = 60.0
SPIKE = 1e3          # far outside any plausible signal, so a leak cannot hide in the noise

BAND = dict(high_pass=0.01, low_pass=0.08)


# ---- short-channel regressors ----

SHORT_HBO = "S4_D4 hbo"          # first short pair, see _haemo below
OTHER_SHORT_HBO = "S5_D5 hbo"


def _haemo(n_long: int = 3, n_short: int = 2, seed: int = 0) -> mne.io.Raw:
    """Haemoglobin Raw whose short pairs sit below MNE's 1 cm short-channel threshold."""
    rng = np.random.default_rng(seed)
    n_times = int(SFREQ * DURATION)

    names, locs, types = [], [], []
    for idx in range(n_long + n_short):
        distance = 0.03 if idx < n_long else 0.008
        src = np.array([idx * 0.03, 0.0, 0.0])
        det = src + np.array([distance, 0.0, 0.0])
        for chromo in ("hbo", "hbr"):
            loc = np.zeros(12)
            loc[0:3] = (src + det) / 2
            loc[3:6] = src
            loc[6:9] = det
            names.append(f"S{idx + 1}_D{idx + 1} {chromo}")
            locs.append(loc)
            types.append(chromo)

    info = mne.create_info(names, SFREQ, types)
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc
    data = 1e-6 * rng.normal(size=(len(names), n_times))
    return mne.io.RawArray(data, info, verbose="error")


@pytest.fixture(scope="module")
def haemo():
    return _haemo()


def _regressors(raw, bads=(), spike=None):
    raw = raw.copy()
    raw.info["bads"] = list(bads)
    if spike is not None:
        raw._data[raw.ch_names.index(spike)] += SPIKE
    return _short_channel_regressors(raw, "mean")


def test_a_strategy_that_does_not_exist_is_refused_rather_than_quietly_averaged(haemo):
    """A `--config` TOML reaches this function past the parser's own choices. Averaging
    instead of what was asked for would put a different regressor in the design matrix than
    the run record says ran, and the two strategies are not variants of one regressor: the
    mean is two columns, the basis is one per short channel."""
    with pytest.raises(ValueError, match="must be 'mean' or 'pca'"):
        _short_channel_regressors(haemo, "nearest")


def test_a_bad_short_channel_does_not_reach_the_basis_either(haemo):
    """The exclusion is the strategy's shared half, so it holds for both of them."""
    raw = haemo.copy()
    raw.info["bads"] = [SHORT_HBO]
    clean = _short_channel_regressors(raw, "pca")

    spiked = haemo.copy()
    spiked.info["bads"] = [SHORT_HBO]
    spiked._data[spiked.ch_names.index(SHORT_HBO)] += SPIKE
    for name, values in clean.items():
        # a sign flip is not a difference: the basis is defined up to it
        assert_allclose(np.abs(values), np.abs(_short_channel_regressors(spiked, "pca")[name]),
                        err_msg=name)


def test_a_bad_short_channel_does_not_reach_the_regressor(haemo):
    clean = _regressors(haemo, bads=[SHORT_HBO])
    spiked = _regressors(haemo, bads=[SHORT_HBO], spike=SHORT_HBO)

    assert set(clean) == {"short_ch_hbo_mean", "short_ch_hbr_mean"}
    for name, values in clean.items():
        assert_allclose(values, spiked[name], err_msg=name)


def test_the_same_spike_moves_the_regressor_while_the_channel_is_good(haemo):
    clean = _regressors(haemo)
    spiked = _regressors(haemo, spike=SHORT_HBO)
    assert not np.allclose(clean["short_ch_hbo_mean"], spiked["short_ch_hbo_mean"])


def test_no_regressors_when_every_short_channel_of_a_chromophore_is_bad(haemo):
    assert _regressors(haemo, bads=[SHORT_HBO, OTHER_SHORT_HBO]) == {}


def test_a_bad_long_channel_leaves_the_regressor_alone(haemo):
    # long channels are not in the regressor to begin with; this pins that the bads
    # filter did not widen into something that drops data it should keep
    clean = _regressors(haemo)
    with_bad = _regressors(haemo, bads=["S1_D1 hbo"])
    for name, values in clean.items():
        assert_allclose(values, with_bad[name], err_msg=name)


# ---- mALFF / zALFF reference ----

def _oscillating(n_ch: int = 4, seed: int = 1) -> np.ndarray:
    """In-band oscillations of differing amplitude, so ALFF varies across channels."""
    rng = np.random.default_rng(seed)
    n_times = int(SFREQ * DURATION)
    t = np.arange(n_times) / SFREQ
    return np.array([
        (1.0 + 0.5 * i) * np.sin(2 * np.pi * 0.05 * t) + 0.1 * rng.normal(size=n_times)
        for i in range(n_ch)
    ])


def _alff(make_raw, bads=(), boost=None):
    data = _oscillating()
    if boost is not None:
        data[boost] *= 100.0
    raw = make_raw(data=data)
    raw.info["bads"] = [raw.ch_names[i] for i in bads]
    return compute_alff(raw, **BAND)


def test_the_malff_reference_ignores_a_bad_channel(make_raw):
    clean = _alff(make_raw, bads=[2])
    boosted = _alff(make_raw, bads=[2], boost=2)

    good = clean["bad"] == False  # noqa: E712 — pandas mask, not a truth test
    assert good.sum() == 3
    assert_allclose(clean.loc[good, "malff"], boosted.loc[good, "malff"])
    assert_allclose(clean.loc[good, "zalff"], boosted.loc[good, "zalff"])


def test_the_same_boost_moves_the_reference_while_the_channel_is_good(make_raw):
    clean = _alff(make_raw)
    boosted = _alff(make_raw, boost=2)
    assert not np.allclose(clean["malff"].to_numpy()[:2], boosted["malff"].to_numpy()[:2])


def test_a_bad_channel_keeps_its_row_and_loses_its_values(make_raw):
    # the row stays so every subject's frame has the same shape and the same order; the
    # values go because a channel preprocessing threw out has no measurement to report
    frame = _alff(make_raw, bads=[2])
    assert frame.loc[2, "bad"]
    assert frame.loc[2, ["alff", "falff", "malff", "zalff"]].isna().all()
    assert len(frame) == 4


def test_every_channel_bad_leaves_the_frame_blank_rather_than_raising(make_raw):
    # the reference falls back to the whole set, but every row is rejected, so every row is
    # blank. A run this bad has to reach the report as an empty panel, not as a traceback
    frame = _alff(make_raw, bads=[0, 1, 2, 3])
    assert frame["bad"].all()
    assert frame[["alff", "falff", "malff", "zalff"]].isna().all().all()


# ---- ROI connectivity ----

ROI_MAP = {"left": ["S0_D0", "S1_D1"], "right": ["S2_D2", "S3_D3"]}


def _roi_fc(make_raw, bads=(), spike=None):
    rng = np.random.default_rng(2)
    n_times = int(SFREQ * DURATION)
    t = np.arange(n_times) / SFREQ
    data = np.array([
        np.sin(2 * np.pi * 0.05 * t + 0.3 * i) + 0.2 * rng.normal(size=n_times)
        for i in range(4)
    ])
    if spike is not None:
        data[spike] += SPIKE * rng.normal(size=n_times)
    raw = make_raw(data=data)
    raw.info["bads"] = [raw.ch_names[i] for i in bads]
    return compute_fc_roi(raw, ROI_MAP, chromophore="hbo")


def test_a_bad_channel_does_not_enter_the_roi_average(make_raw):
    clean = _roi_fc(make_raw, bads=[1])
    noisy = _roi_fc(make_raw, bads=[1], spike=1)
    assert_allclose(clean.to_numpy(), noisy.to_numpy())


def test_the_same_noise_moves_the_roi_correlation_while_the_channel_is_good(make_raw):
    clean = _roi_fc(make_raw)
    noisy = _roi_fc(make_raw, spike=1)
    assert not np.allclose(clean.to_numpy(), noisy.to_numpy())


def test_an_roi_whose_channels_are_all_bad_is_dropped(make_raw):
    fc = _roi_fc(make_raw, bads=[0, 1])
    assert fc.empty                      # one ROI left, and FC needs two
