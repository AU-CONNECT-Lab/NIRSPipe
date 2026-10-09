"""A short channel that is the whole regressor is fitted against a copy of itself.

The `mean` regressor is the mean of the good short channels of a chromophore, and the design
matrix is applied to every channel including the ones it was built from. With one surviving
short channel that mean *is* that channel sample for sample, so its own residual is
numerically zero and every measure read off the residual is an artefact of the method rather
than a measurement.

What is pinned here is the line, at one channel and not at some share, and that the
correction reaches the tables without touching any other channel.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nirspipe.pipeline.glm import _short_channel_regressors, sole_regressor_channels


def _haemo(separations_mm, bads=()):
    """One hbo/hbr pair per separation, positions set so the distances are real."""
    names, locs = [], []
    for i, mm in enumerate(separations_mm):
        src = np.array([i * 0.05, 0.0, 0.0])
        det = src + np.array([mm / 1000.0, 0.0, 0.0])
        for chroma in ("hbo", "hbr"):
            loc = np.zeros(12)
            loc[0:3], loc[3:6], loc[6:9] = (src + det) / 2, src, det
            names.append(f"S{i + 1}_D{i + 1} {chroma}")
            locs.append(loc)
    info = mne.create_info(names, sfreq=10.0, ch_types=["hbo", "hbr"] * len(separations_mm))
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc
    raw = mne.io.RawArray(np.random.default_rng(0).normal(size=(len(names), 600)),
                          info, verbose="ERROR")
    raw.info["bads"] = list(bads)
    return raw


def test_one_surviving_short_channel_is_named():
    raw = _haemo([8.0, 8.5, 30.0], bads=["S2_D2 hbo", "S2_D2 hbr"])
    assert sole_regressor_channels(raw, "mean") == ["S1_D1 hbo", "S1_D1 hbr"]


def test_two_surviving_short_channels_are_not():
    """Their residual is the channel minus the mean of the two, which is a real if small
    quantity; only at one is it identically zero."""
    raw = _haemo([8.0, 8.5, 30.0])
    assert sole_regressor_channels(raw, "mean") == []


def test_the_pca_strategy_falls_the_same_way():
    """A single channel's basis is that channel, however it is decomposed."""
    raw = _haemo([8.0, 8.5, 30.0], bads=["S2_D2 hbo", "S2_D2 hbr"])
    assert sole_regressor_channels(raw, "pca") == ["S1_D1 hbo", "S1_D1 hbr"]


def test_nothing_is_claimed_without_the_regression():
    raw = _haemo([8.0, 30.0])
    assert sole_regressor_channels(raw, None) == []
    assert sole_regressor_channels(raw, False) == []


def test_a_montage_with_no_short_channel_claims_nothing_rather_than_raising():
    """The refusal is `_short_channel_regressors`' to make, and it carries the message that
    names the flag; this one is asked in places that must not raise."""
    assert sole_regressor_channels(_haemo([30.0, 35.0]), "mean") == []


def test_the_regressor_is_still_built_and_still_serves_the_long_channels():
    """The long channels are regressed correctly by a one-channel regressor: it is a fine
    estimate of the systemic signal. Only its own source loses a residual, so the run must
    not be refused."""
    raw = _haemo([8.0, 8.5, 30.0], bads=["S2_D2 hbo", "S2_D2 hbr"])
    out = _short_channel_regressors(raw, "mean")
    assert set(out) == {"short_ch_hbo_mean", "short_ch_hbr_mean"}
    kept = raw.get_data(picks=["S1_D1 hbo"])[0]
    assert np.allclose(out["short_ch_hbo_mean"], kept)


def test_the_warning_names_the_channel_and_says_what_is_blanked(caplog):
    raw = _haemo([8.0, 8.5, 30.0], bads=["S2_D2 hbo", "S2_D2 hbr"])
    with caplog.at_level("WARNING"):
        _short_channel_regressors(raw, "mean")
    said = caplog.text
    assert "S1_D1 hbo" in said and "ALFF and FC are left blank" in said
