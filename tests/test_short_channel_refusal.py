"""Asking for short-channel regression on a montage that has none is refused, not skipped.

The methods text names the regressors that were requested, so a run that quietly drops them
publishes a claim its residual does not support.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.glm import _short_channel_regressors


def _haemo(separations_mm):
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
    return mne.io.RawArray(np.random.default_rng(0).normal(size=(len(names), 600)),
                           info, verbose="ERROR")


def test_a_montage_with_no_short_channel_is_refused():
    raw = _haemo([30.0, 35.0, 40.0])
    with pytest.raises(StageError, match="no channel at or under"):
        _short_channel_regressors(raw, "mean")


def test_the_refusal_names_the_shortest_channel_and_the_flag():
    """The fix is a threshold, so the message has to carry both numbers."""
    raw = _haemo([12.8, 35.0])
    with pytest.raises(StageError) as exc:
        _short_channel_regressors(raw, "mean")
    assert "12.8 mm" in str(exc.value)
    assert "10 mm" in str(exc.value)
    assert "--short-max-dist" in str(exc.value)


def test_a_threshold_that_admits_them_builds_the_regressors():
    """The 12.8 mm montage is usable once the band is told those are short."""
    raw = _haemo([12.8, 35.0])
    out = _short_channel_regressors(raw, "mean", sep_bands=(0.014, 0.015, None))
    assert set(out) == {"short_ch_hbo_mean", "short_ch_hbr_mean"}
    assert len(out["short_ch_hbo_mean"]) == raw.n_times


def test_a_montage_with_short_channels_is_unaffected():
    raw = _haemo([8.0, 35.0])
    out = _short_channel_regressors(raw, "mean")
    assert set(out) == {"short_ch_hbo_mean", "short_ch_hbr_mean"}


def test_a_subject_whose_short_channels_are_all_bad_still_runs(caplog):
    """Per subject rather than per montage: refusing here would kill a whole batch."""
    raw = _haemo([8.0, 35.0])
    raw.info["bads"] = [ch for ch in raw.ch_names if ch.startswith("S1_D1")]
    with caplog.at_level("WARNING"):
        assert _short_channel_regressors(raw, "mean") == {}
    assert "methods text still names it" in caplog.text
