"""The heart rate: each channel's spectral peak above the respiration band, its median over kept channels."""

import logging

import mne
import numpy as np
import pytest

from nirspipe.qc.common.channel_table import heart_rate_note
from nirspipe.qc.metrics.coupling import _heart_rate_metrics
from nirspipe.qc.subject.sqm_record import raw_sections

from tests._synth import CARDIAC_FREQ, _channel_layout, synth_raw

# 3 beats per minute
TOLERANCE_HZ = 0.05


def _recording(pulse_hz, sfreq=10.0, duration=120.0, breath_hz=0.25, breath_amp=0.004,
               flat=()):
    """Intensity whose pair k carries a pulse at ``pulse_hz[k]`` at both wavelengths, over breathing and noise."""
    names, locs, _ = _channel_layout(len(pulse_hz), short=False)
    rng = np.random.default_rng(0)
    t = np.arange(int(sfreq * duration)) / sfreq
    data = np.empty((len(names), t.size))
    for i in range(len(names)):
        k = i // 2
        data[i] = 0.1 * (1.0 + 0.02 * np.sin(2 * np.pi * pulse_hz[k] * t)
                         + breath_amp * np.sin(2 * np.pi * breath_hz * t)
                         + 0.005 * rng.normal(size=t.size))
        if k in flat:
            data[i] = 0.1
    info = mne.create_info(names, sfreq, ["fnirs_cw_amplitude"] * len(names))
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc
    return mne.io.RawArray(data, info, verbose="error")


def test_the_heart_rate_is_the_shared_pulse():
    out = _heart_rate_metrics(_recording([1.2] * 4), 0.5, set())
    assert out["heart_rate"] == pytest.approx(1.2, abs=TOLERANCE_HZ)
    assert len(out["heart_rate_per_channel"]) == 8


def test_breathing_stronger_than_the_pulse_is_left_below_the_search():
    raw = _recording([1.2] * 4, breath_hz=0.4, breath_amp=0.05)
    assert _heart_rate_metrics(raw, 0.5, set())["heart_rate"] == pytest.approx(1.2, abs=TOLERANCE_HZ)
    # a respiration band that ends below the breathing lets it win, so the band is what keeps it out
    assert _heart_rate_metrics(raw, 0.3, set())["heart_rate"] == pytest.approx(0.4, abs=TOLERANCE_HZ)


def test_a_pulse_above_the_cardiac_band_is_found_and_named():
    hr = _heart_rate_metrics(_recording([2.3] * 4), 0.5, set())["heart_rate"]
    assert hr == pytest.approx(2.3, abs=TOLERANCE_HZ)
    assert "outside the cardiac band" in heart_rate_note(hr, (0.7, 1.8))
    assert heart_rate_note(hr, (1.5, 3.1)) is None
    assert heart_rate_note(None, (0.7, 1.8)) is None


def test_nothing_above_where_a_cardiac_band_could_reach_is_reported():
    # at 5 Hz no cardiac band may end above 2.2 Hz, so a 2.4 Hz pulse is out of reach
    hr = _heart_rate_metrics(_recording([2.4] * 4, sfreq=5.0), 0.5, set())["heart_rate"]
    assert hr <= 2.2


def test_rejected_channels_do_not_vote():
    raw = _recording([1.2, 1.2, 2.0, 2.0, 2.0])
    names = raw.ch_names
    rejected = {n for n in names if n.split()[0] in ("S3_D3", "S4_D4", "S5_D5")}
    assert _heart_rate_metrics(raw, 0.5, rejected)["heart_rate"] == pytest.approx(1.2, abs=TOLERANCE_HZ)
    assert _heart_rate_metrics(raw, 0.5, set())["heart_rate"] == pytest.approx(2.0, abs=TOLERANCE_HZ)
    # every channel keeps its own peak, rejected or not
    per_channel = _heart_rate_metrics(raw, 0.5, rejected)["heart_rate_per_channel"]
    assert set(per_channel) == set(names)


def test_a_flat_channel_has_no_peak():
    out = _heart_rate_metrics(_recording([1.2] * 4, flat=(1,)), 0.5, set())
    assert not {"S2_D2 760", "S2_D2 850"} & set(out["heart_rate_per_channel"])
    assert out["heart_rate"] == pytest.approx(1.2, abs=TOLERANCE_HZ)


def test_every_raw_section_carries_its_own_heart_rate():
    raw = synth_raw("01", "rest", duration=120.0, motion_onset=None)
    sci = {ch: 0.9 for ch in raw.ch_names}
    sections, per_channel = raw_sections(raw, sci, [], 0.7, 1.8, resp_h_freq=0.5)
    for name in ("raw", "raw_long", "raw_short"):
        assert sections[name]["heart_rate"] == pytest.approx(CARDIAC_FREQ, abs=TOLERANCE_HZ), name
    assert set(per_channel["raw_long"]["heart_rate_per_channel"]) == {
        ch for ch in raw.ch_names if not ch.startswith("S5_D5")}


def test_a_heart_rate_outside_the_cardiac_band_is_warned_in_the_log(caplog):
    raw = synth_raw("01", "rest", duration=120.0, motion_onset=None)
    sci = {ch: 0.9 for ch in raw.ch_names}
    with caplog.at_level(logging.WARNING):
        raw_sections(raw, sci, [], 1.5, 3.1, resp_h_freq=1.1)
    assert any("outside the cardiac band" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        raw_sections(raw, sci, [], 0.7, 1.8, resp_h_freq=0.5)
    assert not any("outside the cardiac band" in r.getMessage() for r in caplog.records)
