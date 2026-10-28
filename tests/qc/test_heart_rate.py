"""The heart rate: each channel's highest spectral peak above its background, above the respiration band, their median."""

import json
import logging
import re

import mne
import numpy as np
import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.qc.common.channel_table import heart_rate_note
from nirspipe.qc.figures.common._utils import spectrum_fmax
from nirspipe.qc.metrics.coupling import _heart_rate_metrics
from nirspipe.qc.subject.sqm_record import raw_sections

from tests._synth import CARDIAC_FREQ, _channel_layout, make_bids_dataset, synth_raw

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
    out = _heart_rate_metrics(_recording([1.2] * 4), 0.5)
    assert out["heart_rate"] == pytest.approx(1.2, abs=TOLERANCE_HZ)
    assert len(out["heart_rate_per_channel"]) == 8


def test_breathing_ten_times_the_pulse_just_under_the_floor_leaves_it_alone():
    # its tail and its harmonic at 0.9 Hz reach into the search
    raw = _recording([1.2] * 4, duration=600.0, breath_hz=0.45, breath_amp=0.2)
    assert _heart_rate_metrics(raw, 0.5)["heart_rate"] == pytest.approx(1.2, abs=TOLERANCE_HZ)


def test_a_breathing_harmonic_louder_than_the_pulse_is_read_as_the_heart_rate():
    # the limit the docstring states: fifteen times the pulse puts its 0.9 Hz harmonic highest
    raw = _recording([1.2] * 4, duration=600.0, breath_hz=0.45, breath_amp=0.3)
    assert _heart_rate_metrics(raw, 0.5)["heart_rate"] == pytest.approx(0.9, abs=TOLERANCE_HZ)


def _power_law(n, sfreq, rng):
    """Unit-variance noise whose power falls as 1/f^2."""
    f = np.fft.rfftfreq(n, 1 / sfreq)
    amp = np.zeros_like(f)
    amp[1:] = 1 / f[1:]
    x = np.fft.irfft(amp * np.exp(2j * np.pi * rng.random(f.size)), n)
    return x / x.std()


def test_a_slow_pulse_high_on_the_background_wins_over_its_harmonic():
    # 51 bpm with a second harmonic half its size, over a background falling as 1/f^2: the
    # harmonic stands on a lower background, so it rose further above its neighbours
    names, locs, _ = _channel_layout(4, short=False)
    rng = np.random.default_rng(0)
    sfreq = 10.0
    t = np.arange(int(sfreq * 600.0)) / sfreq
    pulse = np.sin(2 * np.pi * 0.85 * t) + 0.5 * np.sin(2 * np.pi * 1.7 * t)
    data = np.array([0.1 * (1.0 + 0.01 * pulse + 0.01 * _power_law(t.size, sfreq, rng)
                            + 0.0002 * rng.normal(size=t.size)) for _ in names])
    info = mne.create_info(names, sfreq, ["fnirs_cw_amplitude"] * len(names))
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc
    raw = mne.io.RawArray(data, info, verbose="error")
    assert _heart_rate_metrics(raw, 0.5)["heart_rate"] == pytest.approx(0.85, abs=TOLERANCE_HZ)


def test_breathing_stronger_than_the_pulse_is_left_below_the_search():
    raw = _recording([1.2] * 4, breath_hz=0.4, breath_amp=0.05)
    assert _heart_rate_metrics(raw, 0.5)["heart_rate"] == pytest.approx(1.2, abs=TOLERANCE_HZ)
    # a respiration band that ends below the breathing lets it win, at 0.4 Hz or its 0.8 Hz
    # harmonic, so the band is what keeps it out
    assert _heart_rate_metrics(raw, 0.3)["heart_rate"] < 1.0


def test_a_pulse_above_the_cardiac_band_is_found_and_named():
    hr = _heart_rate_metrics(_recording([2.3] * 4), 0.5)["heart_rate"]
    assert hr == pytest.approx(2.3, abs=TOLERANCE_HZ)
    assert "outside the cardiac band" in heart_rate_note(hr, (0.7, 1.8))
    assert heart_rate_note(hr, (1.5, 3.1)) is None
    assert heart_rate_note(None, (0.7, 1.8)) is None


def test_nothing_above_where_a_cardiac_band_could_reach_is_reported():
    # at 5 Hz no cardiac band may end above 2.2 Hz, so a 2.4 Hz pulse is out of reach
    hr = _heart_rate_metrics(_recording([2.4] * 4, sfreq=5.0), 0.5)["heart_rate"]
    assert hr <= 2.2


def test_the_heart_rate_is_the_median_channel():
    raw = _recording([1.2, 1.2, 2.0, 2.0, 2.0])
    out = _heart_rate_metrics(raw, 0.5)
    assert out["heart_rate"] == pytest.approx(2.0, abs=TOLERANCE_HZ)
    assert set(out["heart_rate_per_channel"]) == set(raw.ch_names)


def test_a_recording_screening_rejected_whole_still_has_a_heart_rate():
    # a band that misses the pulse rejects every channel, and the heart rate is what says so
    raw = synth_raw("01", "rest", duration=120.0, motion_onset=None)
    sci = {ch: 0.1 for ch in raw.ch_names}
    sections, _ = raw_sections(raw, sci, list(raw.ch_names), 1.5, 2.4, resp_h_freq=0.5)
    assert sections["raw"]["heart_rate"] == pytest.approx(CARDIAC_FREQ, abs=TOLERANCE_HZ)


def test_a_flat_channel_has_no_peak():
    out = _heart_rate_metrics(_recording([1.2] * 4, flat=(1,)), 0.5)
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


def test_the_spectrum_ends_at_2_hz_unless_the_band_or_the_heart_rate_lies_beyond():
    assert spectrum_fmax(10.0, 1.8, 1.38) == 2.0
    assert spectrum_fmax(10.0, 3.1, 2.4) == pytest.approx(3.3)
    assert spectrum_fmax(10.0, 1.8, 2.6) == pytest.approx(2.8)
    assert spectrum_fmax(5.0, 3.1, None) == 2.5


def test_the_raw_report_says_when_the_band_misses_the_pulse(tmp_path):
    # the synthetic pulse is at 1.2 Hz, under this band
    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("rest",))
    out = tmp_path / "qc"
    qc_cli.main([str(bids), str(out), "participant", "--dpf", "6", "--sci-threshold", "0.8",
                 "--cardiac-l-freq", "1.5", "--cardiac-h-freq", "2.4",
                 "--resp-l-freq", "0.1", "--resp-h-freq", "0.5", "--skip-bids-validation"])
    (page,) = out.rglob("*_desc-raw_report.html")
    html = page.read_text(encoding="utf-8")
    data = json.loads(re.search(r"var _STATIC_DATA = (\[.*?\]);\n", html, re.S).group(1))[0]
    assert data["summary"]["heart_rate"] == pytest.approx(CARDIAC_FREQ, abs=TOLERANCE_HZ)
    assert data["summary"]["heart_rate_outside"] is True
    assert any("outside the cardiac band (1.5 to 2.4 Hz)" in n for n in data["notes"])
    # the axis reaches past the band's top, so the band is drawn whole
    assert data["psd"]["figure"]["layout"]["xaxis"]["range"] == [0, pytest.approx(2.6)]
