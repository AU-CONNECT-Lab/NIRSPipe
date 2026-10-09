"""ALFF / fALFF values against analytic truth.

Synthetic sinusoids only, no recorded data: every expectation below follows from
the definition rather than from a previous run of this code. Frequencies are
chosen to land exactly on an FFT bin (f = k * sfreq / n_samples) so spectral
leakage does not blur the expected values.
"""

import numpy as np
import pytest

from nirspipe.pipeline.restingstate import compute_alff

SFREQ = 10.0
DUR = 100.0                       # 1000 samples -> 0.01 Hz per bin
HIGH_PASS, LOW_PASS = 0.01, 0.08  # band is [high_pass, low_pass]
F_IN, F_OUT = 0.05, 1.0           # bins 5 and 100


def _sine(freq, amp=1.0):
    t = np.arange(int(SFREQ * DUR)) / SFREQ
    return amp * np.sin(2 * np.pi * freq * t)


def _raw(data, chromo="hbo"):
    import mne

    data = np.atleast_2d(data)
    names = [f"S{i}_D{i} {chromo}" for i in range(data.shape[0])]
    info = mne.create_info(names, SFREQ, [chromo] * data.shape[0])
    return mne.io.RawArray(data, info, verbose="error")


def _mixed_raw(hbo_data, hbr_data):
    import mne

    hbo_data, hbr_data = np.atleast_2d(hbo_data), np.atleast_2d(hbr_data)
    names = ([f"S{i}_D{i} hbo" for i in range(hbo_data.shape[0])]
             + [f"S{i}_D{i} hbr" for i in range(hbr_data.shape[0])])
    types = ["hbo"] * hbo_data.shape[0] + ["hbr"] * hbr_data.shape[0]
    info = mne.create_info(names, SFREQ, types)
    return mne.io.RawArray(np.vstack([hbo_data, hbr_data]), info, verbose="error")


def _alff(data, chromo="hbo"):
    return compute_alff(_raw(data, chromo), low_pass=LOW_PASS, high_pass=HIGH_PASS)


# ---- fALFF is the band's share of total spectral amplitude ----

def test_falff_is_one_when_all_power_is_in_band():
    assert _alff(_sine(F_IN)).falff[0] == pytest.approx(1.0, abs=1e-6)


def test_falff_is_zero_when_no_power_is_in_band():
    assert _alff(_sine(F_OUT)).falff[0] == pytest.approx(0.0, abs=1e-6)


def test_falff_is_one_half_for_equal_in_and_out_of_band_components():
    assert _alff(_sine(F_IN) + _sine(F_OUT)).falff[0] == pytest.approx(0.5, abs=1e-6)


def test_falff_stays_within_zero_and_one_on_noise():
    noise = np.random.default_rng(0).normal(size=(6, int(SFREQ * DUR)))
    falff = _alff(noise).falff
    assert ((falff >= 0.0) & (falff <= 1.0)).all()


# ---- scaling ----

def test_alff_scales_linearly_with_amplitude():
    single = _alff(_sine(F_IN, amp=1.0)).alff[0]
    doubled = _alff(_sine(F_IN, amp=2.0)).alff[0]
    assert doubled == pytest.approx(2.0 * single, rel=1e-9)


def test_falff_is_unchanged_by_amplitude():
    single = _alff(_sine(F_IN) + _sine(F_OUT)).falff[0]
    doubled = _alff(2 * (_sine(F_IN) + _sine(F_OUT))).falff[0]
    assert doubled == pytest.approx(single, rel=1e-9)


# ---- why the pipeline keeps a separate broadband residual ----

def test_alff_ignores_out_of_band_power_but_falff_does_not():
    # This is the whole reason rest mode regresses a second time without the low-pass:
    # ALFF reads only the band, so one broadband signal serves it unchanged, while
    # fALFF collapses towards 1 the moment its denominator loses the out-of-band power.
    band_only = _alff(_sine(F_IN))
    broadband = _alff(_sine(F_IN) + _sine(F_OUT))

    assert broadband.alff[0] == pytest.approx(band_only.alff[0], rel=1e-9)
    assert band_only.falff[0] == pytest.approx(1.0, abs=1e-6)
    assert broadband.falff[0] == pytest.approx(0.5, abs=1e-6)


# ---- degenerate input ----

def test_flat_channel_yields_zero_rather_than_nan():
    df = _alff(np.zeros(int(SFREQ * DUR)))
    assert df.loc[0, ["alff", "falff", "malff", "zalff"]].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_flat_channel_does_not_poison_its_neighbours():
    data = np.vstack([_sine(F_IN), np.zeros(int(SFREQ * DUR)), _sine(F_IN, amp=2.0)])
    df = _alff(data)
    assert np.isfinite(df[["alff", "falff", "malff", "zalff"]].to_numpy()).all()
    assert df.alff[2] == pytest.approx(2.0 * df.alff[0], rel=1e-9)


# ---- cross-channel standardization ----

def test_malff_and_zalff_standardize_within_each_chromophore():
    hbo = np.vstack([_sine(F_IN, amp=a) for a in (1.0, 2.0, 3.0)])
    hbr = np.vstack([_sine(F_IN, amp=a) for a in (10.0, 20.0, 30.0)])
    df = compute_alff(_mixed_raw(hbo, hbr), low_pass=LOW_PASS, high_pass=HIGH_PASS)

    for chromo in ("hbo", "hbr"):
        grp = df[df.channel.str.endswith(f" {chromo}")]
        assert grp.malff.mean() == pytest.approx(1.0, rel=1e-9)
        assert grp.zalff.mean() == pytest.approx(0.0, abs=1e-9)
        assert grp.zalff.std(ddof=0) == pytest.approx(1.0, rel=1e-9)


def test_zalff_standardizes_by_the_population_standard_deviation():
    # ddof=0: the channels are the whole set being rescaled, not a sample of a larger one
    amps = (1.0, 2.0, 3.0, 5.0)
    df = compute_alff(_raw(np.vstack([_sine(F_IN, amp=a) for a in amps])),
                      low_pass=LOW_PASS, high_pass=HIGH_PASS)
    expected = (df.alff - df.alff.mean()) / df.alff.std(ddof=0)
    assert df.zalff.tolist() == pytest.approx(expected.tolist())


def test_a_single_channel_chromophore_yields_zero_zalff_rather_than_nan():
    # one observation has zero spread, and the grp_std != 0 guard leaves zALFF at 0
    df = compute_alff(_mixed_raw(_sine(F_IN, amp=2.0),
                                 np.vstack([_sine(F_IN, amp=a) for a in (1.0, 2.0, 3.0)])),
                      low_pass=LOW_PASS, high_pass=HIGH_PASS)
    lone = df[df.channel.str.endswith(" hbo")]
    assert len(lone) == 1
    assert lone.zalff.iloc[0] == 0.0
    assert lone.malff.iloc[0] == pytest.approx(1.0)


def test_rescaling_one_chromophore_leaves_the_other_untouched():
    # HbO and HbR sit on different amplitude scales, so a pooled mean would let one
    # chromophore's magnitude distort the other's standardized values.
    hbo = np.vstack([_sine(F_IN, amp=a) for a in (1.0, 2.0, 3.0)])
    hbr = np.vstack([_sine(F_IN, amp=a) for a in (1.0, 2.0, 3.0)])

    before = compute_alff(_mixed_raw(hbo, hbr), low_pass=LOW_PASS, high_pass=HIGH_PASS)
    after = compute_alff(_mixed_raw(hbo, 100.0 * hbr), low_pass=LOW_PASS, high_pass=HIGH_PASS)

    hbo_rows = before.channel.str.endswith(" hbo")
    assert after.zalff[hbo_rows].tolist() == pytest.approx(before.zalff[hbo_rows].tolist())
    assert after.malff[hbo_rows].tolist() == pytest.approx(before.malff[hbo_rows].tolist())
