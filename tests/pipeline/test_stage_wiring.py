"""Class A wiring: what reaches the library, and what each stage refuses.

The math is MNE's; what is ours is that the DPF list arrives unchanged and in order, that a
stage handed the wrong domain stops instead of converting garbage, and that the bandpass
cutoffs are not swapped on their way to the filter.
"""

import mne
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.denoise import bandpass_filter
from fnirs_pipe.pipeline.glm import run_glm_pipeline
from fnirs_pipe.pipeline.prep_pipeline import intensity_to_od, od_to_haemo
from fnirs_pipe.utils.lineage import lineage_of

from tests._synth import synth_raw


@pytest.fixture(scope="module")
def intensity():
    return synth_raw("01", "rest")


@pytest.fixture(scope="module")
def od(intensity):
    return intensity_to_od(intensity.copy())


# ---- DPF reaches Beer-Lambert ----

def _captured_ppf(monkeypatch, od, dpf):
    seen = {}

    def fake(raw, ppf):
        seen["ppf"] = ppf
        return raw.copy()

    monkeypatch.setattr(mne.preprocessing.nirs, "beer_lambert_law", fake)
    od_to_haemo(od.copy(), dpf=dpf)
    return seen["ppf"]


def test_one_dpf_is_passed_as_a_scalar(monkeypatch, od):
    assert _captured_ppf(monkeypatch, od, [6.0]) == 6.0


def test_a_dpf_per_wavelength_is_passed_whole_and_in_order(monkeypatch, od):
    assert _captured_ppf(monkeypatch, od, [6.0, 5.2]) == [6.0, 5.2]


def test_one_dpf_and_the_same_dpf_twice_are_the_same_conversion(od):
    assert_allclose(od_to_haemo(od.copy(), [6.0]).get_data(),
                    od_to_haemo(od.copy(), [6.0, 6.0]).get_data(), rtol=1e-12)


def test_the_order_of_two_dpfs_matters(od):
    # control for the in-order test above: a swap that changed nothing would make it vacuous
    a = od_to_haemo(od.copy(), [6.0, 5.0]).get_data()
    b = od_to_haemo(od.copy(), [5.0, 6.0]).get_data()
    assert not np.allclose(a, b)


# ---- each stage refuses the wrong domain ----

def test_od_conversion_refuses_data_that_is_already_od(od):
    with pytest.raises(StageError, match="needs raw intensity.*got fnirs_od"):
        intensity_to_od(od.copy())


def test_od_conversion_refuses_haemoglobin(od):
    with pytest.raises(StageError, match="needs raw intensity.*got hbo, hbr"):
        intensity_to_od(od_to_haemo(od.copy(), [6.0]))


def test_beer_lambert_refuses_raw_intensity(intensity):
    with pytest.raises(RuntimeError, match="optical density"):
        od_to_haemo(intensity.copy(), [6.0])


def test_beer_lambert_refuses_haemoglobin(od):
    haemo = od_to_haemo(od.copy(), [6.0])
    with pytest.raises(RuntimeError, match="optical density"):
        od_to_haemo(haemo, [6.0])


# ---- GLM outputs need the file they came from ----

def test_glm_outputs_without_a_source_file_are_refused_before_fitting(od, tmp_path):
    haemo = od_to_haemo(od.copy(), [6.0])
    with pytest.raises(ValueError, match="Pass source_path"):
        run_glm_pipeline(haemo, stim_dur=None, hrf_model="spm", noise_model="ols",
                         drift_model="polynomial", high_pass=None, drift_order=1,
                         fir_delays=None, output_dir=str(tmp_path))


def test_a_glm_that_writes_nothing_needs_no_source_file(od):
    haemo = od_to_haemo(od.copy(), [6.0])
    run_glm_pipeline(haemo, stim_dur=None, hrf_model="spm", noise_model="ols",
                     drift_model="polynomial", high_pass=None, drift_order=1, fir_delays=None,
                     events=pd.DataFrame(columns=["trial_type", "onset", "duration"]))


# ---- the residual says whether a task model was in it ----

def _residual_params(od, events):
    haemo = od_to_haemo(od.copy(), [6.0])
    *_, resid = run_glm_pipeline(haemo, stim_dur=None, hrf_model="glover", noise_model="ols",
                                 drift_model="polynomial", high_pass=None, drift_order=1,
                                 fir_delays=None, events=events)
    return lineage_of(resid).params


def test_a_confound_regression_records_no_conditions_and_no_hrf(od):
    params = _residual_params(od, pd.DataFrame(columns=["trial_type", "onset", "duration"]))
    assert params["conditions"] == []
    assert "hrf_model" not in params


def test_a_task_glm_records_its_conditions_and_its_hrf(od):
    events = pd.DataFrame({"trial_type": ["tap", "rest", "tap"], "onset": [20.0, 60.0, 100.0],
                           "duration": [10.0, 10.0, 10.0]})
    params = _residual_params(od, events)
    assert params["conditions"] == ["rest", "tap"]
    assert params["hrf_model"] == "glover"
    # durations came from the table handed in, not from --stim-dur, so neither is claimed
    assert "stim_dur" not in params and "fir_delays" not in params


# ---- the bandpass cutoffs are not swapped ----

SFREQ = 10.0


def _tones(*freqs):
    t = np.arange(int(600 * SFREQ)) / SFREQ
    names = [f"S{i + 1}_D{i + 1} hbo" for i in range(len(freqs))]
    data = np.vstack([np.sin(2 * np.pi * f * t) for f in freqs])
    return mne.io.RawArray(data, mne.create_info(names, SFREQ, ["hbo"] * len(freqs)),
                           verbose="error")


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_the_passband_keeps_its_tone_and_both_stopbands_lose_theirs():
    # one tone per channel: 0.25 Hz inside 0.1-0.5, 0.02 below it, 2 Hz above it
    raw = bandpass_filter(_tones(0.25, 0.02, 2.0), l_freq=0.1, h_freq=0.5)
    edge = int(60 * SFREQ)                            # clear of the filter's start-up
    kept, below, above = raw.get_data()[:, edge:-edge]
    assert_allclose(_rms(kept), 1 / np.sqrt(2), rtol=0.02)
    assert _rms(below) < 0.01
    assert _rms(above) < 0.01


def test_the_stamp_records_the_cutoffs_the_filter_was_given():
    params = lineage_of(bandpass_filter(_tones(0.25), l_freq=0.1, h_freq=0.5)).params
    assert (params["l_freq"], params["h_freq"]) == (0.1, 0.5)
