"""On optical-density input every channel set leaves brightness and saturation out and measures spikes and GVTD."""

import pytest

from nirspipe.pipeline.prep_pipeline import intensity_to_od
from nirspipe.qc.subject.sqm_record import raw_sections

from tests._synth import synth_raw

_LEFT_OUT = ("cv_mean", "snr_mean", "mean_amp_mean", "saturated_pct")
_MEASURED = ("spike_count", "gvtd_filt_mean")
_SETS = ("raw", "raw_long", "raw_short")


@pytest.fixture(scope="module")
def sections():
    od = intensity_to_od(synth_raw("01", "rest", duration=120.0))
    secs, _ = raw_sections(od, {ch: 0.9 for ch in od.ch_names}, [], 0.7, 1.5, resp_h_freq=0.5)
    return secs


@pytest.mark.parametrize("name", _SETS)
def test_brightness_and_its_ceiling_are_left_out_of_every_set(sections, name):
    assert {key: sections[name].get(key) for key in _LEFT_OUT} == dict.fromkeys(_LEFT_OUT)


@pytest.mark.parametrize("name", _SETS)
def test_spikes_and_gvtd_are_measured_in_every_set(sections, name):
    assert all(isinstance(sections[name].get(key), (int, float)) for key in _MEASURED)
