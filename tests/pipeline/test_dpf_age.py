"""--dpf age: the frontal-head equation of Scholkmann & Wolf (2013), one factor per wavelength from the participant's age."""

import argparse

import pytest

from nirspipe.cli import _app_options
from nirspipe.pipeline.dpf import age_dpf, dpf_from_age, resolve_dpf, wavelengths
from nirspipe.pipeline.prep_pipeline import intensity_to_od, od_to_haemo
from nirspipe.qc.boilerplate.generate import step_sentence
from nirspipe.utils.lineage import lineage_of

from tests._synth import synth_raw


# worked by hand from the published constants (NIRS-KIT CalcDPF.m and nirs-toolbox
# frontal_DPF_model.m carry the same six)
@pytest.mark.parametrize("wavelength, age, expected", [
    (760, 30, 6.2966), (850, 30, 5.2343), (760, 0.5, 5.3173), (850, 5, 4.4444),
])
def test_the_equation_gives_the_published_curve(wavelength, age, expected):
    assert age_dpf(wavelength, age) == pytest.approx(expected, abs=1e-4)


def test_numbers_pass_through_untouched():
    od = intensity_to_od(synth_raw("01", "rest"))
    assert resolve_dpf([6.0, 5.5], od, None) == [6.0, 5.5]


def test_age_gives_one_factor_per_wavelength_in_ascending_order():
    od = intensity_to_od(synth_raw("01", "rest"))
    got = resolve_dpf(["age"], od, 30.0)
    assert wavelengths(od) == sorted(wavelengths(od)) and len(got) == len(wavelengths(od))
    assert got == [pytest.approx(age_dpf(w, 30.0)) for w in wavelengths(od)]


def test_age_without_an_age_is_refused():
    od = intensity_to_od(synth_raw("01", "rest"))
    with pytest.raises(ValueError, match="needs the participant's age"):
        resolve_dpf(["age"], od, None)


def test_the_word_mixed_with_numbers_is_refused():
    with pytest.raises(ValueError, match="numbers or the single word"):
        dpf_from_age([6.0, "age"])


def test_the_flag_takes_numbers_or_the_word():
    parser = argparse.ArgumentParser()
    _app_options.add_dpf(parser)
    assert parser.parse_args(["--dpf", "age"]).dpf == ["age"]
    assert parser.parse_args(["--dpf", "6", "5.5"]).dpf == [6.0, 5.5]
    with pytest.raises(SystemExit):
        parser.parse_args(["--dpf", "six"])


def test_the_conversion_records_the_factors_and_the_age():
    od = intensity_to_od(synth_raw("01", "rest"))
    factors = resolve_dpf(["age"], od, 30.0)
    params = lineage_of(od_to_haemo(od, factors, age=30.0)).params
    assert params["dpf"] == factors and params["dpf_age"] == 30.0


def test_the_methods_name_the_age_and_the_equation():
    sentence = step_sentence("beer_lambert", {"dpf": [6.2966, 5.2343], "dpf_age": 30.0})
    assert "6.30 and 5.23" in sentence and "age of 30 years" in sentence
    assert "Scholkmann and Wolf (2013)" in sentence


# ---- outside the range the paper recommends ----

def test_inside_the_recommended_range_there_is_nothing_to_say():
    from nirspipe.pipeline.dpf import dpf_caveat
    assert dpf_caveat(30.0, [760.0, 850.0]) is None
    assert dpf_caveat(0.0, [690.0, 950.0]) is None


def test_an_age_or_a_wavelength_outside_it_is_named():
    from nirspipe.pipeline.dpf import dpf_caveat
    caveat = dpf_caveat(75.0, [660.0, 850.0])
    assert "age 75 years" in caveat and "660 nm" in caveat and "850 nm" not in caveat
    assert "extrapolation" in caveat


def test_the_run_goes_on_and_the_log_says_so(caplog):
    od = intensity_to_od(synth_raw("01", "rest"))
    with caplog.at_level("WARNING"):
        got = resolve_dpf(["age"], od, 80.0)
    assert len(got) == len(wavelengths(od))
    assert any("age 80 years" in r.getMessage() for r in caplog.records)


def test_the_report_notes_it_only_for_an_age_dpf():
    from types import SimpleNamespace
    from nirspipe.qc.subject.report import _note_dpf_range

    raw = synth_raw("01", "rest")
    notes = []
    _note_dpf_range(notes, "01", SimpleNamespace(dpf=["age"], age=80.0), raw)
    _note_dpf_range(notes, "01", SimpleNamespace(dpf=[6.0], age=None), raw)
    _note_dpf_range(notes, "01", SimpleNamespace(dpf=["age"], age=30.0), raw)
    assert len(notes) == 1 and "age 80 years" in notes[0]
