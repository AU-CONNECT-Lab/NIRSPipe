"""The screening decision: which channels a score table rejects, and the lines it uses.

`screen_channels` is fed hand-written score tables, so the rule is pinned without any
measurement in the way. `mark_bad_channels` is then run once on synthetic optical density
whose uncoupled pair is known, plus its all-channels-fail refusal.
"""

import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.prep_pipeline import intensity_to_od, mark_bad_channels
from fnirs_pipe.qc.metrics._helpers import GOOD_FRAC_PASS, PSP_PASS, SCI_PASS
from fnirs_pipe.qc.metrics.screening import resolve_cutoffs, screen_channels

from tests._synth import synth_raw


# ---- screen_channels ----

def test_a_channel_under_the_coupled_window_line_is_rejected_and_says_why():
    bad, why = screen_channels({"good_frac": {"a": GOOD_FRAC_PASS - 0.1, "b": 1.0}})
    assert bad == ["a"]
    assert why == {"a": ["coupled windows"]}


def test_whole_run_sci_and_psp_are_reported_but_reject_nothing():
    scores = {"sci": {"a": 0.1}, "psp": {"a": 0.0}, "good_frac": {"a": 1.0}}
    assert screen_channels(scores) == ([], {})


def test_a_score_exactly_on_the_line_passes():
    assert screen_channels({"good_frac": {"a": GOOD_FRAC_PASS}}) == ([], {})


def test_a_missing_score_rejects_nothing():
    assert screen_channels({"good_frac": {"a": None}}) == ([], {})


def test_a_criterion_that_could_not_be_measured_rejects_nothing():
    # screening_scores stores {} for a scorer that raised; that must not read as "all failed"
    assert screen_channels({"sci": {"a": 0.1, "b": 0.1}, "good_frac": {}}) == ([], {})


def test_a_run_cutoff_overrides_the_default_line():
    scores = {"good_frac": {"a": 0.5, "b": 0.9}}
    bad, _ = screen_channels(scores, {"good_frac": 0.95})
    assert bad == ["a", "b"]


def test_rejected_channels_come_back_in_acquisition_order():
    # order is read off the first criterion carrying scores, sci here, not off good_frac
    scores = {"sci": {"c": 1.0, "a": 1.0, "b": 1.0},
              "good_frac": {"b": 0.0, "a": 0.0, "c": 0.0}}
    assert screen_channels(scores)[0] == ["c", "a", "b"]


# ---- resolve_cutoffs ----

def test_a_none_override_keeps_the_default():
    assert resolve_cutoffs(sci=None, psp=None) == {
        "sci": SCI_PASS, "psp": PSP_PASS, "good_frac": GOOD_FRAC_PASS}


def test_an_override_replaces_only_its_own_line():
    assert resolve_cutoffs(sci=0.9) == {"sci": 0.9, "psp": PSP_PASS, "good_frac": GOOD_FRAC_PASS}


def test_a_config_field_sets_the_line_and_a_loose_override_beats_it():
    class Config:
        sci_threshold = 0.6
        min_good_frac = 0.5

    assert resolve_cutoffs(Config()) == {"sci": 0.6, "psp": PSP_PASS, "good_frac": 0.5}
    assert resolve_cutoffs(Config(), good_frac=0.9)["good_frac"] == 0.9


def test_a_misspelt_criterion_raises_instead_of_being_ignored():
    with pytest.raises(ValueError, match="no such screening criterion"):
        resolve_cutoffs(sci_threshold=0.9)


# ---- mark_bad_channels on synthetic optical density ----

CARDIAC = (0.7, 1.5)


@pytest.fixture
def od():
    return intensity_to_od(synth_raw("01", "rest"))


def test_the_uncoupled_pair_is_rejected_in_both_wavelengths_and_nothing_else(od):
    raw, bad, sci, good_frac = mark_bad_channels(od, 0.8, *CARDIAC)
    assert bad == ["S3_D3 760", "S3_D3 850"]
    assert raw.info["bads"] == bad
    assert good_frac["S3_D3 760"] == 0.0
    assert all(v == 1.0 for ch, v in good_frac.items() if ch not in bad)
    assert set(sci) == set(od.ch_names)


def test_a_line_no_channel_can_clear_is_refused_with_the_flag_to_change(od):
    with pytest.raises(StageError, match=r"every channel failed screening.*--min-good-frac"):
        mark_bad_channels(od, 0.8, *CARDIAC, min_good_frac=1.01)
