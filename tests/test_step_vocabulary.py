"""The map between the step names the pipeline writes and the prose describing them.

The two vocabularies cannot be merged: the pipeline records one motion-correction step
with the method as a parameter, while the Methods text needs one paragraph per method
because the citations differ. So the map is the contract, and this is what holds it.
"""

import json

import pytest

from fnirs_pipe.qc.boilerplate.generate import step_sentence
from fnirs_pipe.qc.boilerplate.vocabulary import (
    STEP_SUMMARY,
    boilerplate_key,
    steps_from_sidecars,
    template_slots,
)

# Every step the pipeline can write into a sidecar. Adding one here without also giving it
# a steps.toml section or a STEP_SUMMARY line is what this list exists to catch.
ALL_STEPS = [
    "load", "od_conversion", "od_passthrough", "sci_pruning", "motion_correction",
    "beer_lambert", "bandpass", "resample", "design_matrix", "glm_fit", "contrasts",
    "glm_residuals", "glm_residuals_broadband", "sqm", "sqm_raw", "alff", "fc",
    "fisher_z", "fc_roi", "fc_seed", "group_sqm_raw", "group_sqm_raw_channels",
    "hyper_wtc", "hyper_wtc_roi", "hyper_isc",
    "group_hyper_wtc", "group_hyper_wtc_roi",
]


def _sidecar(directory, name, step, sources=(), **params):
    (directory / f"{name}.json").write_text(json.dumps(
        {"step": step, "Sources": list(sources), "parameters": params}))


# ---- the map ----

@pytest.mark.parametrize("step, params, expected", [
    ("od_conversion", {}, "od_conversion"),
    ("beer_lambert", {}, "beer_lambert"),
    ("resample", {}, "resample"),
    ("sci_pruning", {}, "sci_marking"),                         # renamed, same thing
    ("motion_correction", {"motion_correction": "tddr"}, "motion_tddr"),
    ("motion_correction", {"motion_correction": "wavelet"}, "motion_wavelet"),
    ("motion_correction", {"motion_correction": "none"}, None),  # nothing to describe
    ("motion_correction", {}, None),
    ("bandpass", {"high_pass": 0.01, "low_pass": 0.5}, "bandpass"),
    ("bandpass", {"high_pass": 0.01}, "highpass"),               # one cutoff, other prose
    ("bandpass", {"low_pass": 0.5}, "lowpass"),
    ("bandpass", {}, None),
    ("load", {}, None),                                          # bookkeeping, not method
    ("sqm_raw", {}, None),
    (None, {}, None),
])
def test_a_step_resolves_to_its_prose_section(step, params, expected):
    assert boilerplate_key(step, params) == expected


def test_only_a_task_run_claims_a_first_level_glm():
    # rest and denoise run the same fitting code to regress out confounds; calling that a
    # first-level GLM in the Methods would be wrong, so it gets its own paragraph. The
    # sidecar cannot tell the two apart, which is why the mode decides
    params = {"hrf_model": "spm", "noise_model": "ar1"}
    assert boilerplate_key("glm_fit", params, mode="glm") == "glm"
    assert boilerplate_key("glm_fit", params, mode="rest") == "confound_regression"
    assert boilerplate_key("glm_fit", params, mode="denoise") == "confound_regression"
    assert boilerplate_key("glm_fit", params) == "confound_regression"


def test_every_step_the_pipeline_writes_can_be_described():
    # params generous enough for every step that reads one, so a step fails here only
    # when nothing describes it at all
    params = {"motion_correction": "tddr", "high_pass": 0.01, "low_pass": 0.5}
    undescribed = [s for s in ALL_STEPS
                   if boilerplate_key(s, params, mode="glm") is None and s not in STEP_SUMMARY]
    assert undescribed == []


# ---- slots ----

def test_the_lower_edge_of_the_band_is_called_two_things():
    # a sidecar's high_pass is the lower edge; the filter templates call it l_freq, and
    # crossing the two silently prints the band backwards
    slots = template_slots("bandpass", {"high_pass": 0.01, "low_pass": 0.5})
    assert slots == {"l_freq": "0.01", "h_freq": "0.5"}


def test_a_dpf_list_becomes_one_string():
    assert template_slots("beer_lambert", {"dpf": [6.0, 6.0]}) == {"dpf": "6.0, 6.0"}


def test_resample_accepts_either_key():
    assert template_slots("resample", {"sfreq": 2.0})["sfreq"] == "2.0"
    assert template_slots("resample", {"resample_sfreq": 2.0})["sfreq"] == "2.0"


# ---- what a run actually did ----

def test_the_steps_follow_the_data_not_the_filenames(tmp_path):
    _sidecar(tmp_path, "sub-01_desc-preproc_nirs", "beer_lambert",
             sources=["/out/sub-01_desc-od_nirs.snirf"], dpf=[6.0])
    _sidecar(tmp_path, "sub-01_desc-od_nirs", "od_conversion", sources=["/bids/in.snirf"])

    assert [k for k, _ in steps_from_sidecars(tmp_path)] == ["od_conversion", "beer_lambert"]


def test_the_glm_paragraph_gathers_slots_from_several_files(tmp_path):
    # glm_residuals carries no hrf_model; glm_fit does. Both map to the one paragraph,
    # so a missing slot on one file has to be filled from the other.
    _sidecar(tmp_path, "sub-01_glm_results", "glm_fit", sources=["/out/in.snirf"],
             hrf_model="spm", noise_model="ar1", drift_model="cosine", drift_high_pass=0.01)

    slots = dict(steps_from_sidecars(tmp_path, mode="glm"))["glm"]
    assert slots["hrf_model"] == "spm"
    assert slots["drift_model"] == "cosine"


def test_a_step_that_ran_twice_is_described_once(tmp_path):
    _sidecar(tmp_path, "sub-01_task-a_desc-od_nirs", "od_conversion", sources=["/bids/a.snirf"])
    _sidecar(tmp_path, "sub-01_task-b_desc-od_nirs", "od_conversion", sources=["/bids/b.snirf"])

    assert [k for k, _ in steps_from_sidecars(tmp_path)] == ["od_conversion"]


def test_a_tree_with_no_sidecars_yields_nothing(tmp_path):
    # the caller falls back to the config, which can only say what was requested
    assert steps_from_sidecars(tmp_path) == []


# ---- one line per step ----

def test_a_method_step_borrows_its_methods_sentence(tmp_path):
    line = step_sentence("resample", {"sfreq": 2.0})
    assert line == "Data were resampled to 2.0 Hz."


def test_the_table_line_carries_no_citations():
    # the paragraph cites, the table does not: a citation per row is noise
    line = step_sentence("od_conversion", {})
    assert "et al." not in line
    assert line.endswith("using MNE-Python.")


def test_a_bookkeeping_step_falls_back_to_its_summary():
    assert step_sentence("sqm_raw", {}) == STEP_SUMMARY["sqm_raw"]


def test_an_unknown_step_says_nothing():
    assert step_sentence("something_new", {}) == ""


# ---- the confound-regression sentence names its own columns ----

@pytest.mark.parametrize("params, expected", [
    ({"short_channel": "mean"}, "the mean short-channel time course of each chromophore"),
    ({"short_channel": "pca"}, "the first principal component of the short channels"),
    ({"drift_model": "cosine", "drift_high_pass": 0.01}, "cosine drift basis (high-pass cutoff: 0.01 Hz)"),
    ({"drift_model": "polynomial", "drift_order": 3}, "an order-3 polynomial drift basis"),
])
def test_the_regressors_named_are_the_ones_that_ran(params, expected):
    assert expected in template_slots("confound_regression", params)["regressors"]


def test_a_regression_with_neither_flag_still_says_something_true():
    # the design matrix always holds an intercept, so the sentence names that rather than
    # claiming columns the model did not carry
    assert template_slots("confound_regression", {})["regressors"] == "a constant term only"
    assert template_slots("confound_regression", {"drift_model": "none"})["regressors"] == (
        "a constant term only")


def test_both_regressor_families_are_named_when_both_ran():
    phrase = template_slots("confound_regression", {
        "short_channel": "mean", "drift_model": "polynomial", "drift_order": 1,
    })["regressors"]
    assert "short-channel" in phrase and "polynomial" in phrase
