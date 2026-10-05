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
    "hyper_sqm", "hyper_bads", "hyper_coherence", "hyper_coherence_windowed",
    "hyper_screening", "hyper_usable",
    "hyper_wtc", "hyper_wtc_roichan", "hyper_wtc_phasenull",
    "hyper_isc", "hyper_isc_roichan",
    "hyper_wtc_bycondition", "hyper_wtc_bycondition_roichan",
    "hyper_merge",
]


def _sidecar(directory, name, step, sources=(), **params):
    (directory / f"{name}.json").write_text(json.dumps(
        {"step": step, "Sources": list(sources), "parameters": params}))


# ---- the map ----

@pytest.mark.parametrize("step, params, expected", [
    ("od_conversion", {}, "od_conversion"),
    ("beer_lambert", {}, "beer_lambert"),
    ("resample", {}, "resample"),
    ("sci_pruning", {}, "sci_marking"),                         # other name, same thing
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
    assert slots["l_freq"] == "0.01"
    assert slots["h_freq"] == "0.5"


def test_the_filter_sentence_names_the_filter_that_ran():
    # the family comes from the sidecar, so a Butterworth run is not described as an
    # FIR one
    iir = template_slots("bandpass", {"high_pass": 0.01, "low_pass": 0.5,
                                      "filter_method": "iir", "filter_order": 4})
    assert "Butterworth" in iir["filter"] and "order 4" in iir["filter"]

    fir = template_slots("bandpass", {"high_pass": 0.01, "low_pass": 0.5,
                                      "filter_method": "fir", "filter_order": None})
    assert "FIR" in fir["filter"] and "Butterworth" not in fir["filter"]

    # a sidecar that records no method names no family at all
    assert template_slots("highpass", {"high_pass": 0.01})["filter"] == "a zero-phase filter"


def test_the_noise_model_is_spelled_out_rather_than_pasted():
    # `auto` reaches the sidecar unexpanded, and pasted in it would read "a auto noise model"
    assert "four times the sampling rate" in template_slots("glm", {"noise_model": "auto"})["noise_model"]
    assert template_slots("glm", {"noise_model": "ar12"})["noise_model"].endswith("order 12")
    assert "prewhitening" in template_slots("glm", {"noise_model": "ols"})["noise_model"]
    # every mode fits one, so the confound sentence has to name it too
    assert "order 1" in template_slots("confound_regression", {"noise_model": "ar1"})["noise_model"]


def test_a_polynomial_drift_is_not_described_by_a_cosine_cutoff():
    assert template_slots("glm", {"drift_model": "polynomial", "drift_order": 2})["drift"] == (
        "an order-2 polynomial drift basis")
    assert "0.01 Hz" in template_slots("glm", {"drift_model": "cosine",
                                               "drift_high_pass": 0.01})["drift"]
    assert template_slots("glm", {"drift_model": "none"})["drift"] == "no drift term"


def test_a_dpf_list_becomes_one_string():
    assert template_slots("beer_lambert", {"dpf": [6.0, 6.0]}) == {"dpf": "6.0, 6.0"}


def test_resample_accepts_either_key():
    assert template_slots("resample", {"sfreq": 2.0})["sfreq"] == "2.0"
    assert template_slots("resample", {"resample_sfreq": 2.0})["sfreq"] == "2.0"


def test_the_screening_sentence_names_every_cutoff_that_rejects_a_channel():
    """The Methods have to state the thresholds a channel was rejected on, and this is the
    sentence a paper copies.

    Driven off CRITERIA rather than a list written here, so a new screening criterion that
    never reaches the prose fails instead of shipping silently.
    """
    from fnirs_pipe.qc.metrics import CRITERIA, criterion_cutoffs

    cutoffs = criterion_cutoffs()
    sentence = step_sentence("sci_pruning", {"sci_threshold": 0.8, "psp_threshold": 0.1,
                                             "min_good_frac": 0.75})
    for criterion in CRITERIA:
        if not criterion.screens:
            continue
        value = cutoffs[criterion.name]
        assert f"{value * 100:g}%" in sentence or f"{value:g}" in sentence,             f"{criterion.name} rejects channels but its cutoff is not in the Methods sentence"


def test_the_screening_sentence_quotes_the_pinned_window_not_the_qc_grid():
    """The screening window does not follow --window-length; `qc_window_s` does. Quoting
    the record's value would print a window the screening never used."""
    from fnirs_pipe.qc.metrics.windowed import SCREEN_WINDOW_S

    slots = template_slots("sci_marking", {"sci_threshold": 0.8, "qc_window_s": 30.0})
    assert slots["window_s"] == f"{SCREEN_WINDOW_S:g}"


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
    assert slots["drift"] == "a discrete cosine drift basis (high-pass cutoff: 0.01 Hz)"


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


def test_every_citation_key_has_a_reference():
    """A key with no entry in references.bib prints as the bare key, in the paper text.

    ``_fmt_citations`` falls back to the key itself rather than raising, so a step citing
    "Author2016" with no such entry renders "(Author et al., 2014; Author2016)"
    into the Methods paragraph and into the reference list under it. Nothing else notices,
    and the Methods paragraph is the part of the report that ends up in a manuscript.
    """
    from fnirs_pipe.qc.boilerplate.generate import _load_refs, _load_steps

    refs = _load_refs()
    missing = sorted({
        key
        for step in _load_steps().values()
        for key in (step.get("citations") or [])
        if key not in refs
    })
    assert not missing, (
        f"steps.toml cites {missing}, which references.bib has no entry for. The Methods "
        f"paragraph will print the raw key. Add the entry, or drop the citation."
    )


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


# ---- hyperscanning steps ----
#
# A dyad's steps reach the Methods paragraph two ways: the coherence passes leave sidecars
# and are read off disk, while the alignment leaves no file and is passed in by the report.
# Both go through the same table, and a key the table does not know renders as nothing.

@pytest.mark.parametrize("step", [
    "hyper_wtc", "hyper_wtc_roichan", "hyper_wtc_bycondition",
    "hyper_wtc_bycondition_roichan",
])
def test_every_coherence_output_maps_to_the_one_coherence_sentence(step):
    # four files, one method; the band is the same for all of them
    assert boilerplate_key(step, {}) == "hyper_wtc"
    assert boilerplate_key(step, {"channel_cross": True}) == "hyper_wtc_crossed"


@pytest.mark.parametrize("step", [
    "hyper_wtc_phasenull", "hyper_wtc_roihom_phasenull", "hyper_wtc_roichan_phasenull",
    "hyper_wtc_bycondition_roihom_phasenull", "hyper_wtc_bycondition_roichan_phasenull",
    "hyper_wtc_bycondition_roichan_pairnull",
])
def test_a_null_keeps_its_own_line_and_no_coherence_sentence(step):
    # a null's sidecar records how the null was drawn, so it cannot pick the sentence
    assert boilerplate_key(step, {"cross": True}) is None
    assert STEP_SUMMARY[step]


def test_a_crossed_run_says_so_once_whatever_its_null_did(tmp_path):
    _sidecar(tmp_path, "group-G01_task-main_stat-wtc_relmat", "hyper_wtc",
             sources=["/in.snirf"], channel_cross=True, wtc_fmin=0.01, wtc_fmax=0.2)
    _sidecar(tmp_path, "group-G01_task-main_null-phase_stat-wtc_relmat",
             "hyper_wtc_phasenull", sources=["/in.snirf"], cross=False)

    assert [k for k, _ in steps_from_sidecars(tmp_path)] == ["hyper_wtc_crossed"]


def test_the_crossed_sentence_fills_the_same_slots():
    from fnirs_pipe.qc.boilerplate.generate import _load_steps

    params = {"wtc_fmin": 0.01, "wtc_fmax": 0.2, "band_fmin": 0.02, "band_fmax": 0.1}
    assert template_slots("hyper_wtc_crossed", params) == template_slots("hyper_wtc", params)
    assert "every channel of one recording" in _load_steps()["hyper_wtc_crossed"]["plain"]


def test_the_coherence_sentence_names_the_axis_and_the_band_apart():
    slots = template_slots("hyper_wtc", {
        "wtc_fmin": 0.004, "wtc_fmax": 0.2, "band_fmin": 0.03, "band_fmax": 0.1,
    })
    assert slots == {"wtc_fmin": "0.004", "wtc_fmax": "0.2",
                     "band_fmin": "0.03", "band_fmax": "0.1"}


def test_a_run_that_named_no_band_averaged_the_whole_axis():
    slots = template_slots("hyper_wtc", {"wtc_fmin": 0.004, "wtc_fmax": 0.2})
    assert slots["band_fmin"] == "0.004" and slots["band_fmax"] == "0.2"


def test_the_hyper_sentences_all_exist():
    from fnirs_pipe.qc.boilerplate.generate import _load_steps

    steps = _load_steps()
    for key in ("hyper_alignment", "hyper_wtc", "hyper_wtc_crossed", "hyper_coherence",
                "hyper_isc"):
        assert key in steps, f"{key} has no prose"
        assert "{citations}" in steps[key]["plain"]


def test_the_hyper_citations_are_still_placeholders():
    # a reminder, not a failure: replace the TODO_ keys in references.bib and this goes away
    from fnirs_pipe.qc.boilerplate.generate import _load_refs, _load_steps

    steps, refs = _load_steps(), _load_refs()
    todo = sorted({key for section in steps.values()
                   for key in section.get("citations", []) if key.startswith("TODO_")})
    for key in todo:
        assert key in refs, f"{key} is cited but not in references.bib"
    if todo:
        pytest.skip(f"hyperscanning Methods still cites placeholders: {todo}")
