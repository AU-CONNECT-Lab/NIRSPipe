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


@pytest.mark.parametrize("step", ["glm_fit", "glm_residuals"])
def test_only_a_design_with_conditions_claims_a_first_level_glm(step):
    # rest and denoise run the same fitting code to regress out confounds; calling that a
    # first-level GLM in the Methods would be wrong, so it gets its own paragraph. The
    # recorded conditions decide, so a reader of the residual alone can tell
    params = {"hrf_model": "spm", "noise_model": "ar1"}
    assert boilerplate_key(step, {**params, "conditions": ["tap"]}) == "glm"
    assert boilerplate_key(step, {**params, "conditions": []}) == "confound_regression"
    # a tree written before the conditions were recorded says so instead of guessing
    assert boilerplate_key(step, params) == "regression_unrecorded"


def test_every_step_the_pipeline_writes_can_be_described():
    # params generous enough for every step that reads one, so a step fails here only
    # when nothing describes it at all
    params = {"motion_correction": "tddr", "high_pass": 0.01, "low_pass": 0.5,
              "conditions": ["tap"]}
    undescribed = [s for s in ALL_STEPS
                   if boilerplate_key(s, params) is None and s not in STEP_SUMMARY]
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
    # the robust solver has an order cap of its own, and is not an unspecified model
    irls = template_slots("glm", {"noise_model": "ar_irls40"})["noise_model"]
    assert "AR-IRLS" in irls and irls.endswith("up to 40")
    assert "four times the sampling rate" in template_slots(
        "glm", {"noise_model": "ar_irls"})["noise_model"]
    # every mode fits one, so the confound sentence has to name it too
    assert "order 1" in template_slots("confound_regression", {"noise_model": "ar1"})["noise_model"]


def test_a_polynomial_drift_is_not_described_by_a_cosine_cutoff():
    regressors = template_slots("glm", {"drift_model": "polynomial", "drift_order": 2})
    assert regressors["regressors"] == "an order-2 polynomial drift basis"
    assert "0.01 Hz" in template_slots("glm", {"drift_model": "cosine",
                                               "drift_high_pass": 0.01})["regressors"]
    assert template_slots("glm", {"drift_model": "none"})["regressors"] == "only a constant term"


def test_the_glm_sentence_names_every_nuisance_column_and_the_hrf_in_words():
    slots = template_slots("glm", {
        "hrf_model": "glover + derivative", "short_channel": "mean",
        "aux_regressors": ["aux_GYRO_X", "aux_GYRO_Y"],
        "drift_model": "cosine", "drift_high_pass": 0.01,
    })
    assert slots["conditions"] == ("with the Glover haemodynamic response function and its "
                                   "time derivative")
    assert slots["regressors"] == (
        "the mean of the retained short channels for each chromophore, the auxiliary signals "
        "GYRO_X and GYRO_Y and a discrete cosine drift basis (high-pass cutoff: 0.01 Hz)")


def test_one_dpf_and_the_same_dpf_twice_read_the_same():
    one = "a differential pathlength factor (DPF) of 6"
    assert template_slots("beer_lambert", {"dpf": [6.0]})["dpf"] == one
    assert template_slots("beer_lambert", {"dpf": [6.0, 6.0]})["dpf"] == one


def test_two_dpfs_are_named_in_the_order_mne_applies_them():
    assert template_slots("beer_lambert", {"dpf": [6.0, 5.2]})["dpf"] == (
        "differential pathlength factors (DPF) of 6 and 5.2, in ascending order of wavelength")


def test_resample_accepts_either_key():
    assert template_slots("resample", {"sfreq": 2.0})["sfreq"] == "2"
    assert template_slots("resample", {"resample_sfreq": 2.0})["sfreq"] == "2"


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


def test_the_screening_sentence_names_the_band_the_scope_and_any_hand_marks():
    slots = template_slots("sci_marking", {
        "sci_threshold": 0.8, "cardiac_l_freq": 0.7, "cardiac_h_freq": 1.5,
        "screen_scope": "task", "screen_scope_counted": "task",
        "bad_channels": ["S1_D1 760", "S1_D1 850", "S2_D2"],
    })
    assert slots["cardiac_band"] == "in the 0.7–1.5 Hz cardiac band"
    assert slots["scope"] == "the windows inside annotated task blocks"
    # a pair named by either wavelength is one channel, named once
    assert slots["manual"] == " Channels S1_D1 and S2_D2 were also marked as bad by hand."

    plain = template_slots("sci_marking", {"sci_threshold": 0.8})
    assert plain["scope"] == "their windows" and plain["manual"] == ""
    # task asked for but no block long enough: the run counted the whole recording
    fell_back = {"sci_threshold": 0.8, "screen_scope": "task", "screen_scope_counted": "run"}
    assert template_slots("sci_marking", fell_back)["scope"] == "their windows"


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
             hrf_model="spm", conditions=["tap"], noise_model="ar1")
    _sidecar(tmp_path, "sub-01_desc-errts_nirs", "glm_residuals", sources=["/out/in.snirf"],
             conditions=["tap"], drift_model="cosine", drift_high_pass=0.01)

    slots = dict(steps_from_sidecars(tmp_path))["glm"]
    assert slots["conditions"] == "with the SPM canonical haemodynamic response function"
    assert slots["regressors"] == "a discrete cosine drift basis (high-pass cutoff: 0.01 Hz)"


def test_a_step_that_ran_twice_is_described_once(tmp_path):
    _sidecar(tmp_path, "sub-01_task-a_desc-od_nirs", "od_conversion", sources=["/bids/a.snirf"])
    _sidecar(tmp_path, "sub-01_task-b_desc-od_nirs", "od_conversion", sources=["/bids/b.snirf"])

    assert [k for k, _ in steps_from_sidecars(tmp_path)] == ["od_conversion"]


def test_a_tree_with_no_sidecars_yields_nothing(tmp_path):
    # nothing recorded, nothing claimed
    assert steps_from_sidecars(tmp_path) == []


# ---- one line per step ----

def test_a_method_step_borrows_its_methods_sentence(tmp_path):
    line = step_sentence("resample", {"sfreq": 2.0})
    assert line == "Data were resampled to 2 Hz."


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
    ({"short_channel": "mean"}, "the mean of the retained short channels for each chromophore"),
    ({"short_channel": "pca"}, "an orthogonal basis spanning the retained short channels"),
    ({"aux_regressors": ["aux_ACC_Z"]}, "the auxiliary signals ACC_Z"),
    ({"drift_model": "cosine", "drift_high_pass": 0.01}, "cosine drift basis (high-pass cutoff: 0.01 Hz)"),
    ({"drift_model": "polynomial", "drift_order": 3}, "an order-3 polynomial drift basis"),
])
def test_the_regressors_named_are_the_ones_that_ran(params, expected):
    assert expected in template_slots("confound_regression", params)["regressors"]


def test_a_regression_with_neither_flag_still_says_something_true():
    # the design matrix always holds an intercept, so the sentence names that rather than
    # claiming columns the model did not carry
    assert template_slots("confound_regression", {})["regressors"] == "only a constant term"
    assert template_slots("confound_regression", {"drift_model": "none"})["regressors"] == (
        "only a constant term")


def test_both_regressor_families_are_named_when_both_ran():
    phrase = template_slots("confound_regression", {
        "short_channel": "mean", "drift_model": "polynomial", "drift_order": 1,
    })["regressors"]
    assert "short channels" in phrase and "polynomial" in phrase


# ---- hyperscanning steps ----
#
# A dyad's steps reach the Methods paragraph two ways: the coherence passes leave sidecars
# and are read off disk, while the alignment leaves no file and is passed in by the report.
# Both go through the same table, and a key the table does not know renders as nothing.

@pytest.mark.parametrize("step", ["hyper_wtc", "hyper_wtc_bycondition"])
def test_every_coherence_output_maps_to_the_one_coherence_sentence(step):
    # the whole run and its conditions are one method; the band is the same for both
    assert boilerplate_key(step, {}) == "hyper_wtc"
    assert boilerplate_key(step, {"channel_cross": True}) == "hyper_wtc_crossed"


@pytest.mark.parametrize("step", [
    "hyper_wtc_roichan", "hyper_wtc_roihom", "hyper_wtc_bycondition_roichan",
    "hyper_wtc_bycondition_roihom", "hyper_isc_roichan",
])
def test_every_region_table_shares_the_one_roi_sentence(step):
    assert boilerplate_key(step, {"channel_cross": True}) == "hyper_roi"


@pytest.mark.parametrize("step", [
    "hyper_wtc_phasenull", "hyper_wtc_roihom_phasenull", "hyper_wtc_roichan_phasenull",
    "hyper_wtc_bycondition_roihom_phasenull", "hyper_wtc_bycondition_roichan_phasenull",
])
def test_a_null_keeps_its_own_line_and_no_coherence_sentence(step):
    # a null's sidecar records how the null was drawn, so it gets its own sentence and never
    # picks the coherence one
    assert boilerplate_key(step, {"cross": True}) == "hyper_phasenull"
    assert STEP_SUMMARY[step]


def test_a_crossed_run_says_so_once_whatever_its_null_did(tmp_path):
    _sidecar(tmp_path, "group-G01_task-main_stat-wtc_relmat", "hyper_wtc",
             sources=["/in.snirf"], channel_cross=True, wtc_fmin=0.01, wtc_fmax=0.2)
    _sidecar(tmp_path, "group-G01_task-main_null-phase_stat-wtc_relmat",
             "hyper_wtc_phasenull", sources=["/in.snirf"], cross=False)

    steps = dict(steps_from_sidecars(tmp_path))
    assert list(steps) == ["hyper_wtc_crossed", "hyper_phasenull"]
    # the null's own crossing reaches its own sentence and nothing else
    assert steps["hyper_phasenull"]["pairs"] == "homologous channel pairs"


def test_the_crossed_sentence_fills_the_same_slots():
    from fnirs_pipe.qc.boilerplate.generate import _load_steps

    params = {"wtc_fmin": 0.01, "wtc_fmax": 0.2, "band_fmin": 0.02, "band_fmax": 0.1}
    assert template_slots("hyper_wtc_crossed", params) == template_slots("hyper_wtc", params)
    assert "every retained long channel of one member" in _load_steps()["hyper_wtc_crossed"]["plain"]


def test_the_coherence_sentence_names_the_band_it_averaged_and_nothing_unasked():
    slots = template_slots("hyper_wtc", {
        "wtc_fmin": 0.004, "wtc_fmax": 0.2, "band_fmin": 0.03, "band_fmax": 0.1,
        "chroma": ["hbo", "hbr"], "mask_coi": True,
    })
    assert (slots["band_fmin"], slots["band_fmax"]) == ("0.03", "0.1")
    assert slots["chroma"] == "HbO and HbR"
    assert slots["coi"] == ", excluding the cone of influence"
    # options this run did not use leave no sentence behind
    assert slots["whiten"] == slots["conditions"] == slots["window"] == slots["bads"] == ""


def test_the_coherence_options_a_run_used_each_get_a_sentence():
    slots = template_slots("hyper_wtc", {
        "band_fmin": 0.02, "band_fmax": 0.1, "wtc_whiten_s": 10.0, "wtc_whiten_order": 100,
        "analysis_window_s": [60.0, 300.0], "bads_scope": "subject",
    })
    assert "order 100 (10 s)" in slots["whiten"]
    assert "60–300 s" in slots["window"]
    assert "any of a member's runs" in slots["bads"]


def test_the_correlation_sentence_states_each_option_that_changed_it():
    slots = template_slots("hyper_isc", {
        "isc_band_hz": [0.06, 0.15], "isc_whiten_max_order": 8, "isc_max_lag_s": 2.0,
    })
    assert "band-pass filtered to 0.06–0.15 Hz and prewhitened" in slots["options"]
    assert "(at most 8)" in slots["options"]
    assert "within ±2 s" in slots["options"]
    # the defaults change nothing, so they add nothing
    assert template_slots("hyper_isc", {"isc_band_hz": None, "isc_whiten_max_order": 0,
                                        "isc_max_lag_s": 0.0}) == {"options": ""}


@pytest.mark.parametrize("pad, said, unsaid", [
    (None, "the same maps over each condition's span", "separate transform"),
    # the cut stops at the recording's ends, so the pad is a ceiling
    (47.14, "up to 47.1 s of recording on either side", "the same maps"),
    (0.0, "cut at its own boundaries", "of recording on either side"),
])
def test_the_condition_sentence_names_the_route_the_tables_record(pad, said, unsaid):
    windows = {"chat": [10.0, 110.0]}
    for key in ("hyper_wtc", "hyper_wtc_crossed"):
        slots = template_slots(key, {"condition_windows_s": windows, "wtc_cond_pad_s": pad})
        assert said in slots["conditions"] and unsaid not in slots["conditions"]


def test_a_run_without_conditions_says_nothing_about_them():
    from fnirs_pipe.qc.boilerplate.generate import _load_steps

    assert template_slots("hyper_wtc", {"wtc_cond_pad_s": 30.0})["conditions"] == ""
    assert "{conditions}" in _load_steps()["hyper_wtc"]["plain"]


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


# ---- the steps behind one file ----
#
# A dyad's paragraph is read up the Sources of the member files the analysis opened, so a
# run that read an early stage is never described by what a later stage of the same
# recording went through.

def _member_chain(directory):
    """od -> sci -> motcorrected -> preproc -> filtered -> errts, plus the fit's own table."""
    def path(desc):
        return str(directory / f"sub-11_task-hold_desc-{desc}_nirs.snirf")

    _sidecar(directory, "sub-11_task-hold_desc-od_nirs", "od_conversion",
             sources=["/bids/sub-11_task-hold_nirs.snirf"])
    _sidecar(directory, "sub-11_task-hold_desc-sci_nirs", "sci_pruning", sources=[path("od")],
             sci_threshold=0.8, cardiac_l_freq=0.7, cardiac_h_freq=1.5)
    _sidecar(directory, "sub-11_task-hold_desc-motcorrected_nirs", "motion_correction",
             sources=[path("sci")], motion_correction="tddr")
    _sidecar(directory, "sub-11_task-hold_desc-preproc_nirs", "beer_lambert",
             sources=[path("motcorrected")], dpf=[6.0])
    _sidecar(directory, "sub-11_task-hold_desc-filtered_nirs", "bandpass",
             sources=[path("preproc")], high_pass=0.01, low_pass=0.2)
    _sidecar(directory, "sub-11_task-hold_desc-errts_nirs", "glm_residuals",
             sources=[path("filtered")], conditions=[], noise_model="ols",
             short_channel="mean")
    _sidecar(directory, "sub-11_task-hold_desc-glm_nirsmap", "glm_fit",
             sources=[path("filtered")], conditions=[], noise_model="ols")
    return path


_PREP = ["od_conversion", "sci_marking", "motion_tddr", "beer_lambert"]


def test_an_early_stage_is_described_up_to_itself(tmp_path):
    from fnirs_pipe.qc.boilerplate.vocabulary import steps_from_lineage

    path = _member_chain(tmp_path)
    assert [k for k, _ in steps_from_lineage(path("preproc"))] == _PREP


def test_the_residual_carries_its_filter_and_its_regression(tmp_path):
    from fnirs_pipe.qc.boilerplate.vocabulary import steps_from_lineage

    path = _member_chain(tmp_path)
    steps = steps_from_lineage(path("errts"))
    assert [k for k, _ in steps] == _PREP + ["bandpass", "confound_regression"]
    assert "short channels" in dict(steps)["confound_regression"]["regressors"]


def test_a_file_with_no_record_has_no_lineage(tmp_path):
    from fnirs_pipe.qc.boilerplate.vocabulary import steps_from_lineage

    _member_chain(tmp_path)
    assert steps_from_lineage(tmp_path / "sub-12_task-hold_desc-errts_nirs.snirf") is None


def test_a_dyad_reads_coherence_then_correlation_then_regions(tmp_path):
    # the tables share one depth, and their names would put the correlation first
    for name, step in (("stat-isc_relmat", "hyper_isc_pairs"),
                       ("seg-a_agg-roi_stat-wtc_relmat", "hyper_wtc_roichan"),
                       ("stat-wtc_relmat", "hyper_wtc")):
        _sidecar(tmp_path, f"group-G1_task-hold_{name}", step,
                 sources=["/deriv/sub-11_task-hold_desc-errts_nirs.snirf"], channel_cross=True)

    assert [k for k, _ in steps_from_sidecars(tmp_path, label="group-G1_task-hold")] == [
        "hyper_wtc_crossed", "hyper_isc", "hyper_roi"]


@pytest.mark.parametrize("params, expected", [
    ({"hrf_model": "glover", "stim_dur": 20.0},
     "as 20 s boxcars convolved with the Glover haemodynamic response function"),
    ({"hrf_model": "spm", "event_table": True},
     "as boxcars of each event's own duration convolved with the SPM canonical "
     "haemodynamic response function"),
    ({"hrf_model": "fir", "fir_delays": [0, 1, 2, 3]},
     "with a finite impulse response (FIR) basis at delays of 0 to 3 scans"),
    # nilearn's FIR shifts the event's boxcar, so the duration belongs in the sentence
    ({"hrf_model": "fir", "fir_delays": [0, 1, 2, 3], "stim_dur": 10.0},
     "with a finite impulse response (FIR) basis (10 s boxcars at delays of 0 to 3 scans)"),
    ({"hrf_model": "fir", "fir_delays": [0, 1, 2, 3], "event_table": True},
     "with a finite impulse response (FIR) basis (boxcars of each event's own duration at "
     "delays of 0 to 3 scans)"),
])
def test_the_glm_sentence_says_how_the_task_regressors_were_built(params, expected):
    assert template_slots("glm", params)["conditions"] == expected


def test_the_wavelet_sentence_states_the_package_design():
    from fnirs_pipe.pipeline.motion import WAVELET, WAVELET_IQR_FACTOR

    line = step_sentence("motion_correction", {"motion_correction": "wavelet"})
    assert f"({WAVELET})" in line and f"{WAVELET_IQR_FACTOR:g} interquartile ranges" in line


def test_bad_channels_are_described_as_marked_and_a_glm_says_it_fitted_them():
    """mne-nirs fits every fNIRS channel and the estimates carry a `bad` column."""
    screening = step_sentence("sci_pruning", {"sci_threshold": 0.8, "psp_threshold": 0.1,
                                              "min_good_frac": 0.75})
    assert "were marked as bad." in screening and "excluded" not in screening
    assert "fitted as well and flagged" in _load_steps_plain("glm")
    for key in ("hyper_wtc", "hyper_wtc_crossed", "hyper_isc", "hyper_coherence"):
        assert "retained long" in _load_steps_plain(key)


def _load_steps_plain(key):
    from fnirs_pipe.qc.boilerplate.generate import _load_steps

    return _load_steps()[key]["plain"]


@pytest.mark.parametrize("step", ["alff", "alff_roi"])
def test_every_alff_table_maps_to_one_sentence_with_its_band(step):
    assert boilerplate_key(step, {}) == "alff"
    slots = template_slots("alff", {"high_pass": 0.01, "low_pass": 0.08})
    assert (slots["l_freq"], slots["h_freq"]) == ("0.01", "0.08")


@pytest.mark.parametrize("step", ["fc", "fc_roi", "fc_seed", "fisher_z"])
def test_every_connectivity_table_maps_to_one_sentence(step):
    assert boilerplate_key(step, {}) == "fc"


def test_a_rest_run_reads_regression_then_alff_then_connectivity(tmp_path):
    """Same depth from the recording: the ranks, not the file names, set the order."""
    _sidecar(tmp_path, "sub-01_task-rest_desc-preproc_nirs", "beer_lambert", dpf=[6.0])
    pre = str(tmp_path / "sub-01_task-rest_desc-preproc_nirs.snirf")
    _sidecar(tmp_path, "sub-01_task-rest_desc-filtered_nirs", "bandpass", sources=[pre],
             high_pass=0.01, low_pass=0.08)
    filt = str(tmp_path / "sub-01_task-rest_desc-filtered_nirs.snirf")
    _sidecar(tmp_path, "sub-01_task-rest_desc-errts_nirs", "glm_residuals", sources=[filt],
             conditions=[], drift_model="cosine", drift_high_pass=0.01, noise_model="ols")
    _sidecar(tmp_path, "sub-01_task-rest_desc-errtsbroad_nirs", "glm_residuals_broadband",
             sources=[pre])
    broad = str(tmp_path / "sub-01_task-rest_desc-errtsbroad_nirs.snirf")
    _sidecar(tmp_path, "sub-01_task-rest_stat-alff_nirsmap", "alff", sources=[broad],
             high_pass=0.01, low_pass=0.08)
    keys = [k for k, _ in steps_from_sidecars(tmp_path)]
    assert keys[-2:] == ["confound_regression", "alff"]


# ---- the re-paired null ----

_PAIRNULL = {"n_iter": 22, "pair_candidates": 22, "pair_pool": "position", "cross": False,
             "pair_cond_pad_s": 141.421}


@pytest.mark.parametrize("step", [
    "hyper_wtc_bycondition_pairnull", "hyper_wtc_bycondition_pairnull_draws",
    "hyper_wtc_bycondition_roihom_pairnull", "hyper_wtc_bycondition_roichan_pairnull",
    "hyper_isc_pairnull", "hyper_isc_bycondition_pairnull",
    "hyper_isc_bycondition_pairnull_draws",
])
def test_every_re_paired_table_maps_to_one_null_sentence(step):
    assert boilerplate_key(step, {"cross": True}) == "hyper_pairnull"


def test_the_level_file_does_not_fill_the_sentence():
    assert boilerplate_key("hyper_wtc_bycondition_pairnull_level", {}) is None


def test_the_null_sentence_names_the_pool_the_pairs_the_count_and_the_pad():
    slots = template_slots("hyper_pairnull", _PAIRNULL)
    assert slots["pool"] == "the member in the same position of each other group"
    assert slots["pairs"] == "homologous channel pairs"
    assert slots["count"] == "All 22 eligible stand-ins were used."
    assert slots["pad"] == "141.4"
    # no correlation settings recorded: only the coherence was re-paired
    assert slots["measures"] == "coherence was"


def test_a_capped_any_pool_crossed_null_with_the_correlation_says_so():
    slots = template_slots("hyper_pairnull", {
        **_PAIRNULL, "n_iter": 10, "pair_candidates": 44, "pair_pool": "any", "cross": True,
        "isc_whiten_max_order": 0})
    assert slots["pool"] == "each member of every other group"
    assert slots["pairs"] == "every channel pair"
    assert slots["count"] == "10 of the 44 eligible stand-ins were used."
    assert slots["measures"] == "coherence and the inter-subject correlation were"


def test_a_dyad_reads_its_null_after_its_regions(tmp_path):
    for name, step, extra in (
            ("cond-all_null-pair_stat-wtc_relmat", "hyper_wtc_bycondition_pairnull", _PAIRNULL),
            ("seg-a_agg-roi_stat-wtc_relmat", "hyper_wtc_roichan", {}),
            ("stat-isc_relmat", "hyper_isc_pairs", {}),
            ("stat-wtc_relmat", "hyper_wtc", {})):
        _sidecar(tmp_path, f"group-G01_task-main_{name}", step, sources=["/in.snirf"], **extra)
    assert [k for k, _ in steps_from_sidecars(tmp_path)] == [
        "hyper_wtc", "hyper_isc", "hyper_roi", "hyper_pairnull"]


# ---- the phase-scrambled nulls ----

def test_the_level_file_of_the_phase_null_does_not_fill_the_sentence():
    assert boilerplate_key("hyper_wtc_phasenull_level", {}) is None


def test_the_phase_null_sentence_names_the_pairs_the_count_and_the_seed():
    slots = template_slots("hyper_phasenull", {"n_iter": 100, "seed": 1, "cross": True})
    assert slots == {"pairs": "every channel pair", "n_iter": "100",
                     "seed": " (random seed 1)"}
    assert template_slots("hyper_phasenull", {"n_iter": 50, "seed": None})["seed"] == ""


def test_the_correlation_names_its_own_null_only_when_one_ran():
    assert "50 times against the second member" in template_slots(
        "hyper_isc", {"isc_phase_null_iter": 50})["options"]
    assert template_slots("hyper_isc", {"isc_phase_null_iter": 0})["options"] == ""


def test_both_nulls_read_after_the_regions_phase_first(tmp_path):
    for name, step, extra in (
            ("cond-all_null-pair_stat-wtc_relmat", "hyper_wtc_bycondition_pairnull", _PAIRNULL),
            ("null-phase_stat-wtc_relmat", "hyper_wtc_phasenull", {"n_iter": 5}),
            ("seg-a_agg-roi_stat-wtc_relmat", "hyper_wtc_roichan", {}),
            ("stat-wtc_relmat", "hyper_wtc", {})):
        _sidecar(tmp_path, f"group-G01_task-main_{name}", step, sources=["/in.snirf"], **extra)
    assert [k for k, _ in steps_from_sidecars(tmp_path)] == [
        "hyper_wtc", "hyper_roi", "hyper_phasenull", "hyper_pairnull"]


# ---- the phase per frequency, and the cohort test ----

@pytest.mark.parametrize("step", ["hyper_wtc_phasescale", "hyper_wtc_bycondition_phasescale"])
def test_the_phase_per_frequency_has_one_sentence(step):
    assert boilerplate_key(step, {}) == "hyper_phasescale"
    assert template_slots("hyper_phasescale", {"mask_coi": True})["coi"] \
        == " outside the cone of influence"
    assert template_slots("hyper_phasescale", {"mask_coi": False})["coi"] == ""


def test_the_phase_reads_after_the_coherence_and_before_the_correlation(tmp_path):
    for name, step in (("stat-isc_relmat", "hyper_isc_pairs"),
                       ("stat-wtcphase_relmat", "hyper_wtc_phasescale"),
                       ("stat-wtc_relmat", "hyper_wtc")):
        _sidecar(tmp_path, f"group-G01_task-main_{name}", step, sources=["/in.snirf"])
    assert [k for k, _ in steps_from_sidecars(tmp_path)] == [
        "hyper_wtc", "hyper_phasescale", "hyper_isc"]


@pytest.mark.parametrize("null, said", [("repaired", "re-paired"),
                                        ("phase", "phase-randomised")])
@pytest.mark.parametrize("table", ["by_cell", "by_occasion", "cohort"])
def test_every_cohort_table_maps_to_one_test_sentence(null, said, table):
    step = f"hyper_{null}_null_{table}"
    assert boilerplate_key(step, {}) == "hyper_groupnull"
    slots = template_slots("hyper_groupnull", {"null_kind": null, "n_resample": 20000,
                                               "p_correction": "none"})
    assert slots == {"null": said, "n_resample": "20000",
                     "measure": "coherence", "tail": "one-tailed", "two_sided": "",
                     "correction": " P values were not corrected for multiple comparisons."}


@pytest.mark.parametrize("test, measure, tail", [
    ("signed", "the inter-subject correlation, Fisher z-transformed,", "two-tailed"),
    ("magnitude", "the absolute inter-subject correlation, Fisher z-transformed,",
     "one-tailed")])
def test_the_cohort_sentence_follows_the_correlations_recorded_test(test, measure, tail):
    slots = template_slots("hyper_groupnull", {"null_kind": "repaired", "measure": "isc",
                                               "isc_test": test, "p_correction": "none"})
    assert slots["measure"] == measure and slots["tail"] == tail
    assert ("smaller of the two tails" in slots["two_sided"]) == (test == "signed")


@pytest.mark.parametrize("method, named", [("fdr_bh", "Benjamini–Hochberg"),
                                           ("fdr_by", "Benjamini–Yekutieli"),
                                           ("holm", "Holm–Bonferroni"),
                                           ("bonferroni", "Bonferroni correction")])
def test_the_cohort_sentence_names_the_correction_that_ran(method, named):
    slots = template_slots("hyper_groupnull", {"null_kind": "repaired", "p_correction": method})
    assert named in slots["correction"]
    assert "within each condition, level and test" in slots["correction"]


def test_a_sidecar_without_the_setting_gets_no_correction_sentence():
    assert template_slots("hyper_groupnull", {"null_kind": "repaired"})["correction"] == ""



# ---- the Monte Carlo level ----

def test_the_monte_carlo_level_is_described_only_when_one_was_drawn():
    slots = template_slots("hyper_wtc", {"wtc_mc_count": 300, "wtc_seed": 1})
    assert "for 300 pairs of first-order autoregressive" in slots["significance"]
    assert "(random seed 1)" in slots["significance"]
    assert "seed" not in template_slots("hyper_wtc", {"wtc_mc_count": 300,
                                                      "wtc_seed": None})["significance"]
    assert template_slots("hyper_wtc", {})["significance"] == ""
