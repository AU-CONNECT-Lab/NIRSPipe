"""A flag typed for a mode or a method that does not read it is named, in the log and in the
run report, where it was dropped without a word. One table says what a run reads."""

import logging

import pytest

from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _unread_typed

GLM_ONLY = {"stim_dur", "hrf_model", "events_path", "contrast_file", "fir_delays"}
# a setting of every stage a mode adds, each named by the flag's own help as read there
REST = {"mode": "rest", "drift_model": "cosine", "motion_correction": "tddr"}
GLM = {"mode": "glm", "drift_model": "cosine", "hrf_model": "spm", "motion_correction": "tddr"}


# ---- the table ----

def test_without_a_mode_no_postprocessing_setting_is_read_but_what_the_report_previews():
    unread = run_cli.unread_settings({"motion_correction": "tddr"})
    assert GLM_ONLY <= set(unread)
    assert {"noise_model", "drift_model", "short_channel", "fc", "min_time", "censor_fill",
            "filter_method", "filter_order", "resample_sfreq", "combine_runs", "p_correction",
            "export_mean_filled"} <= set(unread)
    assert unread["fc"] == "nothing is postprocessed without --mode"
    # the report previews the filter on its spectrum and picks its chromophores
    assert not {"high_pass", "low_pass", "chromophores", "mode", "config", "roi_mapping"} & set(unread)
    # preprocessing is read whatever the mode
    assert not {"sci_threshold", "dpf", "cardiac_l_freq", "motion_correction", "window_length",
                "bad_channels", "by_condition"} & set(unread)


def test_rest_and_denoise_leave_the_task_model_unread():
    for mode in ("rest", "denoise"):
        unread = run_cli.unread_settings({**REST, "mode": mode})
        assert GLM_ONLY <= set(unread), mode
        assert {unread[dest] for dest in GLM_ONLY} == {"only --mode glm reads it"}
        assert not {"noise_model", "drift_model", "short_channel", "censor_fill", "high_pass",
                    "p_correction", "resample_sfreq"} & set(unread), mode


def test_glm_reads_the_task_model_and_the_delays_only_under_fir():
    assert not (GLM_ONLY - {"fir_delays"}) & set(run_cli.unread_settings(GLM))
    assert run_cli.unread_settings(GLM)["fir_delays"] == "only --hrf-model fir reads it"
    assert "fir_delays" not in run_cli.unread_settings({**GLM, "hrf_model": "fir"})


@pytest.mark.parametrize("drift, unread, read", [
    ("cosine", "drift_order", "drift_high_pass"),
    ("polynomial", "drift_high_pass", "drift_order"),
])
def test_a_drift_setting_is_read_only_by_its_own_model(drift, unread, read):
    found = run_cli.unread_settings({**GLM, "drift_model": drift})
    assert unread in found and read not in found


def test_neither_drift_setting_is_read_without_a_drift_model():
    found = run_cli.unread_settings({"mode": "denoise", "motion_correction": "tddr"})
    assert {"drift_high_pass", "drift_order"} <= set(found)


def test_the_time_floor_is_read_where_fc_or_alff_is_computed():
    assert "min_time" not in run_cli.unread_settings(REST)
    assert "min_time" in run_cli.unread_settings(GLM)
    assert "min_time" not in run_cli.unread_settings({**GLM, "fc": True})
    assert "min_time" not in run_cli.unread_settings({**REST, "mode": "denoise", "fc": True})


def test_a_fir_filter_has_no_order_to_read():
    assert "filter_order" not in run_cli.unread_settings(REST)
    assert "filter_order" in run_cli.unread_settings({**REST, "filter_method": "fir"})


def test_the_censoring_settings_are_read_only_when_censoring():
    assert {"gvtd_censor_n_std", "gvtd_min_epoch"} <= set(run_cli.unread_settings(REST))
    assert not {"gvtd_censor_n_std", "gvtd_min_epoch"} & set(
        run_cli.unread_settings({**REST, "gvtd_censor": "long"}))


@pytest.mark.parametrize("method, unread", [
    ("tddr", {"motion_std_thresh", "wavelet_iqr"}), ("none", {"motion_std_thresh", "wavelet_iqr"}),
    ("spline", {"wavelet_iqr"}), ("wavelet", {"motion_std_thresh"}), ("spline_wavelet", set()),
])
def test_a_motion_setting_is_read_only_by_a_method_that_uses_it(method, unread):
    found = set(run_cli.unread_settings({**REST, "motion_correction": method}))
    assert found & {"motion_std_thresh", "wavelet_iqr"} == unread


# ---- only what was typed is named ----

def test_only_a_typed_flag_is_named_and_by_its_flag():
    args = {**REST, "typed": ["stim_dur", "drift_order", "high_pass", "sci_threshold"]}
    assert _unread_typed(args) == [
        ("--stim-dur", "only --mode glm reads it"),
        ("--drift-order", "only a polynomial --drift-model reads it")]
    assert _unread_typed({**REST, "typed": []}) == []
    # a caller that hands no record of what was typed gets no warning
    assert _unread_typed(REST) == []


def test_a_cutoff_typed_without_a_mode_is_named_as_filtering_nothing():
    found = dict(_unread_typed({"motion_correction": "tddr", "typed": ["high_pass", "fc"]}))
    assert found == {
        "--fc": "nothing is postprocessed without --mode",
        "--high-pass": "nothing is filtered without --mode; the report's spectrum previews the filter"}
    assert _unread_typed({**REST, "typed": ["high_pass", "low_pass"]}) == []


# ---- through the command ----

SELECT = ["--population", "adult", "--sci-threshold", "0.8", "--participant-label", "01",
          "--task-label", "tapping", "--skip-bids-validation"]


def _warned(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "typed but not applied" in r.getMessage()]


def test_a_run_names_the_flags_it_will_not_apply_before_any_subject(mini_bids, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        run_cli.main([str(mini_bids), str(tmp_path / "out"), "participant", *SELECT, "--dry-run",
                      "--mode", "rest", "--stim-dur", "10", "--drift-order", "2"])
    assert _warned(caplog) == [
        "--drift-order is typed but not applied: only a polynomial --drift-model reads it",
        "--stim-dur is typed but not applied: only --mode glm reads it"]


def test_a_run_that_applies_everything_typed_says_nothing(mini_bids, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        run_cli.main([str(mini_bids), str(tmp_path / "out"), "participant", *SELECT, "--dry-run",
                      "--mode", "glm", "--stim-dur", "10", "--drift-high-pass", "0.01",
                      "--hrf-model", "fir", "--fir-delays", "0", "1", "--fc", "--min-time", "60"])
    assert _warned(caplog) == []


def test_a_setting_a_config_file_carries_is_not_called_typed(mini_bids, tmp_path, caplog):
    config = tmp_path / "settings.toml"
    config.write_text('[post]\nmode = "rest"\nstim_dur = 10.0\ndrift_order = 2\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        run_cli.main([str(mini_bids), str(tmp_path / "out"), "participant", *SELECT, "--dry-run",
                      "--config", str(config)])
    assert _warned(caplog) == []


@pytest.fixture(scope="module")
def report(mini_bids, tmp_path_factory):
    out = tmp_path_factory.mktemp("unread") / "out"
    run_cli.main([str(mini_bids), str(out), "participant", *SELECT, "--mode", "rest",
                  "--stim-dur", "10", "--drift-order", "2"])
    return (out / "sub-01" / "sub-01_task-tapping_report.html").read_text(encoding="utf-8")


def test_the_run_report_lists_them_under_the_run_command(report):
    block = " ".join(report.split("Typed but not applied:")[1].split("</p>")[0].split())
    assert block == ("<code>--drift-order</code> (only a polynomial --drift-model reads it); "
                     "<code>--stim-dur</code> (only --mode glm reads it).")


def test_a_report_of_a_run_that_applied_everything_has_no_such_line(report):
    # the note sits once, in the run command block, and names nothing that was not typed
    assert report.count("Typed but not applied:") == 1
    assert "<code>--hrf-model</code> (only" not in report
