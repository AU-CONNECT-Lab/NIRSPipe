"""What the generated forms carry for the page's helpers: field kinds for the folded summaries, and Detect's subjects."""

import pytest

from nirspipe.interface.server import WebApp
from tests._synth import make_bids_dataset


def test_detect_lists_the_datasets_subjects(tmp_path):
    bids = make_bids_dataset(tmp_path, subjects=("02", "01"), tasks=("rest",))
    found = WebApp().flask.test_client().get("/subjects", query_string={"bids_dir": str(bids)})
    assert found.get_json() == {"subjects": ["01", "02"]}


def test_detect_says_why_it_found_nothing(tmp_path):
    client = WebApp().flask.test_client()

    assert client.get("/subjects").get_json() == {"error": "Fill the BIDS folder first."}
    missing = tmp_path / "nowhere"
    assert client.get("/subjects", query_string={"bids_dir": str(missing)}).get_json() == {
        "error": f"No folder {missing}."}
    assert client.get("/subjects", query_string={"bids_dir": str(tmp_path)}).get_json() == {
        "error": f"No subjects in {tmp_path}."}


def test_every_field_names_its_kind_and_the_form_names_detect():
    html = WebApp().flask.test_client().get("/raw-qc/qc/").get_data(as_text=True)

    assert 'data-subjects="/subjects"' in html
    assert html.count('class="field ') == html.count("data-kind=")
    for kind in ("switch", "boolean", "choice", "number", "text", "list"):
        assert f'data-kind="{kind}"' in html


# ---- the Analysis page's hints ----

def _hints(**fields):
    from werkzeug.datastructures import MultiDict

    from nirspipe.interface.hints import analysis

    return analysis(MultiDict(fields))


def test_without_a_mode_the_post_flags_that_need_one_are_hidden():
    h = _hints()
    for flag in ("--drift-model", "--short-channel", "--fc", "--export-mean-filled", "--hrf-model",
                 "--events-path"):
        assert flag in h.inactive, flag
    assert h.placeholders["--high-pass"] == "off"


def test_the_mode_names_its_defaults_and_glm_opens_the_task_model():
    glm = _hints(**{"--mode": "glm"})
    assert glm.placeholders["--hrf-model"] == "spm (glm default)"
    assert glm.placeholders["--drift-model"] == "cosine (glm default)"
    assert "--hrf-model" not in glm.inactive and "--fir-delays" in glm.inactive
    assert "--fir-delays" not in _hints(**{"--mode": "glm", "--hrf-model": "fir"}).inactive
    rest = _hints(**{"--mode": "rest"})
    assert rest.placeholders["--low-pass"] == "0.08 (rest default; none switches it off)"
    assert "--hrf-model" in rest.inactive and "--min-time" not in rest.inactive


def test_the_noise_model_shows_in_every_mode_that_fits_a_regression():
    assert "--noise-model" in _hints().inactive
    for mode in ("glm", "rest", "denoise"):
        assert "--noise-model" not in _hints(**{"--mode": mode}).inactive, mode
    assert _hints(**{"--mode": "rest"}).placeholders["--noise-model"] == "ols (rest default)"


def test_a_drift_setting_shows_only_beside_the_model_that_reads_it():
    assert "--drift-high-pass" not in _hints(**{"--mode": "glm"}).inactive
    poly = _hints(**{"--mode": "glm", "--drift-model": "polynomial"})
    assert "--drift-high-pass" in poly.inactive and "--drift-order" not in poly.inactive


def test_a_motion_setting_shows_only_for_a_method_that_reads_it():
    from nirspipe.pipeline.motion import STD_THRESH_DEFAULTS

    assert {"--motion-std-thresh", "--wavelet-iqr"} <= set(_hints().inactive)
    spline = _hints(**{"--motion-correction": "spline"})
    assert "--motion-std-thresh" not in spline.inactive
    assert spline.placeholders["--motion-std-thresh"] == f"{STD_THRESH_DEFAULTS['spline']:g} (spline default)"


def test_the_gvtd_settings_follow_censoring_and_the_fill_follows_the_mode():
    assert {"--gvtd-censor-n-std", "--censor-fill"} <= set(_hints().inactive)
    censored = _hints(**{"--gvtd-censor": "long"})
    assert not {"--gvtd-censor-n-std", "--gvtd-min-epoch"} & set(censored.inactive)
    # a file's own BAD_ spans and its dark edges are filled with no flag naming them
    assert "--censor-fill" not in _hints(**{"--mode": "glm"}).inactive


def test_the_band_note_says_when_high_pass_and_drift_disagree():
    from nirspipe.interface.hints import band_note

    assert band_note(None, "0.01", None, None) is None
    assert band_note("glm", "", "cosine", "").startswith("A cosine drift model needs a cutoff")
    assert "does not span that band" in band_note("glm", "0.01", "polynomial", "")
    assert band_note("glm", "0.02", "cosine", "0.01").startswith("Drift high-pass 0.01 Hz is below")
    assert "doing the same job" in band_note("glm", "0.01", "cosine", "0.01")
    assert band_note("glm", "none", "cosine", "0.01") is None
    # rest fills both from its preset, so an empty form has nothing to warn about
    assert band_note("rest", "", "", "") is None


def test_the_preview_leaves_out_what_the_form_hides():
    client = WebApp().flask.test_client()
    form = {"bids_dir": "/b", "output_dir": "/o", "analysis_level": "participant", "--mode": "denoise",
            "--hrf-model": "spm", "--high-pass": "0.02"}
    shown = client.post("/preview/nirspipe/", data=form).get_json()
    assert "--hrf-model" in shown["hints"]["inactive"]
    assert "--hrf-model" not in shown["preview"] and "--high-pass" in shown["preview"]
    assert client.post("/preview/nirspipe-qc/", data={}).get_json()["hints"] is None


# ---- the form hides what the command's own table says a run would not read ----

def test_without_a_mode_the_settings_only_postprocessing_reads_are_hidden_too():
    hidden = set(_hints().inactive)
    assert {"--filter-method", "--filter-order", "--resample-sfreq", "--combine-runs",
            "--p-correction"} <= hidden
    # the report previews the filter on a prep-only run, so the cutoffs stay
    assert not {"--high-pass", "--low-pass", "--chromophores", "--mode"} & hidden
    assert not {"--filter-method", "--resample-sfreq"} & set(_hints(**{"--mode": "denoise"}).inactive)


def test_a_fir_filter_hides_the_order_it_does_not_have():
    assert "--filter-order" not in _hints(**{"--mode": "rest"}).inactive
    assert "--filter-order" in _hints(**{"--mode": "rest", "--filter-method": "fir"}).inactive


@pytest.mark.parametrize("form", [
    {}, {"--mode": "glm"}, {"--mode": "rest"}, {"--mode": "denoise", "--fc": "on"},
    {"--mode": "glm", "--hrf-model": "fir", "--drift-model": "polynomial"},
    {"--mode": "rest", "--filter-method": "fir", "--gvtd-censor": "long",
     "--motion-correction": "spline_wavelet"},
])
def test_the_form_hides_exactly_the_shown_flags_the_table_names(form):
    from werkzeug.datastructures import MultiDict

    from nirspipe.cli import run as run_cli
    from nirspipe.interface.forms import fields
    from nirspipe.interface.hints import analysis

    parser = run_cli._build_parser()
    shown = {field.name for _title, group in fields(parser) for field in group}
    settings = {name.lstrip("-").replace("-", "_"): value for name, value in form.items()}
    # the form resolves what the mode fills before asking, as a run does
    settings.setdefault("motion_correction", "tddr")
    for dest in ("drift_model", "hrf_model"):
        settings.setdefault(dest, run_cli.mode_defaults(settings.get("mode")).get(dest))
    expected = {"--" + dest.replace("_", "-") for dest in run_cli.unread_settings(settings)}
    hidden = set(analysis(MultiDict(form)).inactive)
    assert hidden & shown == expected & shown
    # every name it hides is a flag of the command, one withheld from the form included
    assert hidden <= set(parser._option_string_actions)
