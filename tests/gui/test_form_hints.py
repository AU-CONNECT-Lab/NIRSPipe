"""What the generated forms carry for the page's helpers: field kinds for the folded summaries, and Detect's subjects."""

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
    html = WebApp().flask.test_client().get("/raw-qc/prep-raw/").get_data(as_text=True)

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


def test_the_censor_settings_follow_what_marks_bad_spans():
    assert {"--gvtd-censor-n-std", "--censor-fill"} <= set(_hints().inactive)
    assert "--censor-fill" not in _hints(**{"--bad-spans": "spans.tsv"}).inactive
    censored = _hints(**{"--gvtd-censor": "long"})
    assert not {"--gvtd-censor-n-std", "--gvtd-min-epoch-s", "--censor-fill"} & set(censored.inactive)


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
    assert client.post("/preview/nirspipe-qc/prep-raw/", data={}).get_json()["hints"] is None
