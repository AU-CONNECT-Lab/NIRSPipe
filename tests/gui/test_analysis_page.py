"""The Analysis page: a run record fills its form, and the stages drawn follow the mode."""

import pytest
from werkzeug.datastructures import MultiDict

from nirspipe.cli.run import read_config
from nirspipe.interface.commands import parser_for
from nirspipe.interface.forms import form_values, to_argv
from nirspipe.interface.server import WebApp
from nirspipe.interface.stages import pipeline_stages

# a "none" that switches a mode default off, a --no- flag, a path with a space, lists
RECORD = """
[prep]
dpf = [6.0, 5.5]
sci_threshold = 0.8
motion_correction = "tddr"
cardiac_l_freq = 0.7
cardiac_h_freq = 2.0
by_condition = true

[post]
mode = "glm"
high_pass = "none"
noise_model = "ar1"
drift_model = "cosine"
combine_runs = false
fc = true
roi_mapping = "C:/maps/roi map.json"

[execution]
participant_label = ["01", "02"]
"""


def _record(tmp_path):
    path = tmp_path / "sub-01_record.toml"
    path.write_text(RECORD, encoding="utf-8")
    return path


def test_a_record_round_trips_through_the_form(tmp_path):
    config = read_config(_record(tmp_path))
    settings = {**config["prep"], **config["post"], **config["execution"]}
    parser = parser_for("nirspipe")
    values = form_values(parser, settings)
    form = MultiDict([(name, value) for name, given in values.items() for value in given]
                     + [("bids_dir", "/bids"), ("output_dir", "/out"),
                        ("analysis_level", "participant")])

    args = vars(parser.parse_args(to_argv(["nirspipe"], parser, form)[1:]))
    for dest, value in settings.items():
        assert args[dest] == value, dest


def test_the_page_fills_its_form_from_a_record(tmp_path):
    client = WebApp().flask.test_client()
    page = client.get("/analysis/run/", query_string={"record": str(_record(tmp_path))})
    html = page.get_data(as_text=True)

    assert 'data-prefilled="1"' in html
    assert 'name="--sci-threshold" value="0.8"' in html
    assert '<option value="glm" selected>' in html
    assert "Loaded the settings of sub-01_record.toml" in html


def test_a_record_the_reader_refuses_is_named(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[prep]\nsci_threshold = 0.8\nnot_a_flag = 1\n", encoding="utf-8")
    html = WebApp().flask.test_client().get(
        "/analysis/run/", query_string={"record": str(path)}).get_data(as_text=True)

    assert "not_a_flag" in html
    assert "data-prefilled" not in html


@pytest.mark.parametrize("form, last", [
    ({}, ["Beer-Lambert Law"]),
    ({"--mode": "denoise"}, ["Bandpass / Resample"]),
    ({"--mode": "denoise", "--drift-model": "cosine"}, ["Confound Regression"]),
    ({"--mode": "glm"}, ["GLM"]),
    ({"--mode": "glm", "--fc": "on"}, ["Connectivity"]),
    ({"--mode": "rest"}, ["ALFF", "Connectivity"]),
])
def test_the_stages_follow_the_mode(form, last):
    assert pipeline_stages(MultiDict(form))[-1] == last


def test_motion_correction_runs_on_optical_density_before_beer_lambert():
    names = [name for level in pipeline_stages(MultiDict()) for name in level]
    assert names.index("Optical Density") < names.index("Motion Correction") < names.index("Beer-Lambert Law")


def test_only_the_analysis_preview_carries_stages():
    client = WebApp().flask.test_client()
    rest = client.post("/preview/nirspipe/", data={"--mode": "rest"}).get_json()
    assert rest["stages"][-1] == ["ALFF", "Connectivity"]
    assert client.post("/preview/nirspipe-qc/", data={}).get_json()["stages"] is None
