"""`--population`: the bands, DPF and template of an age group, each overridden by its own flag."""

import json
import math
import re
import tomllib
from pathlib import Path

import pytest

from nirspipe.cli import _app_options
from nirspipe.cli import qc as qc_cli
from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _make_prep_config
from nirspipe.pipeline.population import POPULATIONS
from nirspipe.qc.boilerplate.vocabulary import _band_population, _dpf_phrase, template_slots
from nirspipe.qc.common.channel_table import template_note

from tests._fingerprint import make_fingerprint_dataset
from tests._synth import WAVELENGTHS

DOCS = Path(__file__).resolve().parents[2] / "docs" / "workflows.md"
# the docs table's rows each population spans, by the ages its row names
SPANS = {
    "infant": ("birth to 3 months", "3 to 12 months"),
    "child": ("1 to 2 years", "2 to 4 years", "4 to 8 years", "8 to 12 years"),
    "adult": ("12 to 18 years", "adults"),
    "older": ("adults",),
}
SELECT = ("--participant-label", "01", "--skip-bids-validation")
ADULT_BANDS = {"cardiac_l_freq": 0.7, "cardiac_h_freq": 1.8, "resp_l_freq": 0.1,
               "resp_h_freq": 0.5}


def _scholkmann_wolf(nm, years):
    # Scholkmann & Wolf (2013) Eq. 7, written out here rather than imported
    return (223.3 + 0.05624 * years ** 0.8493 - 5.723e-7 * nm ** 3 + 0.001245 * nm ** 2
            - 0.9025 * nm)


# ---- the table ----

def _docs_rows() -> dict:
    """Age -> (cardiac low, high, respiration low, high) from the Channel screening table."""
    rows = {}
    for line in DOCS.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not re.fullmatch(r"[\d.]+ to [\d.]+", cells[2]):
            continue
        cardiac = [float(v) for v in cells[2].split(" to ")]
        resp = [float(v) for v in cells[4].split(" to ")]
        rows[cells[0]] = (*cardiac, *resp)
    return rows


@pytest.mark.parametrize("name", list(SPANS))
def test_each_band_is_the_union_of_the_docs_rows_it_spans(name):
    rows = _docs_rows()
    spanned = [rows[age] for age in SPANS[name]]
    union = (min(r[0] for r in spanned), max(r[1] for r in spanned),
             min(r[2] for r in spanned), max(r[3] for r in spanned))
    row = POPULATIONS[name]
    assert (row["cardiac_l_freq"], row["cardiac_h_freq"],
            row["resp_l_freq"], row["resp_h_freq"]) == union


def test_the_populations_cover_every_age_once():
    spans = sorted(tuple(row["ages"]) for row in POPULATIONS.values())
    assert spans[0][0] == 0 and math.isinf(spans[-1][1])
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_each_dpf_age_lies_in_its_population():
    for row in POPULATIONS.values():
        low, high = row["ages"]
        assert low <= row["dpf_age"] <= high


# ---- the command line: what it fills, what overrides it ----

@pytest.fixture(scope="module")
def bids(tmp_path_factory):
    return make_fingerprint_dataset(tmp_path_factory.mktemp("population"))[0]


def _record(out, name="sub-01.toml"):
    return tomllib.loads((out / "sub-01" / "logs" / name).read_text(encoding="utf-8"))


def _dry(bids, out, *flags):
    run_cli.main([str(bids), str(out), "participant", "--sci-threshold", "0.8", *flags,
                  *SELECT, "--dry-run"])
    return _record(out)


def test_adult_fills_the_bands_and_the_dpf(bids, tmp_path):
    record = _dry(bids, tmp_path, "--population", "adult")
    assert {k: record["prep"][k] for k in ADULT_BANDS} == ADULT_BANDS
    assert record["prep"]["population"] == "adult"
    # written as "age" it would read each participant's age on rerun
    assert "dpf" not in record["prep"]
    assert record["prep_sources"] == {**{k: "population" for k in ADULT_BANDS},
                                      "dpf": "population"}


def test_a_typed_flag_wins_over_the_population(bids, tmp_path):
    record = _dry(bids, tmp_path, "--population", "adult", "--cardiac-h-freq", "3.0",
                  "--dpf", "6")
    assert record["prep"]["cardiac_h_freq"] == 3.0 and record["prep"]["dpf"] == [6.0]
    assert "cardiac_h_freq" not in record["prep_sources"]
    assert "dpf" not in record["prep_sources"]


def test_without_population_the_bands_are_still_required(bids, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_cli.main([str(bids), str(tmp_path), "participant", "--sci-threshold", "0.8",
                      "--dpf", "6", *SELECT, "--dry-run"])
    assert "--cardiac-l-freq' (or --population)" in capsys.readouterr().err


def test_the_population_never_sets_the_sci_line(bids, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_cli.main([str(bids), str(tmp_path), "participant", "--population", "adult",
                      *SELECT, "--dry-run"])
    assert "Missing option '--sci-threshold'." in capsys.readouterr().err


def test_a_rerun_from_the_record_is_the_same_run(bids, tmp_path):
    first = _dry(bids, tmp_path / "first", "--population", "child")
    rerun = _dry(bids, tmp_path / "rerun",
                 "--config", str(tmp_path / "first" / "sub-01" / "logs" / "sub-01.toml"))
    assert rerun["prep"] == first["prep"]
    assert rerun["prep_sources"]["dpf"] == "population"


def test_the_preprocessed_run_carries_the_batch_dpf_and_the_bands(bids, tmp_path):
    run_cli.main([str(bids), str(tmp_path), "participant", "--sci-threshold", "0.75",
                  "--population", "adult", *SELECT, "--no-report"])
    preproc = json.loads(next(tmp_path.rglob("*desc-preproc_nirs.json")).read_text())
    params = preproc["parameters"]
    assert params["dpf"] == pytest.approx([_scholkmann_wolf(w, 30.0) for w in WAVELENGTHS])
    assert (params["dpf_age"], params["dpf_population"]) == (30.0, "adult")
    sci = json.loads(next(tmp_path.rglob("*desc-sci_nirs.json")).read_text())["parameters"]
    assert {k: sci[k] for k in ADULT_BANDS} == ADULT_BANDS
    assert sci["population"] == "adult"


# ---- the infant template, by each participant's own age ----

def _ages(root, ages):
    root.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"sub-{s}\t{a}\n" for s, a in ages.items())
    (root / "participants.tsv").write_text("participant_id\tage\n" + rows, encoding="utf-8")
    return root


def _prep_args(root, population, filled, template=None):
    return {"bids_dir": root, "dpf": ["age"], "template": template, "population": population,
            "from_population": filled, "sci_threshold": 0.8, "motion_correction": "tddr",
            **ADULT_BANDS}


def test_an_infant_takes_the_template_nearest_their_age_and_the_batch_dpf_age(tmp_path):
    root = _ages(tmp_path, {"01": 0.5})
    config = _make_prep_config("01", None, _prep_args(root, "infant", ["dpf", "template"]))
    assert (config.template, config.age) == ("6mo", 0.5)


def test_a_typed_template_wins_over_the_infant_age(tmp_path):
    root = _ages(tmp_path, {"01": 0.5})
    config = _make_prep_config("01", None, _prep_args(root, "infant", ["dpf"], "12mo"))
    assert config.template == "12mo"


def test_adult_stays_on_fsaverage_at_the_batch_age(tmp_path):
    root = _ages(tmp_path, {"01": 41.0})
    config = _make_prep_config("01", None, _prep_args(root, "adult", ["dpf"]))
    assert (config.template, config.age) == ("fsaverage", 30.0)


def test_an_infant_without_an_age_stops_naming_both_fixes(tmp_path, capsys):
    root = _ages(tmp_path, {"02": 0.5})
    with pytest.raises(SystemExit):
        _app_options.participant_ages(["age"], ["dpf", "template"], root, ["01"])
    assert "gives no age for sub-01; add it, or pass --template" in capsys.readouterr().err


# ---- raw QC ----

def test_raw_qc_takes_the_bands_dpf_and_template(bids, tmp_path):
    qc_cli.main([str(bids), str(tmp_path), "participant", "--population", "adult",
                 "--sci-threshold", "0.8", *SELECT])
    params = json.loads(next(tmp_path.rglob("*_desc-sqmraw_qc.json")).read_text())["parameters"]
    assert (params["cardiac_l_freq"], params["cardiac_h_freq"]) == (0.7, 1.8)
    assert (params["resp_l_freq"], params["resp_h_freq"]) == (0.1, 0.5)
    assert params["dpf"] == pytest.approx([_scholkmann_wolf(w, 30.0) for w in WAVELENGTHS])
    assert (params["dpf_population"], params["template"]) == ("adult", "fsaverage")
    record = _record(tmp_path, "sub-01_desc-raw.toml")
    assert "dpf" not in record["qc"] and record["qc_sources"]["dpf"] == "population"


def test_raw_qc_refuses_the_population_at_group_level(bids, tmp_path, capsys):
    with pytest.raises(SystemExit):
        qc_cli.main([str(bids), str(tmp_path), "group", "--population", "adult",
                     "--skip-bids-validation"])
    assert "--population is a participant-level setting" in capsys.readouterr().err


# ---- what the pages say ----

def test_the_methods_name_the_population_values():
    assert _dpf_phrase([5.98, 5.23], 30.0, "adult") == (
        "differential pathlength factors (DPF) of 5.98 and 5.23, in ascending order of "
        "wavelength, at 30 years, the package's age for 12 to 65 years, by the general "
        "equation of Scholkmann and Wolf (2013)")
    both = {"population": "infant", "from_population": ["cardiac_l_freq", "cardiac_h_freq"]}
    assert _band_population(both) == ", the package's value for under 1 year"
    one = {"population": "older", "from_population": ["cardiac_l_freq"]}
    assert _band_population(one) == ", its lower edge the package's value for 65 years and over"
    assert _band_population({}) == ""


def test_a_child_run_on_fsaverage_says_so():
    slots = template_slots("optode_landmarks", {"space": "fsaverage", "population": "child"})
    assert "no template for 2 to 18 years" in slots["template_note"]
    assert template_slots("optode_landmarks", {"space": "fsaverage"})["template_note"] == ""
    assert template_note("child", None) and template_note("child", "fsaverage")
    assert template_note("child", "2yr") is None and template_note("adult", None) is None
