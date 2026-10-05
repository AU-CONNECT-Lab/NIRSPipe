"""A dyad's Methods paragraph: the stage its members were read at, then the clock, then the
measures, each sentence backed by a record rather than by what the run asked for."""

import json

import pytest

from fnirs_pipe.qc.hyper.hyper_report import group_methods

LABEL = "group-G1_task-hold"


def _sidecar(directory, name, step, sources=(), **params):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(
        {"step": step, "Sources": list(sources), "parameters": params}))


def _member(nirs, sid):
    def path(desc):
        return str(nirs / f"sub-{sid}_task-hold_desc-{desc}_nirs.snirf")

    _sidecar(nirs, f"sub-{sid}_task-hold_desc-od_nirs", "od_conversion",
             sources=[f"/bids/sub-{sid}_task-hold_nirs.snirf"])
    _sidecar(nirs, f"sub-{sid}_task-hold_desc-preproc_nirs", "beer_lambert",
             sources=[path("od")], dpf=[6.0])
    _sidecar(nirs, f"sub-{sid}_task-hold_desc-filtered_nirs", "bandpass",
             sources=[path("preproc")], high_pass=0.01, low_pass=0.2)
    _sidecar(nirs, f"sub-{sid}_task-hold_desc-errts_nirs", "glm_residuals",
             sources=[path("filtered")], conditions=[], noise_model="ols", drift_model="cosine",
             drift_high_pass=0.01)
    return path


@pytest.fixture
def tree(tmp_path):
    paths = {sid: _member(tmp_path / "deriv" / f"sub-{sid}" / "nirs", sid) for sid in ("11", "12")}
    group = tmp_path / "hyper" / "group-G1" / "nirs"
    _sidecar(group, f"{LABEL}_stat-wtc_relmat", "hyper_wtc",
             sources=[paths["11"]("errts"), paths["12"]("errts")], channel_cross=True,
             band_fmin=0.02, band_fmax=0.1, mask_coi=True, chroma=["hbo", "hbr"])
    return paths, group


def _plain(paths, group, desc, align, notes=None):
    read = [paths[sid](desc) for sid in ("11", "12")]
    return group_methods(read, group, LABEL, align, desc, {"fnirs-pipe": "9.9"},
                         notes if notes is not None else [], LABEL)["plain"]


def test_a_run_that_read_the_beer_lambert_output_is_not_described_as_filtered(tree):
    plain = _plain(*tree, "preproc", {"aligned": True})
    assert "Beer-Lambert" in plain
    assert "bandpass" not in plain and "Nuisance" not in plain


def test_a_run_that_read_the_residual_names_the_regression(tree):
    plain = _plain(*tree, "errts", {"aligned": True})
    assert "bandpass filtered (0.01–0.2 Hz)" in plain
    assert "Nuisance signals were removed" in plain


@pytest.mark.parametrize("align, said, unsaid", [
    ({"aligned": True}, "aligned on their first shared event marker", "were not aligned"),
    ({"aligned": False}, "were not aligned on an event marker", "first shared event marker"),
])
def test_the_clock_sentence_follows_the_stamp(tree, align, said, unsaid):
    plain = _plain(*tree, "errts", align)
    assert said in plain and unsaid not in plain


def test_no_stamp_makes_no_clock_claim(tree):
    plain = _plain(*tree, "errts", {"aligned": None})
    assert "event marker" not in plain


def test_an_unreadable_chain_is_named_rather_than_guessed(tree, tmp_path):
    paths, group = tree
    notes = []
    read = [str(tmp_path / "moved" / "sub-11_task-hold_desc-errts_nirs.snirf")]
    plain = group_methods(read, group, LABEL, {"aligned": True}, "errts",
                          {"fnirs-pipe": "9.9"}, notes, LABEL)["plain"]
    assert "desc-errts files" in plain and "Beer-Lambert" not in plain
    assert notes and "could not be read" in notes[0]
