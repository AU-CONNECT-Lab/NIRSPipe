"""The brain figure: the 3D views with the static optode flat map beside them, one picture."""

import base64
import io

import mne
import numpy as np
import pytest
from PIL import Image

from nirspipe.qc.common.figure_io import figure_namer
from nirspipe.qc.figures.subject import optode_layout
from nirspipe.qc.subject import report
from tests._fingerprint import fingerprint_raw
from tests._synth import synth_raw

RED, GREEN = "#e74c3c", "#27ae60"


def _png_b64(width: int, colour: str = "white") -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, 40), colour).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def raw():
    # every optode at its own place, so each line's two ends name one source and one detector
    return fingerprint_raw("01", "tapping")[0]


def test_the_flat_map_sits_beside_the_views_at_four_fifths_of_their_height(tmp_path, monkeypatch):
    raw = synth_raw("01", "tapping")
    monkeypatch.setattr(report, "quality_brain_views", lambda *a, **k: _png_b64(120))
    monkeypatch.setattr(report, "optode_layout_static", lambda *a, **k: _png_b64(40, "black"))
    names = [n for n in raw.ch_names if n.endswith("760")]
    coords = np.array([raw.info["chs"][raw.ch_names.index(n)]["loc"][:3] for n in names])
    errors = []
    out = report._section_brain({n: 0.9 for n in raw.ch_names}, [], coords,
                                np.ones(len(names), bool), raw, "01", errors, tmp_path,
                                figure_namer("sub-01_task-tapping"), ch_names_brain=names,
                                sci_threshold=0.8)
    assert errors == []
    png = Image.open(tmp_path / out["brain_views_path"].rsplit("/", 1)[-1]).convert("L")
    # a 40 x 40 map scaled to 32 x 32 beside the 120 x 40 views, 4 px of white above and below
    assert png.size == (152, 40)
    dark = np.argwhere(np.asarray(png) < 128)
    assert (dark.min(axis=0).tolist(), dark.max(axis=0).tolist()) == ([4, 120], [35, 151])



def _unplaced(raw):
    from nirspipe.io.optode_space import Placement, hold_placement
    return hold_placement(raw, Placement("none", note="the file has no 3-D optode positions"))


def test_an_unplaced_run_draws_the_flat_map_alone(tmp_path, monkeypatch):
    raw = _unplaced(synth_raw("01", "tapping"))

    def no_views(*a, **k):
        raise AssertionError("3-D views drawn for optodes no reading placed")

    monkeypatch.setattr(report, "quality_brain_views", no_views)
    monkeypatch.setattr(report, "optode_layout_static", lambda *a, **k: _png_b64(40, "black"))
    names = [n for n in raw.ch_names if n.endswith("760")]
    coords = np.array([raw.info["chs"][raw.ch_names.index(n)]["loc"][:3] for n in names])
    errors = []
    out = report._section_brain({n: 0.9 for n in raw.ch_names}, [], coords,
                                np.ones(len(names), bool), raw, "01", errors, tmp_path,
                                figure_namer("sub-01_task-tapping"), ch_names_brain=names,
                                sci_threshold=0.8)
    assert errors == [] and out["brain_views_flat"]
    png = Image.open(tmp_path / out["brain_views_path"].rsplit("/", 1)[-1])
    assert png.size == (40, 40)


def test_an_unplaced_run_projects_nothing_onto_the_cortex(tmp_path, monkeypatch):
    import pandas as pd

    raw = _unplaced(synth_raw("01", "rest"))

    def no_cortex(*a, **k):
        raise AssertionError("ALFF projected onto the cortex for optodes no reading placed")

    monkeypatch.setattr(report, "alff_brain_figure", no_cortex)
    od = mne.preprocessing.nirs.optical_density(raw)
    haemo = _unplaced(mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0))
    alff = pd.DataFrame({"channel": haemo.ch_names, "malff": np.linspace(0.5, 1.5, len(haemo.ch_names))})
    errors = []
    out = report._section_rest(alff, None, "01", errors, tmp_path,
                               figure_namer("sub-01_task-rest"), raw_haemo=haemo)
    assert errors == []
    assert out["rest_alff_brain_path"] is None and out["rest_alff_unplaced"]
    # the flat map stands in for the cortex
    assert (tmp_path / out["rest_alff_flat_path"].rsplit("/", 1)[-1]).exists()

def _drawn_lines(raw, sci, rejected, monkeypatch) -> dict:
    """Each source-detector line of the drawn map, by the labels at its two ends: its colour."""
    figures = []
    monkeypatch.setattr(optode_layout, "fig_png_b64", lambda fig: figures.append(fig) or "")
    optode_layout.optode_layout_static(raw, sci, rejected, sci_threshold=0.8)
    ax = figures[0].axes[0]
    at = {tuple(np.round(t.get_position(), 9)): t.get_text() for t in ax.texts}
    lines = {}
    for line in ax.lines:
        ends = [at.get(tuple(np.round(xy, 9))) for xy in line.get_xydata()]
        if len(ends) == 2 and all(ends):
            lines["_".join(sorted(ends, key=lambda e: e[0] != "S"))] = line.get_color()
    return lines


def test_each_pair_is_one_line_between_its_own_source_and_detector(raw, monkeypatch):
    lines = _drawn_lines(raw, {n: 0.95 for n in raw.ch_names}, [], monkeypatch)
    assert set(lines) == {n.rsplit(" ", 1)[0] for n in raw.ch_names}


# ---- the map draws the run's placement, not the file's frame ----

# a file frame like NIRStar's: turned half a circle about z and moved 130 mm off the origin
_FILE_FROM_HEAD = np.array([[-1.0, 0.0, 0.0, 0.13],
                            [0.0, -1.0, 0.0, 0.12],
                            [0.0, 0.0, 1.0, 0.11],
                            [0.0, 0.0, 0.0, 1.0]])


def _in_file_frame(raw, placement):
    from nirspipe.io.optode_space import hold_placement
    moved = raw.copy()
    for ch in moved.info["chs"]:
        ch["loc"][:9] = mne.transforms.apply_trans(_FILE_FROM_HEAD, ch["loc"][:9].reshape(3, 3)).ravel()
    return hold_placement(moved, placement)


def _head_placed(raw):
    from nirspipe.io.optode_space import Placement, head_to_mri, hold_placement
    return hold_placement(raw.copy(), Placement("head", np.diag([1e3, 1e3, 1e3, 1.0]) @ head_to_mri()))


def test_a_cap_recorded_in_another_frame_lands_where_its_head_frame_twin_does(raw):
    from nirspipe.io.optode_space import Placement, head_to_mri
    from nirspipe.qc.figures.common.topomap import _projected_optodes
    # what a landmark registration finds for that file: back to head, then onto fsaverage mm
    found = np.diag([1e3, 1e3, 1e3, 1.0]) @ head_to_mri() @ np.linalg.inv(_FILE_FROM_HEAD)
    expected, _, _ = _projected_optodes(_head_placed(raw))
    drawn, _, _ = _projected_optodes(_in_file_frame(raw, Placement("landmarks", found)))
    assert drawn.keys() == expected.keys()
    for name, xy in expected.items():
        assert drawn[name] == pytest.approx(tuple(xy), abs=1e-9), name


def test_an_unplaced_cap_is_drawn_as_its_coordinates_stand(raw):
    from nirspipe.io.optode_space import Placement
    from nirspipe.qc.figures.common.topomap import _projected_optodes
    expected, _, outlines = _projected_optodes(_head_placed(raw))
    drawn, _, _ = _projected_optodes(_in_file_frame(raw, Placement("none", note="no reading")))
    # left in the file's frame the cap is not moved back onto the head it was recorded on
    assert all(np.hypot(*(np.asarray(drawn[n]) - expected[n])) > 0.01 for n in expected)


def test_a_rejected_pair_is_red_at_any_sci_and_the_rest_follow_their_sci(raw, monkeypatch):
    pairs = sorted({n.rsplit(" ", 1)[0] for n in raw.ch_names})
    rejected, low = pairs[0], pairs[1]
    sci = {n: 0.96 for n in raw.ch_names}
    sci.update({n: 0.1 for n in raw.ch_names if n.startswith(low + " ")})
    lines = _drawn_lines(raw, sci, [f"{rejected} 760"], monkeypatch)
    assert lines[rejected] == RED
    assert lines[low] == RED
    assert {lines[p] for p in pairs[2:]} == {GREEN}
