"""The brain figure: the 3D views with the static optode flat map beside them, one picture."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from nirspipe.qc.common.figure_io import figure_namer
from nirspipe.qc.figures.subject import optode_layout
from nirspipe.qc.subject import report
from tests._fingerprint import fingerprint_raw
from tests._synth import synth_raw

RED, GREEN = "#e74c3c", "#27ae60"


def _png_b64(width: int) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, 40), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def raw():
    # every optode at its own place, so each line's two ends name one source and one detector
    return fingerprint_raw("01", "tapping")[0]


def test_the_flat_map_sits_beside_the_views(tmp_path, monkeypatch):
    raw = synth_raw("01", "tapping")
    monkeypatch.setattr(report, "quality_brain_views", lambda *a, **k: _png_b64(120))
    monkeypatch.setattr(report, "optode_layout_static", lambda *a, **k: _png_b64(40))
    names = [n for n in raw.ch_names if n.endswith("760")]
    coords = np.array([raw.info["chs"][raw.ch_names.index(n)]["loc"][:3] for n in names])
    errors = []
    out = report._section_brain({n: 0.9 for n in raw.ch_names}, [], coords,
                                np.ones(len(names), bool), raw, "01", errors, tmp_path,
                                figure_namer("sub-01_task-tapping"), ch_names_brain=names,
                                sci_threshold=0.8)
    assert errors == []
    png = tmp_path / out["brain_views_path"].rsplit("/", 1)[-1]
    assert Image.open(png).size == (160, 40)


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


def test_a_rejected_pair_is_red_at_any_sci_and_the_rest_follow_their_sci(raw, monkeypatch):
    pairs = sorted({n.rsplit(" ", 1)[0] for n in raw.ch_names})
    rejected, low = pairs[0], pairs[1]
    sci = {n: 0.96 for n in raw.ch_names}
    sci.update({n: 0.1 for n in raw.ch_names if n.startswith(low + " ")})
    lines = _drawn_lines(raw, sci, [f"{rejected} 760"], monkeypatch)
    assert lines[rejected] == RED
    assert lines[low] == RED
    assert {lines[p] for p in pairs[2:]} == {GREEN}
