"""BAD_ spans drawn by kind on the motion panel and the design matrix: unselected grey, the rest red."""

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from nirspipe.qc.figures.common._utils import SPAN_KIND_COLOR, SPAN_KIND_LABEL, _hex_to_rgba
from nirspipe.qc.figures.common.motion_panel import build_motion_detail_figure
from nirspipe.qc.figures.subject import glm_figures

from tests._synth import synth_raw

SEGMENTS = {"BAD_unselected": [(0.0, 50.0)], "BAD_gvtd": [(120.0, 5.0)],
            "BAD_manual": [(200.0, 20.0)]}
KINDS = ["unselected", "corrupted", "corrupted"]     # SEGMENTS' order
LABELS = [SPAN_KIND_LABEL["unselected"], SPAN_KIND_LABEL["corrupted"]]


@pytest.fixture(scope="module")
def od():
    raw = synth_raw("01", "tapping", duration=300.0)
    return mne.preprocessing.nirs.optical_density(raw, verbose="error")


def _span_fills(fig):
    return {(s.x0, s.x1): s.fillcolor for s in fig.layout.shapes if s.type == "rect"}


def _swatches(fig):
    return [t.name for t in fig.data if t.name in SPAN_KIND_LABEL.values()]


# ---- the motion panel ----

def test_the_motion_panel_colours_each_span_by_its_kind(od):
    fig = build_motion_detail_figure(od, od, od.ch_names[0], segments=SEGMENTS)
    grey, red = (_hex_to_rgba(SPAN_KIND_COLOR[k], 0.12) for k in ("unselected", "corrupted"))
    assert _span_fills(fig) == {(0.0, 50.0): grey, (120.0, 125.0): red, (200.0, 220.0): red}
    assert _swatches(fig) == LABELS


def test_only_the_kinds_present_get_a_swatch(od):
    fig = build_motion_detail_figure(od, od, od.ch_names[0], segments={"BAD_gvtd": [(120.0, 5.0)]})
    assert _swatches(fig) == [SPAN_KIND_LABEL["corrupted"]]


def test_without_spans_the_motion_panel_draws_none(od):
    plain = build_motion_detail_figure(od, od, od.ch_names[0])
    assert not _span_fills(plain) and not _swatches(plain)
    assert plain.to_json() == build_motion_detail_figure(od, od, od.ch_names[0],
                                                         segments={}).to_json()


# ---- the design matrix ----

def _design(seconds: float = 300.0) -> pd.DataFrame:
    times = np.arange(0.0, seconds, 1.0)
    return pd.DataFrame({"tapping": ((times % 60) < 20).astype(float), "constant": 1.0},
                        index=times)


@pytest.fixture
def drawn(monkeypatch):
    figs = []
    monkeypatch.setattr(glm_figures, "fig_png_b64", lambda fig: figs.append(fig) or "")
    yield figs
    for fig in figs:
        plt.close(fig)


def test_the_design_matrix_shades_each_span_by_its_kind(drawn):
    glm_figures.design_matrix_static_figure(_design(), ["tapping"], segments=SEGMENTS)
    (fig,) = drawn
    expected = [to_rgba(SPAN_KIND_COLOR[k], 0.18) for k in KINDS]
    for ax in fig.axes:
        assert [tuple(p.get_facecolor()) for p in ax.patches] == pytest.approx(expected)
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == LABELS


def test_without_spans_the_design_matrix_shades_nothing(drawn):
    glm_figures.design_matrix_static_figure(_design(), ["tapping"])
    (fig,) = drawn
    assert not any(ax.patches for ax in fig.axes)
    assert not fig.legends
