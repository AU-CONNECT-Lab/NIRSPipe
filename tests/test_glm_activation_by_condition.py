"""The GLM activation panel is one figure per condition behind a switch, not one tall image.

Five conditions stacked reached roughly 3500 px, where a condition could not be looked at
on its own and two could not be compared without scrolling between them. What matters about
the split is the part that is easy to lose: the colour scale has to stay shared, or the
switch stops being a comparison and a condition that barely activated fills its own scale
and reads as strong.

`_save_glm_brain` needs pyvista, fsaverage and an offscreen GL context, so it is stubbed
here. That is the point: everything these tests pin sits either side of the render, and the
render itself is covered by the fact that the report treats None as "skip the panel".
"""

import base64
import io

import pandas as pd
import pytest

from fnirs_pipe.qc.figures.subject import glm_figures
from fnirs_pipe.qc.figures.subject.glm_figures import (
    _shared_clim, activation_condition_figures, activation_panel,
)


def _tiny_png(w: int = 12, h: int = 8) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _results(**by_condition) -> dict:
    """{"rest": [0.1, 0.2]} -> {"rest": DataFrame with a Coef. column}."""
    return {name: pd.DataFrame({"ch_name": [f"S1_D{i + 1} hbo" for i in range(len(vals))],
                                "Coef.": list(vals)})
            for name, vals in by_condition.items()}


# ---- the shared colour scale ----

def test_the_colour_scale_covers_every_condition():
    # the weak condition must not be scaled to its own maximum, so the limit comes from
    # the strong one and both are drawn against it
    clim = _shared_clim(_results(weak=[0.01, -0.02], strong=[4.0, -1.0]))
    assert clim["pos_lims"][-1] == pytest.approx(4.0)


def test_an_all_zero_run_still_gets_a_usable_scale():
    # a zero-width scale would make stc.plot raise rather than draw a flat brain
    clim = _shared_clim(_results(flat=[0.0, 0.0]))
    assert clim["pos_lims"][-1] > 0


def test_the_coefficient_column_is_found_under_either_name():
    assert _shared_clim({"a": pd.DataFrame({"theta": [3.0]})})["pos_lims"][-1] == pytest.approx(3.0)


# ---- one figure per condition ----

def test_every_condition_is_rendered_against_the_same_scale(monkeypatch):
    seen = []

    def fake(raw_haemo, results_df, clim, view, size, title):
        seen.append((title, clim["pos_lims"][-1]))
        return _tiny_png()

    monkeypatch.setattr(glm_figures, "_save_glm_brain", fake)
    out = activation_condition_figures(None, _results(weak=[0.01], strong=[4.0]))

    assert [label for label, _ in out] == ["weak", "strong"]
    # both conditions, and both against the strong one's limit rather than their own
    assert [title for title, _ in seen] == ["weak", "strong"]
    assert [limit for _, limit in seen] == pytest.approx([4.0, 4.0])


def test_a_condition_that_failed_to_render_is_left_out(monkeypatch):
    # a blank panel behind a label reads as "this condition had no activation", which is a
    # different claim from "the render failed"
    monkeypatch.setattr(glm_figures, "_save_glm_brain",
                        lambda *a, **k: None if a[5] == "video" else _tiny_png())
    out = activation_condition_figures(None, _results(game=[1.0], video=[1.0]))
    assert [label for label, _ in out] == ["game"]


def test_no_condition_rendering_gives_nothing_rather_than_raising(monkeypatch):
    monkeypatch.setattr(glm_figures, "_save_glm_brain", lambda *a, **k: None)
    assert activation_condition_figures(None, _results(game=[1.0])) == []
    assert activation_panel(None, _results(game=[1.0])) is None


# ---- the stacked image still works for a caller that wants one file ----

def test_the_stacked_panel_still_stacks(monkeypatch):
    monkeypatch.setattr(glm_figures, "_save_glm_brain", lambda *a, **k: _tiny_png(12, 8))
    one = activation_panel(None, _results(a=[1.0]))
    three = activation_panel(None, _results(a=[1.0], b=[1.0], c=[1.0]))
    assert _png_height(three) > _png_height(one)


def _png_height(b64: str) -> int:
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(raw[20:24], "big")
