"""The picked-at-load view must render as the written-out view did, pixel for pixel.

A condition's window used to be baked into a file of its own: the figure was narrowed, its y
axes refitted, and the result saved. Now one file carries every condition's window in a
table and the page applies one on load, which moves part of the work into JavaScript that no
Python test can see. `rescale_y_to_window` is kept as the reference for exactly that reason:
it is what the rendering is *supposed* to be, and the shim has to reproduce it.

So this compares the two by rendering both in a browser and hashing the pixels. It needs
Chrome and reaches the Plotly CDN the saved files name, and skips when either is missing
rather than passing on a pair of blank pages: the size floor below is what makes a blank
render a failure instead of a match.

The last test asks the page a question a screenshot cannot answer: where a reset would go.
The shim rewrites Plotly's own reset target, which is a Plotly internal, so that test is
what catches the rename when the pinned CDN version is bumped.
"""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from fnirs_pipe.qc.condition_views import (
    apply_carpet_window, rescale_y_to_window, zoom_to_condition,
)
from fnirs_pipe.qc.figure_io import _save_multi_fig_html
from fnirs_pipe.qc.report import _carpet_views, _condition_views, _save_plotly_html

WINDOW = (120.0, 320.0)
SLUG = "baseline"
# a 1000x560 blank page is a few KB; either panel drawn is eighty times that, so a render
# that failed cannot pass by matching another failed render
MIN_RENDER_BYTES = 20_000

_CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def _chrome() -> "str | None":
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return next((c for c in _CHROME_CANDIDATES if os.path.exists(c)), None)


pytestmark = pytest.mark.skipif(_chrome() is None,
                                reason="no Chrome or Edge to render the saved figures in")


def _shot(html: Path, png: Path, fragment: str = "") -> bytes:
    url = html.resolve().as_uri() + fragment
    subprocess.run(
        [_chrome(), "--headless=old", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=15000", "--window-size=1000,560",
         f"--user-data-dir={png.parent / '_profile'}", f"--screenshot={png}", url],
        capture_output=True, timeout=120,
    )
    if not png.exists():
        pytest.skip("the browser wrote no screenshot")
    data = png.read_bytes()
    if len(data) < MIN_RENDER_BYTES:
        pytest.skip("the figure did not render; the Plotly CDN is probably unreachable")
    return data


def _sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# Asks the page what a reset would do, which a screenshot cannot show: the shim writes the
# condition's window onto Plotly's own reset target, and a renamed internal would leave the
# modebar's reset button and a double-click opening the whole run again with nothing to say
# the axis moved. Polls because the shim sets it after its relayout resolves.
_RESET_PROBE = """
<div id="reset-probe">pending</div>
<script>(function(){
var n=0,id=setInterval(function(){
var gd=document.querySelector('.plotly-graph-div');
var ax=gd&&gd._fullLayout&&gd._fullLayout.xaxis;
var out=document.getElementById('reset-probe');
if(ax&&ax._rangeInitial0!==undefined){
out.textContent='RESET='+ax._rangeInitial0+','+ax._rangeInitial1;clearInterval(id);}
else if(++n>100){out.textContent='RESET=none';clearInterval(id);}},50);})();</script>
"""


def _reset_target(html: Path, fragment: str) -> "tuple[float, float]":
    """The range a reset would restore, read out of the rendered page."""
    probed = html.with_name("probed.html")
    page = html.read_text(encoding="utf-8")
    probed.write_text(page.replace("</body>", _RESET_PROBE + "</body>", 1), encoding="utf-8")
    out = subprocess.run(
        [_chrome(), "--headless=old", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=15000", "--window-size=1000,560",
         f"--user-data-dir={html.parent / '_probe_profile'}", "--dump-dom",
         probed.resolve().as_uri() + fragment],
        capture_output=True, timeout=120,
    ).stdout.decode("utf-8", "replace")
    if "main-svg" not in out:
        pytest.skip("the figure did not render; the Plotly CDN is probably unreachable")
    found = re.search(r"RESET=([-\d.eE+]+),([-\d.eE+]+)", out)
    assert found, "the shim left Plotly's reset target on the whole run"
    return float(found.group(1)), float(found.group(2))


def _motion_figure():
    """The shape of the per-channel motion panel: a footprint strip, GVTD with a threshold
    rule, the derivative with spike spans behind it, and before/after OD. Quiet up to t=600
    and loud after, so a window over the quiet stretch is the case the y rescale is for."""
    t = np.arange(0, 1000, 0.25)
    rng = np.random.default_rng(0)
    gvtd = np.abs(np.sin(t / 11)) * np.where(t > 600, 0.017, 0.003) + rng.normal(0, 2e-4, t.size)
    deriv = np.abs(np.diff(gvtd, prepend=gvtd[0])) * 8
    od_before = np.sin(t / 7) * np.where(t > 600, 0.35, 0.09)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.08, 0.28, 0.28, 0.36], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=[200, 200, 260, 260], y=[0.3, 0.7, 0.7, 0.3], fill="toself",
                             mode="lines", line=dict(width=0)), row=1, col=1)
    for row, series in ((2, gvtd), (3, deriv)):
        top = float(series.max()) * 1.1
        fig.add_trace(go.Scatter(x=[210, 210, 240, 240, None, 700, 700, 730, 730],
                                 y=[0, top, top, 0, None, 0, top, top, 0], fill="toself",
                                 mode="lines", line=dict(width=0)), row=row, col=1)
    fig.add_trace(go.Scatter(x=t, y=gvtd, mode="lines"), row=2, col=1)
    # the threshold rule and its label, so the notes the views table fills in are not the
    # only annotations on the figure and a slot resolved by position would be wrong
    fig.add_annotation(x=1, xref="x domain", y=0.01, yref="y2", text="thresh=0.0100",
                       showarrow=False, font=dict(size=8), xanchor="right", row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=deriv, mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=od_before, mode="lines"), row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=od_before * 0.8, mode="lines"), row=4, col=1)
    fig.add_vrect(x0=400, x1=500, fillcolor="rgba(231,76,60,0.12)", line_width=0,
                  layer="below", row=2, col=1, annotation_text="block",
                  annotation_position="top left", annotation_font_size=7)
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, float(gvtd.max()) * 1.1], row=2, col=1)
    fig.update_yaxes(range=[0, float(deriv.max()) * 1.1], row=3, col=1)
    fig.update_layout(height=520, showlegend=False)
    return fig


def _carpet_figure():
    """Two GVTD rows over a heatmap, shaped like the real carpet: the rows carry the named
    stat labels a condition view rewrites, and share one y range as the builder gives them."""
    from fnirs_pipe.qc.figures.motion_panel import GVTD_STAT_SLOT

    t = np.arange(0, 1000, 0.5)
    rng = np.random.default_rng(1)
    long_g = np.abs(np.sin(t / 13)) * np.where(t > 600, 0.017, 0.003)
    short_g = np.abs(np.sin(t / 7)) * np.where(t > 600, 0.039, 0.006)
    top = float(max(long_g.max(), short_g.max())) * 1.1

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.2, 0.2, 0.6],
                        vertical_spacing=0.05)
    for row, series, name in ((1, long_g, "long"), (2, short_g, "short")):
        fig.add_trace(go.Scatter(x=[210, 210, 240, 240], y=[0, top, top, 0], fill="toself",
                                 mode="lines", line=dict(width=0)), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=series, mode="lines", name="before"), row=row, col=1)
        fig.add_hline(y=float(series.mean()) * 3, row=row, col=1,
                      line=dict(dash="dash", width=1))
        fig.add_annotation(x=0.998, xref="x domain", y=0.99, yref="y domain",
                           text=f"max {series.max():.2e}", showarrow=False, xanchor="right",
                           name=f"{GVTD_STAT_SLOT}{name}", row=row, col=1)
        fig.update_yaxes(range=[0, top], row=row, col=1)
    fig.add_trace(go.Heatmap(z=rng.normal(0, 1, (40, t.size)), x=t, colorscale="RdBu",
                             zmin=-3, zmax=3, showscale=False), row=3, col=1)
    fig.update_layout(height=520, showlegend=False)
    return fig


def test_a_motion_panel_picked_by_fragment_renders_as_the_written_out_one_did(tmp_path):
    reference = _motion_figure()
    zoom_to_condition(reference, *WINDOW)
    rescale_y_to_window(reference, *WINDOW)
    _save_multi_fig_html([reference], tmp_path / "reference.html")

    one_file = _motion_figure()
    _save_multi_fig_html([one_file], tmp_path / "one_file.html",
                         views=_condition_views(one_file, [(SLUG, *WINDOW)]))

    was = _shot(tmp_path / "reference.html", tmp_path / "reference.png")
    now = _shot(tmp_path / "one_file.html", tmp_path / "one_file.png", f"#{SLUG}")
    assert _sha(now) == _sha(was)


def test_a_carpet_picked_by_fragment_renders_as_the_written_out_one_did(tmp_path):
    # the other saver, and the other div id: _save_plotly_html leaves Plotly to name the div
    reference = _carpet_figure()
    apply_carpet_window(reference, *WINDOW)
    _save_plotly_html(reference, tmp_path / "reference.html")

    one_file = _carpet_figure()
    _save_plotly_html(one_file, tmp_path / "one_file.html",
                      views=_carpet_views(one_file, [(SLUG, *WINDOW)]))

    was = _shot(tmp_path / "reference.html", tmp_path / "reference.png")
    now = _shot(tmp_path / "one_file.html", tmp_path / "one_file.png", f"#{SLUG}")
    assert _sha(now) == _sha(was)


def test_a_file_opened_with_no_fragment_still_shows_the_whole_run(tmp_path):
    # the run's own page reads the same file, so the table must not narrow it by default
    run = _motion_figure()
    _save_multi_fig_html([run], tmp_path / "run.html")

    one_file = _motion_figure()
    _save_multi_fig_html([one_file], tmp_path / "one_file.html",
                         views=_condition_views(one_file, [(SLUG, *WINDOW)]))

    was = _shot(tmp_path / "run.html", tmp_path / "run.png")
    now = _shot(tmp_path / "one_file.html", tmp_path / "one_file_run.png")
    assert _sha(now) == _sha(was)


def test_an_unknown_fragment_falls_back_to_the_whole_run(tmp_path):
    # a renamed condition leaves a link nothing answers, which must not blank the figure
    run = _motion_figure()
    _save_multi_fig_html([run], tmp_path / "run.html")

    one_file = _motion_figure()
    _save_multi_fig_html([one_file], tmp_path / "one_file.html",
                         views=_condition_views(one_file, [(SLUG, *WINDOW)]))

    was = _shot(tmp_path / "run.html", tmp_path / "run.png")
    now = _shot(tmp_path / "one_file.html", tmp_path / "gone.png", "#nosuchcondition")
    assert _sha(now) == _sha(was)


def test_a_reset_returns_to_the_condition_not_to_the_run(tmp_path):
    one_file = _motion_figure()
    _save_multi_fig_html([one_file], tmp_path / "one_file.html",
                         views=_condition_views(one_file, [(SLUG, *WINDOW)]))

    assert _reset_target(tmp_path / "one_file.html", f"#{SLUG}") == pytest.approx(WINDOW)


def _coherence_map():
    """One pairing's whole-run coherence map, plus the arrays the views table re-reads.

    A coherence map carries its phase field as one annotation per arrow, and those are the
    part of it a window cannot inherit: the grid spreads over whatever span it was drawn for,
    so the run's set leaves a 200 s window holding the handful of columns that happen to fall
    inside it. The window carries its own set and the view swaps the whole array in, which is
    the branch of the shim below this exercises.
    """
    from fnirs_pipe.qc.figures.hyper_post_figures import build_wtc_map_interactive

    freqs = np.logspace(-2, np.log10(0.2), 40)
    times = np.arange(0.0, 1000.0, 0.5)
    rng = np.random.default_rng(3)
    wtc = np.clip(np.abs(np.sin(np.add.outer(freqs * 90, times / 40))) * 0.8
                  + rng.normal(0, 0.05, (freqs.size, times.size)), 0.0, 1.0)
    data = {
        "wtc": wtc,
        "phase": np.add.outer(freqs * 40, times / 90) % (2 * np.pi) - np.pi,
        "coi": np.sqrt(2.0) * np.minimum(times, times[-1] - times) + 1e-6,
        "sig": None,
    }
    markers = [{"onset": 120.0, "duration": 200.0, "description": SLUG},
               {"onset": 600.0, "duration": 200.0, "description": "talk"}]
    fig = build_wtc_map_interactive(data, freqs, times, "sub-01 × sub-02", markers,
                                    {SLUG: "#4c72b0", "talk": "#dd8452"}, site_label="leftPFC")
    return fig, data, freqs, times


def test_a_coherence_map_picked_by_fragment_carries_the_window_s_own_arrows(tmp_path):
    from fnirs_pipe.qc.figures.hyper_post_figures import wtc_condition_views
    from fnirs_pipe.qc.figure_io import _save_figure_html

    # the third saver, and the one the dyad report writes its ROI maps with
    reference, data, freqs, times = _coherence_map()
    views = wtc_condition_views(reference, [(SLUG, *WINDOW)], data, freqs, times)
    assert views, "the window holds no sample of the run"

    # what the shim is supposed to arrive at, applied here instead
    reference.update_xaxes(range=list(WINDOW), autorange=False)
    reference.layout.annotations = views[SLUG]["annotations"]
    _save_figure_html(reference, tmp_path / "reference.html")

    one_file, *_ = _coherence_map()
    _save_figure_html(one_file, tmp_path / "one_file.html", views=views)

    was = _shot(tmp_path / "reference.html", tmp_path / "reference.png")
    now = _shot(tmp_path / "one_file.html", tmp_path / "one_file.png", f"#{SLUG}")
    assert _sha(now) == _sha(was)
