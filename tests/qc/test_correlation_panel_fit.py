"""The HbO-HbR panel refits to its frame, and the report lets that frame follow it.

The scripts run in a browser, so these pin the three things that broke when the window was
narrowed rather than the behaviour itself. One case pins what the after stage leaves blank.
"""

import re

import mne
import numpy as np

from nirspipe.qc.common.figure_io import _RESIZE_JS
from nirspipe.qc.common.report_shell import TEMPLATE_DIR
from nirspipe.qc.figures.subject.correlation_panel import fit_js, hbo_hbr_correlation_figure
from tests._synth import synth_raw


def _haemo() -> mne.io.Raw:
    raw = synth_raw("01", "hold", duration=60.0, motion_onset=None)
    od = mne.preprocessing.nirs.optical_density(raw)
    return mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)


def test_the_column_share_is_written_in_rather_than_read_off_the_layout():
    # Plotly writes the square constraint's narrower domain back into layout.xaxis.domain,
    # so a script reading it there could only ever shrink the figure
    haemo = _haemo()
    fig = hbo_hbr_correlation_figure(haemo, raw_after=haemo.copy())
    script = fit_js(fig)
    domain = fig.layout.xaxis.domain
    assert f"COL={domain[1] - domain[0]!r}" in script
    assert "xaxis||{}).domain" not in script


def test_the_width_pin_a_height_relayout_sets_is_dropped():
    haemo = _haemo()
    script = fit_js(hbo_hbr_correlation_figure(haemo, raw_after=haemo.copy()))
    assert "delete gd.layout.width" in script
    assert "gd.style.height=h+'px'" in script


def test_a_channel_only_the_after_stage_marks_bad_is_blank_there():
    # errts marks the short channels its regressor emptied; their residual is float noise
    haemo = _haemo()
    after = haemo.copy()
    after.info["bads"] = ["S5_D5 hbo", "S5_D5 hbr"]
    fig = hbo_hbr_correlation_figure(haemo, raw_after=after)
    before_heat, after_heat = (t for t in fig.data if t.type == "heatmap")
    rows = [i for i, name in enumerate(after_heat.y) if name.startswith("S5_D5")]
    z_after, z_before = (np.asarray(t.z, dtype=float) for t in (after_heat, before_heat))
    assert np.isnan(z_after[rows]).all() and np.isnan(z_after[:, rows]).all()
    assert np.isfinite(z_before[rows]).any()
    assert np.isfinite(np.delete(np.delete(z_after, rows, 0), rows, 1)).any()

    def r_of(name):
        trace = next(t for t in fig.data if t.name == name)
        return dict(zip(trace.customdata, np.asarray(trace.y, dtype=float)))
    before, after_r = r_of("before denoising"), r_of("after denoising")
    assert np.isfinite(before["S5_D5"]) and np.isnan(after_r["S5_D5"])
    assert np.isfinite(after_r["S1_D1"])


def test_the_height_report_watches_the_plot_divs_not_the_body():
    # the figure files carry no doctype, so the root never shrinks below its frame
    assert "observe(document.body)" not in _RESIZE_JS
    assert "o.observe(d)" in _RESIZE_JS


def test_only_the_frames_whose_figures_report_their_height_take_it():
    # the HbO-HbR panel refits to its frame; the trial image swaps views of different heights
    text = (TEMPLATE_DIR / "subject_report.html.j2").read_text(encoding="utf-8")
    frames = re.findall(r"<iframe[^>]*data-fit-height[^>]*>", text)
    assert len(frames) == 2
    assert any("hbo_hbr_path" in f for f in frames)
    assert any('id="trialimg-frame"' in f for f in frames)
    assert 'querySelectorAll("iframe[data-fit-height]")' in text
