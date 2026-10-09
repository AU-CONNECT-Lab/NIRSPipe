"""The HbO-HbR panel refits to its frame, and the report lets that frame follow it.

The scripts run in a browser, so these pin the three things that broke when the window was
narrowed rather than the behaviour itself.
"""

import re

import mne

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


def test_the_height_report_watches_the_plot_divs_not_the_body():
    # the figure files carry no doctype, so the root never shrinks below its frame
    assert "observe(document.body)" not in _RESIZE_JS
    assert "o.observe(d)" in _RESIZE_JS


def test_only_the_refitting_frame_takes_the_figures_height():
    text = (TEMPLATE_DIR / "subject_report.html.j2").read_text(encoding="utf-8")
    frames = re.findall(r"<iframe[^>]*data-fit-height[^>]*>", text)
    assert len(frames) == 1 and "hbo_hbr_path" in frames[0]
    assert 'querySelectorAll("iframe[data-fit-height]")' in text
