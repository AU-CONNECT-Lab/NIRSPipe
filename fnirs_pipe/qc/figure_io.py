"""Shared helpers for getting a figure out of memory and onto disk.

Two families, because the package draws with two libraries. Plotly figures become
standalone iframe-ready HTML; matplotlib figures arrive already encoded and are decoded
to a PNG file. Both end in ``figures/`` beside the report, which is what keeps a report
page small: the page carries a URL per figure and the browser fetches the one being
looked at. Embedding them instead is what took one hyperscanning report to 174 MB.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import mne

PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-3.5.0.min.js"

_IFRAME_CSS = "html,body{margin:0;padding:0;width:100%;}"

# A figure that sets its own width does not stretch to the iframe and sits at the left edge,
# which is right for a panel meant to line up with the one above it and wrong for a single
# head on a wide page. Passed per figure rather than folded into _IFRAME_CSS for that reason.
CENTER_FIGURE_CSS = ".plotly-graph-div{margin-left:auto;margin-right:auto;}"

_RESIZE_JS = (
    "<script>(function(){"
    "function _h(){parent.postMessage({type:'iframe-resize',h:document.body.scrollHeight},'*');}"
    "window.addEventListener('load',_h);"
    "setTimeout(_h,300);"
    "try{new ResizeObserver(_h).observe(document.body);}catch(e){}"
    "})();</script>"
)


def _figure_height(fig, default: int = 500) -> int:
    h = getattr(getattr(fig, "layout", None), "height", None)
    if h:
        return int(h)
    for trace in getattr(fig, "data", ()):
        if getattr(trace, "type", "") == "heatmap":
            y = getattr(trace, "y", None)
            if y is not None:
                return max(300, min(len(y) * 22 + 140, 1400))
    return default


def _save_figure_html(fig, path: Path, extra_js: str = "", extra_css: str = "") -> int:
    """Save a single Plotly figure as standalone iframe-ready HTML. Returns height px."""
    h = _figure_height(fig)
    fig.update_layout(height=h)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(
        full_html=True, include_plotlyjs=False, config={"responsive": True}
    )
    html = html.replace(
        "<head>",
        f'<head>\n<style>{_IFRAME_CSS}{extra_css}</style>\n<script src="{PLOTLY_CDN_URL}"></script>\n{_RESIZE_JS}',
        1,
    )
    if extra_js:
        html = html.replace("</body>", f"{extra_js}\n</body>", 1)
    path.write_text(html, encoding="utf-8")
    return h


def _save_multi_fig_html(figs: list, path: Path) -> int:
    """Stack multiple Plotly figures in one HTML file. Returns total height px."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = (
        '<meta charset="UTF-8">'
        f'<script src="{PLOTLY_CDN_URL}"></script>'
        f'<style>*{{box-sizing:border-box;}}{_IFRAME_CSS}.pfig{{margin-bottom:2px;}}</style>'
        f'{_RESIZE_JS}'
    )
    parts = [f"<!DOCTYPE html><html><head>{head}</head><body>"]
    total_h = 0
    for i, fig in enumerate(figs):
        if fig is None:
            continue
        h = _figure_height(fig)
        total_h += h + 2
        fig.update_layout(height=h)
        fig_html = fig.to_html(
            full_html=False, include_plotlyjs=False,
            div_id=f"pfig{i}", config={"responsive": True},
        )
        parts.append(f'<div class="pfig">{fig_html}</div>')
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return max(total_h, 100)


def _fig_href(figures_dir: Path, name: str) -> str:
    """URL of a figure as the report must link to it, the report sitting above ``figures/``.

    figures/            + carpet_gvtd.html -> "figures/carpet_gvtd.html"
    figures/sub-01_task-rest/ + same       -> "figures/sub-01_task-rest/carpet_gvtd.html"

    Per-run reports put their figures in a subdirectory so several runs of one subject stop
    overwriting each other; a caller that passes a bare ``figures/`` still gets the old URL.
    """
    if figures_dir.parent.name == "figures":
        return f"figures/{figures_dir.name}/{name}"
    return f"figures/{name}"


def _save_b64_png(b64: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))


def save_png(b64: str, figures_dir: Path, name: str) -> "str | None":
    """Decode one base64 PNG into ``figures_dir/name`` and return the report's URL for it.

    ::

      save_png(b64, .../group-d01/figures, "wtc_hbo_S1D1.png")
      -> "figures/wtc_hbo_S1D1.png"

    The two halves of writing a matplotlib figure out, in one call, because every caller
    does both and a caller that saved without taking the href back used to be how a figure
    reached disk and never reached the page. An empty or absent b64 returns None, so a
    builder that declined to draw leaves the page's ``{% if %}`` gate closed.
    """
    if not b64:
        return None
    _save_b64_png(b64, figures_dir / name)
    return _fig_href(figures_dir, name)


def _pair_fname(pair: str) -> str:
    """Strip non-alphanumeric characters so the string is BIDS desc-value safe."""
    return re.sub(r"[^a-zA-Z0-9]", "", pair)


def get_channel_pairs(raw: mne.io.Raw) -> list[str]:
    """Return unique channel pair names (without chroma suffix) in MNE channel order."""
    picks = mne.pick_types(raw.info, fnirs="hbo")
    seen: set[str] = set()
    pairs: list[str] = []
    for i in picks:
        pair = raw.ch_names[i].rsplit(" ", 1)[0]
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def extract_markers(raw: mne.io.Raw) -> list[dict]:
    """Return non-BAD annotations as marker dicts (onset, duration, description)."""
    return [
        {
            "onset":       round(float(a["onset"]), 4),
            "duration":    round(float(a["duration"]), 4),
            "description": str(a["description"]),
        }
        for a in raw.annotations
        if not str(a["description"]).upper().startswith("BAD")
    ]
