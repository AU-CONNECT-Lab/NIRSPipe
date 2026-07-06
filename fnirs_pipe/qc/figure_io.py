"""Shared helpers for writing Plotly figures to standalone HTML files."""

from __future__ import annotations

import re
from pathlib import Path

import mne

PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-3.5.0.min.js"

_IFRAME_CSS = "html,body{margin:0;padding:0;width:100%;}"

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


def _save_figure_html(fig, path: Path, extra_js: str = "") -> int:
    """Save a single Plotly figure as standalone iframe-ready HTML. Returns height px."""
    h = _figure_height(fig)
    fig.update_layout(height=h)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(
        full_html=True, include_plotlyjs=False, config={"responsive": True}
    )
    html = html.replace(
        "<head>",
        f'<head>\n<style>{_IFRAME_CSS}</style>\n<script src="{PLOTLY_CDN_URL}"></script>\n{_RESIZE_JS}',
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
