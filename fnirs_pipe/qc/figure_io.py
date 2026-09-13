"""Shared helpers for getting a figure out of memory and onto disk.

Two families, because the package draws with two libraries. Plotly figures become
standalone iframe-ready HTML; matplotlib figures arrive already encoded and are decoded
to a PNG file. Both end in ``figures/`` beside the report, which is what keeps a report
page small: the page carries a URL per figure and the browser fetches the one being
looked at. Embedding them instead is what took one hyperscanning report to 174 MB.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import mne

PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-3.5.0.min.js"

# No scrollbar inside a figure's frame. The page that holds it sizes the frame to the height
# the figure reports, so a scrollbar there is always transient -- but while it is up it takes
# 15px off the width the figure measures itself against, and a figure that sets its height
# from its width came out that much shorter than the still beside it.
_IFRAME_CSS = ("html,body{margin:0;padding:0;width:100%;}"
               "html{scrollbar-width:none;}html::-webkit-scrollbar{display:none;}")

# A figure that sets its own width does not stretch to the iframe and sits at the left edge,
# which is right for a panel meant to line up with the one above it and wrong for a single
# head on a wide page. Passed per figure rather than folded into _IFRAME_CSS for that reason.
CENTER_FIGURE_CSS = ".plotly-graph-div{margin-left:auto;margin-right:auto;}"

# The figure's own height, reported to the page that frames it. Measured off the plot
# divs and not off `document.body.scrollHeight`, which is floored by the iframe's current
# height: a figure could grow that way but never shrink, so one that sizes itself to the
# page's width left a white band under it at every width narrower than it opened at.
_RESIZE_JS = (
    "<script>(function(){"
    "function _h(){"
    "var d=document.querySelectorAll('.plotly-graph-div'),h=0;"
    # the 2px is the gap between stacked figures, so it goes between them and not after
    # the last one, where it is a hairline of white under a figure that fits exactly
    "for(var i=0;i<d.length;i++)h+=d[i].offsetHeight+(i?2:0);"
    "parent.postMessage({type:'iframe-resize',h:h||document.body.scrollHeight},'*');}"
    "window.addEventListener('load',_h);"
    "setTimeout(_h,300);"
    "try{new ResizeObserver(_h).observe(document.body);}catch(e){}"
    "})();</script>"
)

# One file holding every condition's view, picked by URL fragment, rather than the same
# traces written out once per condition. Applies to the first plot in the file, which is
# the only one a caller passing a views table has. ``window_view_spec`` measured the
# numbers; this spends them, so the run's file and a condition's view of it cannot disagree.
_HASH_VIEW_JS = (
    "<script>(function(){"
    "function apply(){"
    "if(typeof Plotly==='undefined')return;"
    "var t=window.__COND_VIEWS__,gd=document.querySelector('.plotly-graph-div');"
    "if(!t||!gd||!gd.layout)return;"
    "var k=decodeURIComponent(location.hash.replace(/^#/,''));"
    "if(!k||!t[k])return;"
    "var v=t[k],up={},axes=[];"
    "Object.keys(gd.layout).forEach(function(a){"
    "if(/^xaxis[0-9]*$/.test(a)){up[a+'.range']=v.x.slice();up[a+'.autorange']=false;"
    "axes.push([a,v.x]);}});"
    "Object.keys(v.y||{}).forEach(function(a){"
    "up[a+'.range']=v.y[a].slice();up[a+'.autorange']=false;axes.push([a,v.y[a]]);});"
    # A view that carries its own annotations replaces the set outright, which is how a
    # coherence map swaps in the phase arrows measured over its own window; anything else
    # edits the notes in place, by the annotation's own name rather than by an index
    # measured when the file was written, since an annotation added to the figure in
    # between would shift every index and land a row's note on another row
    "if(v.annotations)up.annotations=v.annotations;"
    "else (gd.layout.annotations||[]).forEach(function(a,i){"
    "if(a.name&&(v.ann||{})[a.name]!==undefined)up['annotations['+i+'].text']=v.ann[a.name];});"
    "Plotly.relayout(gd,up).then(function(){"
    "(v.bands||[]).forEach(function(b){"
    "Plotly.restyle(gd,{y:[gd.data[b.i].y.map(function(q){"
    "return q===null?null:(q<=b.lo?b.lo:b.hi);})]},[b.i]);});"
    # Plotly interface: `_rangeInitial0` / `_rangeInitial1` on a computed axis is where both
    # the modebar's reset button and a double-click read their target from, recorded at the
    # first draw and therefore the whole run. Rewritten so a reset lands on the window the
    # page asked for. Checked against the pinned CDN build by
    # tests/test_condition_view_rendering.py.
    "axes.forEach(function(p){var ax=(gd._fullLayout||{})[p[0]];"
    "if(ax){ax._rangeInitial0=p[1][0];ax._rangeInitial1=p[1][1];"
    "ax._autorangeInitial=false;}});"
    "});}"
    "if(document.readyState==='complete')apply();"
    "else window.addEventListener('load',apply);"
    "window.addEventListener('hashchange',function(){location.reload();});"
    "})();</script>"
)

# Plotly's default is `reset+autosize`, which toggles: the first double-click on a view that
# is already at its reset target autoscales over the samples the traces hold, the whole run's.
# Only files carrying a views table get this, so an ordinary figure keeps the default.
_VIEW_CONFIG = {"responsive": True, "doubleClick": "reset"}


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


def _save_figure_html(fig, path: Path, extra_js: str = "", extra_css: str = "",
                      div_id: str | None = None, views: "dict | None" = None) -> int:
    """Save a single Plotly figure as standalone iframe-ready HTML. Returns height px.

    ``views`` maps a condition key to the window the page should open on, so one file serves
    the run and every condition off a URL fragment. See ``_HASH_VIEW_JS``.
    """
    h = _figure_height(fig)
    fig.update_layout(height=h)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"div_id": div_id} if div_id else {}
    html = fig.to_html(
        full_html=True, include_plotlyjs=False,
        config=_VIEW_CONFIG if views else {"responsive": True}, **kwargs
    )
    html = html.replace(
        "<head>",
        f'<head>\n<style>{_IFRAME_CSS}{extra_css}</style>\n<script src="{PLOTLY_CDN_URL}"></script>\n{_RESIZE_JS}',
        1,
    )
    if views:
        html = html.replace(
            "</body>",
            f"<script>window.__COND_VIEWS__={json.dumps(views)};</script>{_HASH_VIEW_JS}</body>",
            1,
        )
    if extra_js:
        html = html.replace("</body>", f"{extra_js}\n</body>", 1)
    path.write_text(html, encoding="utf-8")
    return h


def _save_multi_fig_html(figs: list, path: Path, views: "dict | None" = None) -> int:
    """Stack multiple Plotly figures in one HTML file. Returns total height px.

    ``views`` maps a condition key to the :func:`~fnirs_pipe.qc.condition_views.window_view_spec`
    that names its window, so ``…/motion_detail_S1D1760.html#video`` opens the run's figure
    narrowed to that condition and the bare path opens the run's own view.
    """
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
            div_id=f"pfig{i}",
            config=_VIEW_CONFIG if views else {"responsive": True},
        )
        parts.append(f'<div class="pfig">{fig_html}</div>')
    if views:
        parts.append(f"<script>window.__COND_VIEWS__={json.dumps(views)};</script>")
        parts.append(_HASH_VIEW_JS)
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
    """Return non-BAD annotations as marker dicts (onset, duration, description).

    Unrounded: ``condition_windows`` builds window bounds from these, and a rounded bound
    no longer matches the annotation it came from.
    """
    return [
        {
            "onset":       float(a["onset"]),
            "duration":    float(a["duration"]),
            "description": str(a["description"]),
        }
        for a in raw.annotations
        if not str(a["description"]).upper().startswith("BAD")
    ]
