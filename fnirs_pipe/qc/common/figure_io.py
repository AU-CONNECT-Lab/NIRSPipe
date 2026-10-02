"""Shared helpers for getting a figure out of memory and onto disk.

Two families, because the package draws with two libraries. Plotly figures become
standalone iframe-ready HTML; matplotlib figures arrive already encoded and are decoded
to a PNG file. Both end in ``figures/`` beside the report, which is what keeps a report
page small: the page carries a URL per figure and the browser fetches the one being
looked at.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import mne

from fnirs_pipe.utils import pair_of
from fnirs_pipe.io.naming import figure_name

PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-3.5.0.min.js"

# No scrollbar inside a figure's frame: even a transient one takes width off the figure, and
# a figure that sets its height from its width comes out shorter.
_IFRAME_CSS = ("html,body{margin:0;padding:0;width:100%;}"
               "html{scrollbar-width:none;}html::-webkit-scrollbar{display:none;}")

# A figure that sets its own width does not stretch to the iframe and sits at the left edge,
# which is right for a panel meant to line up with the one above it and wrong for a single
# head on a wide page. Passed per figure rather than folded into _IFRAME_CSS for that reason.
CENTER_FIGURE_CSS = ".plotly-graph-div{margin-left:auto;margin-right:auto;}"

# The figure's own height, reported to the page that frames it. Measured off the plot
# divs and not off `document.body.scrollHeight`, which is floored by the iframe's current
# height and so never lets a figure shrink.
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

    ``views`` maps a condition key to the :func:`~fnirs_pipe.qc.subject.condition_views.window_view_spec`
    that names its window, so ``…_chan-S1D1760_desc-motion_nirs.html#video`` opens the run's
    figure narrowed to that condition and the bare path opens the run's own view.
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


def figure_namer(label: str, condition: "str | None" = None, prefix: str = ""):
    """Names every figure of one run, or of one condition of it, with the label bound in.

    ::

      fig_name = figure_namer("sub-01_task-rest")
      fig_name("carpet")                  -> "sub-01_task-rest_desc-carpet_nirs.html"
      fig_name("detail", channel="S1D1")  -> "sub-01_task-rest_chan-S1D1_desc-detail_nirs.html"
      figure_namer("sub-01_task-rest", "game1")("carpet")
                                          -> "sub-01_task-rest_cond-game1_desc-carpet_nirs.html"
      figure_namer("sub-01_task-rest", prefix="raw")("carpet")
                                          -> "sub-01_task-rest_desc-rawcarpet_nirs.html"

    Binding the label here is what lets one ``figures/`` directory hold every run of a
    subject, each name saying which run it is; binding the condition does the same for a
    condition's panels.

    ``prefix`` is how the raw viewer keeps its panels apart from the pipeline report's in
    that shared folder: the two draw the same panels at different stages of one run, so
    both would otherwise ask for ``desc-carpet``. Set in one place rather than spelled into
    each panel's name, so the two writers call their panels the same thing.
    """
    def fig_name(desc: str, *, suffix: str = "nirs", extension: str = ".html",
                 **entities) -> str:
        entities.setdefault("condition", condition)
        return figure_name(label, prefix + desc, suffix=suffix, extension=extension,
                           **entities)

    # what this namer is bound to, for a caller that has the namer and needs to say so
    fig_name.label = label
    fig_name.condition = condition
    return fig_name


def _fig_href(name: str) -> str:
    """URL of a figure as the report must link to it, the report sitting above ``figures/``.

    ``"sub-01_task-rest_desc-carpet_nirs.html"``
      -> ``"figures/sub-01_task-rest_desc-carpet_nirs.html"``

    One line, and it stays a function because it is the one place that knows a report sits
    directly above its figures.
    """
    return f"figures/{name}"


def _save_b64_png(b64: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))


def fig_png_b64(fig, dpi: int = 300, **savefig_kw) -> str:
    """A matplotlib figure as a base64 PNG, closed afterwards so pyplot lets go of it."""
    # the caller drew it through pyplot, so this import is already loaded and picks no backend
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", **savefig_kw)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def save_png(b64: str, figures_dir: Path, name: str) -> "str | None":
    """Decode one base64 PNG into ``figures_dir/name`` and return the report's URL for it.

    ::

      save_png(b64, .../group-01/figures,
               "group-G1_task-rest_chromo-hbo_chan-S1D1_desc-wtcmap_nirs.png")
      -> "figures/group-G1_task-rest_chromo-hbo_chan-S1D1_desc-wtcmap_nirs.png"

    The two halves of writing a matplotlib figure out, in one call, because every caller
    does both and saving without taking the href back leaves a figure on disk that never
    reaches the page. An empty or absent b64 returns None, so a
    builder that declined to draw leaves the page's ``{% if %}`` gate closed.
    """
    if not b64:
        return None
    _save_b64_png(b64, figures_dir / name)
    return _fig_href(name)


def pair_slug(pair: "tuple[str, str] | None", n_pairings: int) -> str:
    """``""`` while a group holds one pairing, else ``_<sub1>x<sub2>``.

    Every inter-brain figure is of two members, so a group of three writes three of
    everything and each needs a name of its own. A dyad has exactly one pairing, where the
    slug would distinguish nothing and rename every file for no reason, so it is empty
    there.

    ::

      pair_slug(("sub-a", "sub-b"), 1)  ->  ""
      pair_slug(("sub-a", "sub-b"), 3)  ->  "_subaxsubb"
    """
    if pair is None or n_pairings < 2:
        return ""
    return "_" + "x".join(_pair_fname(sid) for sid in pair)


def _pair_fname(pair: str) -> str:
    """Strip non-alphanumeric characters so the string is BIDS desc-value safe."""
    return re.sub(r"[^a-zA-Z0-9]", "", pair)


def get_channel_pairs(raw: mne.io.Raw) -> list[str]:
    """Return unique channel pair names (without chroma suffix) in MNE channel order."""
    picks = mne.pick_types(raw.info, fnirs="hbo")
    seen: set[str] = set()
    pairs: list[str] = []
    for i in picks:
        pair = pair_of(raw.ch_names[i])
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
