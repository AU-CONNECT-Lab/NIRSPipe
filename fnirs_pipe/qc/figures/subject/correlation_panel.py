"""HbO–HbR correlation panel.

Top row:    one channel × channel Pearson r heatmap per stage, before and after denoising,
            on one colour scale so the pair subtracts by eye.
Bottom row: per-pair HbO–HbR r as a dumbbell, before → after, grouped by source-detector
            separation.

The two stages used to be two separate figures, each with its own bar chart sorted by its
own r, so the same channel sat at a different height in each and comparing one channel
across the step meant finding it twice in two differently ordered lists. One figure, one
channel order, and the change is the length of a connector.

Separation is what the grouping is for. The HbO–HbR anticorrelation that the −0.3
threshold tests for is a property of cortical haemodynamics, so it says nothing about a
short channel, which only ever sees scalp. Sorted together the short channels land at one
end of the list and read as the worst channels on the montage when they are simply not
being asked the same question. The heatmap rows follow the same grouping, so the block
dividers inside each chromophore mark the long/short boundary rather than wherever the
acquisition order happened to switch.

Both panels read their colour off one reversed RdBu scale, so a shade means the same r
whether it is a matrix cell or a dot. The verdict is the dashed −0.3 rule rather than a
third hue: the green/amber/red fills this replaced separated amber from green by ΔE 5.8
under protanopia, so the one distinction the panel exists to make was unreadable to a
red-green colourblind reader. A single threshold rule keeps its green, having nothing to
be confused against.
"""

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# The separation groups, in the order they are drawn. "mid" is the 10-15 mm gap that
# long_short_channels leaves unclaimed; it is usually empty.
_GROUP_ORDER = ["long", "mid", "short"]
_GROUP_LABEL = {
    "long":  "long channels",
    "mid":   "mid-range channels",
    "short": "short channels",
}

# Plotly's RdBu runs red→blue, so it is reversed everywhere to put red at r = +1. A white
# midpoint rather than a tinted one is what keeps a correlation matrix reading as "nothing
# here" in the middle instead of beige. The dots below take their fill from the same map,
# so one colour means one r across the figure.
_SCALE, _REVERSE = "RdBu", True

_MUTED, _GRID, _BASELINE = "#888888", "#eeeeee", "#444444"
# a white fill at r near zero would vanish on a white surface, so the marks carry a ring
_MARK_EDGE = "#9aa0a6"

_R_THRESHOLD = -0.3

# What the two stages are called wherever they are named. The GLM variants say the task
# is gone from the residual, which is what stops -0.3 from being read against it.
_BEFORE_LABEL = "before denoising"
_AFTER_LABEL = "after denoising"
_AFTER_LABEL_GLM = "after GLM (task removed)"

# ---- Layout ----
# The heatmap is square-constrained, so its side is min(column width, row height). A row
# height fixed in advance is what made the row the binding one on a wide page: 15 px cells
# with the spare width spent on blank range either side of the matrix, which is also what
# pushed the y tick labels away from it. The row is sized off the channel count instead,
# and `constrain="domain"` below shrinks the axis rather than padding its range, so the
# labels stay against the matrix whichever dimension binds. These are the height the file is
# written with; `fit_js` replaces it with the one that fits the page the figure is opened on.
_CELL_PX = 22
_HEAT_MIN_PX, _HEAT_MAX_PX = 520, 1100
_DUMBBELL_PX = 300
_V_SPACING_PX = 130


def _pair_key(ch_name: str) -> str:
    return ch_name.rsplit(" ", 1)[0]


def _pair_group(raw_haemo: mne.io.Raw, sep_bands=None) -> "dict[str, str]":
    """Map each S-D pair to ``long`` / ``mid`` / ``short``.

    ``long_short_channels`` returns names with the chromophore suffix and leaves the
    10-15 mm band in neither list, so::

        ["S1_D1 hbo", "S1_D8 hbo"] -> {"S1_D1": "long", "S1_D8": "short"}
    """
    from fnirs_pipe.qc.metrics import long_short_channels

    long_names, short_names = long_short_channels(raw_haemo, sep_bands)
    groups = {_pair_key(n): "mid" for n in raw_haemo.ch_names}
    groups.update({_pair_key(n): "long" for n in long_names})
    groups.update({_pair_key(n): "short" for n in short_names})
    return groups


def _channel_order(raw_haemo: mne.io.Raw, groups: "dict[str, str]") -> "list[str]":
    """HbO block then HbR block, each sorted by separation group then name.

    The two blocks end up carrying the same pair order, so a cell in the HbO × HbR corner
    sits at the crossing of one pair's two rows.
    """
    rank = {g: i for i, g in enumerate(_GROUP_ORDER)}
    names = [n for n in raw_haemo.ch_names if n.endswith(("hbo", "hbr"))]
    return sorted(names, key=lambda n: (n.endswith("hbr"),
                                        rank.get(groups.get(_pair_key(n), "mid"), 9),
                                        _pair_key(n)))


def _stage(raw: mne.io.Raw, order: "list[str]"):
    """One stage's correlation matrix and per-pair r, on a channel order fixed elsewhere.

    ``order`` is the before stage's channel order, so both stages index the same rows and
    a cell at (i, j) means the same pair of channels in both heatmaps.
    """
    idx = {n: i for i, n in enumerate(raw.ch_names)}
    picks = [idx[n] for n in order if n in idx]
    if len(picks) != len(order):
        return None, None
    data = raw.get_data(picks=picks)
    n = len(order)
    corr = np.corrcoef(data).astype(float)
    np.fill_diagonal(corr, np.nan)
    # Lower triangle only: the matrix is symmetric, so the upper half is the same values
    # read the other way round. The HbO x HbR block, which is what the panel is for,
    # survives once in the lower left.
    corr[np.triu_indices(n, k=1)] = np.nan

    by_name = dict(zip(order, data))
    pair_r = {}
    for name in order:
        k = _pair_key(name)
        hbo, hbr = f"{k} hbo", f"{k} hbr"
        if k not in pair_r and hbo in by_name and hbr in by_name:
            pair_r[k] = float(np.corrcoef(by_name[hbo], by_name[hbr])[0, 1])
    return corr, pair_r


def _add_heatmap(fig, corr, order, col, show_scale, heat_px):
    # float32 halves the serialised payload and still resolves r to ~1e-7, far below what
    # the colour scale or the hover readout distinguishes
    fig.add_trace(go.Heatmap(
        z=np.asarray(corr, dtype=np.float32), x=order, y=order,
        zmin=-1.0, zmax=1.0, colorscale=_SCALE, reversescale=_REVERSE,
        showscale=show_scale,
        # in pixels and hung from the top of the row, so `fit_js` restores it by setting one
        # number: a fraction of the figure's height would have to be recomputed with the rest
        colorbar=dict(title="Pearson r", lenmode="pixels", len=heat_px * 0.92,
                      yanchor="top", y=1.0, thickness=12,
                      tickvals=[-1, -0.5, 0, 0.5, 1]),
        hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<extra></extra>",
    ), row=1, col=col)


def _add_dividers(fig, groups, order, n_hbo, col):
    """An L hugging the diagonal at each block boundary.

    A full-width rule would run out over the blank upper triangle and read as part of the
    plot, so each divider stops where the diagonal is.
    """
    n_ch = len(order)

    def _divider(pos, color, dash, width):
        for x0, y0, x1, y1 in ((-0.5, pos, pos, pos), (pos, pos, pos, n_ch - 0.5)):
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, layer="above",
                          line=dict(color=color, width=width, dash=dash), row=1, col=col)

    _divider(n_hbo - 0.5, "#888", "dash", 0.8)
    # separation boundaries inside each chromophore block, lighter than the HbO/HbR one
    for offset, names in ((0, order[:n_hbo]), (n_hbo, order[n_hbo:])):
        seen = [groups.get(_pair_key(n), "mid") for n in names]
        for i in range(1, len(seen)):
            if seen[i] != seen[i - 1]:
                _divider(offset + i - 0.5, "#ccc", "dot", 0.7)


def _add_dumbbell(fig, groups, r_before, r_after, row, col,
                  after_label=_AFTER_LABEL, task_modelled=False):
    """Per-pair r, one x position per pair, before as an open ring and after filled.

    Ordered best→worst inside each group on the *before* value, so one order serves both
    stages and the −0.3 crossing happens once along the row.
    """
    xs, labels, before, after, spans = [], [], [], [], []
    x = 0.0
    for g in _GROUP_ORDER:
        keys = sorted((k for k in r_before if groups.get(k) == g),
                      key=lambda k: r_before[k])
        if not keys:
            continue
        start = x
        for k in keys:
            xs.append(x)
            labels.append(k)
            before.append(r_before[k])
            after.append(r_after.get(k) if r_after else None)
            x += 1.0
        spans.append((start, x - 1.0, _GROUP_LABEL[g]))
        x += 1.4          # a gap between groups, so the grouping needs no rule
    if not xs:
        return None, None

    xs = np.asarray(xs, dtype=float)
    before = np.asarray(before, dtype=float)
    paired = r_after is not None and all(v is not None for v in after)
    tip = np.asarray(after, dtype=float) if paired else before

    fig.add_hline(y=0.0, line=dict(color=_BASELINE, width=1.0), row=row, col=col)
    fig.add_hline(y=_R_THRESHOLD, line=dict(color="#27ae60", width=1.0, dash="dash"),
                  opacity=0.7, row=row, col=col)

    if paired:
        seg_x, seg_y = [], []
        for xi, b, a in zip(xs, before, tip):
            seg_x += [xi, xi, None]
            seg_y += [b, a, None]
        fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", showlegend=False,
                                 line=dict(color=_MUTED, width=1.6), opacity=0.55,
                                 hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=xs, y=before, mode="markers", name=_BEFORE_LABEL, customdata=labels,
            marker=dict(size=9, color="white", line=dict(color=_MUTED, width=1.4)),
            hovertemplate="%{customdata}<br>before r = %{y:.3f}<extra></extra>",
        ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=xs, y=tip, mode="markers", customdata=labels,
        name=after_label if paired else "HbO–HbR r", showlegend=paired,
        marker=dict(size=13, color=tip, cmin=-1.0, cmax=1.0, colorscale=_SCALE,
                    reversescale=_REVERSE, line=dict(color=_MARK_EDGE, width=1.0)),
        hovertemplate=("%{customdata}<br>" + ("after " if paired else "")
                       + "r = %{y:.3f}<extra></extra>"),
    ), row=row, col=col)

    rule_text = "r = −0.3 (before)" if task_modelled else "r = −0.3"
    fig.add_annotation(x=xs[-1] + 0.9, y=_R_THRESHOLD, text=rule_text, showarrow=False,
                       font=dict(color="#27ae60", size=10), xanchor="right",
                       yanchor="bottom", row=row, col=col)
    if len(spans) > 1:
        for xa, xb, name in spans:
            fig.add_annotation(x=(xa + xb) / 2, y=1.16, text=f"<b>{name}</b>",
                               showarrow=False, font=dict(color="#555", size=11),
                               row=row, col=col)
    fig.add_annotation(
        x=0, y=1.30, xanchor="left", showarrow=False, font=dict(size=13),
        text="Per-pair HbO–HbR r, best→worst" + (" within group" if len(spans) > 1 else ""),
        row=row, col=col)
    return xs, labels


def hbo_hbr_correlation_figure(
    raw_haemo: mne.io.Raw,
    title: str = "HbO–HbR Signal Quality",
    sep_bands=None,
    raw_after: "mne.io.Raw | None" = None,
    task_modelled: bool = False,
) -> "go.Figure | None":
    """Return the correlation panel as a Plotly figure, or None on an empty montage.

    ``raw_after`` is the denoised recording. Given it, the panel carries both stages: two
    heatmaps on one colour scale and a dumbbell per pair. Left out, or not matching the
    before stage's channels, it degrades to the one-stage panel: one heatmap and one dot
    per pair, which is what a run with no denoising and what a condition page both get.

    ``task_modelled`` says the after stage had a task model taken out of it as well as the
    confounds, which is what a GLM run's residual is. The -0.3 rule tests for the HbO-HbR
    anticorrelation of cortical haemodynamics, and a model that explained part of that
    shared variance leaves a weaker anticorrelation behind without anything having gone
    wrong. The rule then reads the before stage only, and both the rule and the after stage
    are labelled to say so.
    """
    groups = _pair_group(raw_haemo, sep_bands)
    order = _channel_order(raw_haemo, groups)
    if not order:
        return None

    corr_b, r_b = _stage(raw_haemo, order)
    if corr_b is None:
        return None
    corr_a, r_a = (_stage(raw_after, order) if raw_after is not None else (None, None))
    two_stage = corr_a is not None

    after_label = _AFTER_LABEL_GLM if task_modelled else _AFTER_LABEL

    n_ch = len(order)
    n_hbo = sum(1 for n in order if n.endswith("hbo"))
    cols = 2 if two_stage else 1

    heat_px = float(np.clip(n_ch * _CELL_PX, _HEAT_MIN_PX, _HEAT_MAX_PX))
    height = heat_px + _V_SPACING_PX + _DUMBBELL_PX
    dumb_frac = _DUMBBELL_PX / height
    row_heights = [heat_px / (heat_px + _DUMBBELL_PX),
                   _DUMBBELL_PX / (heat_px + _DUMBBELL_PX)]
    v_spacing = _V_SPACING_PX / height

    fig = make_subplots(
        rows=2, cols=cols, row_heights=row_heights, vertical_spacing=v_spacing,
        horizontal_spacing=0.06,
        specs=[[{}, {}], [{"colspan": 2}, None]] if two_stage else [[{}], [{}]],
        subplot_titles=(_BEFORE_LABEL, after_label) if two_stage else (),
    )

    for col, corr in ((1, corr_b), (2, corr_a))[:cols]:
        _add_heatmap(fig, corr, order, col, show_scale=(col == cols),
                     heat_px=heat_px)
        _add_dividers(fig, groups, order, n_hbo, col)

    dumbbell_row = 2
    xs, labels = _add_dumbbell(fig, groups, r_b, r_a, dumbbell_row, 1,
                               after_label=after_label, task_modelled=task_modelled)

    # Every channel keeps its label: unlike the static panel this replaced, an unreadable
    # tick here is one scroll-zoom away from being readable, so subsampling them buys
    # nothing. The size only has to keep 44-ish channels legible unzoomed.
    tick_fs = int(np.clip(480 / max(n_ch, 1), 5, 10))
    for col in range(1, cols + 1):
        x_axis = "x" if col == 1 else "x2"
        fig.update_xaxes(showgrid=False, zeroline=False, tickangle=45,
                         constrain="domain", tickfont=dict(size=tick_fs), row=1, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, autorange="reversed",
                         scaleanchor=x_axis, constrain="domain",
                         constraintoward="top", showticklabels=(col == 1),
                         tickfont=dict(size=tick_fs), row=1, col=col)

    if xs is not None:
        fig.update_xaxes(tickmode="array", tickvals=xs, ticktext=labels, tickangle=90,
                         tickfont=dict(size=8), showgrid=False, zeroline=False,
                         range=[-1.0, xs[-1] + 1.0], row=dumbbell_row, col=1)
    fig.update_yaxes(title_text="HbO–HbR r", tickvals=[-1, -0.5, 0, 0.5, 1],
                     # headroom above r = 1 for the group headers, which would otherwise
                     # land on the marks
                     range=[-1.12, 1.34], gridcolor=_GRID, zeroline=False,
                     tickfont=dict(size=9), row=dumbbell_row, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        height=int(height), plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=90, r=70, t=70, b=90),
        # above the dumbbell: at lower right it sat on the short channels, which is exactly
        # where this panel puts its most positive values
        legend=dict(orientation="h", xanchor="right", yanchor="bottom",
                    x=1.0, y=dumb_frac - v_spacing / 2, font=dict(size=10)),
    )
    return fig


def _axis_name(ref) -> str:
    """``"y3"`` -> ``"yaxis3"``, an unset ref being the first axis."""
    ref = ref or "y"
    return "yaxis" + ref[1:]


def fit_js(fig: "go.Figure") -> str:
    """Script that re-fits the panel to the width the browser actually gives it.

    The height is written into the file and the width is the page's, so one of the two always
    has slack and a square-constrained matrix cannot use it. ``constrain="domain"`` moves that
    slack out from between the labels and the matrix, but it lands between the heatmap and the
    dumbbell instead, which is a hole in the middle of the figure. This is the only place the
    two can be reconciled: measure the column on load, set the height that makes the matrix
    square in it, and move the row domains so the dumbbell keeps its own height rather than
    scaling with the figure.

    Idempotent and re-run on resize. A page where the width cannot be read leaves the written
    height alone, which is the figure as it is drawn today.
    """
    heat = sorted({_axis_name(tr.yaxis) for tr in fig.data if tr.type == "heatmap"})
    dumb = sorted({_axis_name(tr.yaxis) for tr in fig.data} - set(heat))
    # a montage with no pairs to dumbbell leaves an empty second row whose domain this has
    # nothing to set, so the figure keeps the height it was written with
    if not heat or not dumb:
        return ""
    return (
        "<script>(function(){"
        f"var HEAT={heat!r}.map(String),DUMB={(dumb[0] if dumb else '')!r},"
        f"MIN={_HEAT_MIN_PX},MAX={_HEAT_MAX_PX},D={_DUMBBELL_PX},GAP={_V_SPACING_PX},n=0;"
        "function fit(){"
        "var gd=document.querySelector('.plotly-graph-div');"
        "if(!gd||typeof Plotly==='undefined'||!gd.layout)return;"
        # the first subplot's drag layer is its plot area, so this is the width the matrix
        # actually got. Deriving it from the container and the domains lands about 30 px out,
        # the axis constraint and the colour bar having moved things in between
        "var L=gd.layout,m=L.margin||{},drag=gd.querySelector('.nsewdrag');"
        "if(!drag)return;"
        "var w=drag.getBoundingClientRect().width;"
        "if(!(w>0))return;"
        "var heat=Math.min(Math.max(w,MIN),MAX);"
        "var h=Math.round(heat+(DUMB?GAP+D:0)+(m.t||0)+(m.b||0));"
        "if(Math.abs(h-(L.height||0))<4)return;"
        "var inner=h-(m.t||0)-(m.b||0),up={height:h};"
        "HEAT.forEach(function(a){up[a+'.domain']=[1-heat/inner,1];});"
        "if(DUMB){up[DUMB+'.domain']=[0,D/inner];up['legend.y']=D/inner;}"
        "Plotly.relayout(gd,up).then(function(){"
        "var i=gd.data.findIndex(function(t){return t.type==='heatmap'&&t.showscale;});"
        "if(i>=0)Plotly.restyle(gd,{'colorbar.len':heat*0.92},[i]);"
        # the height it just set can move the width: a page tall enough to scroll takes the
        # scrollbar's width off the column. Bounded, so a layout that oscillates settles
        "if(++n<3)setTimeout(fit,0);});}"
        "if(document.readyState==='complete')fit();"
        "else window.addEventListener('load',fit);"
        "window.addEventListener('resize',function(){n=0;fit();});"
        "})();</script>"
    )
