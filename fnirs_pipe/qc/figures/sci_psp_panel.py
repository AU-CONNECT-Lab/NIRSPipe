import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_GOOD_COLOR = "#C5E0B3"
_BAD_COLOR  = "#F8786E"
_MAX_TS_PTS = 3000
_SPACING    = 3

# translucent fill colors for annotation segments in timeseries_figure
_SEGMENT_PALETTE = [
    "rgba(231,76,60,0.20)",
    "rgba(230,126,34,0.20)",
    "rgba(155,89,182,0.20)",
    "rgba(26,188,156,0.20)",
    "rgba(41,128,185,0.20)",
    "rgba(243,156,18,0.20)",
]

# Pass as post_script to fig.write_html() to enable click-to-highlight.
# Click any channel element (pass/fail square, lollipop dot, or TS trace) to
# highlight that channel across all three columns. Double-click to reset.
CHANNEL_CLICK_JS = """
(function () {
  var gd = document.querySelector('.js-plotly-plot');
  var _active = null;

  function highlight(ch) {
    var n=gd.data.length, tsI=[], ops=[], widths=[];
    for (var i=0; i<n; i++) {
      var tr=gd.data[i];
      if (tr.type==='scattergl') {
        tsI.push(i);
        var sel=(tr.name===ch||tr.legendgroup===ch);
        ops.push(sel?1.0:0.05);
        widths.push(sel?2.2:0.9);
      } else if (tr.type==='scatter'&&tr.mode==='markers') {
        // pass/fail squares or lollipop dots — dim non-selected via selectedpoints
        var pool = Array.isArray(tr.customdata) ? tr.customdata
                 : Array.isArray(tr.text)       ? tr.text.map(function(t){return t.split(': ')[0];})
                 : [];
        var idx = pool.indexOf(ch);
        if (idx>=0) Plotly.restyle(gd,
          {selectedpoints:[[idx]],'selected.marker.opacity':[1.0],'unselected.marker.opacity':[0.12]},[i]);
      }
    }
    if (tsI.length) Plotly.restyle(gd,{opacity:ops,'line.width':widths},tsI);
  }

  function reset() {
    var n=gd.data.length, tsI=[], k=0;
    for (var i=0; i<n; i++) {
      var tr=gd.data[i];
      if (tr.type==='scattergl') { tsI.push(i); k++; }
      else if (tr.type==='scatter'&&tr.mode==='markers')
        Plotly.restyle(gd,{selectedpoints:[null]},[i]);
    }
    if (k) Plotly.restyle(gd,{opacity:Array(k).fill(1),'line.width':Array(k).fill(0.9)},tsI);
  }

  gd.on('plotly_click', function(evt) {
    if (!evt.points.length) return;
    var pt=evt.points[0], tr=gd.data[pt.curveNumber];
    var ch = (tr.type==='scatter'&&tr.mode==='markers'&&tr.marker&&tr.marker.symbol==='square')
             ? pt.text.split(': ')[0]
             : pt.customdata!=null
               ? String(Array.isArray(pt.customdata)?pt.customdata[0]:pt.customdata)
               : tr.name||null;
    if (!ch) return;
    if (_active===ch) { _active=null; reset(); return; }
    _active=ch; highlight(ch);
  });

  gd.on('plotly_doubleclick', function() { _active=null; reset(); });
})();
"""


# Do not delete this function; it may be useful for future pass/fail column in quality_panel.
def binary_heatmap_figure(
    ch_names: list[str],
    good_mask: np.ndarray,
    metric_label: str = "Pass/Fail",
) -> go.Figure:
    n_ch = len(ch_names)
    colors = [_GOOD_COLOR if g else _BAD_COLOR for g in good_mask]

    fig = go.Figure(go.Scatter(
        x=[0] * n_ch,
        y=[i * _SPACING for i in range(n_ch)],
        mode="markers",
        marker=dict(
            symbol="square",
            size=12,
            color=colors,
            line=dict(width=0.5, color="#aaa"),
        ),
        text=[f"{n}: {'PASS' if g else 'FAIL'}" for n, g in zip(ch_names, good_mask)],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5, 0.5]),
        yaxis=dict(
            tickvals=[i * _SPACING for i in range(n_ch)],
            ticktext=ch_names,
            autorange="reversed",
            tickfont=dict(size=9),
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=30, b=40),
    )
    return fig


def timeseries_figure(
    raw: mne.io.Raw,
    ch_names: list[str],
    colors: list[str],
    motion_spans: list[tuple[float, float]] | None = None,
    segments: dict[str, list[tuple[float, float]]] | None = None,
) -> go.Figure:
    ch_idx = [raw.ch_names.index(c) for c in ch_names if c in raw.ch_names]
    data, times = raw.get_data(picks=ch_idx, return_times=True)

    if len(times) > _MAX_TS_PTS:
        step = max(1, len(times) // _MAX_TS_PTS)
        times = times[::step]
        data  = data[:, ::step]

    n_ch = len(ch_idx)
    band_colors = ["rgba(0,0,0,0.04)", "rgba(0,0,0,0.0)"]
    shapes = [dict(
        type="rect", xref="paper", yref="y", layer="below",
        x0=0, x1=1,
        y0=i * _SPACING - _SPACING / 2,
        y1=i * _SPACING + _SPACING / 2,
        fillcolor=band_colors[i % 2], line=dict(width=0),
    ) for i in range(n_ch)]

    fig = go.Figure()
    for i, (ts, name, color) in enumerate(zip(data, ch_names, colors)):
        std = ts.std()
        normed = (ts - ts.mean()) / std if std > 0 else ts - ts.mean()
        fig.add_trace(go.Scattergl(
            x=times, y=normed + i * _SPACING,
            name=name, legendgroup=name, mode="lines",
            line=dict(width=0.9, color=color),
            showlegend=True,
            hovertemplate=f"{name}<extra></extra>",
        ))

    if segments:
        for k, (label, spans) in enumerate(segments.items()):
            fill = _SEGMENT_PALETTE[k % len(_SEGMENT_PALETTE)]
            for j, (onset, duration) in enumerate(spans):
                kw = dict(annotation_text=label, annotation_position="top left",
                          annotation_font_size=8) if j == 0 else {}
                fig.add_vrect(x0=onset, x1=onset + duration,
                              fillcolor=fill, line_width=0, layer="below", **kw)
    elif motion_spans:
        for onset, duration in motion_spans:
            fig.add_vrect(x0=onset, x1=onset + duration,
                          fillcolor="gray", opacity=0.20,
                          line_width=0, layer="below")

    fig.update_layout(
        xaxis=dict(title="Time (s)", gridcolor="#eeeeee", zerolinecolor="#cccccc"),
        yaxis=dict(
            tickvals=[i * _SPACING for i in range(n_ch)],
            ticktext=ch_names,
            autorange="reversed",
            gridcolor="#eeeeee",
        ),
        shapes=shapes,
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(350, n_ch * 22 + 80),
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


def lollipop_scores_figure(
    ch_names: list[str],
    mean_scores: np.ndarray,
    colors: list[str],
    threshold: float,
    metric_label: str = "Score",
) -> go.Figure:
    n_ch = len(ch_names)
    stem_x, stem_y = [], []
    for i, val in enumerate(mean_scores):
        stem_x += [0.0, float(val), None]
        stem_y += [i * _SPACING, i * _SPACING, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stem_x, y=stem_y, mode="lines",
        line=dict(color="#aaaaaa", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=mean_scores.tolist(),
        y=[i * _SPACING for i in range(n_ch)],
        mode="markers",
        marker=dict(size=8, color=colors, line=dict(width=0.5, color="#333")),
        customdata=ch_names,
        showlegend=False,
        hovertemplate="%{customdata}: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line=dict(dash="dash", color="#888", width=1))
    fig.update_layout(
        xaxis_title=metric_label,
        yaxis=dict(autorange="reversed", showticklabels=False),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def sci_segment_heatmap_figure(
    ch_names: list[str],
    scores: np.ndarray,
    win_times: np.ndarray,
    threshold: float,
    metric_label: str = "Score",
) -> go.Figure:
    """Continuous 2-D heatmap: channels (Y) × time windows (X), value = metric score.

    Colorscale centres at `threshold` (yellow), below = red, above = green.
    Complements the binary pass/fail panel with gradient detail.
    """
    wt = np.asarray(win_times).ravel()
    n_ticks = 12
    step = max(1, len(wt) // n_ticks)
    tick_vals = wt[::step].tolist()

    fig = go.Figure(go.Heatmap(
        z=scores,
        x=wt.tolist(),
        y=ch_names,
        colorscale="RdYlGn",
        zmid=threshold,
        colorbar=dict(title=metric_label, thickness=14, len=0.85),
        hovertemplate=(
            "Ch: %{y}<br>"
            "t = %{x:.1f} s<br>"
            + metric_label + " = %{z:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"{metric_label} per segment (threshold = {threshold})",
        xaxis=dict(
            title="Window time (s)",
            gridcolor="#eeeeee",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=[f"{v:.0f}" for v in tick_vals],
            tickangle=0,
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=min(max(220, len(ch_names) * 7 + 80), 480),
        margin=dict(l=110, r=80, t=60, b=40),
    )
    return fig


def quality_panel(
    raw: mne.io.Raw,
    ch_names: list[str],
    scores: np.ndarray,
    win_times: np.ndarray,
    threshold: float,
    title: str = "Signal Quality",
    metric_label: str = "Score",
) -> go.Figure:
    valid_rows = [i for i, c in enumerate(ch_names) if c in raw.ch_names]
    ch_names   = [ch_names[i] for i in valid_rows]
    ch_idx     = [raw.ch_names.index(c) for c in ch_names]
    scores     = scores[valid_rows]

    mean_scores = scores.mean(axis=1)
    good = mean_scores >= threshold
    colors = [_GOOD_COLOR if g else _BAD_COLOR for g in good]

    # binary_heatmap_figure is kept but not shown; use lollipop for pass/fail summary
    fig_ts = timeseries_figure(raw, ch_names, colors)
    fig_lp = lollipop_scores_figure(ch_names, mean_scores, colors, threshold, metric_label)

    n_ch = len(ch_names)
    combined = make_subplots(
        rows=1, cols=2,
        column_widths=[0.88, 0.12],
        shared_yaxes=True,
        horizontal_spacing=0.01,
        subplot_titles=["Time series", metric_label],
    )

    for trace in fig_ts.data:
        combined.add_trace(trace, row=1, col=1)
    for trace in fig_lp.data:
        combined.add_trace(trace, row=1, col=2)

    combined.add_vline(x=threshold, line=dict(dash="dash", color="#888", width=1),
                       row=1, col=2)

    combined.update_yaxes(
        tickvals=[i * _SPACING for i in range(n_ch)],
        ticktext=ch_names,
        autorange="reversed",
        tickfont=dict(size=9),
        row=1, col=1,
    )
    combined.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=2)
    combined.update_xaxes(title_text="Time (s)", row=1, col=1)
    combined.update_xaxes(title_text=metric_label, row=1, col=2)

    combined.update_layout(
        title=title,
        height=min(max(400, n_ch * 14 + 120), 700),
        margin=dict(l=110, r=30, t=80, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return combined