"""Per-channel before/after strip: one metric, two stages, one row per channel.

Three sections of the subject report ask the same question of different numbers. Did the
motion correction cost any scalp coupling (SCI, PSP across ``desc-sci`` to
``desc-motcorrected``); did the denoising push HbO and HbR apart (HbO-HbR r across
``desc-preproc`` to ``desc-errts``); did it leave the evoked response intact (CNR across
the same pair). One figure builder serves all three, so a reader who has learned to read
one has learned to read the others.

The shape is a dumbbell: the before value as a hollow marker, the after value as a filled
one, a line between them, sorted so the channels that moved the wrong way sit at the top.
A slope chart would fit as many channels in less height, but crossing lines make it
impossible to follow one channel, and following one channel is the whole point here.
"""

import numpy as np
import plotly.graph_objects as go

_BEFORE_COLOR = "#95a5a6"
_WORSE_COLOR  = "#e74c3c"
_BETTER_COLOR = "#27ae60"
_FLAT_COLOR   = "#bdc3c7"

# below this much movement a channel is called unchanged, as a fraction of the metric's
# spread across channels. Without it every channel gets a direction and the figure reads as
# if the stage did something everywhere, when most of the colour is rounding.
_FLAT_FRAC = 0.02


def stage_dumbbell_figure(
    before: dict[str, float],
    after: dict[str, float],
    *,
    title: str,
    x_title: str,
    before_label: str,
    after_label: str,
    higher_is_better: bool = True,
    reference: "float | None" = None,
    reference_label: str = "",
) -> "go.Figure | None":
    """Dumbbell of one metric at two stages, one row per channel.

    ``before`` and ``after`` are ``{channel: value}``; only channels present in both are
    drawn, since a channel missing from one stage has no pair to compare. ``reference``
    draws a vertical guide, for a threshold or for the zero line of a correlation.

    Returns None when the two dicts share no channel, which is what a run missing the
    second stage looks like.

    Example::

        before = {"S1_D1": 0.90, "S1_D2": 0.80}
        after  = {"S1_D1": 0.88, "S1_D2": 0.55}
        -> two rows, S1_D2 at the top in red (it fell 0.25), S1_D1 below it in grey
    """
    shared = [ch for ch in before if ch in after]
    if not shared:
        return None

    deltas = {ch: after[ch] - before[ch] for ch in shared}
    spread = float(np.ptp([*before.values(), *after.values()])) or 1.0
    flat_tol = spread * _FLAT_FRAC

    def direction(ch: str) -> str:
        d = deltas[ch]
        if abs(d) < flat_tol:
            return "flat"
        improved = d > 0 if higher_is_better else d < 0
        return "better" if improved else "worse"

    # worst movement first: sorting on the signed delta puts the channels the stage hurt at
    # the top whichever way the metric runs, which is what a reader is scanning for
    sign = 1 if higher_is_better else -1
    shared.sort(key=lambda ch: sign * deltas[ch])

    colors = {"better": _BETTER_COLOR, "worse": _WORSE_COLOR, "flat": _FLAT_COLOR}
    y = list(range(len(shared)))

    fig = go.Figure()

    # connector first so the markers sit on top of it
    for i, ch in enumerate(shared):
        fig.add_trace(go.Scatter(
            x=[before[ch], after[ch]], y=[i, i],
            mode="lines",
            line=dict(color=colors[direction(ch)], width=2),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=[before[ch] for ch in shared], y=y,
        mode="markers", name=before_label,
        marker=dict(size=8, color="white", line=dict(color=_BEFORE_COLOR, width=2)),
        customdata=shared,
        hovertemplate="%{customdata}<br>" + before_label + ": %{x:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[after[ch] for ch in shared], y=y,
        mode="markers", name=after_label,
        marker=dict(size=9, color=[colors[direction(ch)] for ch in shared]),
        customdata=[[ch, deltas[ch]] for ch in shared],
        hovertemplate=("%{customdata[0]}<br>" + after_label
                       + ": %{x:.3f}<br>change: %{customdata[1]:+.3f}<extra></extra>"),
    ))

    if reference is not None:
        fig.add_vline(x=reference, line=dict(color="#888", width=1, dash="dash"),
                      annotation_text=reference_label, annotation_font_size=9)

    n_worse = sum(direction(ch) == "worse" for ch in shared)
    mean_before = float(np.mean([before[ch] for ch in shared]))
    mean_after = float(np.mean([after[ch] for ch in shared]))
    subtitle = (f"mean {mean_before:.3f} → {mean_after:.3f}"
                f"  |  {n_worse} of {len(shared)} channels moved the wrong way")

    fig.update_layout(
        title=dict(text=f"{title}<br><span style='font-size:11px;color:#666'>{subtitle}</span>"),
        xaxis=dict(title=x_title, gridcolor="#eeeeee", zerolinecolor="#cccccc"),
        yaxis=dict(
            tickvals=y, ticktext=shared,
            tickfont=dict(size=7 if len(shared) > 40 else 9),
            autorange="reversed", showgrid=False, zeroline=False,
        ),
        height=max(280, min(len(shared) * 16 + 140, 1400)),
        margin=dict(l=110, r=30, t=70, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=10)),
    )
    return fig
