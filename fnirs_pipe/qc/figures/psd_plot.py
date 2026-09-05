"""PSD by pipeline stage: one row per chromophore, one line per stage, band annotations."""

import logging

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

from fnirs_pipe.pipeline.denoise import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_FILTER_ORDER,
    filter_kwargs,
)

from ._utils import HBO_COLOR as _HBO_COLOR, HBR_COLOR as _HBR_COLOR
from ._utils import physio_bands as _physio_bands

# colors for the physiological band annotations (band frequencies come from _physio_bands)
_BAND_COLORS = {
    "Mayer":   "rgba(46,204,113,0.12)",
    "Resp":    "rgba(241,196,15,0.10)",
    "Cardiac": "rgba(231,76,60,0.10)",
}

# Stage line colors, light to dark in reading order. Chromophore is the row, so colour is
# free to carry the stage; a reader follows one line down the ramp to see what each step did.
_STAGE_COLORS = ["#b0b7bd", "#7f8c8d", "#e67e22", "#8e44ad", "#2c3e50"]


def _simulate_bandpass(
    raw_haemo: mne.io.Raw,
    l_freq: float | None,
    h_freq: float | None,
    method: str,
    order: int,
) -> "tuple[str, mne.io.Raw] | None":
    """The bandpass this figure shows when no real stage file was passed in.

    Kept for the prep-only report, where no post-processing has run and there is no second
    file to read. The label says it is a simulation, so the two reports cannot be confused.

    The design comes from filter_kwargs, so the curve is one the pipeline would build.
    """
    if l_freq is None and h_freq is None:
        return None
    kwargs = filter_kwargs(raw_haemo.info["sfreq"], raw_haemo.n_times,
                           l_freq, h_freq, method, order)
    filtered = raw_haemo.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False, **kwargs)
    edges = "  ".join(part for part in (f"HP {l_freq} Hz" if l_freq is not None else "",
                                        f"LP {h_freq} Hz" if h_freq is not None else "") if part)
    return f"simulated bandpass ({edges})", filtered


def _stage_psd(raw: mne.io.Raw, fmax: float, ch_type: str) -> "tuple[np.ndarray, np.ndarray] | None":
    """Mean PSD across the channels of one chromophore, in dB. Returns (freqs, dB)."""
    picks = [c for c in raw.ch_names if c.endswith(f" {ch_type}")]
    if not picks:
        return None
    # a resampled stage has a lower Nyquist than the one before it, so each stage is
    # clamped to its own rather than to the figure's
    stage_fmax = min(fmax, raw.info["sfreq"] / 2)
    psd = raw.compute_psd(fmax=stage_fmax, picks=picks, verbose=False)
    return psd.freqs, 10 * np.log10(psd.get_data().mean(axis=0) + 1e-30)


def _add_band_annotations(fig: go.Figure, fmax: float, bands: list, rows: int = 2) -> None:
    for name, x0, x1 in bands:
        if x0 > fmax:
            continue
        for row in range(1, rows + 1):
            fig.add_vrect(
                x0=x0, x1=min(x1, fmax),
                fillcolor=_BAND_COLORS.get(name, "rgba(120,120,120,0.10)"),
                line_width=0, layer="below", row=row, col=1,
            )

    for name, x0, x1 in bands:
        label_x = (x0 + min(x1, fmax)) / 2
        if label_x > fmax:
            continue
        for row in range(1, rows + 1):
            ax = "" if row == 1 else str(row)
            fig.add_annotation(
                x=label_x, y=1.0,
                xref=f"x{ax}", yref=f"y{ax} domain",
                text=name, showarrow=False,
                font=dict(size=8, color="#555"),
                textangle=-90, xanchor="center", yanchor="top",
            )


def psd_figure(
    raw_haemo: mne.io.Raw,
    l_freq: float | None = None,
    h_freq: float | None = 0.4,
    filter_method: str = DEFAULT_FILTER_METHOD,
    filter_order: int = DEFAULT_FILTER_ORDER,
    fmax: float = 2.0,
    title: str = "Power spectral density by stage",
    cardiac: "tuple[float, float] | None" = None,
    resp: "tuple[float, float] | None" = None,
    stages: "list[tuple[str, mne.io.Raw]] | None" = None,
    first_label: str = "desc-preproc (no filtering)",
) -> go.Figure:
    """PSD of every pipeline stage on one pair of axes, HbO above HbR.

    ``raw_haemo`` is the first stage, the Beer-Lambert output. ``stages`` are the ones
    after it as ``[(label, raw), ...]``, read off disk by the caller. Passing None falls
    back to simulating the bandpass in memory, which is what a prep-only run gets.

    Individual channels of the first stage are drawn faintly so the starting spread is
    visible; every stage after that is its channel mean alone, or the figure becomes a
    wash. Physiological bands are shaded and the filter cutoffs marked with dashed lines.

    Args:
        raw_haemo:          Beer-Lambert output (HbO/HbR channels), the first stage.
        l_freq:             High-pass cutoff, for the cutoff marker and the simulation.
        h_freq:             Low-pass cutoff, same.
        filter_method:      Filter design for the simulation only; must match the pipeline's.
        filter_order:       Butterworth order, simulation only.
        fmax:               Maximum frequency to display (Hz), clamped per stage to Nyquist.
        title:              Figure title.
        stages:             Later stages as (label, raw); None simulates the bandpass.
        first_label:        Legend label for ``raw_haemo``.
    """
    nyquist = raw_haemo.info["sfreq"] / 2
    if fmax > nyquist:
        logger.warning("fmax %.2f Hz exceeds Nyquist %.2f Hz; clamping to Nyquist", fmax, nyquist)
        fmax = nyquist

    if stages is None:
        simulated = _simulate_bandpass(raw_haemo, l_freq, h_freq, filter_method, filter_order)
        stages = [simulated] if simulated is not None else []
    all_stages = [(first_label, raw_haemo), *stages]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=["HbO", "HbR"],
    )

    # the first stage's channels, faint, so the mean lines are read against a real spread
    for row, (ch_type, color) in enumerate([("hbo", _HBO_COLOR), ("hbr", _HBR_COLOR)], start=1):
        picks = [c for c in raw_haemo.ch_names if c.endswith(f" {ch_type}")]
        if not picks:
            continue
        psd = raw_haemo.compute_psd(fmax=fmax, picks=picks, verbose=False)
        freqs = psd.freqs.tolist()
        for ts in psd.get_data():
            fig.add_trace(go.Scatter(
                x=freqs, y=(10 * np.log10(ts + 1e-30)).tolist(),
                mode="lines", line=dict(width=0.6, color=color), opacity=0.15,
                showlegend=False, hoverinfo="skip",
            ), row=row, col=1)

    for s_idx, (label, raw) in enumerate(all_stages):
        color = _STAGE_COLORS[s_idx % len(_STAGE_COLORS)]
        for row, ch_type in enumerate(("hbo", "hbr"), start=1):
            result = _stage_psd(raw, fmax, ch_type)
            if result is None:
                continue
            freqs, db = result
            fig.add_trace(go.Scatter(
                x=freqs.tolist(), y=db.tolist(),
                mode="lines", line=dict(width=2.0, color=color),
                name=label, legendgroup=label, showlegend=(row == 1),
                hovertemplate=f"{label}<br>%{{x:.3f}} Hz<br>%{{y:.1f}} dB<extra></extra>",
            ), row=row, col=1)

    _add_band_annotations(fig, fmax=fmax, bands=_physio_bands(cardiac, resp))

    for cutoff in (l_freq, h_freq):
        if cutoff is not None and cutoff <= fmax:
            fig.add_vline(x=cutoff, line=dict(color="#555", width=1, dash="dash"))

    fig.update_xaxes(gridcolor="#eeeeee", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", gridcolor="#eeeeee", row=2, col=1)
    fig.update_yaxes(title_text="Power (dB)", gridcolor="#eeeeee")
    fig.update_layout(
        title=title,
        height=620,
        margin=dict(l=70, r=30, t=90, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=10), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig
