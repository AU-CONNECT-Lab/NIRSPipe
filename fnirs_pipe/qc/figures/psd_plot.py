"""PSD before/after bandpass: side-by-side Plotly figure with band annotations."""

import logging

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

from ._utils import HBO_COLOR as _HBO_COLOR, HBR_COLOR as _HBR_COLOR
from ._utils import HBO_MEAN_COLOR as _HBO_MEAN_COLOR, HBR_MEAN_COLOR as _HBR_MEAN_COLOR
from ._utils import physio_bands as _physio_bands

# colors for the physiological band annotations (band frequencies come from _physio_bands)
_BAND_COLORS = {
    "Mayer":   "rgba(46,204,113,0.12)",
    "Resp":    "rgba(241,196,15,0.10)",
    "Cardiac": "rgba(231,76,60,0.10)",
}


def _filter_response_trace(
    l_freq: float | None,
    h_freq: float | None,
    h_trans_bandwidth: float,
    sfreq: float,
    fmax: float,
) -> go.Scatter | None:
    try:
        from scipy.signal import firwin, freqz
    except ImportError:
        return None
    nyq = sfreq / 2
    numtaps = max(int(sfreq * 6.6 / max(h_trans_bandwidth, 0.01)), 15) | 1
    if h_freq is not None and l_freq is None:
        b = firwin(numtaps, h_freq / nyq)
    elif l_freq is not None and h_freq is None:
        b = firwin(numtaps, l_freq / nyq, pass_zero=False)
    elif l_freq is not None and h_freq is not None:
        b = firwin(numtaps, [l_freq / nyq, h_freq / nyq], pass_zero=False)
    else:
        return None
    w, h = freqz(b, worN=4096, fs=sfreq)
    mask = w <= fmax
    db = 20 * np.log10(np.abs(h[mask]) + 1e-12)
    return go.Scatter(
        x=w[mask].tolist(), y=db.tolist(),
        mode="lines",
        line=dict(width=1.5, color="#555555", dash="dash"),
        name="Filter response",
        showlegend=True,
        hovertemplate="Filter<br>%{x:.3f} Hz<br>%{y:.1f} dB<extra></extra>",
    )


def _add_band_annotations(fig: go.Figure, fmax: float, bands: list) -> None:
    for name, x0, x1 in bands:
        if x0 > fmax:
            continue
        for row in (1, 2):
            fig.add_vrect(
                x0=x0, x1=min(x1, fmax),
                fillcolor=_BAND_COLORS.get(name, "rgba(120,120,120,0.10)"),
                line_width=0, layer="below", row=row, col=1,
            )

    for name, x0, x1 in bands:
        label_x = (x0 + min(x1, fmax)) / 2
        if label_x > fmax:
            continue
        for row in (1, 2):
            ax = "" if row == 1 else "2"
            fig.add_annotation(
                x=label_x, y=1.0,
                xref=f"x{ax}", yref=f"y{ax} domain",
                text=name, showarrow=False,
                font=dict(size=8, color="#555"),
                textangle=-90, xanchor="center", yanchor="top",
            )


def _psd_traces(
    freqs: np.ndarray,
    ch_data: np.ndarray,
    ch_type: str,
    color: str,
    mean_color: str,
    show_legend: bool,
) -> list[go.Scatter]:
    """Individual channel traces (low opacity) + mean trace."""
    traces = []
    label = "HbO" if ch_type == "hbo" else "HbR"

    for i, ts in enumerate(ch_data):
        traces.append(go.Scatter(
            x=freqs.tolist(), y=(10 * np.log10(ts + 1e-30)).tolist(),
            mode="lines",
            line=dict(width=0.6, color=color),
            opacity=0.2,
            showlegend=False,
            hoverinfo="skip",
        ))

    mean_ts = ch_data.mean(axis=0)
    traces.append(go.Scatter(
        x=freqs.tolist(), y=(10 * np.log10(mean_ts + 1e-30)).tolist(),
        mode="lines",
        line=dict(width=2.0, color=mean_color),
        name=label,
        showlegend=show_legend,
        hovertemplate=f"{label}<br>%{{x:.3f}} Hz<br>%{{y:.1f}} dB<extra></extra>",
    ))
    return traces


def psd_figure(
    raw_haemo: mne.io.Raw,
    l_freq: float | None = None,
    h_freq: float | None = 0.4,
    h_trans_bandwidth: float = 0.1,
    fmax: float = 2.0,
    title: str = "Power Spectral Density — before / after bandpass",
    cardiac: "tuple[float, float] | None" = None,
    resp: "tuple[float, float] | None" = None,
) -> go.Figure:
    """Side-by-side PSD figure: before (left) and after (right) bandpass filter.

    Individual channel traces are shown with low opacity; the mean per type
    (HbO red, HbR blue) is drawn bold on top. Physiological frequency bands
    are annotated as coloured background regions.

    Args:
        raw_haemo:          Raw object after Beer-Lambert (HbO/HbR channels).
        l_freq:             High-pass cutoff (None = no high-pass).
        h_freq:             Low-pass cutoff in Hz.
        h_trans_bandwidth:  Transition bandwidth for the low-pass filter.
        fmax:               Maximum frequency to display (Hz).
        title:              Figure title.
    """
    nyquist = raw_haemo.info["sfreq"] / 2
    if fmax > nyquist:
        logger.warning("fmax %.2f Hz exceeds Nyquist %.2f Hz; clamping to Nyquist", fmax, nyquist)
        fmax = nyquist

    psd_before = raw_haemo.compute_psd(fmax=fmax, verbose=False)

    raw_filtered = raw_haemo.copy().filter(
        l_freq=l_freq, h_freq=h_freq,
        h_trans_bandwidth=h_trans_bandwidth,
        verbose=False,
    )
    psd_after = raw_filtered.compute_psd(fmax=fmax, verbose=False)

    freqs = psd_before.freqs
    ch_names = psd_before.info["ch_names"]
    hbo_chs = [c for c in ch_names if c.endswith(" hbo")]
    hbr_chs = [c for c in ch_names if c.endswith(" hbr")]
    data_b_hbo = psd_before.get_data(picks=hbo_chs) if hbo_chs else np.empty((0, len(freqs)))
    data_b_hbr = psd_before.get_data(picks=hbr_chs) if hbr_chs else np.empty((0, len(freqs)))
    data_a_hbo = psd_after.get_data(picks=hbo_chs) if hbo_chs else np.empty((0, len(freqs)))
    data_a_hbr = psd_after.get_data(picks=hbr_chs) if hbr_chs else np.empty((0, len(freqs)))

    filter_str = ""
    if l_freq is not None:
        filter_str += f"HP {l_freq} Hz"
    if h_freq is not None:
        filter_str += f"{'  ' if filter_str else ''}LP {h_freq} Hz"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=["Before bandpass", f"After bandpass ({filter_str})"],
    )

    for trace in _psd_traces(freqs, data_b_hbo, "hbo", _HBO_COLOR, _HBO_MEAN_COLOR, True):
        fig.add_trace(trace, row=1, col=1)
    for trace in _psd_traces(freqs, data_b_hbr, "hbr", _HBR_COLOR, _HBR_MEAN_COLOR, True):
        fig.add_trace(trace, row=1, col=1)
    for trace in _psd_traces(freqs, data_a_hbo, "hbo", _HBO_COLOR, _HBO_MEAN_COLOR, False):
        fig.add_trace(trace, row=2, col=1)
    for trace in _psd_traces(freqs, data_a_hbr, "hbr", _HBR_COLOR, _HBR_MEAN_COLOR, False):
        fig.add_trace(trace, row=2, col=1)

    _add_band_annotations(fig, fmax=fmax, bands=_physio_bands(cardiac, resp))

    fr_trace = _filter_response_trace(l_freq, h_freq, h_trans_bandwidth,
                                      raw_haemo.info["sfreq"], fmax)
    if fr_trace is not None:
        fig.add_trace(fr_trace, row=2, col=1)

    fig.update_xaxes(gridcolor="#eeeeee", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", gridcolor="#eeeeee", row=2, col=1)
    fig.update_yaxes(title_text="Power (dB)", gridcolor="#eeeeee")
    fig.update_layout(
        title=title,
        height=580,
        margin=dict(l=70, r=30, t=80, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=11)),
    )
    return fig
