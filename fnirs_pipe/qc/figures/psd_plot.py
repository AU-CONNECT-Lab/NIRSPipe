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
    filter_description,
    filter_kwargs,
    filter_response,
)
from fnirs_pipe.utils.lineage import lineage_of

from ._utils import HBO_COLOR as _HBO_COLOR, HBR_COLOR as _HBR_COLOR
from ._utils import physio_bands as _physio_bands

# colors for the physiological band annotations (band frequencies come from _physio_bands)
_BAND_COLORS = {
    "Mayer":   "rgba(46,204,113,0.12)",
    "Resp":    "rgba(241,196,15,0.10)",
    "Cardiac": "rgba(231,76,60,0.10)",
}

# Stage is the row and chromophore is the colour, so HbO and HbR keep the identity they have
# in every other figure of the report and a step is read by looking down the column.
_RESPONSE_COLOR = "#7f8c8d"
_RESPONSE_FLOOR = -90  # dB; below this the curve is numerical, not something to read


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
    """Per-channel PSD of one chromophore, in dB. Returns (freqs, dB of shape (n_ch, n_freqs))."""
    picks = [c for c in raw.ch_names if c.endswith(f" {ch_type}")]
    if not picks:
        return None
    # a resampled stage has a lower Nyquist than the one before it, so each stage is
    # clamped to its own rather than to the figure's
    stage_fmax = min(fmax, raw.info["sfreq"] / 2)
    psd = raw.compute_psd(fmax=stage_fmax, picks=picks, verbose=False)
    return psd.freqs, 10 * np.log10(psd.get_data() + 1e-30)


def _recorded_filter(stages, l_freq, h_freq, method, order):
    """The filter a loaded stage went through, falling back to what the caller passed.

    Stage files carry it on their lineage stamp, restored from the sidecar on read, so the
    figure labels the filter the run built rather than the one its arguments describe.
    """
    for _, raw in stages:
        params = (lin.params if (lin := lineage_of(raw)) else None) or {}
        if params.get("low_pass") is not None or params.get("high_pass") is not None:
            return (params.get("high_pass"), params.get("low_pass"),
                    params.get("filter_method") or method,
                    params.get("filter_order") or order)
    return l_freq, h_freq, method, order


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
                x=label_x, y=0.93,
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
    """One row per pipeline stage, HbO and HbR together, over the filter's own response.

    ``raw_haemo`` is the first stage, the Beer-Lambert output. ``stages`` are the ones
    after it as ``[(label, raw), ...]``, read off disk by the caller. Passing None falls
    back to simulating the bandpass in memory, which is what a prep-only run gets.

    Stage is the row so that a step is read by looking down the column, and every row shares
    one power axis or the comparison would be against a rescaled yardstick. The last row is
    the filter's own frequency response, which answers "what did the bandpass do" directly
    rather than leaving it to be inferred from the gap between two noisy curves. It is a
    separate row rather than a second axis on the data rows: two scales on one plot invent
    an alignment the data does not have.

    Individual channels are drawn faintly behind each stage's channel mean, so a mean is
    read against a real spread. Physiological bands are shaded and the cutoffs marked.

    Args:
        raw_haemo:          Beer-Lambert output (HbO/HbR channels), the first stage.
        l_freq:             High-pass cutoff. Overridden by what the stage files record.
        h_freq:             Low-pass cutoff, same.
        filter_method:      Filter design, same.
        filter_order:       Butterworth order, same.
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
    l_freq, h_freq, filter_method, filter_order = _recorded_filter(
        stages, l_freq, h_freq, filter_method, filter_order)

    n_data = len(all_stages)
    has_response = l_freq is not None or h_freq is not None
    n_rows = n_data + (1 if has_response else 0)
    # the response is a reference, not a measurement, so it gets a shorter row
    heights = [1.0] * n_data + ([0.62] if has_response else [])
    titles = [label for label, _ in all_stages] + (["Filter response"] if has_response else [])

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[h / sum(heights) for h in heights], subplot_titles=titles,
    )

    lo, hi = np.inf, -np.inf
    for row, (label, raw) in enumerate(all_stages, start=1):
        for ch_type, color in (("hbo", _HBO_COLOR), ("hbr", _HBR_COLOR)):
            result = _stage_psd(raw, fmax, ch_type)
            if result is None:
                continue
            freqs, db = result
            x = freqs.tolist()
            for channel in db:
                fig.add_trace(go.Scatter(
                    x=x, y=channel.tolist(), mode="lines",
                    line=dict(width=0.6, color=color), opacity=0.15,
                    showlegend=False, hoverinfo="skip",
                ), row=row, col=1)
            mean_db = db.mean(axis=0)
            name = ch_type.upper().replace("HBO", "HbO").replace("HBR", "HbR")
            fig.add_trace(go.Scatter(
                x=x, y=mean_db.tolist(), mode="lines",
                line=dict(width=2.0, color=color),
                name=name, legendgroup=name, showlegend=(row == 1),
                hovertemplate=f"{label} {name}<br>%{{x:.3f}} Hz<br>%{{y:.1f}} dB<extra></extra>",
            ), row=row, col=1)
            lo, hi = min(lo, float(db.min())), max(hi, float(db.max()))

    if has_response:
        freqs, db = filter_response(raw_haemo.info["sfreq"], raw_haemo.n_times,
                                    l_freq, h_freq, filter_method, filter_order)
        inside = freqs <= fmax
        fig.add_trace(go.Scatter(
            x=freqs[inside].tolist(),
            y=np.maximum(db[inside], _RESPONSE_FLOOR).tolist(),
            mode="lines", line=dict(width=1.5, color=_RESPONSE_COLOR, dash="dash"),
            showlegend=False,
            hovertemplate="filter<br>%{x:.3f} Hz<br>%{y:.1f} dB<extra></extra>",
        ), row=n_rows, col=1)
        fig.update_yaxes(title_text="Attenuation (dB)", range=[_RESPONSE_FLOOR, 6],
                         gridcolor="#eeeeee", row=n_rows, col=1)

    _add_band_annotations(fig, fmax=fmax, bands=_physio_bands(cardiac, resp), rows=n_data)

    for cutoff in (l_freq, h_freq):
        if cutoff is not None and cutoff <= fmax:
            fig.add_vline(x=cutoff, line=dict(color="#555", width=1, dash="dash"))

    if np.isfinite(lo):
        pad = 0.04 * (hi - lo) or 1.0
        for row in range(1, n_data + 1):
            fig.update_yaxes(range=[lo - pad, hi + pad], row=row, col=1)
    fig.update_yaxes(title_text="Power (dB)", gridcolor="#eeeeee",
                     row=(n_data + 1) // 2, col=1)
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_xaxes(title_text="Frequency (Hz)", row=n_rows, col=1)
    for note in fig.layout.annotations[:n_rows]:
        note.update(font=dict(size=11, color="#555"), xanchor="left", x=0)
    fig.update_layout(
        title=dict(text=f"{title}<br><span style='font-size:11px;color:#777'>"
                        f"{filter_description(l_freq, h_freq, filter_method, filter_order)}</span>"),
        height=120 + 210 * sum(heights),
        margin=dict(l=70, r=30, t=110, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=10), orientation="h",
                    yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    return fig
