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
from ._utils import HBO_MEAN_COLOR as _HBO_MEAN_COLOR, HBR_MEAN_COLOR as _HBR_MEAN_COLOR
from ._utils import add_band_shading as _add_band_shading
from ._utils import physio_bands as _physio_bands

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


def _passband_level(freqs: np.ndarray, db: np.ndarray, l_freq, h_freq) -> float:
    """Mean power inside the passband, the level the filter's 0 dB is drawn at.

    _passband_level(freqs, hbo_mean_db, 0.02, 0.2) -> the dB the curve sits at between the
    cutoffs, so a response curve added to it lands on top of the data it should explain.
    """
    inside = np.ones(len(freqs), dtype=bool)
    if l_freq is not None:
        inside &= freqs >= l_freq
    if h_freq is not None:
        inside &= freqs <= h_freq
    return float(db[inside].mean()) if inside.any() else float(db.mean())


def psd_figure(
    raw_haemo: mne.io.Raw,
    l_freq: float | None = None,
    h_freq: float | None = 0.4,
    filter_method: str = DEFAULT_FILTER_METHOD,
    filter_order: int = DEFAULT_FILTER_ORDER,
    fmax: float = 2.0,
    title: str = "Power spectral density, before and after the bandpass",
    cardiac: "tuple[float, float] | None" = None,
    resp: "tuple[float, float] | None" = None,
    stages: "list[tuple[str, mne.io.Raw]] | None" = None,
    first_label: str = "Before bandpass",
) -> go.Figure:
    """The bandpass, before over after, HbO and HbR together, with the filter drawn on it.

    ``raw_haemo`` is the Beer-Lambert output, the filter's input. ``stages`` are what the run
    wrote after it as ``[(label, raw), ...]``, read off disk by the caller, normally just the
    filtered file. Passing None falls back to simulating the bandpass in memory, which is
    what a prep-only run gets.

    Stage is the row so a step is read by looking down the column, and both rows share one
    power axis or the comparison would be against a rescaled yardstick. The filter's own
    response is drawn dashed over the row it produced, shifted so 0 dB sits at that row's
    passband level: it stays on the one power axis that way, and the data curve can be read
    straight against the shape the filter should have given it.

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
        first_label:        Row title for ``raw_haemo``.
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

    n_rows = len(all_stages)
    titles = [label for label, _ in all_stages]
    if n_rows >= 2 and titles[1].startswith("desc-") and (l_freq is not None or h_freq is not None):
        edges = "  ".join(part for part in (f"HP {l_freq} Hz" if l_freq is not None else "",
                                            f"LP {h_freq} Hz" if h_freq is not None else "") if part)
        titles[1] = f"After bandpass ({edges})"
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=titles,
    )

    lo, hi = np.inf, -np.inf
    anchor = None  # passband level of the filter's output row, for the response curve
    for row, (label, raw) in enumerate(all_stages, start=1):
        for ch_type, color, mean_color in (("hbo", _HBO_COLOR, _HBO_MEAN_COLOR),
                                           ("hbr", _HBR_COLOR, _HBR_MEAN_COLOR)):
            result = _stage_psd(raw, fmax, ch_type)
            if result is None:
                continue
            freqs, db = result
            x = freqs.tolist()
            for channel in db:
                fig.add_trace(go.Scatter(
                    x=x, y=channel.tolist(), mode="lines",
                    line=dict(width=0.6, color=color), opacity=0.2,
                    showlegend=False, hoverinfo="skip",
                ), row=row, col=1)
            mean_db = db.mean(axis=0)
            name = "HbO" if ch_type == "hbo" else "HbR"
            fig.add_trace(go.Scatter(
                x=x, y=mean_db.tolist(), mode="lines",
                line=dict(width=2.0, color=mean_color),
                name=name, legendgroup=name, showlegend=(row == 1),
                hovertemplate=f"{label} {name}<br>%{{x:.3f}} Hz<br>%{{y:.1f}} dB<extra></extra>",
            ), row=row, col=1)
            lo, hi = min(lo, float(db.min())), max(hi, float(db.max()))
            if row == 2 and ch_type == "hbo":
                anchor = _passband_level(freqs, mean_db, l_freq, h_freq)

    # the filter goes over row 2, the stage it produced; row 1 is its input
    if anchor is not None and n_rows >= 2:
        freqs, db = filter_response(raw_haemo.info["sfreq"], raw_haemo.n_times,
                                    l_freq, h_freq, filter_method, filter_order)
        inside = freqs <= fmax
        attenuation = np.maximum(db[inside], _RESPONSE_FLOOR)
        fig.add_trace(go.Scatter(
            x=freqs[inside].tolist(),
            # clipped to the data's own floor rather than extending the axis to reach it:
            # the filter's stopband runs tens of dB below anything the PSD can measure
            y=np.maximum(attenuation + anchor, lo).tolist(),
            mode="lines", line=dict(width=1.5, color=_RESPONSE_COLOR, dash="dash"),
            name="Filter response", legendgroup="Filter response",
            hovertemplate="filter<br>%{x:.3f} Hz<br>%{text:.1f} dB<extra></extra>",
            text=attenuation.tolist(),
        ), row=2, col=1)

    _add_band_shading(fig, fmax=fmax, bands=_physio_bands(cardiac, resp), rows=n_rows)

    for cutoff in (l_freq, h_freq):
        if cutoff is not None and cutoff <= fmax:
            fig.add_vline(x=cutoff, line=dict(color="#555", width=1, dash="dash"))

    if np.isfinite(lo):
        pad = 0.04 * (hi - lo) or 1.0
        fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_xaxes(title_text="Frequency (Hz)", row=n_rows, col=1)
    fig.update_yaxes(title_text="Power (dB)", gridcolor="#eeeeee")
    fig.update_layout(
        title=dict(text=f"{title}<br><span style='font-size:11px;color:#777'>"
                        f"{filter_description(l_freq, h_freq, filter_method, filter_order)}</span>"),
        height=110 + 210 * n_rows,
        margin=dict(l=70, r=30, t=95, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=11)),
    )
    return fig
