"""GLM visualisation figures.

activation_condition_figures(): per-condition brain surface projections, rendered off-screen.
activation_flat_figure():  the betas on the optode flat map, for optodes no reading placed.
"""

import base64
import io
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from nirspipe.pipeline.chromophores import chroma_channels, chroma_of
from nirspipe.qc.common.figure_io import fig_png_b64, stack_png_b64
from nirspipe.utils import pair_of
from nirspipe.utils.logging import get_logger

from nirspipe.qc.figures.common._brain_utils import RENDER_LOCK
from nirspipe.qc.figures.common._cortex import project_to_cortex, render_cortex
from nirspipe.qc.figures.common._surface_overlay import activation_rgba
from nirspipe.qc.figures.common._utils import CHROMA_STYLE
from nirspipe.qc.figures.common._utils import SPAN_KIND_COLOR, SPAN_KIND_LABEL
from nirspipe.qc.figures.common.matrix_map import CORRELATION_SCALE
from nirspipe.qc.figures.subject.rest_figures import flat_value_maps
from nirspipe.utils.spans import span_kind

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger("qc.figures.glm")

_DPI = 300

# --- activation brain --------------------------------------------------------

def _clim_vmax(clim: dict) -> float:
    lims = clim.get("pos_lims") or clim.get("lims") or (0.0, 0.0, 1.0)
    return abs(float(lims[-1]))


def _add_shared_colorbar(fig, clim: dict, chroma: str = "hbo", quantity: str = "beta") -> None:
    """One horizontal scale under the three views, in micromolar.

    The renderer draws a colourbar into every view; this draws it once instead, in the unit
    the betas are quoted in elsewhere rather than in bare molar.

    MNE spaces ``pos_lims``/``lims`` evenly about zero, so a plain symmetric norm over
    RdBu_r reproduces exactly what was rendered.
    """
    vmax = _clim_vmax(clim)
    cax = fig.add_axes([0.36, 0.035, 0.28, 0.022])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-vmax, vmax), cmap="RdBu_r"),
                      cax=cax, orientation="horizontal")
    ticks = np.linspace(-vmax, vmax, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t * 1e6:.3g}" for t in ticks])
    cb.set_label(f"{CHROMA_STYLE[chroma][0]} {quantity} (µM)", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_linewidth(0.5)


def _render_activation(stc, clim: dict, subjects_dir, size: tuple[int, int],
                       view_configs: list[dict]) -> list[np.ndarray]:
    """Draw ``stc`` on its template's pial surface as ``stc.plot`` does, without its Qt window.

    Same defaults as ``stc.plot``: transparent below fmin, 10 smoothing steps, grey binary
    curvature. One RGB image per ``view_configs``.
    """
    fmin, fmid, fmax = clim["pos_lims"]
    return render_cortex(
        stc, lambda curv, smoothed: activation_rgba(curv, smoothed[:, 0], fmin, fmid, fmax),
        subjects_dir, size, view_configs)


def _save_glm_brain(
    raw_haemo: mne.io.Raw,
    results_df: "pd.DataFrame",
    clim: dict,
    size: tuple[int, int] = (800, 700),
    title: str = "",
    chroma: str = "hbo",
    quantity: str = "beta",
) -> str:
    view_configs = [
        {'azimuth': 180, 'elevation': 90},  # Lateral
        {'azimuth': 0,   'elevation': 0},   # Dorsal
        {'azimuth': 90,  'elevation': 90},  # Frontal
    ]

    ch_col = _ch_col(results_df)
    drawn = chroma_channels(raw_haemo, chroma)

    # stc_near_sensors._get_channel_positions skips bad channels but evoked.data
    # includes them, causing a matmul shape mismatch, so drop bads before passing
    bads = [b for b in drawn.info["bads"] if b in drawn.ch_names]
    if bads:
        drawn.drop_channels(bads)

    if ch_col is not None:
        results_df = results_df[results_df[ch_col].isin(set(drawn.ch_names))].copy()

    coef_col = _coef_col(results_df)

    # the projection steps of plot_glm_surface_projection; the render is _render_activation
    if ch_col is not None:
        results_df = results_df.set_index(ch_col).loc[drawn.ch_names].reset_index()
    stc, subjects_dir = project_to_cortex(drawn, results_df[coef_col].values[:, np.newaxis])
    # ``size`` is the layout at 100 px per inch; the views are rendered at the saved density
    render_px = (size[0] * _DPI // 100, size[1] * _DPI // 100)
    with RENDER_LOCK:
        view_images = _render_activation(stc, clim, subjects_dir, render_px, view_configs)

    combined_img = np.hstack(view_images)

    import matplotlib.pyplot as plt
    fig_width = (size[0] * len(view_configs)) / 100
    fig_height = size[1] / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=_DPI)
    ax.imshow(combined_img)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=16, pad=10)
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.10)
    _add_shared_colorbar(fig, clim, chroma, quantity)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _coef_col(df) -> str:
    return "Coef." if "Coef." in df.columns else (
           "theta" if "theta" in df.columns else df.columns[-1])


def _ch_col(df) -> "str | None":
    return next((c for c in ("ch_name", "Channel", "channel") if c in df.columns), None)


def _plotted_rows(df, raw_haemo: "mne.io.Raw | None", chroma: str = "hbo"):
    """The rows :func:`_save_glm_brain` will actually project: ``chroma``'s, and not a bad channel.

    _plotted_rows(df_with_hbo_and_hbr_rows, raw) -> only the good HbO rows

    The colour scale has to be measured on the same set that gets drawn, or a bad channel's
    beta sets a limit no drawn channel comes near.
    """
    ch_col = _ch_col(df)
    if ch_col is None:
        return df
    names = df[ch_col].astype(str)
    keep = names.map(chroma_of) == chroma
    bads = set(raw_haemo.info.get("bads") or ()) if raw_haemo is not None else set()
    # HbT is bad where either half is
    bads |= {f"{pair_of(b)} hbt" for b in bads}
    if bads:
        keep &= ~names.isin(bads)
    return df[keep]


def _shared_clim(results_dict: "dict[str, pd.DataFrame]",
                 raw_haemo: "mne.io.Raw | None" = None, chroma: str = "hbo") -> dict:
    """One colour scale over every condition, so two of them can be read against each other.

    The limit is a high percentile of the drawn rows rather than their maximum, so one
    surviving outlier channel cannot flatten the rest. There is no absolute floor: at the
    scale of haemoglobin betas a fixed one would grey out every condition.
    """
    vals = [_plotted_rows(df, raw_haemo, chroma)[_coef_col(df)].to_numpy(dtype=float)
            for df in results_dict.values()]
    mag = np.abs(np.concatenate(vals)) if vals else np.array([])
    mag = mag[np.isfinite(mag)]

    v = float(np.percentile(mag, 99.5)) if mag.size else 0.0
    if v <= 0 and mag.size:
        v = float(mag.max())
    if v <= 0:
        # a flat run still gets a non-zero scale
        v = 1.0
    return dict(kind="value", pos_lims=(0, v / 2, v))


def activation_condition_figures(
    raw_haemo: mne.io.Raw,
    results_dict: "dict[str, pd.DataFrame]",
    clim: dict | None = None,
    size: tuple[int, int] = (800, 700),
    failed: "list[tuple[str, str]] | None" = None,
    quantity: str = "beta",
) -> "list[tuple[str, str]]":
    """One brain render per condition, a row per chromophore the results hold, as ``[(label, png_b64), ...]``.

    activation_condition_figures(haemo, {"rest": df1, "task": df2})
    -> [("rest", "iVBOR..."), ("task", "iVBOR...")]

    The report shows these behind one switch rather than stacked, so a condition can be
    looked at on its own. Each chromophore's colour scale is shared by every condition, see
    :func:`_shared_clim`.

    A condition whose render failed is left out rather than kept as a blank panel, so the
    switcher never offers a label with nothing behind it; ``failed`` collects
    ``(label, reason)`` for each one, for the caller to report.
    """
    held = {chroma_of(n) for df in results_dict.values() if _ch_col(df) is not None
            for n in df[_ch_col(df)].astype(str)}
    chromas = [c for c in CHROMA_STYLE if c in held] or ["hbo"]
    clims = {c: clim or _shared_clim(results_dict, raw_haemo, c) for c in chromas}
    rendered: list[tuple[str, str]] = []
    for cond, df in results_dict.items():
        try:
            rows = [_save_glm_brain(raw_haemo, df, clims[c], size,
                                    f"{cond} - {CHROMA_STYLE[c][0]}", chroma=c,
                                    quantity=quantity)
                    for c in chromas]
            rendered.append((str(cond), stack_png_b64(rows)))
        except Exception as exc:
            logger.exception("GLM activation render failed for %s", cond)
            if failed is not None:
                failed.append((str(cond), str(exc)))
    return rendered


def activation_flat_figure(
    raw_haemo: mne.io.Raw,
    results_dict: "dict[str, pd.DataFrame]",
    sep_bands=None,
    quantity: str = "beta",
) -> "go.Figure | None":
    """Betas on the optode flat map, a head per condition and a row per chromophore on the scale :func:`activation_condition_figures` shares: its stand-in when the optodes could not be placed."""
    held = {chroma_of(n) for df in results_dict.values() if _ch_col(df) is not None
            for n in df[_ch_col(df)].astype(str)}
    rows = []
    for chroma in (c for c in CHROMA_STYLE if c in held):
        v = _shared_clim(results_dict, raw_haemo, chroma)["pos_lims"][2] * 1e6
        panels = []
        for cond, df in results_dict.items():
            drawn = _plotted_rows(df, raw_haemo, chroma)
            panels.append((str(cond), chroma,
                           dict(zip(drawn[_ch_col(df)].astype(str),
                                    drawn[_coef_col(df)].astype(float) * 1e6))))
        rows.append({"panels": panels, "scale": (-v, v),
                     "bar": f"{CHROMA_STYLE[chroma][0]} {quantity} (µM)"})
    return flat_value_maps(raw_haemo, rows, f"{quantity.capitalize()}s on the optode flat map",
                           CORRELATION_SCALE, sep_bands)


def design_matrix_static_figure(
    design_matrix: "pd.DataFrame",
    conditions: list[str],
    segments: "dict[str, list[tuple[float, float]]] | None" = None,
    title: str = "Design Matrix",
) -> str:
    """design matrix: stacked time-series rows, one per condition + sum row.

    Returns base64-encoded PNG. *segments*, ``{description: [(onset, duration)]}``, are shaded
    by kind: ``BAD_unselected`` grey, every other ``BAD_`` span red.
    """
    times = design_matrix.index.values.astype(float)
    regressors = design_matrix[conditions].values
    n_cond = len(conditions)
    n_rows = n_cond + 1

    colors = [
        "#2980b9", "#c0392b", "#27ae60", "#8e44ad",
        "#d35400", "#16a085", "#2c3e50", "#f39c12",
    ]

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(12, max(2.5, n_rows * 1.3 + 0.6)),
        sharex=True,
    )
    if n_rows == 1:
        axes = [axes]

    reg_sum = regressors.sum(axis=1)
    reg_sum_norm = reg_sum / max(float(reg_sum.max()), 1e-9)

    def _shade(ax):
        if not segments:
            return
        for desc, spans in segments.items():
            for onset, duration in spans:
                ax.axvspan(onset, onset + duration,
                           color=SPAN_KIND_COLOR[span_kind(desc)], alpha=0.18, linewidth=0)

    axes[0].plot(times, reg_sum_norm, color="#333333", linewidth=1.0)
    axes[0].set_ylabel("regressor sum", fontsize=7, labelpad=2)
    axes[0].set_ylim(-0.05, 1.15)
    axes[0].tick_params(labelsize=7)
    axes[0].spines[["top", "right"]].set_visible(False)
    _shade(axes[0])

    for k, name in enumerate(conditions):
        col_data = regressors[:, k]
        col_norm = col_data / max(float(col_data.max()), 1e-9)
        color = colors[k % len(colors)]
        ax = axes[k + 1]
        ax.plot(times, col_norm, color=color, linewidth=1.0)
        ax.set_ylabel(name, fontsize=7, labelpad=2)
        ax.set_ylim(-0.05, 1.15)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        _shade(ax)

    axes[-1].set_xlabel("Time (s)", fontsize=8)
    fig.suptitle(title, fontsize=10)
    if segments:
        kinds = [k for k in ("unselected", "corrupted") if any(span_kind(d) == k for d in segments)]
        fig.legend(handles=[Patch(color=SPAN_KIND_COLOR[k], alpha=0.18, label=SPAN_KIND_LABEL[k])
                            for k in kinds],
                   loc="upper right", fontsize=7, frameon=False, ncol=len(kinds))
    fig.tight_layout()

    return fig_png_b64(fig)


def design_matrix_heatmap(design_matrix: "pd.DataFrame") -> str:
    """Every column the model fitted, nuisance regressors included, as nilearn draws it."""
    from nilearn.plotting import plot_design_matrix as _plot_dm

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    _plot_dm(design_matrix, axes=ax)
    ax.tick_params(axis="x", labelsize=5)
    ax.tick_params(axis="y", labelsize=7)
    return fig_png_b64(fig)


# ---- whitening diagnostic ----
_WHITE_COLOR = "#555555"
_BAND_FILL = "rgba(0,0,0,0.06)"


def design_peak_frequencies(design_matrix: "pd.DataFrame", conditions: "list[str]") -> dict:
    """Each condition regressor's strongest frequency, read off its periodogram.

    ``{"tap": <15 s on, 30 s off>}`` -> ``{"tap": 0.0222}``, one cycle per 45 s.
    """
    times = np.asarray(design_matrix.index, float)
    sfreq = 1.0 / float(np.median(np.diff(times)))
    out = {}
    for name in conditions:
        x = design_matrix[name].to_numpy(float)
        power = np.abs(np.fft.rfft(x - x.mean())) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / sfreq)
        if len(power) > 1 and power[1:].max() > 0:
            out[name] = float(freqs[1 + int(np.argmax(power[1:]))])
    return out


def whitening_figure(spectrum: "pd.DataFrame", channels: "pd.DataFrame",
                     design_freqs: dict, lowband: "tuple[float, float]") -> "go.Figure | None":
    """The whitened residual's spectrum against white, then each channel's lag-1 autocorrelation."""
    chromas = [(c, label, color) for c, (label, color) in CHROMA_STYLE.items()
               if f"{c}_whitened" in spectrum.columns]
    if not chromas or spectrum.empty:
        return None
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                        subplot_titles=["Residual power relative to white noise",
                                        "Lag-1 autocorrelation of the whitened residual"])
    freqs = spectrum["freq"].to_numpy(float)
    values = np.concatenate([spectrum[f"{c}_{kind}"].to_numpy(float)
                             for c, _, _ in chromas for kind in ("whitened", "residual")])
    values = values[np.isfinite(values) & (values > 0)]
    lo, hi = float(values.min()), float(values.max())
    fig.add_trace(go.Scatter(x=[lowband[0], lowband[1], lowband[1], lowband[0]],
                             y=[lo, lo, hi, hi], mode="lines", fill="toself", fillcolor=_BAND_FILL,
                             line=dict(width=0), hoverinfo="skip", showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[freqs[0], freqs[-1]], y=[1.0, 1.0], mode="lines",
                             line=dict(color=_WHITE_COLOR, width=1, dash="dot"),
                             name="white", hoverinfo="skip"), row=1, col=1)
    for name, f in design_freqs.items():
        fig.add_trace(go.Scatter(x=[f, f], y=[lo, hi], mode="lines",
                                 line=dict(color=_WHITE_COLOR, width=1, dash="dash"),
                                 showlegend=False,
                                 hovertemplate=f"{name}<br>{f:.4f} Hz<extra></extra>"),
                      row=1, col=1)
    for c, label, color in chromas:
        fig.add_trace(go.Scatter(x=freqs, y=spectrum[f"{c}_residual"].to_numpy(float),
                                 mode="lines", line=dict(color=color, width=1), opacity=0.45,
                                 name=f"{label} before whitening",
                                 hovertemplate=f"{label} before<br>%{{x:.4f}} Hz<br>%{{y:.3g}}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=freqs, y=spectrum[f"{c}_whitened"].to_numpy(float),
                                 mode="lines", line=dict(color=color, width=2),
                                 name=f"{label} whitened",
                                 hovertemplate=f"{label} whitened<br>%{{x:.4f}} Hz<br>%{{y:.3g}}<extra></extra>"),
                      row=1, col=1)

    good = channels[channels["long"].astype(bool) & ~channels["bad"].astype(bool)
                    & channels["acf1_whitened"].notna()]
    pairs = sorted({str(n).rsplit(" ", 1)[0] for n in good["ch_name"]})
    if pairs:
        n = int(good["n_samples"].median())
        bound = 1.96 / np.sqrt(n)
        fig.add_trace(go.Scatter(x=[pairs[0], pairs[-1], pairs[-1], pairs[0]],
                                 y=[-bound, -bound, bound, bound], mode="lines", fill="toself",
                                 fillcolor=_BAND_FILL, line=dict(width=0), hoverinfo="skip",
                                 showlegend=False), row=2, col=1)
        for c, label, color in chromas:
            rows = good[good["Chroma"] == c]
            fig.add_trace(go.Scatter(x=[str(v).rsplit(" ", 1)[0] for v in rows["ch_name"]],
                                     y=rows["acf1_whitened"].to_numpy(float), mode="markers",
                                     marker=dict(color=color, size=7), showlegend=False,
                                     hovertemplate=f"%{{x}} {label}<br>%{{y:.3f}}<extra></extra>"),
                          row=2, col=1)
    fig.update_xaxes(type="log", title_text="Frequency (Hz)", gridcolor="#eeeeee", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Power / white", gridcolor="#eeeeee", row=1, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=pairs, tickangle=-60, row=2, col=1)
    fig.update_yaxes(title_text="lag-1 r", gridcolor="#eeeeee", zeroline=True,
                     zerolinecolor="#999999", row=2, col=1)
    fig.update_layout(height=720, margin=dict(l=70, r=30, t=60, b=90),
                      plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(font=dict(size=11), orientation="h",
                                  x=1, xanchor="right", y=1.04, yanchor="bottom"))
    return fig
