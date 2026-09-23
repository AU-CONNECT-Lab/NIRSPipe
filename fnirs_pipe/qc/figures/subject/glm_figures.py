"""GLM visualisation figures.

design_matrix_figure():   regressor time series + censoring + BC/AC bars.
activation_brain_figure(): brain surface projection via mne_nirs (solid surface).
activation_panel():        per-condition rows of brain projections.
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
from PIL import Image as _PILImage
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from fnirs_pipe.utils.logging import get_logger

from fnirs_pipe.qc.figures.common._brain_utils import to_head
from fnirs_pipe.qc.figures.common._utils import HBO_COLOR, HBR_COLOR

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger("qc.figures.glm")

_COND_COLORS = [
    "#2980b9", "#c0392b", "#27ae60", "#8e44ad",
    "#d35400", "#16a085", "#2c3e50", "#f39c12",
]


def _censored_mask(times: np.ndarray,
                   segments: dict[str, list[tuple[float, float]]] | None) -> np.ndarray:
    """Boolean mask: True = timepoint is NOT censored."""
    mask = np.ones(len(times), dtype=bool)
    if not segments:
        return mask
    for spans in segments.values():
        for onset, duration in spans:
            mask &= ~((times >= onset) & (times < onset + duration))
    return mask


def design_matrix_figure(
    regressors: np.ndarray,
    times: np.ndarray,
    condition_names: list[str],
    segments: dict[str, list[tuple[float, float]]] | None = None,
    title: str = "Regressors of interest and combined censoring (gray)",
) -> go.Figure:
    """Design matrix figure.

    Layout: (n_conditions + 1) rows × 2 columns::

      Col 1 (85%): regressor time series; top row = sum of all regressors.
      Col 2 (15%): BC / AC bar chart per row.

    Censored spans are shaded gray; individual annotation types can be
    distinguished by passing them in `segments`.

    Args:
        regressors:       (n_times, n_conditions) array, values in [0, 1].
        times:            (n_times,) time axis in seconds.
        condition_names:  Names for each column of `regressors`.
        segments:         {label: [(onset, duration), ...]} for censoring overlay.
                          Build from raw.annotations (see motion_panel.py).
        title:            Figure title.
    """
    n_cond  = len(condition_names)
    n_rows  = n_cond + 1
    good_mask = _censored_mask(times, segments)

    row_heights = [1.0 / n_rows] * n_rows

    fig = make_subplots(
        rows=n_rows, cols=2,
        column_widths=[0.85, 0.15],
        shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.03,
        row_heights=row_heights,
    )

    regressor_sum = regressors.sum(axis=1)
    regressor_sum_norm = regressor_sum / max(regressor_sum.max(), 1e-9)

    def _add_row(row: int, y: np.ndarray, name: str, color: str) -> None:
        # time series trace
        fig.add_trace(go.Scatter(
            x=times.tolist(), y=y.tolist(),
            mode="lines", name=name,
            line=dict(width=1.2, color=color),
            showlegend=False,
        ), row=row, col=1)

        # gray shading for all censored spans
        if segments:
            for spans in segments.values():
                for onset, duration in spans:
                    fig.add_vrect(
                        x0=onset, x1=onset + duration,
                        fillcolor="rgba(150,150,150,0.25)", line_width=0,
                        layer="below", row=row, col=1,
                    )

        # BC / AC bars
        bc_val = float(y.mean())
        ac_val = float(y[good_mask].mean()) if good_mask.any() else bc_val
        fig.add_trace(go.Bar(
            x=["BC", "AC"], y=[bc_val, ac_val],
            marker_color=[color, color],
            marker_opacity=[0.45, 1.0],
            showlegend=False,
            hovertemplate="%{x}: %{y:.3f}<extra>" + name + "</extra>",
        ), row=row, col=2)

        fig.update_yaxes(
            title_text=name, title_font=dict(size=9),
            tickfont=dict(size=8), range=[-0.05, 1.1],
            row=row, col=1,
        )
        fig.update_yaxes(tickfont=dict(size=8), row=row, col=2)

    # row 1: sum of regressors
    _add_row(1, regressor_sum_norm, "regressor sum", "#333333")

    # rows 2..n+1: individual conditions
    for k, (name, color) in enumerate(zip(condition_names, _COND_COLORS)):
        col_data = regressors[:, k]
        col_norm = col_data / max(col_data.max(), 1e-9)
        _add_row(k + 2, col_norm, name, color)

    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
    fig.update_layout(
        title=title,
        height=max(300, n_rows * 130 + 80),
        margin=dict(l=90, r=20, t=60, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        barmode="group",
        bargap=0.3,
    )
    return fig


# --- activation brain --------------------------------------------------------

def _clim_vmax(clim: dict) -> float:
    lims = clim.get("pos_lims") or clim.get("lims") or (0.0, 0.0, 1.0)
    return abs(float(lims[-1]))


def _add_shared_colorbar(fig, clim: dict) -> None:
    """One horizontal scale under the three views, in micromolar.

    The renderer draws a colourbar into every view, so a three-view strip carries the same
    scale three times, each squeezed narrow enough that its scientific-notation labels run
    into one another. This draws it once instead, wide enough to read, in the unit the
    betas are quoted in elsewhere rather than in bare molar.

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
    cb.set_label("HbO beta (µM)", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_linewidth(0.5)


def _save_glm_brain(
    raw_haemo: mne.io.Raw,
    results_df: "pd.DataFrame",
    clim: dict,
    view: str = "lat",
    size: tuple[int, int] = (800, 700),
    title: str = "",
) -> str | None:
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
        from mne import EvokedArray, read_source_spaces
        from mne.source_estimate import stc_near_sensors
        from mne._fiff.constants import FIFF
        from mne.utils import get_subjects_dir
        import os
        mne.viz.set_3d_backend('pyvistaqt')
    except ImportError:
        return None

    view_configs = [
        {'azimuth': 180, 'elevation': 90},  # Lateral
        {'azimuth': 0,   'elevation': 0},   # Dorsal
        {'azimuth': 90,  'elevation': 90},  # Frontal
    ]

    try:
        # HbO only, the surface-projection convention; the HbR betas are reported by the
        # beta heatmap and the beta table, which are both per chromophore
        ch_col = next((c for c in ("ch_name", "Channel", "channel") if c in results_df.columns), None)
        raw_hbo = raw_haemo.copy().pick("hbo")

        # stc_near_sensors._get_channel_positions skips bad channels but evoked.data
        # includes them, causing a matmul shape mismatch — drop bads before passing
        bads_in_hbo = [b for b in raw_hbo.info["bads"] if b in raw_hbo.ch_names]
        if bads_in_hbo:
            raw_hbo.drop_channels(bads_in_hbo)

        hbo_ch_names = set(raw_hbo.ch_names)

        if ch_col is not None:
            results_df = results_df[results_df[ch_col].str.endswith("hbo")].copy()
            results_df = results_df[results_df[ch_col].isin(hbo_ch_names)].copy()

        coef_col = _coef_col(results_df)

        # Replicate plot_glm_surface_projection internally so we can pass
        # time_viewer=False — required for offscreen rendering (no iren available)
        # https://mne.tools/mne-nirs/stable/_modules/mne_nirs/visualisation/_plot_GLM_surface_projection.html#plot_glm_surface_projection
        if ch_col is not None:
            results_df = results_df.set_index(ch_col).loc[raw_hbo.ch_names].reset_index()
        ea = EvokedArray(results_df[coef_col].values[:, np.newaxis], raw_hbo.info.copy())
        # trans="fsaverage" makes stc_near_sensors read loc as head coords, so move the
        # optodes into that frame rather than just relabelling them
        for idx in range(len(ea.ch_names)):
            loc = ea.info["chs"][idx]["loc"]
            loc[:9] = to_head(loc[:9].reshape(3, 3), raw_hbo.info).reshape(9)
            ea.info["chs"][idx]["coord_frame"] = FIFF.FIFFV_COORD_HEAD

        subjects_dir = get_subjects_dir(raise_error=True)
        src_path = os.path.join(subjects_dir, "fsaverage", "bem", "fsaverage-ico-5-src.fif")
        src = read_source_spaces(src_path)

        picks = np.arange(len(ea.ch_names))
        stc = stc_near_sensors(
            evoked=ea, picks=picks, subject="fsaverage", trans="fsaverage",
            distance=0.03, mode="weighted", surface="pial",
            subjects_dir=subjects_dir, src=src, project=True, verbose=False,
        )
        brain = stc.plot(
            src=src, subjects_dir=subjects_dir, hemi="both", surface="pial",
            initial_time=0, clim=clim, size=size, colormap="RdBu_r",
            background="w", colorbar=False, time_viewer=False, verbose=False,
        )

        view_images = []
        for cfg in view_configs:
            brain.show_view(azimuth=cfg['azimuth'], elevation=cfg['elevation'], distance=480)
            if hasattr(brain, 'plotter'):
                brain.plotter.render()
            view_images.append(brain.screenshot())

        combined_img = np.hstack(view_images)
        brain.close()

        import matplotlib.pyplot as plt
        fig_width = (size[0] * len(view_configs)) / 100
        fig_height = size[1] / 100
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        ax.imshow(combined_img)
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=16, pad=10)
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.10)
        _add_shared_colorbar(fig, clim)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    except Exception:
        logger.exception("GLM brain render failed")
        return None


def _b64_to_figure(b64: str | None, title: str, img_height: int) -> go.Figure:
    fig = go.Figure()
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title, paper_bgcolor="#ffffff", height=img_height + 60,
                      margin=dict(l=0, r=0, t=50, b=10))
    if b64 is None:
        fig.add_annotation(text="Brain render unavailable (pyvista/mne_nirs not installed).",
                           showarrow=False, font=dict(size=13))
        return fig
    fig.add_layout_image(dict(
        source=f"data:image/png;base64,{b64}",
        xref="paper", yref="paper",
        x=0, y=1, sizex=1, sizey=1,
        sizing="contain", layer="above",
    ))
    return fig


def activation_brain_figure(
    raw_haemo: mne.io.Raw,
    results_df: "pd.DataFrame",
    title: str = "Activation",
    clim: dict | None = None,
    view: str = "dorsal",
    size: tuple[int, int] = (800, 700),
) -> go.Figure:
    if clim is None:
        clim = _shared_clim({title: results_df}, raw_haemo)

    b64 = _save_glm_brain(raw_haemo, results_df, clim, view, size, title)
    return _b64_to_figure(b64, title, size[1])


def _coef_col(df) -> str:
    return "Coef." if "Coef." in df.columns else (
           "theta" if "theta" in df.columns else df.columns[-1])


def _plotted_rows(df, raw_haemo: "mne.io.Raw | None"):
    """The rows :func:`_save_glm_brain` will actually project: HbO, and not a bad channel.

    _plotted_rows(df_with_hbo_and_hbr_rows, raw) -> only the good HbO rows

    The colour scale has to be measured on the same set that gets drawn. A bad channel's
    beta can sit orders of magnitude above the rest, and left in, it sets a limit no drawn
    channel comes near, flattening every real one to background grey.
    """
    ch_col = next((c for c in ("ch_name", "Channel", "channel") if c in df.columns), None)
    if ch_col is None:
        return df
    names = df[ch_col].astype(str)
    keep = names.str.endswith("hbo")
    bads = set(raw_haemo.info.get("bads") or ()) if raw_haemo is not None else set()
    if bads:
        keep &= ~names.isin(bads)
    return df[keep]


def _shared_clim(results_dict: "dict[str, pd.DataFrame]",
                 raw_haemo: "mne.io.Raw | None" = None) -> dict:
    """One colour scale over every condition, so two of them can be read against each other.

    Scaling each condition to its own maximum would make a condition that barely activated
    look like one that activated strongly, since both would fill their own scale.

    The limit is a high percentile of the drawn rows rather than their maximum, so one
    surviving outlier channel cannot flatten the rest. There is deliberately no absolute
    floor: haemoglobin betas sit around 1e-7 M, so any fixed floor would outrun the data
    and grey out every condition.
    """
    vals = [_plotted_rows(df, raw_haemo)[_coef_col(df)].to_numpy(dtype=float)
            for df in results_dict.values()]
    mag = np.abs(np.concatenate(vals)) if vals else np.array([])
    mag = mag[np.isfinite(mag)]

    v = float(np.percentile(mag, 99.5)) if mag.size else 0.0
    if v <= 0 and mag.size:
        v = float(mag.max())
    if v <= 0:
        # a zero-width scale makes stc.plot raise rather than draw a flat brain
        v = 1.0
    return dict(kind="value", pos_lims=(0, v / 2, v))


def activation_condition_figures(
    raw_haemo: mne.io.Raw,
    results_dict: "dict[str, pd.DataFrame]",
    clim: dict | None = None,
    view: str = "dorsal",
    size: tuple[int, int] = (800, 700),
) -> "list[tuple[str, str]]":
    """One brain render per condition, as ``[(label, png_b64), ...]``.

    activation_condition_figures(haemo, {"rest": df1, "talk": df2})
    -> [("rest", "iVBOR..."), ("talk", "iVBOR...")]

    The report shows these behind one switch rather than stacked, so a condition can be
    looked at on its own. The colour scale is shared, see :func:`_shared_clim`.

    A condition whose render failed is left out rather than kept as a blank panel, so the
    switcher never offers a label with nothing behind it.
    """
    if clim is None:
        clim = _shared_clim(results_dict, raw_haemo)
    rendered: list[tuple[str, str]] = []
    for cond, df in results_dict.items():
        b64 = _save_glm_brain(raw_haemo, df, clim, view, size, str(cond))
        if b64 is not None:
            rendered.append((str(cond), b64))
    return rendered


def activation_panel(
    raw_haemo: mne.io.Raw,
    results_dict: "dict[str, pd.DataFrame]",
    title: str = "Activation by condition",
    clim: dict | None = None,
    view: str = "dorsal",
    size: tuple[int, int] = (800, 700),
) -> str | None:
    """Every condition's render stacked into one image, for a caller that wants one file.

    The subject report does not use this any more: five conditions stack to roughly 3500 px,
    where no single condition can be looked at and two cannot be compared without scrolling
    between them. It renders each condition through
    :func:`activation_condition_figures` and switches between them instead.
    """
    rendered = activation_condition_figures(raw_haemo, results_dict, clim, view, size)
    if not rendered:
        return None

    row_labels = [label for label, _ in rendered]
    row_imgs = [np.array(_PILImage.open(io.BytesIO(base64.b64decode(b64))))
                for _, b64 in rendered]

    label_px = 30
    row_h = row_imgs[0].shape[0] + label_px
    combined_w = row_imgs[0].shape[1]
    dpi = 100

    fig_mpl, axes = plt.subplots(
        len(row_imgs), 1,
        figsize=(combined_w / dpi, row_h * len(row_imgs) / dpi),
        dpi=dpi,
    )
    if len(row_imgs) == 1:
        axes = [axes]
    for ax, arr, label in zip(axes, row_imgs, row_labels):
        ax.imshow(arr)
        ax.axis("off")
        ax.set_title(label, fontsize=13, pad=4)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.05)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig_mpl)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def per_channel_hrf_figure(
    times: np.ndarray,
    hrfs_hbo: np.ndarray,
    hrfs_hbr: np.ndarray,
    std_hbo: np.ndarray,
    std_hbr: np.ndarray,
    ch_names: list[str],
    q_hbo: "np.ndarray | None" = None,
    q_hbr: "np.ndarray | None" = None,
    hrfs_ctrl_hbo: "np.ndarray | None" = None,
    hrfs_ctrl_hbr: "np.ndarray | None" = None,
    title: str = "Per-channel HRF",
    n_cols: int = 9,
) -> go.Figure:
    """Grid of per-channel HRF timecourses with HbO/HbR uncertainty bands.

    Args:
        times:          Time axis (seconds) for the HRF window.
        hrfs_hbo:       (n_ch, n_times) mean HbO HRF per channel.
        hrfs_hbr:       (n_ch, n_times) mean HbR HRF per channel.
        std_hbo:        (n_ch, n_times) standard deviation for HbO.
        std_hbr:        (n_ch, n_times) standard deviation for HbR.
        ch_names:       Channel base names (without hbo/hbr suffix).
        q_hbo:          (n_ch,) FDR-corrected q-values for HbO.
        q_hbr:          (n_ch,) FDR-corrected q-values for HbR.
        hrfs_ctrl_hbo:  Optional (n_ch, n_times) control-condition HbO.
        hrfs_ctrl_hbr:  Optional (n_ch, n_times) control-condition HbR.
        title:          Figure title.
        n_cols:         Number of columns in the grid (default 9).
    """
    n_ch = len(ch_names)
    n_rows = max(1, int(np.ceil(n_ch / n_cols)))
    t = times.tolist()

    subplot_titles = []
    for i, name in enumerate(ch_names):
        parts = [name]
        if q_hbo is not None and i < len(q_hbo) and q_hbo[i] > 0:
            parts.append(f"q={np.log10(q_hbo[i]):.1f}")
        subplot_titles.append("  ".join(parts))
    # pad to n_rows * n_cols
    subplot_titles += [""] * (n_rows * n_cols - n_ch)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.06,
        horizontal_spacing=0.02,
        subplot_titles=subplot_titles,
    )

    # suppress subplot title font from make_subplots default (too large)
    for ann in fig.layout.annotations:
        ann.font = dict(size=7)

    hbo_color = HBO_COLOR
    hbr_color = HBR_COLOR
    ctrl_hbo_color = "#e8a0a0"
    ctrl_hbr_color = "#a0c4e8"

    for i in range(n_ch):
        row = i // n_cols + 1
        col = i % n_cols + 1
        show_legend = i == 0

        # HbO uncertainty band
        y_hi = (hrfs_hbo[i] + std_hbo[i]).tolist()
        y_lo = (hrfs_hbo[i] - std_hbo[i]).tolist()
        fig.add_trace(go.Scatter(
            x=t, y=y_hi, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=t, y=y_lo, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(192,57,43,0.15)",
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=t, y=hrfs_hbo[i].tolist(), mode="lines",
            line=dict(width=1.2, color=hbo_color),
            name="HbO", showlegend=show_legend,
            legendgroup="hbo",
            hovertemplate=f"{ch_names[i]} HbO<extra></extra>",
        ), row=row, col=col)

        # HbR uncertainty band
        y_hi_r = (hrfs_hbr[i] + std_hbr[i]).tolist()
        y_lo_r = (hrfs_hbr[i] - std_hbr[i]).tolist()
        fig.add_trace(go.Scatter(
            x=t, y=y_hi_r, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=t, y=y_lo_r, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(41,128,185,0.15)",
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=t, y=hrfs_hbr[i].tolist(), mode="lines",
            line=dict(width=1.2, color=hbr_color),
            name="HbR", showlegend=show_legend,
            legendgroup="hbr",
            hovertemplate=f"{ch_names[i]} HbR<extra></extra>",
        ), row=row, col=col)

        # optional control condition
        if hrfs_ctrl_hbo is not None and i < len(hrfs_ctrl_hbo):
            fig.add_trace(go.Scatter(
                x=t, y=hrfs_ctrl_hbo[i].tolist(), mode="lines",
                line=dict(width=0.8, color=ctrl_hbo_color, dash="dot"),
                name="HbO ctrl", showlegend=show_legend,
                legendgroup="hbo_ctrl",
                hovertemplate=f"{ch_names[i]} HbO ctrl<extra></extra>",
            ), row=row, col=col)
        if hrfs_ctrl_hbr is not None and i < len(hrfs_ctrl_hbr):
            fig.add_trace(go.Scatter(
                x=t, y=hrfs_ctrl_hbr[i].tolist(), mode="lines",
                line=dict(width=0.8, color=ctrl_hbr_color, dash="dot"),
                name="HbR ctrl", showlegend=show_legend,
                legendgroup="hbr_ctrl",
                hovertemplate=f"{ch_names[i]} HbR ctrl<extra></extra>",
            ), row=row, col=col)

        # zero line
        fig.add_hline(y=0, line_width=0.5, line_color="#cccccc",
                      row=row, col=col)

    fig.update_xaxes(tickfont=dict(size=7), title_font=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=7))
    fig.update_xaxes(title_text="Time (s)", row=n_rows)
    fig.update_layout(
        title=title,
        height=max(300, n_rows * 130 + 80),
        margin=dict(l=50, r=30, t=80, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=9)),
    )
    return fig


def design_matrix_static_figure(
    design_matrix: "pd.DataFrame",
    conditions: list[str],
    segments: "dict[str, list[tuple[float, float]]] | None" = None,
    title: str = "Design Matrix",
) -> str:
    """design matrix: stacked time-series rows, one per condition + sum row.

    Returns base64-encoded PNG. Censored spans are shaded gray when *segments* is given.
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
        for spans in segments.values():
            for onset, duration in spans:
                ax.axvspan(onset, onset + duration,
                           color="gray", alpha=0.18, linewidth=0)

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
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def design_matrix_heatmap(
    design_matrix: "pd.DataFrame",
    conditions: "list[str] | None" = None,
    title: str = "Design matrix",
) -> str:
    from nilearn.plotting import plot_design_matrix as _plot_dm

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    _plot_dm(design_matrix, axes=ax)
    ax.tick_params(axis="x", labelsize=5)
    ax.tick_params(axis="y", labelsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def glm_betas_figure(
    glm_df: "pd.DataFrame",
    conditions: "list[str]",
    title: str = "GLM: per-channel beta (theta)",
) -> str:
    """Per-channel HbO / HbR beta values as a heatmap (channels × conditions).

    `glm_df` is the output of `glm_est.to_dataframe()`.
    Only the listed `conditions` are shown (drift/constant excluded).
    """
    glm_df = glm_df.reset_index()
    if "Contrast" not in glm_df.columns:
        for alt in ("contrast", "Regressor", "regressor", "condition", "Condition"):
            if alt in glm_df.columns:
                glm_df = glm_df.rename(columns={alt: "Contrast"})
                break
    if "Contrast" not in glm_df.columns:
        return None

    glm_df["Contrast"] = glm_df["Contrast"].astype(str)
    df = glm_df[glm_df["Contrast"].isin(conditions)].copy()
    if df.empty:
        return None

    df["pair"] = df["ch_name"].str.replace(r"\s+(hbo|hbr)$", "", regex=True)
    df["chroma"] = df["ch_name"].str.extract(r"(hbo|hbr)$")[0]

    def _pivot(chroma):
        sub = df[df["chroma"] == chroma]
        if sub.empty:
            return None
        return sub.pivot_table(index="pair", columns="Contrast",
                               values="theta", aggfunc="mean")[conditions]

    hbo_pv = _pivot("hbo")
    hbr_pv = _pivot("hbr")
    if hbo_pv is None and hbr_pv is None:
        return None

    panels = [(hbo_pv, "HbO", HBO_COLOR), (hbr_pv, "HbR", HBR_COLOR)]
    panels = [(p, lbl, c) for p, lbl, c in panels if p is not None]
    n_panels = len(panels)

    pair_names = panels[0][0].index.tolist()
    n_ch = len(pair_names)
    n_cond = len(conditions)
    fig_h = max(4, n_ch * 0.22 + 1.5)

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(max(6, n_cond * n_panels * 0.9 + 2), fig_h),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (pv, lbl, color) in zip(axes, panels):
        mat = pv.reindex(pair_names).values.astype(float)
        vmax = np.nanpercentile(np.abs(mat), 95)
        vmax = max(vmax, 1e-6)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n_cond))
        ax.set_xticklabels(conditions, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(n_ch))
        if ax is axes[0]:
            ax.set_yticklabels(pair_names, fontsize=7)
        ax.set_title(lbl, fontsize=10, color=color)
        plt.colorbar(im, ax=ax, shrink=0.7, label="β (theta)", pad=0.02)

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()