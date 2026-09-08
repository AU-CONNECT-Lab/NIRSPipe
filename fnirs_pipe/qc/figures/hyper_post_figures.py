"""Figure builders for the hyperscanning post-processing QC report."""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
import plotly.graph_objects as go

from fnirs_pipe.io.snirf import long_channel_picks
from fnirs_pipe.pipeline.synchrony import long_axis_over
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.hyper_post")


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _log_freq_ticks(freqs: np.ndarray) -> "tuple[list[float], list[float]]":
    """(labelled ticks, unlabelled ticks) for a WTC frequency axis, in Hz.

    e.g. a 0.004-0.2 Hz axis gives ([0.01, 0.1], [0.005 ... 0.09]): the decades carry the
    labels and every intermediate digit gets a bare tick.

    Left to itself plotly picks "D1" over a range of that width, which labels all nine digits
    of every decade: 0.008 and 0.009 then sit a fifth as far apart as 0.1 and 0.2 and the text
    collides. Worse, the rule flips to "D2" once the band is a little narrower, so two figures
    from one report disagree about what a tick means. Naming the ticks fixes both, and the
    same list drives the matplotlib panels so the two kinds of figure agree.
    """
    lo, hi = float(np.min(freqs)), float(np.max(freqs))
    if not (lo > 0 and hi > lo):
        return [], []

    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    steps = [m * 10.0 ** d for d in decades for m in range(1, 10)]
    major = [f for f in steps if f == 10.0 ** round(np.log10(f)) and lo <= f <= hi]
    # a band narrower than two decades carries at most one of them, which is not an axis:
    # fall back to the 1-2-5 pattern, and to the band's own ends when even that is empty.
    # Still one rule the whole way down, rather than plotly's two
    if len(major) < 2:
        major = [f for f in steps if round(f / 10.0 ** np.floor(np.log10(f))) in (1, 2, 5)
                 and lo <= f <= hi]
    if len(major) < 2:
        major = list(np.geomspace(lo, hi, 3))
    return major, [f for f in steps if lo <= f <= hi and f not in major]


def _freq_label(f: float, superscript: str = "html") -> str:
    """"0.01" as "10^-2" where it is a whole power of ten, else three significant digits."""
    exponent = np.log10(f)
    if abs(exponent - round(exponent)) >= 1e-9:
        return f"{f:.3g}"
    e = int(round(exponent))
    return f"10<sup>{e}</sup>" if superscript == "html" else f"$10^{{{e}}}$"


def _log_freq_axis(freqs: np.ndarray) -> dict:
    """plotly y-axis for a WTC map: log frequency, decades labelled, low frequency at the top.

    Reversed so frequency increases downward, which is how the wavelet coherence figures in
    the literature are drawn. Ticks come from :func:`_log_freq_ticks`.
    """
    major, minor = _log_freq_ticks(freqs)
    axis = dict(title="Frequency (Hz)", type="log", autorange="reversed", gridcolor="#444")
    if not major:
        return axis
    axis.update(
        tickmode="array",
        tickvals=major,
        ticktext=[_freq_label(f) for f in major],
        minor=dict(tickvals=minor, ticks="outside", ticklen=3, showgrid=False),
    )
    return axis


def _apply_log_freq_axis(ax, freqs: np.ndarray) -> None:
    """The matplotlib half of :func:`_log_freq_axis`, so both kinds of figure tick alike."""
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    major, minor = _log_freq_ticks(freqs)
    ax.set_yscale("log")
    ax.set_ylim(float(np.max(freqs)), float(np.min(freqs)))   # frequency increases downward
    if not major:
        return
    ax.yaxis.set_major_locator(FixedLocator(major))
    ax.yaxis.set_major_formatter(FixedFormatter([_freq_label(f, "tex") for f in major]))
    ax.yaxis.set_minor_locator(FixedLocator(minor))
    ax.yaxis.set_minor_formatter(NullFormatter())


def build_wtc_channel(
    wtc_data: dict,
    freqs: np.ndarray,
    times: np.ndarray,
    pair_label: str,
    markers_list: list[dict],
    cond_colors: dict[str, str],
) -> go.Figure | None:
    """WTC heatmap for one channel pair: time × log-frequency, colour = coherence [0–1].

    Frequency runs downward on a log axis labelled at the decades; see :func:`_log_freq_axis`.

    ::

      wtc_data: {"wtc": ndarray(n_freqs, n_times), "coi": ndarray(n_times),
                 "sig": ndarray(n_freqs) | None}

    COI boundary drawn as a white dashed line; regions below it may be edge-affected.
    When "sig" is present, a black contour outlines where coherence exceeds the
    Monte Carlo significance level (WTC / sig > 1).
    """
    if wtc_data is None or len(freqs) == 0 or len(times) == 0:
        return None

    wtc_arr = wtc_data["wtc"]   # (n_freqs, n_times)
    coi     = wtc_data["coi"]   # (n_times) — max reliable period in seconds
    sig     = wtc_data.get("sig")  # (n_freqs) per-frequency significance level, or None

    # COI → minimum reliable frequency at each time point
    with np.errstate(divide="ignore", invalid="ignore"):
        freq_coi = np.where(coi > 1e-10, 1.0 / coi, freqs.max())
    freq_coi = np.clip(freq_coi, float(freqs.min()), float(freqs.max()))

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=times.tolist(),
        y=freqs.tolist(),
        z=np.round(wtc_arr, 3).tolist(),
        zmin=0, zmax=1,
        colorscale="Viridis",
        colorbar=dict(title="WTC", thickness=12, len=0.6, y=0.5),
        hovertemplate="t=%{x:.1f}s<br>f=%{y:.4f}Hz<br>WTC=%{z:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=times.tolist(),
        y=freq_coi.tolist(),
        mode="lines",
        line=dict(color="white", width=1.5, dash="dot"),
        name="COI",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Significance contour: outline where coherence beats the Monte Carlo level (WTC / sig > 1).
    if sig is not None and len(sig) == wtc_arr.shape[0]:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = wtc_arr / np.asarray(sig, dtype=float)[:, None]
        fig.add_trace(go.Contour(
            x=times.tolist(),
            y=freqs.tolist(),
            z=ratio.tolist(),
            contours=dict(type="constraint", operation=">", value=1.0),
            line=dict(color="black", width=1.2),
            fillcolor="rgba(0,0,0,0)",
            showscale=False,
            name="p<0.05",
            showlegend=True,
            hoverinfo="skip",
        ))

    shapes = []
    for m in markers_list:
        color = cond_colors.get(m["description"], "#f39c12")
        if m["duration"] > 0.1:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=m["onset"], x1=m["onset"] + m["duration"], y0=0, y1=1,
                fillcolor=_hex_to_rgba(color, 0.12),
                line=dict(width=0), layer="below",
            ))

    fig.update_layout(
        shapes=shapes,
        xaxis=dict(title=f"Time (s)  [{pair_label}]", gridcolor="#444"),
        yaxis=_log_freq_axis(freqs),
        height=300,
        margin=dict(l=60, r=80, t=10, b=40),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(font=dict(size=9)),
    )
    return fig


def build_wtc_roi_matrix(
    band_df,
    roi_labels: list[str],
    subject_ids: list[str],
    band_fmin: float,
    band_fmax: float,
) -> go.Figure | None:
    """ROI x ROI heatmap of band-mean coherence, rows sub1's ROIs, columns sub2's.

    The crossed result is n**2 pairings and only the homologous ones reach the page as
    heatmaps. This carries the rest: one cell per ROI pair holding the number already written
    to `hyper-wtc-roichan.tsv`, so the off-diagonal pairs are visible without embedding their
    maps. Reads the `label` / `label2` columns, so it needs a crossed frame; an uncrossed one
    has no `label2` and returns None.
    """
    if band_df is None or "label2" not in getattr(band_df, "columns", []):
        return None

    lookup = {(r.label, r.label2): r.coherence for r in band_df.itertuples()}
    z = [[lookup.get((row, col)) for col in roi_labels] for row in roi_labels]
    if all(v is None for line in z for v in line):
        return None

    sub1 = subject_ids[0] if subject_ids else "sub1"
    sub2 = subject_ids[1] if len(subject_ids) > 1 else "sub2"

    fig = go.Figure(go.Heatmap(
        z=z, x=roi_labels, y=roi_labels,
        colorscale="Viridis", zmin=0, zmax=1,
        colorbar=dict(title="coherence"),
        hovertemplate=f"{sub1} %{{y}} × {sub2} %{{x}}<br>coherence %{{z:.3f}}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title=f"{sub2} ROI", side="top"),
        yaxis=dict(title=f"{sub1} ROI", autorange="reversed"),
        title=dict(text=f"Band mean {band_fmin:.3g}-{band_fmax:.3g} Hz", font=dict(size=12)),
        height=380,
        margin=dict(l=90, r=40, t=70, b=40),
        paper_bgcolor="white",
    )
    return fig


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_wtc_cross_matrix(
    band_df,
    labels: list[str],
    subject_ids: list[str],
    band_fmin: float,
    band_fmax: float,
    kind: str = "channel",
) -> str | None:
    """Band-mean coherence for every pairing across the two brains, as one heatmap.

    Rows are sub1's sites, columns sub2's, so cell (i, j) is sub1's site i against sub2's
    site j and the diagonal is the homologous pairing the rest of the report shows. This is
    the whole point of ``--wtc-channel-cross``: the crossed pairs are computed and written to
    the TSV, and without this figure the only ones anybody looks at are the n on the diagonal.

    Needs a crossed frame, recognised by its ``label2`` column; returns None without one.
    ``kind`` names the sites in the axis titles ("channel" or "ROI"). A blank cell is a
    pairing that failed or was dropped for resting on too few channels.

    Read cell by cell the off-diagonal is exploratory: single pairings are noisy and a
    correction over n**2 of them leaves little. The structure is what it is for.
    """
    if band_df is None or "label2" not in getattr(band_df, "columns", []):
        return None

    lookup = {(r.label, r.label2): r.coherence for r in band_df.itertuples()}
    z = np.array([[lookup.get((row, col), np.nan) for col in labels] for row in labels],
                 dtype=float)
    # labels is the montage, so a blank row is a channel one member lost and a blank
    # column one the other lost; the shape is the same for every dyad
    if not np.isfinite(z).any():
        return None

    sub1 = subject_ids[0] if subject_ids else "sub1"
    sub2 = subject_ids[1] if len(subject_ids) > 1 else "sub2"

    n = len(labels)
    side = max(4.0, min(0.34 * n + 1.6, 11.0))
    fig, ax = plt.subplots(figsize=(side + 1.4, side))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(z, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=max(5, min(8, 120 // max(n, 1))))
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=max(5, min(8, 120 // max(n, 1))))
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel(f"{sub2} {kind}", fontsize=9, labelpad=8)
    ax.set_ylabel(f"{sub1} {kind}", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    mean = float(np.nanmean(z))
    ax.set_title(f"Band mean {band_fmin:.3g}-{band_fmax:.3g} Hz   (grand mean {mean:.3f})",
                 fontsize=10, pad=26)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label="coherence")
    return _png_b64(fig)


def _phase_arrows(ax, times: np.ndarray, freqs: np.ndarray, phase, n_time: int = 14,
                  n_freq: int = 10) -> None:
    """Draw the relative-phase field over a coherence map, thinned to a readable grid.

    Right means in phase, left antiphase, and up means the row's subject leads by a quarter
    cycle. The map carries one phase per pixel, tens of thousands of them, so it is sampled
    onto a coarse grid; the arrows are directions and not magnitudes, so every one is the
    same length and drawing fewer loses nothing.
    """
    if phase is None:
        return
    phase = np.asarray(phase, dtype=float)
    if phase.shape != (len(freqs), len(times)):
        return

    fi = np.unique(np.linspace(0, len(freqs) - 1, min(n_freq, len(freqs))).astype(int))
    ti = np.unique(np.linspace(0, len(times) - 1, min(n_time, len(times))).astype(int))
    grid_t, grid_f = np.meshgrid(times[ti], freqs[fi])
    angle = phase[np.ix_(fi, ti)]
    ax.quiver(grid_t, grid_f, np.cos(angle), np.sin(angle),
              color="black", scale=22, width=0.006, headwidth=4, headlength=5,
              pivot="mid", zorder=3)


def build_wtc_roi_grid(
    roi_wtc,
    roi_labels: list[str],
    pair_key,
    subject_ids: list[str],
) -> str | None:
    """Every ROI-by-ROI coherence map on one grid, rows sub1's ROIs and columns sub2's.

    The per-ROI selector in the report carries one map at a time and, on a crossed run, only
    the diagonal of them. A few ROIs cross to a few dozen maps, which is small enough to draw
    at once and is the view that shows whether an off-diagonal pairing is coupled at a
    different time or a different frequency from the homologous one. Channels are left out of
    this treatment on purpose: 14 of them cross to 196 maps, and no page wants that.

    An uncrossed result has only the homologous pairings and gets a single row of them rather
    than nothing: the arrows are drawn here and not on the interactive maps, so this has to be
    where the lead-lag of a run without crossing is readable too.
    """
    if roi_wtc is None or pair_key is None or not roi_labels:
        return None
    pairs = roi_wtc.pairs.get(pair_key, {})
    if not pairs:
        return None
    crossed = any(isinstance(k, tuple) for k in pairs)

    freqs, times = np.asarray(roi_wtc.freqs), np.asarray(roi_wtc.times)
    if freqs.size == 0 or times.size == 0:
        return None

    n = len(roi_labels)
    rows = roi_labels if crossed else [None]
    fig, axes = plt.subplots(len(rows), n, figsize=(2.6 * n + 1.2, 2.1 * len(rows) + 0.6),
                             squeeze=False, sharex=True, sharey=True)
    mesh = None
    for i, roi1 in enumerate(rows):
        for j, roi2 in enumerate(roi_labels):
            ax = axes[i][j]
            data = pairs.get((roi1, roi2) if crossed else roi2)
            if data is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color="#888")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            mesh = ax.pcolormesh(times, freqs, np.asarray(data["wtc"], dtype=float),
                                 cmap="viridis", vmin=0, vmax=1, shading="nearest")
            _apply_log_freq_axis(ax, freqs)

            # below the cone of influence the map is padding, not measurement
            coi = np.asarray(data["coi"], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                freq_coi = np.where(coi > 1e-10, 1.0 / coi, freqs.max())
            ax.plot(times, np.clip(freq_coi, freqs.min(), freqs.max()),
                    color="white", lw=1.2, ls="--")
            _phase_arrows(ax, times, freqs, data.get("phase"))

            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(roi2, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{roi1}\nFrequency (Hz)" if roi1 else "Frequency (Hz)",
                              fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

    if mesh is None:
        plt.close(fig)
        return None

    sub1 = subject_ids[0] if subject_ids else "sub1"
    sub2 = subject_ids[1] if len(subject_ids) > 1 else "sub2"
    fig.colorbar(mesh, ax=axes, shrink=0.6, pad=0.02, label="WTC")
    heading = (f"Row: {sub1}'s ROI    Column: {sub2}'s ROI" if crossed
               else f"{sub1} × {sub2}, ROI by ROI")
    # a single-row grid has no spare height inside, so the heading sits above the figure and
    # the arrow key below it; bbox_inches="tight" takes in both
    fig.suptitle(heading, fontsize=11, y=1 + 0.30 / fig.get_figheight())
    fig.text(0.5, -0.4 / fig.get_figheight(),
             f"arrows: right = in phase, left = antiphase, "
             f"up = {sub1} leads by a quarter cycle",
             ha="center", fontsize=9, color="#444444")
    return _png_b64(fig)


def compute_isc(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    ch_type: str = "hbo",
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Compute inter-brain Pearson r matrix (n_ch × n_ch) over long channels.

    matrix[i, j] = Pearson r between sub1_ch_i and sub2_ch_j.
    Diagonal = same-channel ISC.

    Both axes are the *montage's* long channels, rejected ones included, so every dyad's
    matrix has one shape and a group analysis can stack them however their rejections
    differ. That is the convention
    :func:`fnirs_pipe.pipeline.restingstate.compute_fc` follows for the same reason. A
    rejected channel of sub1 leaves a blank row and one of sub2 a blank column -- never
    both, since sub1's copy of a channel is not needed to correlate sub2's against
    everything else. The axes carried sub1's *surviving* channels until 0.30.0, which left
    a matrix whose shape moved with the rejections and dropped sub2's own channels wherever
    sub1 had lost the same one.

    Position is not a safe key: a participant with one more rejected channel than the other
    shifts every channel after it, so column j would hold a different pair than its label
    claims. Everything here is looked up by S-D label.

    Rejections arrive on ``raw.info["bads"]``, which is where
    :func:`fnirs_pipe.pipeline.hyperscanning.load_group_haemo` puts them and the only place
    the WTC path reads them from. This used to take the resolved rejections a second time as
    a ``bad_channels`` argument and never look at it.

    Args:
        ch_type: "hbo" or "hbr".
    """
    if len(subject_ids) < 2:
        return None, None
    raw1 = aligned_raws.get(subject_ids[0])
    raw2 = aligned_raws.get(subject_ids[1])
    if raw1 is None or raw2 is None:
        return None, None

    def _by_label(raw: mne.io.Raw) -> dict[str, int]:
        """{label: index} over what this member kept, bads dropped: what gets correlated."""
        return {raw.ch_names[p].rsplit(" ", 1)[0]: p
                for p in long_channel_picks(raw, ch_type)}

    # the axis is the montage, the maps are what survived: one shape, blanks where a channel
    # went. The axis rule is shared with the crossed WTC matrix, which drew it from the first
    # member alone until 0.30.0
    ch_names = long_axis_over([raw1, raw2], ch_type)
    map1, map2 = _by_label(raw1), _by_label(raw2)
    if not ch_names:
        return None, None

    # alignment trims the pair to a common length, but nothing here depends on that having run
    n_times = min(raw1.n_times, raw2.n_times)

    def _rows(raw: mne.io.Raw, by_label: dict[str, int], who: str) -> np.ndarray:
        """One row per axis label, NaN for a label this subject has no usable channel at."""
        out = np.full((len(ch_names), n_times), np.nan)
        have = [c for c in ch_names if c in by_label]
        if have:
            out[[ch_names.index(c) for c in have]] = \
                raw.get_data(picks=[by_label[c] for c in have])[:, :n_times]
        if (blank := [c for c in ch_names if c not in by_label]):
            logger.warning("ISC (%s): %s has no usable %s, leaving those blank",
                           ch_type, who, ", ".join(blank))
        return out

    data1 = _rows(raw1, map1, subject_ids[0])
    data2 = _rows(raw2, map2, subject_ids[1])

    def _zscore(x: np.ndarray) -> np.ndarray:
        mu  = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std[std < 1e-12] = 1.0
        return (x - mu) / std

    d1 = _zscore(data1)
    d2 = _zscore(data2)
    isc_mat = (d1 @ d2.T) / d1.shape[1]
    np.clip(isc_mat, -1.0, 1.0, out=isc_mat)

    # a rejected channel contributed a row of NaN above, which the products carry
    return isc_mat, ch_names


def build_isc_panel(
    isc_mat: np.ndarray,
    ch_names: list[str],
    subject_ids: list[str],
    ch_type: str = "hbo",
    isc_threshold: float = 0.3,
) -> str:
    """Return base64 PNG of 2-panel ISC summary: ISC matrix | connectogram.

    Args:
        isc_mat:       n × n ISC matrix from compute_isc().
        ch_names:      Channel labels (n,), without type suffix.
        subject_ids:   [sub1_id, sub2_id, ...].
        ch_type:       "hbo" or "hbr", shown in titles.
        isc_threshold: Minimum ``|ISC|`` arc threshold forwarded to connectogram.
    """
    from PIL import Image
    from fnirs_pipe.qc.figures.connectogram import isc_connectogram as _isc_conn

    if isc_mat is None or len(ch_names) == 0:
        return ""

    n          = len(ch_names)
    sub1_label = subject_ids[0] if subject_ids else "Sub1"
    sub2_label = subject_ids[1] if len(subject_ids) > 1 else "Sub2"
    type_label = ch_type.upper()

    # height driven by matrix size so it can be square; cap to [5, 9]
    sq = float(np.clip(n * 0.20, 5.0, 9.0))
    fig = plt.figure(figsize=(sq * 2.4, sq + 1.0))
    gs  = fig.add_gridspec(1, 2, width_ratios=[2, 2], wspace=0.35)
    ax_matrix = fig.add_subplot(gs[0])
    ax_circle = fig.add_subplot(gs[1])

    # left: channel × channel ISC heatmap (bad channels masked to grey)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d0d0d0")
    im = ax_matrix.imshow(isc_mat, cmap=cmap, vmin=-1, vmax=1,
                           aspect="equal", interpolation="nearest")
    step = max(1, n // 20)
    idxs = list(range(0, n, step))
    ax_matrix.set_xticks(idxs)
    ax_matrix.set_xticklabels([ch_names[i] for i in idxs],
                               fontsize=6, rotation=45, ha="right")
    ax_matrix.set_yticks(idxs)
    ax_matrix.set_yticklabels([ch_names[i] for i in idxs], fontsize=6)
    ax_matrix.set_xlabel(sub2_label, fontsize=8)
    ax_matrix.set_ylabel(sub1_label, fontsize=8)
    ax_matrix.set_title(f"ISC matrix ({type_label})", fontsize=9, pad=4)
    plt.colorbar(im, ax=ax_matrix, shrink=0.75, label="Pearson r", pad=0.02)

    # right: connectogram embedded as image
    try:
        b64 = _isc_conn(
            np.nan_to_num(isc_mat, nan=0.0), ch_names,
            (sub1_label, sub2_label),
            threshold=isc_threshold,
            title=f"ISC {type_label}",
        )
        circle_bytes = base64.b64decode(b64)
        circle_img   = np.array(Image.open(io.BytesIO(circle_bytes)).convert("RGB"))
        ax_circle.imshow(circle_img)
        ax_circle.axis("off")
    except Exception as exc:
        logger.debug("ISC connectogram embed failed: %s", exc)
        ax_circle.text(0.5, 0.5, f"Connectogram\nunavailable\n{exc}",
                       ha="center", va="center",
                       transform=ax_circle.transAxes, fontsize=8, color="#888")
        ax_circle.axis("off")

    fig.suptitle(
        f"Inter-brain Synchrony ({type_label})  —  {sub1_label} × {sub2_label}",
        fontsize=10, y=1.01,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
