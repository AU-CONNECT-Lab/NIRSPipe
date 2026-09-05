"""Interpolated scalp topomaps of the evoked response, rendered by MNE.

The layout figure in raw_figures puts each channel's waveform where the channel sits;
this instead interpolates the response into a continuous field at a few time points,
which is what makes a spatial pattern legible. It is the only spatial view of activity
available before the GLM stage, and the only one at all when the 3-D stack (pyvista,
fsaverage) that the GLM surface projection needs is not installed.
"""

from __future__ import annotations

import base64
import io

import mne
import numpy as np

from fnirs_pipe.qc.figures._utils import epochable_events
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures")

_CBAR_WIDTH_RATIO = 0.10


def _condition_evokeds(
    raw_haemo: mne.io.Raw, epoch_tmin: float, epoch_tmax: float,
) -> "dict[str, mne.Evoked]":
    """Condition -> evoked, epoched on the (non-BAD) annotations. Empty dict if there are none."""
    if not any(not str(a["description"]).upper().startswith("BAD") for a in raw_haemo.annotations):
        return {}
    events, event_id = epochable_events(raw_haemo, epoch_tmin, epoch_tmax)
    if len(events) == 0:
        return {}
    epochs = mne.Epochs(
        raw_haemo, events, event_id, tmin=epoch_tmin, tmax=epoch_tmax,
        baseline=(epoch_tmin, 0), preload=True, verbose=False,
    )
    out = {}
    for cond in event_id:
        try:
            ev = epochs[cond].average()
        except Exception:
            continue
        # bad channels are still in the object and would otherwise pull the interpolation
        # toward whatever they happen to hold
        bads = [b for b in ev.info["bads"] if b in ev.ch_names]
        out[str(cond)] = ev.drop_channels(bads) if bads else ev
    return out


def _default_times(epoch_tmax: float) -> np.ndarray:
    """Five points from onset to the end of the canonical response, e.g. 0, 3.75, 7.5, 11.25, 15."""
    return np.linspace(0.0, min(epoch_tmax, 15.0), 5)


def evoked_topomap_static(
    raw_haemo: mne.io.Raw,
    times: "np.ndarray | list[float] | None" = None,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    chromophores: "tuple[str, ...]" = ("hbo", "hbr"),
) -> "str | None":
    """Base64 PNG of interpolated topomaps: one row per condition x chromophore, one column
    per time point. None when the run has no events or the montage carries no positions.

    Each chromophore gets one colour scale across every condition and time point, so panels
    can be compared; HbO and HbR keep separate scales, since HbR is the smaller signal.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        evokeds = _condition_evokeds(raw_haemo, epoch_tmin, epoch_tmax)
    except Exception as exc:
        logger.warning("evoked topomap epoching failed: %s", exc)
        return None
    if not evokeds:
        return None

    t = np.asarray(_default_times(epoch_tmax) if times is None else times, dtype=float)
    lo, hi = next(iter(evokeds.values())).times[[0, -1]]
    t = np.unique(np.clip(t, lo, hi))
    if not len(t):
        return None

    # rows first, so an empty chromophore (hbr absent) drops out before the grid is sized
    rows = [
        (cond, chromo)
        for chromo in chromophores
        for cond, ev in evokeds.items()
        if len(mne.pick_types(ev.info, fnirs=chromo))
    ]
    if not rows:
        return None

    vlim = {}
    for chromo in {c for _, c in rows}:
        vals = [
            ev.copy().pick(chromo).data[:, np.searchsorted(ev.times, t)] * 1e6
            for ev in evokeds.values()
            if len(mne.pick_types(ev.info, fnirs=chromo))
        ]
        v = float(np.nanmax(np.abs(np.concatenate(vals, axis=1)))) if vals else 0.0
        vlim[chromo] = (-v, v) if v > 0 else (None, None)

    n_cols = len(t)
    fig, axes = plt.subplots(
        len(rows), n_cols + 1,
        figsize=(1.55 * n_cols + 0.9, 1.85 * len(rows)),
        gridspec_kw={"width_ratios": [1] * n_cols + [_CBAR_WIDTH_RATIO]},
        squeeze=False,
    )
    # one colourbar per chromophore, on its last row: every row of a chromophore shares
    # the same scale, so repeating it once per condition says nothing
    cbar_rows = {max(i for i, (_, c) in enumerate(rows) if c == chromo)
                 for chromo in {c for _, c in rows}}

    try:
        for ri, (cond, chromo) in enumerate(rows):
            row = list(axes[ri])
            want_cbar = ri in cbar_rows
            if not want_cbar:
                row[n_cols].axis("off")
            evokeds[cond].plot_topomap(
                times=t, ch_type=chromo,
                axes=row if want_cbar else row[:n_cols], colorbar=want_cbar,
                extrapolate="local",  # optodes cover part of the head; do not paint the rest
                vlim=vlim[chromo], cmap="RdBu_r", contours=6, sensors=True,
                time_unit="s", show=False,
            )
            if ri:  # the time titles repeat down the grid; keep only the top row's
                for ax in row[:n_cols]:
                    ax.set_title("")
            pos = row[0].get_position()
            fig.text(
                pos.x0 - 0.006, (pos.y0 + pos.y1) / 2,
                f"{cond} {chromo.upper()}",
                rotation=90, va="center", ha="right", fontsize=8,
            )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception as exc:
        logger.warning("evoked topomap render failed: %s", exc)
        return None
    finally:
        plt.close(fig)
