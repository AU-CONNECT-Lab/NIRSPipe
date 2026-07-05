"""Denoising before/after comparison carpet (HbO + HbR, optional ROI grouping)."""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

_MAX_PTS = 2000


def _decimate_cols(arr: np.ndarray, max_pts: int) -> np.ndarray:
    if arr.shape[1] > max_pts:
        return arr[:, :: arr.shape[1] // max_pts]
    return arr


def _shared_scale(before: np.ndarray, after: np.ndarray, z: float):
    """Demean each stage by its own mean, divide BOTH by the before-std so denoising's
    variance reduction shows (after gets paler). z = (x - mean_t) / std_t(before), clipped."""
    if before.shape[0] == 0:
        return before, after
    std_b = before.std(axis=1, keepdims=True)
    std_b[std_b == 0] = 1.0
    bz = (before - before.mean(axis=1, keepdims=True)) / std_b
    az = (after - after.mean(axis=1, keepdims=True)) / std_b
    return np.clip(bz, -z, z), np.clip(az, -z, z)


def carpet_compare_figure(
    before_haemo: mne.io.Raw,
    after_haemo: mne.io.Raw,
    roi_map: "dict | None" = None,
    z_threshold: float = 2.5,
) -> str:
    """Before/after denoising carpet: HbO and HbR columns, before/after rows, shared per-channel scaling.

    Both stages divide by the before-std (see _shared_scale) so reduced fluctuation after
    denoising renders paler. roi_map ({ROI label: [channel names]}, project convention) groups
    rows by ROI (argsort + colour strip) when provided, else channel order. HbR is kept separate
    from HbO (they are anti-correlated). Returns a base64-encoded PNG.
    """
    after_set = set(after_haemo.ch_names)
    hbo_names = [c for c in before_haemo.ch_names if c.endswith(" hbo") and c in after_set]
    hbr_names = [c for c in before_haemo.ch_names if c.endswith(" hbr") and c in after_set]

    def _stage(names):
        b = before_haemo.get_data(picks=names) if names else np.empty((0, len(before_haemo.times)))
        a = after_haemo.get_data(picks=names) if names else np.empty((0, len(after_haemo.times)))
        return b, a

    hbo_bz, hbo_az = _shared_scale(*_stage(hbo_names), z_threshold)
    hbr_bz, hbr_az = _shared_scale(*_stage(hbr_names), z_threshold)

    # ROI order shared across chromophores (assumes hbo/hbr rows correspond by S-D pair)
    n_ch = len(hbo_names)
    order = np.arange(n_ch)
    roi_codes = None
    roi_names: list[str] = []
    if roi_map and n_ch:
        # roi_map is {ROI label: [channel names]} (project convention); invert to per-channel
        ch_to_roi = {ch: label for label, chans in roi_map.items() for ch in chans}
        labels = [str(ch_to_roi.get(c, ch_to_roi.get(c[:-4], ""))) for c in hbo_names]  # try full name then S-D base
        roi_names = sorted(set(labels))
        code_of = {name: i for i, name in enumerate(roi_names)}
        codes = np.array([code_of[l] for l in labels])
        order = np.argsort(codes, kind="stable")
        roi_codes = codes[order]

    def _prep(arr):
        arr = arr[order] if arr.shape[0] == len(order) and len(order) else arr
        return _decimate_cols(arr, _MAX_PTS)

    panels = {
        (0, 0): _prep(hbo_bz), (0, 1): _prep(hbr_bz),
        (1, 0): _prep(hbo_az), (1, 1): _prep(hbr_az),
    }

    has_roi = roi_codes is not None
    col0 = 1 if has_roi else 0
    ncols = 2 + col0
    width_ratios = ([0.05] if has_roi else []) + [1, 1]
    fig = plt.figure(figsize=(12, max(3.0, n_ch * 0.09) + 1.2))
    gs = fig.add_gridspec(2, ncols, width_ratios=width_ratios, hspace=0.18, wspace=0.06)

    row_titles = ["Before", "After (denoised)"]
    col_titles = ["HbO", "HbR"]
    im = None
    for r in range(2):
        if has_roi:
            ax_s = fig.add_subplot(gs[r, 0])
            ax_s.imshow(roi_codes[:, None], aspect="auto", cmap="gist_ncar", interpolation="none")
            ax_s.set_xticks([])
            locs = [float(np.mean(np.where(roi_codes == c)[0])) for c in np.unique(roi_codes)]
            ax_s.set_yticks(locs)
            ax_s.set_yticklabels([roi_names[c] for c in np.unique(roi_codes)], fontsize=6)
            ax_s.tick_params(length=0)
        for c in range(2):
            ax = fig.add_subplot(gs[r, col0 + c])
            data = panels[(r, c)]
            if data.shape[0]:
                im = ax.imshow(data, aspect="auto", cmap="gray_r",
                               vmin=-z_threshold, vmax=z_threshold, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=9)
            if r == 1:
                ax.set_xlabel("Time", fontsize=8)

    if im is not None:
        fig.colorbar(im, ax=fig.get_axes(), fraction=0.015, pad=0.02,
                     label="z (scaled by before SD)")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
