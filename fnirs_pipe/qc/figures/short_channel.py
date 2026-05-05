"""Short-channel QC figure: PSD comparison of short vs long separation channels."""

import io
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np


def short_channel_figure(
    raw: mne.io.Raw,
    fmax: float = 2.0,
) -> str | None:
    """PSD overlay of short-channel (< 1 cm) vs long-channel signals.

    Short channels contain mostly systemic / superficial noise. Overlaying
    their PSDs on the long-channel PSDs shows whether the cardiac peak is
    relatively stronger in short channels (expected) and whether regression
    would be effective.

    Returns a base64-encoded PNG string, or None if no short channels exist.
    """
    picks = mne.pick_types(raw.info, fnirs=True)
    if not len(picks):
        return None

    dists = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    short_mask = dists < 0.01
    long_mask = ~short_mask

    if not short_mask.any():
        return None

    short_picks = picks[short_mask]
    long_picks = picks[long_mask]

    psd = raw.compute_psd(fmax=fmax, picks=picks, verbose=False)
    freqs = psd.freqs
    data_all = psd.get_data()

    # index into data_all using positions within `picks`
    short_idx = np.where(short_mask)[0]
    long_idx = np.where(long_mask)[0]
    data_short = data_all[short_idx]
    data_long = data_all[long_idx]

    fig, ax = plt.subplots(figsize=(10, 4))

    db_long  = 10 * np.log10(data_long  + 1e-30)
    db_short = 10 * np.log10(data_short + 1e-30)

    for row in db_long:
        ax.plot(freqs, row, lw=0.5, color="#2980b9", alpha=0.15)
    ax.plot(freqs, db_long.mean(axis=0), lw=2.0, color="#2980b9",
            label=f"Long channels (n={len(long_idx)})")

    for row in db_short:
        ax.plot(freqs, row, lw=0.5, color="#e74c3c", alpha=0.30)
    ax.plot(freqs, db_short.mean(axis=0), lw=2.0, color="#e74c3c",
            label=f"Short channels (n={len(short_idx)})")

    ax.axvspan(0.08, 0.12, color="#2ecc71", alpha=0.10)
    ax.axvspan(0.70, 1.50, color="#e74c3c", alpha=0.08)
    ax.text(0.10, 1.0, "Mayer", ha="center", va="top", fontsize=7, color="#27ae60",
            transform=ax.get_xaxis_transform())
    ax.text(1.10, 1.0, "Cardiac", ha="center", va="top", fontsize=7, color="#c0392b",
            transform=ax.get_xaxis_transform())

    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("Power (dB)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="both", color="#eeeeee", linewidth=0.6)
    ax.set_xlim(0, fmax)
    fig.suptitle("Short vs long channel PSD — overlay", fontsize=11)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
