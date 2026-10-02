"""What the coherence modules share: the channel axis, the sampling rate, row z-scoring."""

from __future__ import annotations

from collections.abc import Iterable

import mne
import numpy as np

from fnirs_pipe.utils import pair_of
from fnirs_pipe.io.snirf import long_channel_picks


def _shared_sfreq(raws: dict[str, mne.io.Raw]) -> float:
    """Sampling rate common to every recording, or an error naming the offenders.

    Every metric here reads the rate off one participant and applies it to the pair, so a
    mismatch does not fail, it silently mislabels the frequency axis of the other. Alignment
    equalises duration, not rate, and the input stage is the caller's choice, so two
    participants can arrive resampled differently.
    """
    if not raws:
        raise ValueError("no recordings to read a sampling rate from")
    rates = {sid: round(float(raw.info["sfreq"]), 4) for sid, raw in raws.items()}
    if len(set(rates.values())) > 1:
        raise ValueError(
            f"recordings differ in sampling rate: {rates}. Resample them to a common rate "
            "before computing inter-brain metrics."
        )
    return next(iter(rates.values()))


def _long_by_label(
    raw: mne.io.Raw, ch_type: str = "hbo", sep_bands=None,
) -> dict[str, int]:
    """{S-D label: channel index} over one chromophore's long channels, bads already dropped.

    The label is the key every inter-brain metric matches on. Position cannot be: two
    participants with different channels rejected no longer agree on what index 3 is.

    ``ch_type`` is "hbo" or "hbr", and the two are parallel from here down: a member's HbO
    pairs only with the other member's HbO, and the two results are never averaged, since
    HbO and HbR anti-correlate by construction.
    """
    picks = long_channel_picks(raw, ch_type, sep_bands=sep_bands)
    if not picks:
        raise ValueError(
            f"no usable long {ch_type.upper()} channel: every one is either short-distance "
            "or marked bad"
        )
    return {pair_of(raw.ch_names[p]): p for p in picks}


def _long_signals(
    raw: mne.io.Raw, ch_type: str = "hbo", sep_bands=None,
) -> dict[str, np.ndarray]:
    """{S-D label: time course} over one chromophore's long channels, bads already dropped."""
    return {
        label: raw.get_data(picks=[p])[0].astype(np.float64)
        for label, p in _long_by_label(raw, ch_type, sep_bands).items()
    }


def long_axis_over(
    raws: "Iterable[mne.io.Raw]", ch_type: str = "hbo", sep_bands=None,
) -> list[str]:
    """The axis a *dyad's* channel-by-channel matrix is indexed by: the union of the members'
    montages, in the first member's order.

    ::

      member A: S1_D1, S2_D2        member B: S2_D2, S3_D3
      -> ["S1_D1", "S2_D2", "S3_D3"]

    A label only the other member carries still has a row or a column of its own.

    **Bads are kept**, unlike in :func:`_long_by_label`, so two dyads that lost different
    channels still produce matrices of one shape, and an empty cell stays distinct from a
    channel that was never in the montage. A 20-channel montage with 2 rejected gives 20
    labels, of which 2 index an all-blank row or column.

    One recording is a group of one, so ``long_axis_over([raw], ch_type)`` is the same rule
    over a single montage.
    """
    axis: list[str] = []
    for raw in raws:
        for p in long_channel_picks(raw, ch_type, exclude=[], sep_bands=sep_bands):
            label = pair_of(raw.ch_names[p])
            if label not in axis:
                axis.append(label)
    return axis


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """Each row to zero mean and unit deviation, a flat row left alone rather than divided by 0."""
    mu  = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mu) / np.where(std < 1e-12, 1.0, std)


def _chroma_tuple(chroma, name: str) -> "tuple[str, ...]":
    """The chromophores asked for, in order and once each, or a ValueError naming ``name``."""
    out = tuple(dict.fromkeys(chroma))
    if not out or set(out) - {"hbo", "hbr"}:
        raise ValueError(f"{name} must be some of ('hbo', 'hbr'), got {chroma!r}")
    return out
