"""Magnitude-squared coherence over a band: one number per channel pair, no time axis.

  compute_pairwise_coherence   Cheap, and enough when the question is whether a pair is
                               locked at all rather than when.
  screening_coherence          The same estimator turned on one recording's own channels,
                               which is what the channel screening reads.
"""

from __future__ import annotations

from itertools import combinations

import mne
import numpy as np
import pandas as pd
from scipy.signal import coherence

from fnirs_pipe.pipeline.hyper._helpers import _long_by_label, _shared_sfreq
from fnirs_pipe.pipeline.hyper.surrogate import phase_scramble
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.coherence")


def compute_pairwise_coherence(
    raws: dict[str, mne.io.Raw],
    fmin: float = 0.01,
    fmax: float = 0.10,
    sep_bands=None,
) -> pd.DataFrame:
    r"""Magnitude-squared coherence per HbO channel, averaged over [fmin, fmax] Hz.

    .. math::

        C_{xy}(f) = \frac{|P_{xy}(f)|^2}{P_{xx}(f)\, P_{yy}(f)}

    :math:`P_{xy}` is the cross-spectral density and :math:`P_{xx}, P_{yy}` the auto-spectra
    (Welch). The scalar per channel pair is :math:`C_{xy}` averaged over the band.
    Long channels only, matched across participants by S-D label; a label one of them lacks
    keeps its row with a NaN coherence.
    Returns a DataFrame with columns: ch_name, sub1, sub2, coherence.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for pairwise coherence")

    ref_raw = raws[subject_ids[0]]
    sfreq = _shared_sfreq(raws)
    nperseg = welch_nperseg(ref_raw.n_times)

    rows: list[dict] = []
    for sub1, sub2 in combinations(subject_ids, 2):
        raw1, raw2 = raws[sub1], raws[sub2]
        # HbO only, deliberately: this is the raw-report screening coherence, not a result
        map1, map2 = (_long_by_label(raw1, sep_bands=sep_bands),
                      _long_by_label(raw2, sep_bands=sep_bands))
        data1 = raw1.get_data(picks=list(map1.values()))
        data2 = raw2.get_data(picks=list(map2.values()))
        row_of = {label: i for i, label in enumerate(map2)}

        for i, label in enumerate(map1):
            j = row_of.get(label)
            if j is None:
                logger.warning("coherence: %s has no %s, leaving the pair blank", sub2, label)
                rows.append({"ch_name": label, "sub1": sub1, "sub2": sub2,
                             "coherence": float("nan")})
                continue
            mean_coh = _band_coherence(data1[i], data2[j], sfreq, nperseg, fmin, fmax)
            rows.append({"ch_name": label, "sub1": sub1, "sub2": sub2, "coherence": mean_coh})

    return pd.DataFrame(rows, columns=["ch_name", "sub1", "sub2", "coherence"])


# Surrogate pairings a screening percentile is read against. 100 is `write_wtc_null`'s own
# default, so the two nulls a dyad is measured by are drawn the same number of times.
SCREEN_NULL_ITER = 100


def welch_nperseg(n_times: int) -> int:
    """Segment length the band coherence is estimated with, for a stretch of ``n_times``.

    ::

      a 3900 s run at 10 Hz -> 512;  a 300 s block -> 512;  a 20 s window -> 64

    A quarter of the stretch, capped at 512 and floored at 64. One function because the number
    decides both the estimate and its floor: magnitude-squared coherence sits near 1/(number
    of segments) when nothing is coupled, so every caller's values sit on one scale.
    """
    return min(512, max(64, n_times // 4))


def _band_coherence(a, b, sfreq: float, nperseg: int, fmin: float, fmax: float) -> float:
    freqs, coh = coherence(a, b, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.mean(coh[mask])) if mask.any() else float("nan")


def screening_coherence(
    raws: dict[str, mne.io.Raw],
    fmin: float = 0.01,
    fmax: float = 0.10,
    windows: "list[tuple[str, float, float]] | None" = None,
    n_iter: int = SCREEN_NULL_ITER,
    seed: "int | None" = None,
    sep_bands=None,
) -> pd.DataFrame:
    r"""Band coherence per window and channel, with the percentile of its own surrogate null.

    ::

      screening_coherence(raws, windows=[("game1", 520.0, 1420.0)])
      -> rows for "whole run" and "game1", each channel carrying coherence and percentile

    **The percentile is the readable number, not the coherence.** Magnitude-squared coherence
    has a floor near :math:`1/n_{seg}` where :math:`n_{seg}` is the number of Welch segments,
    and that count falls with the window, so two windows' raw values are not comparable; each
    value's rank inside a null drawn for *that* window is.

    The null pairs one member against a phase-scrambled copy of the other, which is the
    surrogate :func:`~fnirs_pipe.pipeline.hyper.wtc_null.write_wtc_null` uses on the post report. One
    definition across a dyad's two pages, so "above the null" means one thing on both.

    Returns a DataFrame with columns: window, ch_name, sub1, sub2, coherence, null_mean,
    null_p95, percentile, window_percentile, n_seg, window_s. ``percentile`` is per channel
    and ``window_percentile`` is the window's own, repeated down its rows; grade on the
    second, since the first is a rank over a hundred draws and moves with the draw.
    """
    subject_ids = list(raws)
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for screening coherence")

    sfreq = _shared_sfreq(raws)
    rng = np.random.default_rng(seed)
    n_times = min(r.n_times for r in raws.values())
    spans = [("whole run", 0.0, n_times / sfreq), *(windows or [])]

    rows: list[dict] = []
    for sub1, sub2 in combinations(subject_ids, 2):
        map1 = _long_by_label(raws[sub1], sep_bands=sep_bands)
        map2 = _long_by_label(raws[sub2], sep_bands=sep_bands)
        labels = [name for name in map1 if name in map2]
        for missing in (name for name in map1 if name not in map2):
            logger.warning("screening coherence: %s has no %s, dropping the pair",
                           sub2, missing)
        if not labels:
            continue
        d1 = raws[sub1].get_data(picks=[map1[n] for n in labels])[:, :n_times]
        d2 = raws[sub2].get_data(picks=[map2[n] for n in labels])[:, :n_times]

        for name, tstart, tstop in spans:
            i0, i1 = max(0, int(tstart * sfreq)), min(n_times, int(tstop * sfreq))
            seg = i1 - i0
            nperseg = welch_nperseg(seg)
            if seg <= nperseg:
                logger.warning("screening coherence: %r is %.0f s, too short to score", name,
                               seg / sfreq)
                continue
            n_seg = max(1, (seg - nperseg) // (nperseg // 2) + 1)
            real = np.array([_band_coherence(d1[i, i0:i1], d2[i, i0:i1], sfreq, nperseg,
                                             fmin, fmax) for i in range(len(labels))])
            null = np.empty((n_iter, len(labels)))
            for k in range(n_iter):
                for i in range(len(labels)):
                    null[k, i] = _band_coherence(
                        d1[i, i0:i1], phase_scramble(d2[i, i0:i1], rng),
                        sfreq, nperseg, fmin, fmax)
            # window percentile: the channel mean ranked among the null's channel means; a
            # window with no Welch bin in the band has no coherence, so it has no rank either
            mean_null = null.mean(axis=1)
            window_pct = (float((mean_null < real.mean()).mean() * 100)
                          if np.isfinite(real.mean()) else float("nan"))
            for i, label in enumerate(labels):
                rows.append({
                    "window": name, "ch_name": label, "sub1": sub1, "sub2": sub2,
                    "coherence": float(real[i]),
                    "null_mean": float(null[:, i].mean()),
                    "null_p95": float(np.percentile(null[:, i], 95)),
                    "percentile": (float((null[:, i] < real[i]).mean() * 100)
                                   if np.isfinite(real[i]) else float("nan")),
                    "window_percentile": window_pct,
                    "n_seg": int(n_seg), "window_s": round(seg / sfreq, 1),
                })

    return pd.DataFrame(rows, columns=["window", "ch_name", "sub1", "sub2", "coherence",
                                       "null_mean", "null_p95", "percentile",
                                       "window_percentile", "n_seg", "window_s"])
