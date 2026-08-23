"""Synchrony metrics for a hyperscanning group: wavelet coherence and band coherence.

Two ways of asking how locked two recordings are.

  compute_wtc / compute_wtc_roi   Wavelet transform coherence, resolved in both time and
                                  frequency, so a pair that only synchronises during part
                                  of the task still shows it. Per channel, or per ROI once
                                  channels have been averaged. Backed by pycwt.
  compute_pairwise_coherence      One magnitude-squared coherence number per channel pair,
                                  averaged over a band. Cheap, and enough when the question
                                  is whether a pair is locked at all.

The callers live in pipeline/hyperscanning.py, which handles the dyad bookkeeping these
functions assume has already happened: recordings loaded, aligned, trimmed to a common
length and normalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import mne
import numpy as np
import pandas as pd
from scipy.signal import coherence

from fnirs_pipe.utils.logging import get_logger

# the caller's logger name, kept so existing log filters still match
logger = get_logger("pipeline.hyperscanning")


@dataclass
class WTCResult:
    """Pairwise wavelet transform coherence per HbO channel.

    ::

      pairs[(sub1, sub2)][ch_name] = {"wtc": ndarray(n_freqs, n_times),
                                       "coi": ndarray(n_times)}

    freqs: ascending Hz.  times: decimated aligned time axis (seconds).
    """
    pairs: dict
    freqs: np.ndarray
    times: np.ndarray


def _pairwise_wtc(
    sig1: np.ndarray,
    sig2: np.ndarray,
    dt: float,
    step: int,
    fmin: float,
    fmax: float,
    significance: bool = False,
    cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    r"""Morlet wavelet transform coherence for one signal pair (0 = independent, 1 = locked).

    .. math::

        R^2(f, t) = \frac{\left| S(W_{xy}) \right|^2}{S(|W_x|^2)\, S(|W_y|^2)}

    :math:`W_x, W_y` are the continuous wavelet transforms, :math:`W_{xy} = W_x W_y^{*}`
    the cross-wavelet spectrum, and :math:`S` a smoothing operator in time and scale.
    Output is sorted to ascending frequency, band-limited to ``[fmin, fmax]`` and
    decimated by ``step``. The leading slice to ``len(sig1)`` is defensive: this pycwt
    pads internally for the FFT but unpads before returning, so the slice is a no-op
    until a version does not. With ``significance`` the fourth return is the
    per-frequency Monte Carlo significance level (else None).

    ``cache`` is pycwt's on-disk store of significance curves. It is keyed on the AR1
    coefficients and the wavelet grid but not on the seed or the surrogate count, and lives
    in the user's home directory, so a seeded run has to switch it off or a curve computed
    under unknown settings can stand in for the one that was asked for. Seeding itself
    happens once per run in the caller, not here.

    Backend: `pycwt.wct <https://pycwt.readthedocs.io/en/development/reference/#pycwt.wct>`_.
    """
    import pycwt

    kwargs: dict = {} if cache else {"cache": False}
    WCT, _, coi, freqs, signif = pycwt.wct(
        sig1, sig2, dt=dt,
        dj=1.0 / 12,  # 12 sub-octaves per octave (pycwt default; frequency-axis resolution)
        sig=significance, normalize=True, **kwargs,
    )
    n_sig = len(sig1)
    WCT   = WCT[:, :n_sig]
    coi   = coi[:n_sig]

    order      = np.argsort(freqs)
    freqs_s    = freqs[order]
    WCT_s      = WCT[order]
    band       = (freqs_s >= fmin) & (freqs_s <= fmax)
    WCT_band   = WCT_s[band][:, ::step].astype(np.float32)
    freqs_band = freqs_s[band]
    coi_dec    = coi[::step].astype(np.float32)

    # Per-frequency significance is constant over time: reorder + band-limit only, no decimation.
    sig_band = None
    if significance and np.ndim(signif) == 1 and len(signif) == len(freqs):
        sig_band = np.asarray(signif)[order][band].astype(np.float32)
    return WCT_band, freqs_band, coi_dec, sig_band


def _wtc_over_pairs(
    raws: dict[str, mne.io.Raw],
    signals: dict[str, dict[str, np.ndarray]],
    labels: list[str],
    fmin: float,
    fmax: float,
    significance: bool = False,
    seed: int | None = None,
) -> WTCResult:
    """Run pairwise Morlet WTC over precomputed per-subject {label: signal} maps.

    Signals are matched by label; a label absent for either subject yields None for that pair.
    Time axis decimated to ~1 Hz for display; frequency axis filtered to [fmin, fmax] Hz.
    significance adds a per-frequency Monte Carlo level to each pair (slow; ~300 surrogate runs).

    ``seed`` makes those levels reproducible. It is applied once here rather than per pair:
    the surrogates come from numpy's global legacy RNG inside pycwt, which takes no seed
    argument, and seeding every pair with one number would hand nearly identical surrogates
    to channels with similar autocorrelation, turning the Monte Carlo error into a bias
    shared by the whole montage. Seeding once lets the stream advance, so each pair draws
    fresh numbers. The cost is that a pair's level depends on how many ran before it, so
    changing the channel set moves the levels of everything after it.
    """
    subject_ids = list(raws.keys())
    ref_raw = raws[subject_ids[0]]
    sfreq   = float(ref_raw.info["sfreq"])
    dt      = 1.0 / sfreq
    step    = max(1, int(round(sfreq)))

    result_pairs: dict = {}
    shared_freqs: np.ndarray | None = None
    shared_times: np.ndarray | None = None

    rng_state = np.random.get_state() if seed is not None else None
    if seed is not None:
        np.random.seed(seed)
    try:
        for sub1, sub2 in combinations(subject_ids, 2):
            sig_map1, sig_map2 = signals[sub1], signals[sub2]
            pair_data: dict[str, dict | None] = {}
            for label in labels:
                sig1, sig2 = sig_map1.get(label), sig_map2.get(label)
                if sig1 is None or sig2 is None:
                    pair_data[label] = None
                    continue
                try:
                    WCT_band, freqs_band, coi_dec, sig_band = _pairwise_wtc(
                        sig1, sig2, dt, step, fmin, fmax, significance,
                        cache=seed is None)
                    if shared_freqs is None:
                        shared_freqs = freqs_band
                        shared_times = ref_raw.times[::step]
                    pair_data[label] = {"wtc": WCT_band, "coi": coi_dec, "sig": sig_band}
                except Exception as exc:
                    logger.warning("WTC failed %s-%s label %s: %s", sub1, sub2, label, exc)
                    pair_data[label] = None
            result_pairs[(sub1, sub2)] = pair_data
    finally:
        if rng_state is not None:
            np.random.set_state(rng_state)

    return WTCResult(
        pairs=result_pairs,
        freqs=shared_freqs if shared_freqs is not None else np.array([]),
        times=shared_times if shared_times is not None else np.array([]),
    )


def compute_wtc(
    raws: dict[str, mne.io.Raw],
    fmin: float = 0.004,
    fmax: float = 0.20,
    significance: bool = False,
    seed: int | None = None,
) -> WTCResult:
    """Compute pairwise WTC per HbO channel using pycwt Morlet wavelet.

    Channels are matched by S-D label across subjects; time axis decimated to ~1 Hz.
    significance adds a Monte Carlo significance level per pair (slow; see _wtc_over_pairs),
    and seed makes it reproducible.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for WTC")

    signals: dict[str, dict[str, np.ndarray]] = {}
    for sid, raw in raws.items():
        picks = mne.pick_types(raw.info, fnirs="hbo")
        signals[sid] = {
            raw.ch_names[p].rsplit(" ", 1)[0]: raw.get_data(picks=[p])[0].astype(np.float64)
            for p in picks
        }

    return _wtc_over_pairs(
        raws, signals, list(signals[subject_ids[0]]), fmin, fmax, significance, seed)


def _roi_averaged_signals(
    raw: mne.io.Raw,
    roi_map: dict[str, list[str]],
    bad_pairs: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Average HbO channels within each ROI → {roi_name: 1D signal}.

    bad_pairs (S-D labels without suffix) are excluded from the average.
    ROIs left with no usable channel are skipped.
    """
    bad_pairs = bad_pairs or set()
    picks = mne.pick_types(raw.info, fnirs="hbo")
    label_to_data = {
        raw.ch_names[p].rsplit(" ", 1)[0]: raw.get_data(picks=[p])[0].astype(np.float64)
        for p in picks
    }
    out: dict[str, np.ndarray] = {}
    for roi, chs in roi_map.items():
        rows = [label_to_data[c] for c in chs if c in label_to_data and c not in bad_pairs]
        if rows:
            out[roi] = np.mean(rows, axis=0)
    return out


def compute_wtc_roi(
    raws: dict[str, mne.io.Raw],
    roi_map: dict[str, list[str]],
    bad_channels: dict[str, list[str]] | None = None,
    fmin: float = 0.004,
    fmax: float = 0.20,
    significance: bool = False,
    seed: int | None = None,
) -> WTCResult:
    """Compute pairwise WTC on ROI-averaged HbO signals.

    Averages each ROI's HbO channels into one representative signal per subject,
    excluding the union of all subjects' bad_channels so every subject's ROI signal
    is built from the same channel set. Then runs the same Morlet WTC as compute_wtc.
    Returned WTCResult.pairs is keyed by ROI name instead of channel.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for WTC")

    bad_channels = bad_channels or {}
    bad_pairs_union = {
        ch.rsplit(" ", 1)[0]
        for sid in subject_ids
        for ch in bad_channels.get(sid, [])
    }
    signals = {
        sid: _roi_averaged_signals(raw, roi_map, bad_pairs_union)
        for sid, raw in raws.items()
    }

    return _wtc_over_pairs(raws, signals, list(roi_map), fmin, fmax, significance, seed)


def compute_pairwise_coherence(
    raws: dict[str, mne.io.Raw],
    fmin: float = 0.01,
    fmax: float = 0.10,
) -> pd.DataFrame:
    r"""Magnitude-squared coherence per HbO channel, averaged over [fmin, fmax] Hz.

    .. math::

        C_{xy}(f) = \frac{|P_{xy}(f)|^2}{P_{xx}(f)\, P_{yy}(f)}

    :math:`P_{xy}` is the cross-spectral density and :math:`P_{xx}, P_{yy}` the auto-spectra
    (Welch). The scalar per channel pair is :math:`C_{xy}` averaged over the band.
    Channels are matched by index; all subjects must share the same channel layout.
    Returns a DataFrame with columns: ch_name, sub1, sub2, coherence.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for pairwise coherence")

    ref_raw = raws[subject_ids[0]]
    sfreq = ref_raw.info["sfreq"]
    nperseg = min(512, max(64, ref_raw.n_times // 4))

    rows: list[dict] = []
    for sub1, sub2 in combinations(subject_ids, 2):
        raw1, raw2 = raws[sub1], raws[sub2]
        picks1 = mne.pick_types(raw1.info, fnirs="hbo")
        picks2 = mne.pick_types(raw2.info, fnirs="hbo")
        data1 = raw1.get_data(picks=picks1)
        data2 = raw2.get_data(picks=picks2)
        ch_names1 = [raw1.ch_names[p] for p in picks1]
        ch_names2 = [raw2.ch_names[p] for p in picks2]

        for i in range(min(len(picks1), len(picks2))):
            ch_label = ch_names1[i].rsplit(" ", 1)[0] if " " in ch_names1[i] else ch_names1[i]
            freqs, coh = coherence(data1[i], data2[i], fs=sfreq, nperseg=nperseg)
            mask = (freqs >= fmin) & (freqs <= fmax)
            mean_coh = float(np.mean(coh[mask])) if mask.any() else float("nan")
            rows.append({"ch_name": ch_label, "sub1": sub1, "sub2": sub2, "coherence": mean_coh})

    return pd.DataFrame(rows, columns=["ch_name", "sub1", "sub2", "coherence"])
