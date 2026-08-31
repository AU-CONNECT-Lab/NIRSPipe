"""Synchrony metrics for a hyperscanning group: wavelet coherence and band coherence.

Two ways of asking how locked two recordings are.

  compute_wtc / compute_wtc_roi   Wavelet transform coherence, resolved in both time and
                                  frequency, so a pair that only synchronises during part
                                  of the task still shows it. Per channel, or per ROI once
                                  channels have been averaged. Backed by pycwt.
  compute_pairwise_coherence      One magnitude-squared coherence number per channel pair,
                                  averaged over a band. Cheap, and enough when the question
                                  is whether a pair is locked at all.
  wtc_band_mean                   Collapses a WTC map to one number per channel, which is
                                  the form a group analysis wants.

The callers live in pipeline/hyperscanning.py, which handles the dyad bookkeeping these
functions assume has already happened: recordings loaded, aligned, trimmed to a common
length and normalised. All of them read long channels only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import mne
import numpy as np
import pandas as pd
from scipy.signal import coherence

from fnirs_pipe.io.snirf import long_channel_picks
from fnirs_pipe.utils.logging import get_logger

# the caller's logger name, kept so existing log filters still match
logger = get_logger("pipeline.hyperscanning")


def _shared_sfreq(raws: dict[str, mne.io.Raw]) -> float:
    """Sampling rate common to every recording, or an error naming the offenders.

    Every metric here reads the rate off one participant and applies it to the pair, so a
    mismatch does not fail, it silently mislabels the frequency axis of the other. Alignment
    equalises duration, not rate, and the input stage is now the caller's choice, so two
    participants can arrive resampled differently.
    """
    rates = {sid: round(float(raw.info["sfreq"]), 4) for sid, raw in raws.items()}
    if len(set(rates.values())) > 1:
        raise ValueError(
            f"recordings differ in sampling rate: {rates}. Resample them to a common rate "
            "before computing inter-brain metrics."
        )
    return next(iter(rates.values()))


def _long_hbo_by_label(raw: mne.io.Raw) -> dict[str, int]:
    """{S-D label: channel index} over long HbO channels only, bads already dropped.

    The label is the key every inter-brain metric matches on. Position cannot be: two
    participants with different channels rejected no longer agree on what index 3 is.
    """
    picks = long_channel_picks(raw, "hbo")
    if not picks:
        raise ValueError(
            "no usable long HbO channel: every one is either short-distance or marked bad"
        )
    return {raw.ch_names[p].rsplit(" ", 1)[0]: p for p in picks}


def _long_hbo_signals(raw: mne.io.Raw) -> dict[str, np.ndarray]:
    """{S-D label: HbO time course} over long channels only, bads already dropped."""
    return {
        label: raw.get_data(picks=[p])[0].astype(np.float64)
        for label, p in _long_hbo_by_label(raw).items()
    }


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


# Morlet (w0 = 6) Fourier factor: period = _FLAMBDA * scale, so frequency = 1 / period.
_FLAMBDA = 4 * np.pi / (6 + np.sqrt(2 + 6 ** 2))

# Scales of margin kept on each side of the requested band when limiting the scale range.
# pycwt smooths the coherence across neighbouring scales with a boxcar of round(2 * 0.6 / dj)
# points, 14 at the default dj, so the outermost 7 scales of whatever range is computed are
# convolved against the zero padding at the edge. Keeping more margin than that leaves every
# scale inside the band with the same neighbours it would have had.
_SCALE_MARGIN = 12


def _scale_range(dt: float, dj: float, fmin: float, fmax: float, n: int) -> tuple[float, int]:
    """(s0, J) covering [fmin, fmax] plus margin, on pycwt's own default scale grid.

    pycwt starts its scales at ``2 * dt / flambda`` and steps them by ``2 ** dj``, which puts
    grid frequency ``j`` at ``1 / (2 * dt) * 2 ** (-j * dj)``: the Nyquist frequency halved
    once per octave. Choosing s0 at one of those grid points rather than exactly at the band
    edge keeps the computed scales a subset of the default ones, so the frequencies that
    survive the band filter are the same numbers either way.
    """
    s0_default = 2 * dt / _FLAMBDA
    nyquist_ratio = 1.0 / (2 * dt)

    # first grid index whose frequency is at or below fmax, then margin above it
    k = int(np.ceil(np.log2(nyquist_ratio / fmax) / dj)) - _SCALE_MARGIN
    k = max(k, 0)
    s0 = s0_default * 2 ** (k * dj)

    # last grid index whose frequency is at or above fmin, then margin below it
    j_band = int(np.floor(np.log2(nyquist_ratio / fmin) / dj)) + _SCALE_MARGIN
    # never past what the record length supports, which is where pycwt's own default stops
    j_max = int(round(np.log2(n * dt / s0) / dj))
    return s0, max(min(j_band - k, j_max), 1)


def _pairwise_wtc(
    sig1: np.ndarray,
    sig2: np.ndarray,
    dt: float,
    step: int,
    fmin: float,
    fmax: float,
    significance: bool = False,
    cache: bool = True,
    mc_count: int = 300,
    limit_scales: bool = True,
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

    ``mc_count`` is the number of surrogate series behind each significance level and is
    what the runtime is spent on. It reaches pycwt through ``**kwargs``, so it is ignored
    unless ``significance`` is set.

    ``limit_scales`` computes only the scales the ``[fmin, fmax]`` filter is going to keep,
    plus margin, instead of pycwt's default range from ``2 * dt`` down to whatever the record
    length allows. A narrow band over a long record leaves most of the default range unused,
    and the saving is proportional. See :func:`_scale_range` for why the retained numbers do
    not change.

    Backend: `pycwt.wct <https://pycwt.readthedocs.io/en/development/reference/#pycwt.wct>`_.
    """
    import pycwt

    dj = 1.0 / 12  # 12 sub-octaves per octave (pycwt default; frequency-axis resolution)
    kwargs: dict = {} if cache else {"cache": False}
    if significance:
        kwargs["mc_count"] = mc_count
    if limit_scales:
        s0, J = _scale_range(dt, dj, fmin, fmax, len(sig1))
        kwargs.update(s0=s0, J=J)
    WCT, _, coi, freqs, signif = pycwt.wct(
        sig1, sig2, dt=dt, dj=dj, sig=significance, normalize=True, **kwargs,
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
    mc_count: int = 300,
    cross: bool = False,
    limit_scales: bool = True,
) -> WTCResult:
    """Run pairwise Morlet WTC over precomputed per-subject {label: signal} maps.

    Signals are matched by label: the first subject's ``S1_D1`` against the second's
    ``S1_D1``, and nothing else, so ``n`` labels give ``n`` results keyed by the label
    string. ``cross`` instead crosses every label with every other, ``n**2`` results keyed
    by the ``(label_sub1, label_sub2)`` tuple, of which the homologous ones are the
    diagonal. A label absent for either subject yields None for that pair.

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
    sfreq   = _shared_sfreq(raws)
    dt      = 1.0 / sfreq
    step    = max(1, int(round(sfreq)))

    result_pairs: dict = {}
    shared_freqs: np.ndarray | None = None
    shared_times: np.ndarray | None = None

    label_pairs = [(a, b) for a in labels for b in labels] if cross else [(a, a) for a in labels]

    rng_state = np.random.get_state() if seed is not None else None
    if seed is not None:
        np.random.seed(seed)
    try:
        for sub1, sub2 in combinations(subject_ids, 2):
            sig_map1, sig_map2 = signals[sub1], signals[sub2]
            pair_data: dict[str | tuple[str, str], dict | None] = {}
            for label1, label2 in label_pairs:
                key = (label1, label2) if cross else label1
                sig1, sig2 = sig_map1.get(label1), sig_map2.get(label2)
                if sig1 is None or sig2 is None:
                    pair_data[key] = None
                    continue
                try:
                    WCT_band, freqs_band, coi_dec, sig_band = _pairwise_wtc(
                        sig1, sig2, dt, step, fmin, fmax, significance,
                        cache=seed is None, mc_count=mc_count,
                        limit_scales=limit_scales)
                    if shared_freqs is None:
                        shared_freqs = freqs_band
                        shared_times = ref_raw.times[::step]
                    pair_data[key] = {"wtc": WCT_band, "coi": coi_dec, "sig": sig_band}
                except Exception as exc:
                    logger.warning("WTC failed %s-%s label %s: %s", sub1, sub2, key, exc)
                    pair_data[key] = None
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
    mc_count: int = 300,
    cross: bool = False,
    limit_scales: bool = True,
) -> WTCResult:
    """Compute pairwise WTC per long HbO channel using pycwt Morlet wavelet.

    Channels are matched by S-D label across subjects; time axis decimated to ~1 Hz.
    Short-distance channels are excluded (see long_channel_picks), as are bads.
    significance adds a Monte Carlo significance level per pair (slow; see _wtc_over_pairs),
    and seed makes it reproducible.

    ``cross`` crosses every channel with every other rather than pairing like with like, so
    n channels give n**2 results keyed by ``(label_sub1, label_sub2)`` instead of n keyed by
    the label. It tests whether one person's channel couples to a different site on the
    other's head. The cost is quadratic in the channel count and single channels are noisier
    than the ROI averages ``compute_wtc_roi`` crosses, so the off-diagonal is exploratory.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for WTC")

    signals = {sid: _long_hbo_signals(raw) for sid, raw in raws.items()}

    return _wtc_over_pairs(
        raws, signals, list(signals[subject_ids[0]]), fmin, fmax, significance, seed,
        mc_count, cross, limit_scales)


def _roi_averaged_signals(
    raw: mne.io.Raw,
    roi_map: dict[str, list[str]],
    bad_pairs: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Average long HbO channels within each ROI → {roi_name: 1D signal}.

    bad_pairs (S-D labels without suffix) are excluded from the average, as are any short
    channels an ROI happens to list. ROIs left with no usable channel are skipped.
    """
    bad_pairs = bad_pairs or set()
    label_to_data = _long_hbo_signals(raw)
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
    mc_count: int = 300,
    cross: bool = False,
    limit_scales: bool = True,
) -> WTCResult:
    """Compute pairwise WTC on ROI-averaged HbO signals.

    Averages each ROI's HbO channels into one representative signal per subject,
    excluding the union of all subjects' bad_channels so every subject's ROI signal
    is built from the same channel set. Then runs the same Morlet WTC as compute_wtc.
    Returned WTCResult.pairs is keyed by ROI name instead of channel.

    ``cross`` crosses each subject's ROIs with the other's rather than pairing like with
    like, so four ROIs give sixteen results keyed by ``(roi_sub1, roi_sub2)`` instead of
    four keyed by ROI name. Off-diagonal entries are what tells you whether one person's
    PFC couples to the other's TPJ. Averaging within an ROI first is what makes this
    affordable: the same crossing over raw channels is quadratic in a much larger number.
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

    return _wtc_over_pairs(
        raws, signals, list(roi_map), fmin, fmax, significance, seed, mc_count, cross,
        limit_scales)


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
    Long channels only, matched across participants by S-D label; a label one of them lacks
    keeps its row with a NaN coherence.
    Returns a DataFrame with columns: ch_name, sub1, sub2, coherence.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for pairwise coherence")

    ref_raw = raws[subject_ids[0]]
    sfreq = _shared_sfreq(raws)
    nperseg = min(512, max(64, ref_raw.n_times // 4))

    rows: list[dict] = []
    for sub1, sub2 in combinations(subject_ids, 2):
        raw1, raw2 = raws[sub1], raws[sub2]
        map1, map2 = _long_hbo_by_label(raw1), _long_hbo_by_label(raw2)
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
            freqs, coh = coherence(data1[i], data2[j], fs=sfreq, nperseg=nperseg)
            mask = (freqs >= fmin) & (freqs <= fmax)
            mean_coh = float(np.mean(coh[mask])) if mask.any() else float("nan")
            rows.append({"ch_name": label, "sub1": sub1, "sub2": sub2, "coherence": mean_coh})

    return pd.DataFrame(rows, columns=["ch_name", "sub1", "sub2", "coherence"])


def wtc_band_mean(
    result: WTCResult,
    fmin: float,
    fmax: float,
    mask_coi: bool = True,
) -> pd.DataFrame:
    r"""Collapse each WTC map to one number per pair and label: the band mean inside the COI.

    A time-frequency map is what you look at; a single number per channel is what enters a
    group analysis. This averages :math:`R^2(f, t)` over the frequencies of ``[fmin, fmax]``
    and over time,

    .. math::

        \overline{R^2} = \frac{1}{|V|} \sum_{(f, t) \in V} R^2(f, t),

    where :math:`V` is the set of cells inside the band and, with ``mask_coi``, inside the
    cone of influence. pycwt reports the COI as the longest period still free of edge effects
    at each time point, so a cell is kept when :math:`f \ge 1 / \mathrm{coi}(t)`. Cells outside
    it are wavelet coefficients padded against the edges of the record: near 1 whatever the
    data does, and enough of them at the low-frequency end to carry a whole row.

    ``n_valid_frac`` is the share of band cells that survived the mask, so a value resting on
    a handful of time points is visible instead of implied. It is 1.0 when ``mask_coi`` is off.

    Returns one row per (pair, label) with columns sub1, sub2, label, coherence, n_valid_frac.
    A crossed result (see ``compute_wtc_roi``) is keyed by a label pair rather than one label,
    and gains a ``label2`` column after ``label``: ``label`` is what sub1 contributed, ``label2``
    what sub2 did, and the homologous rows are the ones where they agree.
    Labels that failed to compute keep their row, with NaN coherence and n_valid_frac 0.
    """
    freqs = np.asarray(result.freqs, dtype=float)
    band  = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        span = f"{freqs.min():.4f}-{freqs.max():.4f} Hz" if freqs.size else "empty"
        raise ValueError(
            f"no WTC frequency bin inside [{fmin}, {fmax}] Hz; the computed axis spans {span}. "
            "Widen the band, or recompute the WTC over a wider fmin/fmax."
        )
    band_freqs = freqs[band]

    crossed = any(isinstance(k, tuple)
                  for labels in result.pairs.values() for k in labels)

    rows: list[dict] = []
    for (sub1, sub2), labels in result.pairs.items():
        for label, data in labels.items():
            label1, label2 = label if isinstance(label, tuple) else (label, label)
            head = {"sub1": sub1, "sub2": sub2, "label": label1}
            if crossed:
                head["label2"] = label2
            if data is None:
                rows.append({**head, "coherence": float("nan"), "n_valid_frac": 0.0})
                continue

            wtc = np.asarray(data["wtc"], dtype=float)[band]
            if mask_coi:
                coi = np.asarray(data["coi"], dtype=float)
                # coi is a period in seconds; 1/coi is the lowest frequency still reliable at
                # that time. A coi of 0 (the very edges) leaves nothing reliable there.
                with np.errstate(divide="ignore"):
                    f_edge = np.where(coi > 1e-10, 1.0 / coi, np.inf)
                wtc = np.where(band_freqs[:, None] >= f_edge[None, :], wtc, np.nan)

            valid = np.isfinite(wtc)
            n_valid_frac = float(valid.mean()) if valid.size else 0.0
            rows.append({
                **head,
                "coherence": float(wtc[valid].mean()) if valid.any() else float("nan"),
                "n_valid_frac": n_valid_frac,
            })

    columns = ["sub1", "sub2", "label"] + (["label2"] if crossed else [])
    return pd.DataFrame(rows, columns=columns + ["coherence", "n_valid_frac"])
