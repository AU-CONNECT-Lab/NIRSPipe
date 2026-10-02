"""Wavelet transform coherence: the transform, and the numbers a finished map collapses to.

  compute_wtc          Coherence resolved in both time and frequency, per channel pair, so a
                       pair that only synchronises during part of the task still shows it.
                       Backed by pycwt.
  window_result        Cuts a finished map to a stretch of it, which is not the same number
                       as transforming that stretch on its own.
  wtc_band_mean        Collapses a map to one value per channel pair, the form a group
                       analysis wants.
  wtc_phase_by_scale   The same collapse kept per scale, for the phase.

The callers live in pipeline/hyper/__init__.py, which handles the dyad bookkeeping these
functions assume has already happened: recordings loaded, aligned, trimmed to a common
length and normalised. All of them read long channels only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import mne
import numpy as np
import pandas as pd
from scipy.fft import next_fast_len
from scipy.signal import convolve2d

from fnirs_pipe.pipeline.hyper._helpers import _long_signals, _shared_sfreq, long_axis_over
from fnirs_pipe.utils import fisher_r_to_z
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.wtc")


@dataclass
class WTCResult:
    """Pairwise wavelet transform coherence per HbO channel.

    ::

      pairs[(sub1, sub2)][ch_name] = {"wtc": ndarray(n_freqs, n_times),
                                       "coi": ndarray(n_times),
                                       "phase": ndarray(n_freqs, n_times)}

    freqs: ascending Hz.  times: decimated aligned time axis (seconds).
    """
    pairs: dict
    freqs: np.ndarray
    times: np.ndarray


# Morlet (w0 = 6) Fourier factor: period = _FLAMBDA * scale, so frequency = 1 / period.
_FLAMBDA = 4 * np.pi / (6 + np.sqrt(2 + 6 ** 2))

# Sub-octaves per octave, pycwt's default. Fixed because _SCALE_MARGIN is sized for it;
# reported in every WTC sidecar.
WTC_DJ = 1.0 / 12

# Width of the coherence's scale-direction boxcar, in log2(scale) units, so it spans
# _SCALE_SMOOTH_DJ0 / dj grid steps: 7.2 at the default dj. pycwt's own Morlet spans twice
# this; see _morlet.
_SCALE_SMOOTH_DJ0 = 0.6

# Scales of margin kept on each side of the requested band when limiting the scale range.
# The scale smoothing reaches _SCALE_SMOOTH_DJ0 / (2 * dj) scales either way, 3.6 at the
# default dj, so the outermost few scales of whatever range is computed are convolved against
# the zero padding at the edge. Keeping more margin than that leaves every scale inside the
# band with the same neighbours it would have had.
_SCALE_MARGIN = 12


def _scale_window(dj: float) -> np.ndarray:
    """The scale-direction boxcar, ``_SCALE_SMOOTH_DJ0 / dj`` grid steps wide, summing to 1.

    ::

      _scale_window(1 / 12)  ->  [0.014, 0.139 x 7, 0.014]      (7.2 steps over 9 points)

    A width of 0.6 in log2(scale) is 0.6 / dj grid steps, which is not a whole number of
    them, so each tap takes the share of its own grid cell the boxcar actually covers: 1 for
    the cells wholly inside, the remainder for the two it ends in. The length is a property
    of the grid, the width is the one the definition fixes.
    """
    half = _SCALE_SMOOTH_DJ0 / (2 * dj)
    reach = max(0, int(np.ceil(half - 0.5)))
    offsets = np.arange(-reach, reach + 1)
    win = np.clip(half - (np.abs(offsets) - 0.5), 0.0, 1.0)
    return win / win.sum()


def _fft_length(n: int, dt: float, s_max: float) -> int:
    """Samples to transform over: enough padding to clear the widest wavelet, never more than
    the power of two pycwt would round up to.

    ::

      _fft_length(39611, 0.0983, 183.9)  ->  45360, where pycwt rounds up to 65536

    Padding stops the circular convolution wrapping the end of the record onto its start, so
    what it has to clear is how far the widest wavelet reaches in from an edge. That is
    :func:`cone_margin_s` at the **largest computed scale**, which sits ``_SCALE_MARGIN``
    scales below ``fmin`` rather than at it. The minimum keeps a recording shorter than its
    own padding from being transformed over more samples than pycwt would use.
    """
    margin = int(np.ceil(cone_margin_s(1.0 / (_FLAMBDA * s_max)) / dt))
    return min(next_fast_len(n + margin), 1 << (max(n, 1) - 1).bit_length())


def _morlet():
    r"""The mother wavelet every coherence here is computed with.

    A Morlet whose smoothing operator :math:`S` spans ``_SCALE_SMOOTH_DJ0`` in log2(scale),
    half the width of pycwt's own.

    Returned as an instance rather than a class so the scale window is built once per
    transform, and named apart from the stock Morlet because pycwt keys its Monte Carlo
    significance cache on the name.
    """
    import pycwt
    from pycwt.helpers import fft

    class _Morlet(pycwt.Morlet):
        def __init__(self):
            super().__init__()
            self.name = "Morlet-dj0"
            # set by _wavelet_grid so the transform and its smoothing share one length; left
            # None when pycwt drives, where it is read off the scales instead
            self.n_fft = None

        def smooth(self, W, dt, dj, scales):
            n = W.shape[1]
            pad = {"n": self.n_fft or _fft_length(n, dt, float(np.max(scales)))}
            # time: the wavelet's own Gaussian envelope per scale, applied in Fourier
            k2 = (2 * np.pi * fft.fftfreq(pad["n"])) ** 2
            F = np.exp(-0.5 * (np.asarray(scales)[:, None] / dt) ** 2 * k2)
            T = fft.ifft(F * fft.fft(W, axis=1, **pad), axis=1, **pad)[:, :n]
            if np.isreal(W).all():
                T = T.real
            return convolve2d(T, _scale_window(dj)[:, None], "same")

    return _Morlet()


def cone_margin_s(band_fmin: float, factor: float = 2.0) -> float:
    r"""Seconds of recording to keep either side of a window so its band is inside the cone.

    ::

      cone_margin_s(0.06)  ->  47.1     cone_margin_s(0.02)  ->  141.4

    A wavelet coefficient near the end of a series is computed partly against the padding
    beyond it, and the cone of influence is where that contamination has decayed to
    :math:`e^{-2}`. For the Morlet wavelet the cone reaches :math:`\sqrt{2}\,P` in from each
    edge at period :math:`P`, so the lowest frequency a band mean uses, ``band_fmin``, is the
    one that reaches furthest:

    .. math::

        \mathrm{margin} = \mathrm{factor} \times \frac{\sqrt{2}}{f_{\min}}

    Returned in seconds, so a caller crops ``[t0 - margin, t1 + margin]`` and windows the
    result back to ``[t0, t1]``.
    """
    if band_fmin <= 0:
        raise ValueError(f"band_fmin must be positive, got {band_fmin}")
    return float(factor) * np.sqrt(2.0) / float(band_fmin)


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


def _decim_step(sfreq: float) -> int:
    """Samples per retained column of a coherence map: one per second, at least one.

    It bounds how short a window --wtc-by-condition can describe, and is reported in the
    sidecar as a time resolution.
    """
    return max(1, int(round(sfreq)))


def wtc_grid_params(raws: dict) -> dict:
    """The wavelet grid the maps sit on, for a sidecar: fixed, but reportable.

    wtc_grid_params(raws_at_10_Hz)
      -> {"wtc_dj": 0.0833, "wtc_time_step_s": 1.0, "wtc_scale_smooth_dj0": 0.6}

    None is configurable, and all three change what a band mean is an average over, so a
    reader comparing two studies' coherences needs them on the file. Read off the
    recordings rather than passed in, so they cannot disagree with what ran.
    """
    sfreq = _shared_sfreq(raws)
    return {"wtc_dj": round(WTC_DJ, 6),
            "wtc_time_step_s": round(_decim_step(sfreq) / sfreq, 6),
            "wtc_scale_smooth_dj0": _SCALE_SMOOTH_DJ0}


@dataclass
class _WaveletGrid:
    """The scale grid every channel of one transform shares, built once per run."""
    mother: object
    s0: float
    J: int
    sj: np.ndarray
    scales: np.ndarray
    n_fft: int


@dataclass
class _ChannelWavelet:
    """One signal's transform and smoothed auto spectrum, reusable across its pairings."""
    W: np.ndarray
    S: np.ndarray
    coi: np.ndarray
    freqs: np.ndarray


def _wavelet_grid(dt: float, n: int, fmin: float, fmax: float,
                  limit_scales: bool) -> _WaveletGrid:
    """The scale grid pycwt would pick for signals of length ``n``, resolved up front."""
    mother = _morlet()
    if limit_scales:
        s0, J = _scale_range(dt, WTC_DJ, fmin, fmax, n)
    else:
        s0 = 2 * dt / mother.flambda()
        J = int(np.round(np.log2(n * dt / s0) / WTC_DJ))
    sj = s0 * 2 ** (np.arange(0, J + 1) * WTC_DJ)
    n_fft = _fft_length(n, dt, float(sj[-1]))
    mother.n_fft = n_fft
    return _WaveletGrid(mother=mother, s0=s0, J=J, sj=sj,
                        scales=np.ones([1, n]) * sj[:, None], n_fft=n_fft)


def _cwt(sig: np.ndarray, dt: float, grid: _WaveletGrid):
    r"""The continuous wavelet transform :math:`W_x`, on the grid's own padding length.

    The same arithmetic in the same order as ``pycwt.cwt``; only the number of samples
    transformed over differs, and :func:`_fft_length` picks one that leaves every returned
    value unchanged. Owned here rather than monkeypatched into pycwt, which chooses its length
    inside its own module and would go back to a power of two without saying so.

    Returns ``(W, sj, freqs, coi)``.
    """
    from pycwt.helpers import fft

    n0 = len(sig)
    n_fft = grid.n_fft
    ftfreqs = 2 * np.pi * fft.fftfreq(n_fft, dt)
    sj_col = grid.sj[:, None]
    psi_ft_bar = ((sj_col * ftfreqs[1] * n_fft) ** 0.5
                  * np.conjugate(grid.mother.psi_ft(sj_col * ftfreqs)))
    W = fft.ifft(fft.fft(sig, n=n_fft) * psi_ft_bar, axis=1, n=n_fft)

    sj = grid.sj
    freqs = 1 / (grid.mother.flambda() * sj)
    # pycwt drops any scale whose row came back all-NaN, and the caller's grid check reads
    # the shortened sj to catch it
    sel = np.invert(np.isnan(W).all(axis=1))
    if np.any(sel):
        sj, freqs, W = sj[sel], freqs[sel], W[sel, :]

    coi = n0 / 2 - np.abs(np.arange(0, n0) - (n0 - 1) / 2)
    coi = grid.mother.flambda() * grid.mother.coi() * dt * coi
    return W[:, :n0], sj, freqs, coi


def _prepare_channel(sig: np.ndarray, dt: float, grid: _WaveletGrid) -> _ChannelWavelet:
    r"""One signal's half of the coherence: :math:`W_x` and the smoothed :math:`S(|W_x|^2)`.

    Computed once per channel rather than once per pairing, which is what crossing repeats.
    """
    if len(sig) != grid.scales.shape[1]:
        raise ValueError(f"signal length {len(sig)} differs from the grid's "
                         f"{grid.scales.shape[1]}")
    y = (sig - sig.mean()) / sig.std()
    W, sj, freqs, coi = _cwt(y, dt, grid)
    # pycwt drops any scale whose row came back all-NaN, which would leave two channels on
    # different grids and misalign the cross spectrum without changing its shape
    if not np.array_equal(sj, grid.sj):
        raise ValueError("wavelet scales differ from the run's grid")
    S = grid.mother.smooth(np.abs(W) ** 2 / grid.scales, dt, WTC_DJ, sj)
    return _ChannelWavelet(W=W, S=S, coi=coi, freqs=freqs)


def _pair_from_prepared(p1: _ChannelWavelet, p2: _ChannelWavelet, dt: float,
                        grid: _WaveletGrid) -> tuple[np.ndarray, np.ndarray]:
    r"""The half that needs both signals: the cross spectrum, its smoothing and the ratio.

    :math:`R^2 = |S(W_{xy})|^2 / (S(|W_x|^2)\, S(|W_y|^2))`, phase taken from
    :math:`W_{xy} = W_x W_y^{*}`.
    """
    W12 = p1.W * p2.W.conj()
    phase = np.angle(W12)
    # in place, and after the phase: the cross spectrum is the largest array a pairing holds
    W12 /= grid.scales
    S12 = grid.mother.smooth(W12, dt, WTC_DJ, grid.sj)
    return np.abs(S12) ** 2 / (p1.S * p2.S), phase


def _trim_pair(
    WCT: np.ndarray,
    aWCT: np.ndarray,
    coi: np.ndarray,
    freqs: np.ndarray,
    n_sig: int,
    step: int,
    fmin: float,
    fmax: float,
    signif: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Sort to ascending frequency, band-limit to ``[fmin, fmax]`` and decimate by ``step``.

    The leading slice to ``n_sig`` is defensive: this pycwt pads internally for the FFT but
    unpads before returning, so the slice is a no-op until a version does not.
    """
    WCT  = WCT[:, :n_sig]
    aWCT = aWCT[:, :n_sig]
    coi  = coi[:n_sig]

    order      = np.argsort(freqs)
    freqs_s    = freqs[order]
    WCT_s      = WCT[order]
    band       = (freqs_s >= fmin) & (freqs_s <= fmax)
    WCT_band   = WCT_s[band][:, ::step].astype(np.float32)
    phase_band = aWCT[order][band][:, ::step].astype(np.float32)
    freqs_band = freqs_s[band]
    coi_dec    = coi[::step].astype(np.float32)

    # Per-frequency significance is constant over time: reorder + band-limit only, no decimation.
    sig_band = None
    if signif is not None and np.ndim(signif) == 1 and len(signif) == len(freqs):
        sig_band = np.asarray(signif)[order][band].astype(np.float32)
    return WCT_band, freqs_band, coi_dec, sig_band, phase_band


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
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

    The fifth is the relative phase in radians, on the same grid: 0 points right and means
    the two are in phase, :math:`\pi` points left and means antiphase, and a quarter turn up
    means the first signal leads the second by a quarter cycle. It is what the arrows on a
    coherence map are drawn from, and coherence alone cannot distinguish a pair that moves
    together from one that moves together a few seconds apart.

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

    Backend: ``pycwt.wct``.
    """
    import pycwt

    dj = WTC_DJ
    kwargs: dict = {} if cache else {"cache": False}
    if significance:
        kwargs["mc_count"] = mc_count
    if limit_scales:
        s0, J = _scale_range(dt, dj, fmin, fmax, len(sig1))
        kwargs.update(s0=s0, J=J)
    # the same mother the fast path uses, and pycwt hands it to its Monte Carlo too, so the
    # significance level is drawn under the smoothing the coherence was computed with
    WCT, aWCT, coi, freqs, signif = pycwt.wct(
        sig1, sig2, dt=dt, dj=dj, sig=significance, normalize=True,
        wavelet=_morlet(), **kwargs,
    )
    return _trim_pair(WCT, aWCT, coi, freqs, len(sig1), step, fmin, fmax,
                      signif if significance else None)


def _check_cached_grid(cache1: "dict[tuple[str, str], _ChannelWavelet]", subject: str,
                       grid: _WaveletGrid) -> None:
    """Raise if a carried transform was built on a different scale grid or signal length."""
    want = (grid.sj.size, grid.scales.shape[1])
    for (sub, label), prep in cache1.items():
        if sub == subject and prep.W.shape != want:
            raise ValueError(f"cached transform for {sub} {label} was built on a different "
                             f"grid: {prep.W.shape} against {want}")


def _wtc_over_pairs(
    raws: dict[str, mne.io.Raw],
    signals: dict[str, dict[str, np.ndarray]],
    fmin: float,
    fmax: float,
    significance: bool = False,
    seed: int | None = None,
    mc_count: int = 300,
    cross: bool = False,
    limit_scales: bool = True,
    axis: "list[str] | None" = None,
    cache1: "dict[tuple[str, str], _ChannelWavelet] | None" = None,
) -> WTCResult:
    """Run pairwise Morlet WTC over precomputed per-subject {label: signal} maps.

    Signals are matched by label: the first subject's ``S1_D1`` against the second's
    ``S1_D1``, and nothing else, so the homologous set is the labels both subjects kept.
    ``cross`` instead crosses every label of one with every label of the other, keyed by the
    ``(label_sub1, label_sub2)`` tuple, of which the homologous ones are the diagonal.

    ``axis`` is the montage's labels, rejections included, and is what the keys are drawn
    from: a label one member has no usable channel at gets its key with ``None`` behind it
    rather than no key at all, so every dyad's table has one shape (the ``exclude=[]`` rule
    :func:`~fnirs_pipe.io.snirf.long_channel_picks` documents). Left at None the keys come
    from the surviving channels, which is the shape a caller with no montage to hand can
    produce.

    **Each side contributes its own surviving channels**, so a channel only the second
    subject kept still reaches its pairings and the result does not depend on which member
    is listed first.

    Time axis decimated to ~1 Hz for display; frequency axis filtered to [fmin, fmax] Hz.
    significance adds a per-frequency Monte Carlo level to each pair (slow; ~300 surrogate runs).

    Each channel is transformed once and reused across its pairings
    (:func:`_prepare_channel`), which is where the time goes when ``cross`` squares the pair
    count. The significance path stays on :func:`_pairwise_wtc`, whose cost is the surrogates.

    ``cache1`` extends that reuse across *calls*, for a caller that runs this many times over
    an unchanged first side: pass a dict and the first subject's transforms are read from it
    and written back to it, keyed ``(subject, label)``. :func:`compute_wtc_phase_null` is the one
    caller, and it is where the saving is, since it scrambles only the second side. Left at
    None the first side is prepared once per label and dropped, which is what a single pass
    needs and what keeps a crossed run from holding a second montage of transforms.

    ``seed`` makes those levels reproducible. It seeds pycwt's Monte Carlo only;
    :func:`compute_wtc_phase_null` takes the same number for its own surrogate generator, so one
    value makes a whole run reproducible without the two sharing a stream.
    It seeds numpy's global RNG once per call, not per pair, since per-pair seeding would
    hand channels with similar autocorrelation near-identical surrogates. A pair's level
    therefore depends on how many pairs ran before it, so changing the channel set moves the
    levels of everything after it.
    """
    subject_ids = list(raws.keys())
    ref_raw = raws[subject_ids[0]]
    sfreq   = _shared_sfreq(raws)
    dt      = 1.0 / sfreq
    step    = _decim_step(sfreq)

    result_pairs: dict = {}
    shared_freqs: np.ndarray | None = None
    shared_times: np.ndarray | None = None

    # one grid for the whole run, so a channel's transform can be reused across its pairings
    grid = None
    if not significance:
        first = next((s for m in signals.values() for s in m.values() if s is not None), None)
        if first is not None:
            grid = _wavelet_grid(dt, len(first), fmin, fmax, limit_scales)

    rng_state = np.random.get_state() if seed is not None else None
    if seed is not None:
        np.random.seed(seed)
    try:
        for sub1, sub2 in combinations(subject_ids, 2):
            sig_map1, sig_map2 = signals[sub1], signals[sub2]
            # the montage when it was given, so a rejection blanks a row or a column of the
            # result instead of shrinking it. Each side keeps its own axis in the fallback,
            # so a channel only the second subject kept still reaches its pairings
            axis1 = axis if axis is not None else list(sig_map1)
            axis2 = axis if axis is not None else list(sig_map2)
            label_pairs = ([(a, b) for a in axis1 for b in axis2] if cross
                           else [(a, a) for a in axis1
                                 if axis is not None or a in sig_map2])
            pair_data: dict[str | tuple[str, str], dict | None] = {}
            n_blank = 0
            # crossing revisits every label of side 2 once per label of side 1, and never
            # revisits side 1, so within one call only side 2 is worth holding on to.
            # Across calls it is the other way round, which is what `cache1` is for
            cache2: dict[str, _ChannelWavelet] = {}
            prep1_label: str | None = None
            prep1: _ChannelWavelet | None = None
            if cache1 is not None and grid is not None:
                # checked here rather than at the lookup: a stale entry is the caller's
                # mistake and applies to every pairing, where the loop's own `except` is for
                # one pair's data. _prepare_channel's grid checks cannot run on a reused
                # transform, and a stale one misaligns the cross spectrum without changing
                # its shape
                _check_cached_grid(cache1, sub1, grid)
            for label1, label2 in label_pairs:
                key = (label1, label2) if cross else label1
                sig1, sig2 = sig_map1.get(label1), sig_map2.get(label2)
                if sig1 is None or sig2 is None:
                    # a rejection, not a failure: the key is kept so the row survives, and
                    # counted rather than warned about one pair at a time, since crossing
                    # turns two rejected channels into fifty-odd blank pairings
                    pair_data[key] = None
                    n_blank += 1
                    continue
                try:
                    if grid is None:
                        WCT_band, freqs_band, coi_dec, sig_band, phase_band = _pairwise_wtc(
                            sig1, sig2, dt, step, fmin, fmax, significance,
                            cache=seed is None, mc_count=mc_count,
                            limit_scales=limit_scales)
                    else:
                        if cache1 is None:
                            if label1 != prep1_label:
                                prep1 = _prepare_channel(sig1, dt, grid)
                                prep1_label = label1
                        else:
                            prep1 = cache1.get((sub1, label1))
                            if prep1 is None:
                                prep1 = _prepare_channel(sig1, dt, grid)
                                cache1[(sub1, label1)] = prep1
                        prep2 = cache2.get(label2)
                        if prep2 is None:
                            prep2 = _prepare_channel(sig2, dt, grid)
                            if cross:
                                cache2[label2] = prep2
                        WCT, aWCT = _pair_from_prepared(prep1, prep2, dt, grid)
                        WCT_band, freqs_band, coi_dec, sig_band, phase_band = _trim_pair(
                            WCT, aWCT, prep1.coi, prep1.freqs, len(sig1), step, fmin, fmax)
                    if shared_freqs is None:
                        shared_freqs = freqs_band
                        shared_times = ref_raw.times[::step]
                    pair_data[key] = {"wtc": WCT_band, "coi": coi_dec, "sig": sig_band,
                                      "phase": phase_band}
                except Exception as exc:
                    logger.warning("WTC failed %s-%s label %s: %s", sub1, sub2, key, exc)
                    pair_data[key] = None
            if n_blank:
                logger.info("%s-%s: %d of %d pairing(s) left blank, one side having no "
                            "usable channel there", sub1, sub2, n_blank, len(label_pairs))
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
    ch_type: str = "hbo",
    sep_bands=None,
) -> WTCResult:
    """Compute pairwise WTC per long channel of one chromophore, using a Morlet wavelet.

    Channels are matched by S-D label across subjects; time axis decimated to ~1 Hz.
    Short-distance channels are excluded (see long_channel_picks), as are bads.
    significance adds a Monte Carlo significance level per pair (slow; see _wtc_over_pairs),
    and seed makes it reproducible.

    ``ch_type`` is one chromophore, "hbo" or "hbr". Running both is this function called
    twice and costs exactly twice as much: nothing about the statistic changes, and the two
    never mix, since a member's HbO pairs only with the other member's HbO.

    ``cross`` crosses every channel with every other rather than pairing like with like, so
    n channels give n**2 results keyed by ``(label_sub1, label_sub2)`` instead of n keyed by
    the label. The cost is quadratic in the channel count, and it is also what
    :func:`roi_mean_of_channels` needs to build the cross-ROI matrix.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for WTC")

    signals = {sid: _long_signals(raw, ch_type, sep_bands) for sid, raw in raws.items()}

    return _wtc_over_pairs(
        raws, signals, fmin, fmax, significance, seed, mc_count, cross, limit_scales,
        axis=long_axis_over(raws.values(), ch_type, sep_bands))


def window_result(result: WTCResult, tstart: float, tstop: float) -> WTCResult:
    """The same WTC restricted to a time window, so a band mean over it describes one condition.

    ::

      a 1200 s result + (300, 600)  ->  the same maps holding only those 300 s

    This is how a condition is read out of a whole-record transform, and it is not the same
    number as transforming that condition on its own: a cut window has two edges of its own,
    so more of its band cells fall outside the cone. Windowing carries the whole record's
    cone instead, which only reaches into the ends of the recording.

    ``sig`` is carried through unchanged: a Monte Carlo level is per frequency and constant
    over time, so a window of it is itself.
    """
    times = np.asarray(result.times, dtype=float)
    keep = (times >= float(tstart)) & (times <= float(tstop))
    if not keep.any():
        raise ValueError(
            f"window [{tstart}, {tstop}] holds no sample of a result spanning "
            f"{times.min():.1f}-{times.max():.1f} s"
        )
    pairs = {
        pair_key: {
            label: (None if data is None else {
                "wtc": np.asarray(data["wtc"])[:, keep],
                "coi": np.asarray(data["coi"])[keep],
                "sig": data.get("sig"),
                **({"sig_source": data["sig_source"]} if "sig_source" in data else {}),
                "phase": np.asarray(data["phase"])[:, keep],
            })
            for label, data in labels.items()
        }
        for pair_key, labels in result.pairs.items()
    }
    return WTCResult(pairs=pairs, freqs=result.freqs, times=times[keep])


def _circular_stats(angles: np.ndarray) -> tuple[float, float, int]:
    r"""Circular mean and spread of a set of phase angles, in radians, and how many there were.

    ::

      [10 deg, 350 deg]  ->  (0 deg, 0.42 rad, 2), not the 180 deg an arithmetic mean gives

    .. math::

        \bar{a} = \arg(X, Y), \quad X = \sum_i \cos a_i, \quad Y = \sum_i \sin a_i

    and the spread is the circular standard deviation :math:`s = \sqrt{-2 \ln(R/n)}` with
    :math:`R = \sqrt{X^2 + Y^2}`, which is 0 when every angle agrees and grows without bound
    as they spread round the circle. Both are the forms Grinsted et al. (2004) define for a
    wavelet phase.

    An empty input, or one with no finite angle, gives NaN at ``n = 0``.
    """
    a = np.asarray(angles, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), 0
    X, Y = float(np.cos(a).sum()), float(np.sin(a).sum())
    # R/n is the resultant length. Floored so a set spread evenly round the circle reports a
    # large spread rather than dividing by zero, and capped at 1 because angles that all
    # agree land a hair above it and would take the square root of a negative
    resultant = min(max(float(np.hypot(X, Y)) / a.size, 1e-12), 1.0)
    return float(np.arctan2(Y, X)), float(np.sqrt(-2.0 * np.log(resultant))), int(a.size)


def _band_rows(sig, band: np.ndarray) -> "np.ndarray | None":
    """``sig`` cut to the band's rows, or None when no level was computed or it is the wrong length."""
    if sig is None:
        return None
    arr = np.asarray(sig, dtype=float)
    return arr[band] if arr.shape[:1] == band.shape else None


def _phase_cells(
    wtc: np.ndarray, in_coi: np.ndarray, sig: "np.ndarray | None", mask_coi: bool,
) -> np.ndarray:
    """Which band cells a phase angle may be averaged over: inside the cone, above the level.

    The relative phase of two uncorrelated series is a uniformly random direction, so a mean
    taken over cells that are not coupled measures the shape of the map rather than a lead.
    ``sig`` is the per-frequency Monte Carlo level when one was computed, and is applied on
    the band rows; without one only the cone is required and the caller reads ``phase_n`` to
    see how weak the mask was.
    """
    keep = in_coi if mask_coi else np.ones(wtc.shape, dtype=bool)
    if sig is not None and np.asarray(sig).shape[:1] == wtc.shape[:1]:
        keep = keep & (wtc >= np.asarray(sig, dtype=float)[:, None])
    return keep


def wtc_band_mean(
    result: WTCResult,
    fmin: float,
    fmax: float,
    mask_coi: bool = True,
) -> pd.DataFrame:
    r"""Collapse each WTC map to one number per pair and label: the band mean.

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

    ``mask_coi`` is **on by default**; cells outside the cone are padding. Masking discards
    more of a short segment than of a long one, so ``n_valid_frac`` is reported either way.
    ``--no-wtc-mask-coi`` averages the whole band.

    ``n_valid_frac`` is the share of band cells that lie inside the cone of influence. **It is
    reported whether or not the mask is applied**, so the share is visible as a quality number
    even when every cell was averaged.

    ``coherence_z`` is the Fisher r-to-z of ``coherence`` (:math:`\operatorname{arctanh}`,
    clipped just below 1).

    ``phase_angle`` is the circular mean of the relative phase over the same band, in degrees,
    positive meaning the first member leads; ``phase_sd`` is its circular standard deviation,
    also in degrees and unbounded above; ``phase_n`` is how many cells they were taken over.
    Coherence says whether a pair is locked, the angle says which of them leads, and the two
    are independent: a pair can be fully coherent at any fixed lag. The cells are chosen by
    :func:`_phase_cells`, which is a stricter mask than the coherence uses, and the angle is
    only readable where ``phase_sd`` is small. **It is a band mean, so it cannot be divided by
    the frequency to get a lag in seconds**; :func:`wtc_phase_by_scale` is that number.

    Returns one row per (pair, label) with columns sub1, sub2, label, coherence, coherence_z,
    n_valid_frac, phase_angle, phase_sd, phase_n. A crossed result is keyed by a label pair
    rather than one label, and gains a
    ``label2`` column after ``label``: ``label`` is what sub1 contributed, ``label2`` what sub2
    did, and the homologous rows are the ones where they agree.
    A label with no map behind it keeps its row, with NaN in every measured column. That
    covers both a pairing one member had no usable channel at and one the transform failed
    on. ``n_valid_frac`` is NaN there rather than 0: the share of band cells inside the cone
    is undefined when there are no cells, and a 0 would be averaged as a real share by
    :func:`roi_mean_of_channels`, pulling an ROI's reported share down by however many of
    its channels were rejected.
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
                rows.append({**head, "coherence": float("nan"),
                             "coherence_z": float("nan"),
                             "n_valid_frac": float("nan"),
                             "phase_angle": float("nan"),
                             "phase_sd": float("nan"), "phase_n": 0})
                continue

            wtc = wtc_raw = np.asarray(data["wtc"], dtype=float)[band]
            coi = np.asarray(data["coi"], dtype=float)
            # coi is a period in seconds; 1/coi is the lowest frequency still reliable at
            # that time. A coi of 0 (the very edges) leaves nothing reliable there.
            with np.errstate(divide="ignore"):
                f_edge = np.where(coi > 1e-10, 1.0 / coi, np.inf)
            in_coi = band_freqs[:, None] >= f_edge[None, :]
            # measured whether or not it is applied, so the share stays a reportable number
            n_valid_frac = float(in_coi.mean()) if in_coi.size else 0.0
            if mask_coi:
                wtc = np.where(in_coi, wtc, np.nan)

            valid = np.isfinite(wtc)
            coherence = float(wtc[valid].mean()) if valid.any() else float("nan")

            phase = data.get("phase")
            if phase is None:
                angle, spread, n_phase = float("nan"), float("nan"), 0
            else:
                keep = _phase_cells(wtc_raw, in_coi, _band_rows(data.get("sig"), band),
                                    mask_coi)
                angle, spread, n_phase = _circular_stats(
                    np.asarray(phase, dtype=float)[band][keep])

            rows.append({
                **head,
                "coherence": coherence,
                "coherence_z": float(fisher_r_to_z(coherence)),
                "n_valid_frac": n_valid_frac,
                "phase_angle": np.degrees(angle),
                "phase_sd": np.degrees(spread),
                "phase_n": n_phase,
            })

    columns = ["sub1", "sub2", "label"] + (["label2"] if crossed else [])
    return pd.DataFrame(rows, columns=columns + ["coherence", "coherence_z", "n_valid_frac",
                                                 "phase_angle", "phase_sd", "phase_n"])


def wtc_phase_by_scale(
    result: WTCResult,
    fmin: float,
    fmax: float,
    mask_coi: bool = True,
) -> pd.DataFrame:
    r"""The same circular statistic as :func:`wtc_band_mean`, but one row per frequency.

    ::

      a pair coherent at 0.1 Hz with the first member 1.25 s ahead
      ->  freq 0.1, phase_angle +45.0, lag_s +1.25

    A band mean collapses scales whose phase differs, so its angle belongs to no particular
    frequency and cannot be read as a delay. Resolved per scale it can: a phase of
    :math:`\theta` at frequency :math:`f` is :math:`\theta / 2 \pi f` seconds, which is what
    ``lag_s`` carries, positive meaning the first member leads.

    ``lag_s`` wraps. An angle is only known modulo a turn, so a delay longer than half the
    period at that frequency comes back as a short one of the other sign, and only a
    ``lag_s`` that agrees across neighbouring scales is a delay rather than an artefact of
    where it folded. That agreement is the check Grinsted et al. (2004) describe.

    Returns one row per (pair, label, freq), the label pair and a ``label2`` column on a
    crossed result, as in :func:`wtc_band_mean`. A pairing with no map behind it contributes
    no rows rather than a NaN one: there is no frequency axis to hang them on.
    """
    freqs = np.asarray(result.freqs, dtype=float)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        span = f"{freqs.min():.4f}-{freqs.max():.4f} Hz" if freqs.size else "empty"
        raise ValueError(
            f"no WTC frequency bin inside [{fmin}, {fmax}] Hz; the computed axis spans {span}."
        )
    band_freqs = freqs[band]
    crossed = any(isinstance(k, tuple)
                  for labels in result.pairs.values() for k in labels)

    rows: list[dict] = []
    for (sub1, sub2), labels in result.pairs.items():
        for label, data in labels.items():
            if data is None or data.get("phase") is None:
                continue
            label1, label2 = label if isinstance(label, tuple) else (label, label)
            head = {"sub1": sub1, "sub2": sub2, "label": label1}
            if crossed:
                head["label2"] = label2

            wtc = np.asarray(data["wtc"], dtype=float)[band]
            coi = np.asarray(data["coi"], dtype=float)
            with np.errstate(divide="ignore"):
                f_edge = np.where(coi > 1e-10, 1.0 / coi, np.inf)
            in_coi = band_freqs[:, None] >= f_edge[None, :]
            keep = _phase_cells(wtc, in_coi, _band_rows(data.get("sig"), band), mask_coi)
            phase = np.asarray(data["phase"], dtype=float)[band]

            for i, f in enumerate(band_freqs):
                angle, spread, n = _circular_stats(phase[i][keep[i]])
                rows.append({
                    **head, "freq": float(f),
                    "phase_angle": np.degrees(angle),
                    "phase_sd": np.degrees(spread),
                    "phase_n": n,
                    "lag_s": angle / (2.0 * np.pi * f) if f > 0 else float("nan"),
                })

    columns = (["sub1", "sub2", "label"] + (["label2"] if crossed else [])
               + ["freq", "phase_angle", "phase_sd", "phase_n", "lag_s"])
    return pd.DataFrame(rows, columns=columns)
