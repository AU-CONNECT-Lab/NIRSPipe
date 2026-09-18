"""Synchrony metrics for a hyperscanning group: wavelet coherence and band coherence.

Two ways of asking how locked two recordings are.

  compute_wtc                     Wavelet transform coherence, resolved in both time and
                                  frequency, so a pair that only synchronises during part
                                  of the task still shows it. Per channel pair. Backed by
                                  pycwt.
  compute_wtc_phase_null              The same, against a phase-scrambled partner: the null a
                                  hyperscanning result is actually compared against.
  compute_pairwise_coherence      One magnitude-squared coherence number per channel pair,
                                  averaged over a band. Cheap, and enough when the question
                                  is whether a pair is locked at all.
  wtc_band_mean                   Collapses a WTC map to one number per channel, which is
                                  the form a group analysis wants.
  roi_mean_of_channels            Groups those numbers into ROIs, and
  roi_maps_from_channels          groups the maps the same way so the figure matches.

ROI coherence is always computed per channel pair and then averaged. Averaging the signals
into one ROI trace first is a different and less sensitive number, and the studies that
compared the two report the channel route detecting effects the signal route misses.

The callers live in pipeline/hyperscanning.py, which handles the dyad bookkeeping these
functions assume has already happened: recordings loaded, aligned, trimmed to a common
length and normalised. All of them read long channels only.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import mne
import numpy as np
import pandas as pd
from scipy.fft import next_fast_len
from scipy.linalg import solve_toeplitz
from scipy.signal import coherence, convolve2d, lfilter

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
    return {raw.ch_names[p].rsplit(" ", 1)[0]: p for p in picks}


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

    One member's montage is not the axis. A label only the other member carries still has a
    row or a column of its own, so drawing the axis from the first member alone drops it.

    **Bads are kept**, unlike in :func:`_long_by_label`, which drops them because it is
    choosing what to compute on. An axis has to outlive a rejection: two dyads that lost
    different channels still have to produce matrices of one shape to be stacked, and a
    reader has to be able to tell an empty cell from a channel that was never in the
    montage. A 20-channel montage with 2 rejected gives 20 labels, of which 2 index an
    all-blank row or column.

    One recording is a group of one, so ``long_axis_over([raw], ch_type)`` is the same rule
    over a single montage.
    """
    axis: list[str] = []
    for raw in raws:
        for p in long_channel_picks(raw, ch_type, exclude=[], sep_bands=sep_bands):
            label = raw.ch_names[p].rsplit(" ", 1)[0]
            if label not in axis:
                axis.append(label)
    return axis


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

# Sub-octaves per octave on the wavelet scale grid: the frequency-axis resolution, and
# pycwt's own default. Not configurable, and _SCALE_MARGIN below is why: the coherence is
# smoothed across neighbouring scales over a fixed span in log2(scale), so a different dj
# needs a different margin for --wtc-limit-scales to keep returning the same coherences.
# Reported in every WTC sidecar instead, since it decides how many time-frequency cells a
# band mean averages over.
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
    scales below ``fmin`` rather than at it. The minimum is what keeps a recording shorter
    than its own padding from being transformed over more samples than it is today.
    """
    margin = int(np.ceil(cone_margin_s(1.0 / (_FLAMBDA * s_max)) / dt))
    return min(next_fast_len(n + margin), 1 << (max(n, 1) - 1).bit_length())


def _morlet():
    r"""The mother wavelet every coherence here is computed with.

    A Morlet whose smoothing operator :math:`S` spans ``_SCALE_SMOOTH_DJ0`` in log2(scale),
    which is the width the coherence is defined with. pycwt's own Morlet spans twice that,
    and a wider window pulls :math:`R^2` down: the cross term :math:`S(W_{xy})` loses
    magnitude as phases from neighbouring scales cancel, while the auto terms
    :math:`S(|W_x|^2)` have nothing to cancel. Measured at about 0.05 on a band mean.

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

    ``factor`` is 2 rather than 1 because the cone is a contour and not a wall: contamination
    is small past it, not absent.

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

    The maps are for reading and for averaging over a band, and neither needs the sampling
    rate: a long recording at 10 Hz runs to tens of thousands of columns per pair per
    frequency, far more than any figure resolves or any band mean moves on. What it does limit
    is how short a window --wtc-by-condition can describe, so it is reported in the sidecar
    as the time resolution it produces rather than as this count.
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

    Backend: `pycwt.wct <https://pycwt.readthedocs.io/en/development/reference/#pycwt.wct>`_.
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
    rather than no key at all. That is the package's one rule for a channel-by-channel
    result, the same ``exclude=[]`` against ``exclude="bads"`` split
    :func:`~fnirs_pipe.io.snirf.long_channel_picks` documents, and it is what gives every
    dyad a table of one shape: a cohort table missing thirty rows for one dyad cannot say
    whether those pairs were rejected or never in the montage, and a merged frame cannot be
    subtracted row by row from its null. Left at None the keys come from the surviving
    channels, which is the shape a caller with no montage to hand can produce.

    **Each side contributes its own surviving channels.** Drawing both axes from the first
    subject's list would drop every pairing involving a channel the second subject kept and
    the first had rejected -- pairings that never needed the first subject's copy of that
    channel -- and would make the result depend on which member the pairs table happens to
    list first. The published pipelines cross the two lists independently
    and blank only the row or only the column a rejection belongs to (St. Clair et al.
    2025).

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
    It is applied once here rather than per pair:
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
    never mix, since a member's HbO pairs only with the other member's HbO. The reason to
    run both is a consistency check rather than two results -- HbO has the larger amplitude
    and the better SNR, HbR is the less contaminated by scalp and systemic circulation, so a
    coupling in HbO with nothing in HbR is a caution flag.

    ``cross`` crosses every channel with every other rather than pairing like with like, so
    n channels give n**2 results keyed by ``(label_sub1, label_sub2)`` instead of n keyed by
    the label. It tests whether one person's channel couples to a different site on the
    other's head. The cost is quadratic in the channel count, and it is also what
    :func:`roi_mean_of_channels` needs to build the cross-ROI matrix. Read cell by cell the
    off-diagonal is exploratory: single channels are noisy and the correction over n**2 pairs
    leaves little.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for WTC")

    signals = {sid: _long_signals(raw, ch_type, sep_bands) for sid, raw in raws.items()}

    return _wtc_over_pairs(
        raws, signals, fmin, fmax, significance, seed, mc_count, cross, limit_scales,
        axis=long_axis_over(raws.values(), ch_type, sep_bands))


def phase_scramble(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate with the same power spectrum as ``sig`` and its phases randomised.

    The null a hyperscanning result needs is "this dyad against a dyad that never
    interacted", not "this dyad against red noise". Scrambling one side's phases destroys
    every temporal relationship while leaving each signal's own spectrum and autocorrelation
    intact, so coherence computed against the surrogate is the coherence two unrelated
    recordings of this kind produce.

    Phases are randomised under Hermitian symmetry, so the inverse transform is real and the
    magnitude spectrum is preserved exactly. DC and, on an even-length record, Nyquist have
    no mirror partner and keep their sign instead of taking a phase.
    """
    n = len(sig)
    spectrum = np.fft.rfft(sig)
    scrambled = np.abs(spectrum) * np.exp(1j * rng.uniform(-np.pi, np.pi, size=spectrum.shape))
    # carried over rather than rebuilt: both are real, and |x| would flip a negative one,
    # which for DC means flipping the sign of the mean
    scrambled[0] = spectrum[0]
    if n % 2 == 0:
        scrambled[-1] = spectrum[-1]
    return np.fft.irfft(scrambled, n=n)


# ---- Phase-scrambled null ----


@dataclass
class NullDraws:
    """One chromophore's phase-scrambled null: the draws, the levels, and the summary of both.

    The draws are kept rather than averaged on the spot because the number worth reading off
    a null is not its mean but where a real value falls inside it, and the real table is
    written by a step that runs after this one. :meth:`summarise` is that step's half.
    """

    draws: "list[pd.DataFrame]"
    cond_draws: "list[pd.DataFrame]"
    keys: "list[str]"
    # (sub1, sub2, label) -> the coherence a cell clears at each frequency to beat the null
    levels: dict

    def summarise(self, real: "pd.DataFrame | None" = None,
                  real_by_cond: "pd.DataFrame | None" = None,
                  ) -> "tuple[pd.DataFrame, pd.DataFrame | None]":
        """The whole-run and per-condition null tables, ranked against ``real`` if given.

        ``real`` is the true-dyad band-mean table this null sits beside, matched on the key
        columns, and turns on the ``percentile`` column: the share of a cell's draws its real
        value beat. Exact rather than interpolated from the stored quantiles, and the same
        definition :func:`screening_coherence` uses, so "above the null" means one thing
        across the report. ``real_by_cond`` does the same for the per-condition table.
        """
        return (
            _average_iterations(self.draws, self.keys, real=real),
            (_average_iterations(self.cond_draws, ["condition"] + self.keys,
                                 real=real_by_cond) if self.cond_draws else None),
        )

    def summarise_roi(self, roi_map: dict[str, list[str]],
                      real: "pd.DataFrame | None" = None,
                      real_by_cond: "pd.DataFrame | None" = None,
                      min_channels: int = 2,
                      ) -> "tuple[pd.DataFrame, pd.DataFrame | None]":
        """The same two tables at ROI level, for :func:`roi_mean_of_homologous`.

        **Each iteration is grouped into ROIs before the iterations are summarised**, which is
        the whole point and not an implementation detail. An ROI value is the mean of that
        region's channels, so its null is the distribution of that mean, and that distribution
        depends on how the channels' draws move together within an iteration. Summarise first
        and the covariance is gone: all that is left is each channel's own ``null_sd``, from
        which the mean's spread can only be bracketed between ``sd / sqrt(k)`` and ``sd``.

        It costs nothing. The draws being averaged are the ones already taken for the channel
        table; no surrogate is transformed twice.

        Only the homologous ROI value can be ranked this way. A crossed ``(roi, roi)`` cell
        also holds the within-region cross pairings, which a homologous null never draws.
        """
        keys = ["sub1", "sub2", "label"]
        whole = [roi_mean_of_homologous(f, roi_map, min_channels=min_channels)
                 for f in self.draws]
        cond = None
        if self.cond_draws:
            cond = []
            for frame in self.cond_draws:
                for condition, part in frame.groupby("condition", sort=False):
                    grouped = roi_mean_of_homologous(part, roi_map,
                                                     min_channels=min_channels)
                    grouped.insert(0, "condition", condition)
                    cond.append(grouped)
        return (
            _average_iterations(whole, keys, real=real),
            (_average_iterations(cond, ["condition"] + keys, real=real_by_cond)
             if cond else None),
        )


# Quantile of the surrogate coherence a cell has to clear before its phase arrow is drawn.
# 0.95 is the alpha NULL_ALPHA_PCT grades on, so "above the null" reads the same on a
# coherence map as on the screening panel.
NULL_ARROW_QUANTILE = 0.95

# Bins the surrogate coherences are counted into, per frequency. Coherence is bounded on
# [0, 1], so a fixed grid is exact to 1/_NULL_HIST_BINS and, unlike keeping the draws,
# costs the same whatever n_iter is: the alternative is n_iter copies of a whole map.
_NULL_HIST_BINS = 1000


def _accumulate_null_hist(hists: dict, result: "WTCResult", mask_coi: bool) -> None:
    """Count one iteration's surrogate coherences into a per-pair, per-frequency histogram.

    ::

      a 51 x 3962 surrogate map -> 51 rows of counts, added to whatever earlier iterations left

    Pooled over time as well as over iterations, because the null being estimated is the
    distribution of a single cell at that frequency and every in-cone cell of a surrogate map
    is a draw from it. ``hists`` is mutated in place, keyed the way ``result.pairs`` is.
    """
    freqs = np.asarray(result.freqs, dtype=float)
    for (sub1, sub2), labels in result.pairs.items():
        for label, data in labels.items():
            if data is None:
                continue
            wtc = np.asarray(data["wtc"], dtype=float)
            if wtc.shape[0] != len(freqs):
                continue
            keep = (freqs[:, None] >= 1.0 / np.asarray(data["coi"], dtype=float)[None, :]
                    if mask_coi else np.ones(wtc.shape, dtype=bool))
            if not keep.any():
                continue
            # one bincount over the whole map rather than one per row: the row index is
            # folded into the bin index, so the flat counts reshape straight back
            col = np.clip((wtc * _NULL_HIST_BINS).astype(np.int64), 0, _NULL_HIST_BINS - 1)
            row = np.broadcast_to(np.arange(wtc.shape[0])[:, None], wtc.shape)
            flat = row[keep] * _NULL_HIST_BINS + col[keep]
            counts = np.bincount(flat, minlength=wtc.shape[0] * _NULL_HIST_BINS)
            key = (sub1, sub2, label)
            if key not in hists:
                hists[key] = np.zeros((wtc.shape[0], _NULL_HIST_BINS), dtype=np.int64)
            hists[key] += counts.reshape(hists[key].shape)


def _null_level(hist: np.ndarray, quantile: float = NULL_ARROW_QUANTILE) -> np.ndarray:
    """The quantile of each frequency's accumulated surrogate counts, as a coherence.

    ::

      a 51 x 1000 count matrix -> ndarray(51,), the level a cell clears to beat the null

    A frequency no surrogate cell ever landed on, every one of its cells having been outside
    the cone, gets NaN rather than 0: no level was measured there, and a 0 would pass every
    cell as significant.
    """
    total = hist.sum(axis=1)
    cum = np.cumsum(hist, axis=1)
    out = np.full(hist.shape[0], np.nan)
    for i in np.flatnonzero(total):
        j = int(np.searchsorted(cum[i], quantile * total[i], side="left"))
        # bin centre, so the level sits inside the bin its count was recorded in
        out[i] = (min(j, hist.shape[1] - 1) + 0.5) / hist.shape[1]
    return out


def compute_wtc_phase_null(
    raws: dict[str, mne.io.Raw],
    band_fmin: float,
    band_fmax: float,
    n_iter: int = 100,
    fmin: float = 0.004,
    fmax: float = 0.20,
    seed: int | None = None,
    cross: bool = False,
    limit_scales: bool = True,
    mask_coi: bool = True,
    ch_type: str = "hbo",
    sep_bands=None,
    windows: "list[tuple[str, float, float]] | None" = None,
    analysis_window: "tuple[float, float] | None" = None,
) -> "NullDraws":
    """Phase-scrambled band means: WTC against a phase-scrambled partner, averaged over ``n_iter``.

    One subject's signals are replaced by surrogates and the whole pairwise WTC is rerun, once
    per iteration; the band means are averaged across iterations. The result has the columns
    ``wtc_band_mean`` returns, so a true-dyad table and this one subtract or test cell by cell,
    plus ``null_sd``, ``null_p95`` and ``n_iter``: the mean alone cannot say where in its null
    a real value sits, and a null nobody can rank against is only half of one.

    Returns a :class:`NullDraws`, which holds the per-iteration draws as well as their
    summary: ranking a real value inside its null needs the draws, and the caller that has
    the real table to rank runs after this one.

    Cost is ``n_iter`` times a full WTC run. Significance contours are never computed here:
    this table *is* the null, so a second null inside it would be redundant and slow. The
    surrogate maps are not saved either, but they are no longer only averaged: each one is
    counted into a per-frequency histogram on the way past, and ``NullDraws.levels`` is
    ``{(sub1, sub2, label): ndarray(n_freqs,)}``, the coherence a cell has to clear at each
    frequency to beat the null. That is what the phase arrows are drawn against, and it has
    to be per frequency: surrogate coherence is not flat in frequency, it rises at both ends
    of the computed range, so one scalar threshold over the whole map draws arrows
    preferentially at the band edges.

    ``seed`` drives the phase randomisation and nothing else. Passing the same value as the
    real run is what makes the pair reproducible together.

    ``ch_type`` runs the null on one chromophore, and has to match the real table it is
    compared against: a null computed on HbO says nothing about an HbR coupling.

    ``windows`` is ``[(label, tstart, tstop)]``, and turns the second return value into a
    per-condition null with a ``condition`` column. Each window is read off the same
    whole-run transform the whole-run table is averaged from, by :func:`window_result`, so
    one pass of ``n_iter`` transforms produces both tables and the null is windowed exactly
    the way the real table it is compared against was.

    ``analysis_window`` is ``--tstart``/``--tend``, and does the same job for the whole-run
    row that ``windows`` does for the per-condition ones: the real whole-run table describes
    that stretch, so its null has to as well.

    A whole-run null against a windowed real table is anticonservative on the short windows,
    because a long record's surrogate coherence is lower than a short window's. Passing
    ``windows`` and ``analysis_window`` is what removes that.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}")

    subject_ids = list(raws.keys())
    if len(subject_ids) != 2:
        # _wtc_over_pairs walks every combination, and only one subject is scrambled, so a
        # third member would give pairs of two real recordings sitting in a table labelled
        # null. Refused rather than warned: a wrong null reads exactly like a right one.
        raise ValueError(
            f"phase-scrambled WTC needs exactly 2 subjects, got {len(subject_ids)}: "
            f"{subject_ids}. Only one side is scrambled, so a larger group would leave "
            "real-against-real pairs in a table labelled null. Run it per dyad."
        )

    true_signals = {sid: _long_signals(raw, ch_type, sep_bands) for sid, raw in raws.items()}
    # scramble the second subject only: scrambling both would test surrogate against
    # surrogate, which is a different and weaker null
    scrambled_id = subject_ids[1]
    rng = np.random.default_rng(seed)

    frames: list[pd.DataFrame] = []
    cond_frames: list[pd.DataFrame] = []
    hists: dict[tuple, np.ndarray] = {}
    # the unscrambled side is the same signal in every iteration, so its transforms are
    # computed once and reused. Roughly a third of the run: two thirds of an iteration is
    # _prepare_channel and half of those prepares were this side's. Costs one montage of
    # transforms resident (~70 MB per channel on an hour-long recording), which does not
    # grow with n_iter
    cache1: dict[tuple[str, str], _ChannelWavelet] = {}
    for i in range(n_iter):
        signals = dict(true_signals)
        signals[scrambled_id] = {
            label: phase_scramble(sig, rng)
            for label, sig in true_signals[scrambled_id].items()
        }
        result = _wtc_over_pairs(
            raws, signals, fmin, fmax, significance=False, seed=None,
            cross=cross, limit_scales=limit_scales, cache1=cache1,
            # the same axis the real table is built on, without which the null cannot be
            # subtracted from it row by row
            axis=long_axis_over(raws.values(), ch_type, sep_bands))
        # --tstart/--tend, read off this iteration's transform the way the real table reads
        # it off its own. Without it the whole-run row of the null describes the recording
        # while the row it is compared against describes the window
        run_result = (result if analysis_window is None
                      else window_result(result, *analysis_window))
        frames.append(wtc_band_mean(run_result, band_fmin, band_fmax, mask_coi=mask_coi))
        # counted off the whole-run transform, not the windowed read: a condition is a slice
        # of the same map, so its cells are draws from the same per-frequency null
        _accumulate_null_hist(hists, result, mask_coi)
        # windowed off this iteration's own transform, never recomputed on the cut: the
        # real table is windowed the same way, and a null built differently from the table
        # it is subtracted from measures the difference between the two routes
        for label, tstart, tstop in (windows or []):
            part = wtc_band_mean(window_result(result, tstart, tstop),
                                 band_fmin, band_fmax, mask_coi=mask_coi)
            part.insert(0, "condition", label)
            cond_frames.append(part)
        if (i + 1) % 10 == 0:
            logger.info("phase-scrambled WTC: %d/%d iterations", i + 1, n_iter)

    keys = ["sub1", "sub2", "label"] + (["label2"] if "label2" in frames[0].columns else [])
    return NullDraws(draws=frames, cond_draws=cond_frames, keys=keys,
                      levels={key: _null_level(hist) for key, hist in hists.items()})


def _average_iterations(frames: "list[pd.DataFrame]", keys: "list[str]",
                        real: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """One band-mean table summarising the iterations that produced it, distribution and all.

    ::

      [iter1 rows, iter2 rows], ["sub1", "sub2", "label"]  ->  one row per channel pair

    ``null_mean`` is the mean of the draws and ``null_mean_z`` its Fisher z, taken from the
    averaged value rather than averaged itself. **Not called ``coherence``**, which is what
    every other table's measured value is called: this column is the null's own centre, and a
    reader who takes it for the real value compares the null with itself. ``null_sd``,
    ``null_p95`` and ``n_iter`` describe the spread it came out of, since a null summarised by
    its mean alone cannot say whether a real value above it is anywhere near unusual.

    ``real`` is the true-dyad table, matched on ``keys``, and adds ``percentile``: the share
    of a cell's draws its real value beat. A cell the real table has no row for, or whose
    real value is NaN, gets NaN rather than a rank against nothing.
    """
    stacked = pd.concat(frames, ignore_index=True)
    grouped = stacked.groupby(keys, sort=False)
    out = (grouped.agg(null_mean=("coherence", "mean"),
                       null_sd=("coherence", "std"),
                       n_iter=("coherence", "count"),
                       n_valid_frac=("n_valid_frac", "mean"))
                  .reset_index())
    draws = {key: part.to_numpy(dtype=float) for key, part in grouped["coherence"]}
    out.insert(out.columns.get_loc("null_sd"), "null_mean_z",
               out["null_mean"].map(_fisher_z))
    out.insert(out.columns.get_loc("n_iter"), "null_p95",
               [_p95(draws[k]) for k in _row_keys(out, keys)])
    if real is not None:
        out.insert(out.columns.get_loc("n_iter"), "percentile",
                   _null_percentile(out, keys, draws, real))
    return out


def _p95(draw: np.ndarray) -> float:
    """95th percentile of the draws that exist. A pair blank in every iteration has none."""
    finite = draw[np.isfinite(draw)]
    return float(np.percentile(finite, 95)) if finite.size else float("nan")


def _row_keys(frame: pd.DataFrame, keys: "list[str]") -> list:
    """The groupby key of each row, shaped the way pandas hands it back: scalar for one key."""
    if len(keys) == 1:
        return frame[keys[0]].tolist()
    return list(map(tuple, frame[keys].to_numpy()))


def _null_percentile(out: pd.DataFrame, keys: "list[str]", draws: dict,
                     real: pd.DataFrame) -> "list[float]":
    """Where each real value falls among its cell's own surrogate draws, as a percentage.

    ::

      real 0.31 against draws [0.22, 0.25, 0.29, 0.33]  ->  75.0

    Counting rather than interpolating a stored quantile: at the iteration counts a null is
    affordable at, the two disagree by more than the number is worth.

    A homologous null against a crossed real table ranks the crossed table's diagonal: the
    rows where ``label`` and ``label2`` agree are the pairings the null was drawn for.
    """
    missing = [k for k in keys if k not in real.columns]
    if missing:
        logger.warning("null percentile skipped: the real table has no %s column(s)",
                       ", ".join(missing))
        return [float("nan")] * len(out)
    # a crossed table holds one row per label pair, so keying on `label` alone would take
    # whichever pairing that label came first in rather than its homologous one
    if "label2" in real.columns and "label2" not in keys:
        real = real[real["label"] == real["label2"]]
    truth = real.set_index(keys)["coherence"]
    if truth.index.has_duplicates:
        logger.warning("null percentile: %d real row(s) share a key, ranking against the "
                       "first of each", int(truth.index.duplicated().sum()))
        truth = truth[~truth.index.duplicated()]
    values = []
    for key in _row_keys(out, keys):
        real_value = truth.get(key, float("nan"))
        draw = draws.get(key)
        # a NaN draw compares False, so a pair blank in every iteration would rank 0th:
        # "the real value beat none of them", which is a claim about draws that do not exist
        usable = draw is not None and np.isfinite(draw).any()
        values.append(float((draw < real_value).mean() * 100)
                      if usable and np.isfinite(real_value) else float("nan"))
    return values


def _mean_phase(phases: "list[np.ndarray]") -> "np.ndarray | None":
    """Circular mean of several phase maps, cell by cell.

    e.g. [179 deg, -179 deg] gives 180 deg, not the 0 deg an arithmetic mean would.
    """
    if not phases:
        return None
    stack = np.stack([np.asarray(p, dtype=float) for p in phases])
    return np.angle(np.exp(1j * stack).mean(axis=0)).astype(np.float32)


def window_result(result: WTCResult, tstart: float, tstop: float) -> WTCResult:
    """The same WTC restricted to a time window, so a band mean over it describes one condition.

    ::

      a 1200 s result + (300, 600)  ->  the same maps holding only those 300 s

    This is how a condition is read out of a whole-record transform, and it is not the same
    number as transforming that condition on its own. A cut window has two edges of its own,
    and the cone of influence reaches further at longer periods, so a short condition
    transformed alone has a larger share of its band cells sitting outside the cone. Those
    cells are coefficients padded against the window's own edges, near 1 whatever the data
    does, so transforming each condition separately inflates the band mean by an amount that
    tracks window length, which in a design whose conditions differ in length is confounded
    with the contrast. Windowing carries the whole record's cone instead, which only reaches
    into the ends of the recording.

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
                "phase": np.asarray(data["phase"])[:, keep],
            })
            for label, data in labels.items()
        }
        for pair_key, labels in result.pairs.items()
    }
    return WTCResult(pairs=pairs, freqs=result.freqs, times=times[keep])


def roi_maps_from_channels(
    result: WTCResult,
    roi_map: dict[str, list[str]],
) -> WTCResult:
    """Average the channel-pair WTC maps cell by cell into one map per ROI pair.

    The ROI number the field reports is a mean over channel-pair coherences, so the figure
    that belongs beside it is the mean over those channels' maps rather than a separate
    transform on ROI-averaged signals. Averaging the maps first and the band second gives the
    same number as averaging the band first and the channels second, so the picture and the
    table finally agree.

    ::

      {"S1_D1": map, "S1_D2": map}  + {"lPFC": ["S1_D1", "S1_D2"]}  ->  {"lPFC": mean map}

    The COI depends only on record length and sampling rate, so every member shares one and
    it is carried through unchanged. ``sig`` is dropped: a Monte Carlo level belongs to the
    pair it was computed for and does not average.

    Phase averages as a direction, not as a number: the arithmetic mean of 179 degrees and
    -179 degrees is 0, the one direction neither member points in. The members are summed as
    unit vectors and the angle read off the sum, so a set of members that disagree gives a
    short resultant and, drawn as an arrow, a direction no more confident than they were.
    """
    ch_to_roi = {ch: roi for roi, chs in roi_map.items() for ch in chs}

    pairs: dict = {}
    for pair_key, labels in result.pairs.items():
        bucket: dict = {}
        for label, data in labels.items():
            if data is None:
                continue
            label1, label2 = label if isinstance(label, tuple) else (label, label)
            roi1, roi2 = ch_to_roi.get(label1), ch_to_roi.get(label2)
            if roi1 is None or roi2 is None:
                continue
            key = (roi1, roi2) if isinstance(label, tuple) else roi1
            bucket.setdefault(key, []).append(data)
        pairs[pair_key] = {
            key: {"wtc": np.mean([d["wtc"] for d in members], axis=0),
                  "coi": members[0]["coi"], "sig": None,
                  "phase": _mean_phase([d["phase"] for d in members
                                       if d.get("phase") is not None])}
            for key, members in bucket.items()
        }

    return WTCResult(pairs=pairs, freqs=result.freqs, times=result.times)


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
            freqs, coh = coherence(data1[i], data2[j], fs=sfreq, nperseg=nperseg)
            mask = (freqs >= fmin) & (freqs <= fmax)
            mean_coh = float(np.mean(coh[mask])) if mask.any() else float("nan")
            rows.append({"ch_name": label, "sub1": sub1, "sub2": sub2, "coherence": mean_coh})

    return pd.DataFrame(rows, columns=["ch_name", "sub1", "sub2", "coherence"])


# Surrogate pairings a screening percentile is read against. 100 is `write_wtc_null`'s own
# default, so the two nulls a dyad is measured by are drawn the same number of times.
SCREEN_NULL_ITER = 100


def welch_nperseg(n_times: int) -> int:
    """Segment length the band coherence is estimated with, for a stretch of ``n_times``.

    ::

      a 3900 s run at 10 Hz -> 512;  a 300 s block -> 512;  a 20 s window -> 64

    A quarter of the stretch so a short window still yields several segments, capped at 512
    so a long one does not spend its resolution on a frequency nobody reads, floored at 64 so
    the band has bins at all. One function because the number decides both the estimate and
    its floor: magnitude-squared coherence sits near 1/(number of segments) when nothing is
    coupled, so two callers with two copies of this rule would print values a reader compares
    that are not on the same scale.
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
    and that count falls with the window, so on one recording the floor moves by an order of
    magnitude between a 300 s block and the whole run. Two windows' raw values are therefore
    not comparable and neither is readable alone; each value's rank inside a null drawn for
    *that* window is both.

    The null pairs one member against a phase-scrambled copy of the other, which is the
    surrogate :func:`~fnirs_pipe.pipeline.wtc_null.write_wtc_null` uses on the post report. One
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
            # Two percentiles, because they answer different questions and only the
            # second is the window's verdict. Per channel, a rank among that channel's own
            # draws, which at an affordable iteration count is noisy. Per window, the rank
            # of the channel mean among the null's channel means: the channels are pooled
            # before the comparison, so it tests the dyad rather than fourteen channels.
            mean_null = null.mean(axis=1)
            window_pct = float((mean_null < real.mean()).mean() * 100)
            for i, label in enumerate(labels):
                rows.append({
                    "window": name, "ch_name": label, "sub1": sub1, "sub2": sub2,
                    "coherence": float(real[i]),
                    "null_mean": float(null[:, i].mean()),
                    "null_p95": float(np.percentile(null[:, i], 95)),
                    "percentile": float((null[:, i] < real[i]).mean() * 100),
                    "window_percentile": window_pct,
                    "n_seg": int(n_seg), "window_s": round(seg / sfreq, 1),
                })

    return pd.DataFrame(rows, columns=["window", "ch_name", "sub1", "sub2", "coherence",
                                       "null_mean", "null_p95", "percentile",
                                       "window_percentile", "n_seg", "window_s"])


def _fisher_z(r: float) -> float:
    """Fisher r-to-z of one value, clipped as :func:`restingstate.fisher_z` clips a matrix."""
    if not np.isfinite(r):
        return float("nan")
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


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

    ``mask_coi`` is **on by default**. Cells outside the cone are padding, so averaging them
    reports the record's edges as coupling, which is the whole reason the cone is drawn. Very
    few published studies say either way, and that is a gap in reporting rather than a
    consensus to average everything: of 30 WTC studies extracted, one mentions the cone at all
    and it excludes. Masking does discard more of a short segment than of a long one, so it
    moves conditions of different length by different amounts, which is why ``n_valid_frac`` is
    reported either way and why the windowing in :func:`window_result` matters more than this
    flag: on a whole-record transform there is almost nothing outside the cone to drop.
    ``--no-wtc-mask-coi`` averages the whole band.

    ``n_valid_frac`` is the share of band cells that lie inside the cone of influence. **It is
    reported whether or not the mask is applied**, so the share is visible as a quality number
    even when every cell was averaged.

    ``coherence_z`` is the Fisher r-to-z of ``coherence`` (:math:`\operatorname{arctanh}`,
    clipped just below 1), which is what group statistics should average: coherence is bounded
    on [0, 1], so its mean across dyads is biased toward the interior.

    Returns one row per (pair, label) with columns sub1, sub2, label, coherence, coherence_z,
    n_valid_frac. A crossed result is keyed by a label pair rather than one label, and gains a
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
                             "n_valid_frac": float("nan")})
                continue

            wtc = np.asarray(data["wtc"], dtype=float)[band]
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
            rows.append({
                **head,
                "coherence": coherence,
                "coherence_z": _fisher_z(coherence),
                "n_valid_frac": n_valid_frac,
            })

    columns = ["sub1", "sub2", "label"] + (["label2"] if crossed else [])
    return pd.DataFrame(rows, columns=columns + ["coherence", "coherence_z", "n_valid_frac"])


def roi_mean_of_channels(
    band_df: pd.DataFrame,
    roi_map: dict[str, list[str]],
    min_channels: int = 2,
) -> pd.DataFrame:
    """Average channel-level band means within each ROI: the ROI number the WTC literature reports.

    The field computes coherence per channel pair and averages those values into ROI
    clusters. The alternative, one WTC on the ROI-averaged signal, is a different number
    because coherence is bounded and nonlinear, and it is the less sensitive of the two.

    ``band_df`` is what :func:`wtc_band_mean` returns for a channel-level result; a crossed
    one carries ``label2`` and is grouped on both sides into the ROI-by-ROI matrix. ``n_ch``
    counts the channel pairs behind each mean, so an ROI thinned by rejection is visible.
    Channels no ROI lists are dropped.

    ``min_channels`` drops a cell resting on fewer than that many channel pairs, the rule the
    published pipelines use to stop one surviving optode from standing in for a region. It
    counts pairs, so on a crossed frame a cell needs ``min_channels`` combinations rather than
    that many channels on each side.

    ``coherence_z`` is recomputed from the averaged coherence rather than averaged itself, so
    it stays the Fisher z of the number in the same row.
    """
    ch_to_roi = {ch: roi for roi, chs in roi_map.items() for ch in chs}
    df = band_df.copy()
    label_cols = ["label"] + (["label2"] if "label2" in df.columns else [])
    for col in label_cols:
        df[col] = df[col].map(ch_to_roi)
    df = df.dropna(subset=label_cols)

    keys = ["sub1", "sub2"] + label_cols
    out = (
        df.groupby(keys, sort=False)
          .agg(coherence=("coherence", "mean"),
               n_valid_frac=("n_valid_frac", "mean"),
               n_ch=("coherence", "count"))
          .reset_index()
    )
    if min_channels > 1:
        thin = out["n_ch"] < min_channels
        if thin.any():
            logger.info("ROI means: %d cell(s) under %d channel pairs, dropped",
                        int(thin.sum()), min_channels)
        out = out[~thin].reset_index(drop=True)
    out.insert(out.columns.get_loc("n_valid_frac"), "coherence_z",
               out["coherence"].map(_fisher_z))
    return out


def roi_mean_of_homologous(
    band_df: pd.DataFrame,
    roi_map: dict[str, list[str]],
    min_channels: int = 2,
) -> pd.DataFrame:
    """Average an ROI's *homologous* channel pairs: one value per ROI, however the run was made.

    ::

      right_pfc holds S1_D1, S1_D2, S2_D1, S2_D2
      -> the mean of the four (S1_D1, S1_D1), (S1_D2, S1_D2), ... coherences

    :func:`roi_mean_of_channels` groups whatever it is handed, so on a crossed frame its
    ``(roi, roi)`` diagonal is the mean of every pairing inside the ROI, sixteen of them here,
    of which four are homologous. That is a different quantity from the one the literature
    reports and from the one an uncrossed run produces, and a flag whose job is to add the
    off-diagonal cells should not silently redefine the diagonal. This function is the
    reported number: the homologous mean, identical whether or not the run crossed.

    It is also the only ROI value the homologous phase-scrambled null can rank, since the null
    draws exactly these pairings.
    """
    df = band_df
    if "label2" in df.columns:
        df = df[df["label"] == df["label2"]].drop(columns=["label2"])
    return roi_mean_of_channels(df, roi_map, min_channels=min_channels)


# ---- Inter-subject correlation ----
# Beside the wavelet coherence: same question in the time domain, same two helpers
# (`_shared_sfreq`, `long_axis_over`), and `window_result` documents itself against it.

ISC_MAX_AR_ORDER = 32


def _ar_whiten(x: np.ndarray, max_order: int = ISC_MAX_AR_ORDER) -> tuple[np.ndarray, int]:
    r"""Residuals of the best autoregressive fit to ``x``, and the order that won.

    ::

      a slow, strongly autocorrelated trace  ->  a near-white one of the same length, 22

    A haemodynamic trace is heavily autocorrelated: the response is a low-pass filter on
    whatever drove it, so neighbouring samples are near copies and a correlation between two
    such traces has far fewer independent observations than it has samples. Fitting

    .. math::

        x_t = \sum_{k=1}^{p} a_k\, x_{t-k} + e_t

    and keeping :math:`e_t` leaves a series whose own samples are close to independent, so
    the correlation between two of them sits on the scale its sample count implies.

    Coefficients come from the Yule-Walker equations at each order, and ``p`` is the one
    minimising the Bayesian information criterion over ``1..max_order``, which trades the
    variance explained against the coefficients spent. The residual keeps the input's length:
    the filter is applied from the start rather than from sample ``p``, so the first ``p``
    samples are a startup transient the caller drops.

    An all-NaN row, a constant one, or one too short for a single lag comes back unchanged
    at order 0.
    """
    finite = np.isfinite(x)
    if not finite.all() or x.size < 8:
        return x, 0
    centred = x - x.mean()
    n = centred.size
    n_lag = min(int(max_order), n // 4)
    if n_lag < 1:
        return x, 0
    # only the first n_lag lags are needed, and each is one dot product; a full correlation
    # would be quadratic in the record length and this runs once per surrogate iteration
    acov = np.array([centred @ centred] + [centred[:-k] @ centred[k:]
                                           for k in range(1, n_lag + 1)]) / n
    if acov[0] <= 0:
        return x, 0

    best_bic, best_coef, best_order = np.inf, None, 0
    for p in range(1, n_lag + 1):
        try:
            coef = solve_toeplitz((acov[:p], acov[:p]), acov[1:p + 1])
        except np.linalg.LinAlgError:
            break
        resid_var = acov[0] - coef @ acov[1:p + 1]
        if resid_var <= 0:
            break
        bic = n * np.log(resid_var) + p * np.log(n)
        if bic < best_bic:
            best_bic, best_coef, best_order = bic, coef, p
    if best_order == 0:
        return x, 0
    return lfilter(np.r_[1.0, -best_coef], [1.0], x), best_order


def _whiten_rows(data: np.ndarray, max_order: int) -> tuple[np.ndarray, list[int]]:
    """:func:`_ar_whiten` over every row, with the startup transient dropped from all of them.

    Each row gets its own order, so the transient to discard is the longest of them: cutting
    per row would leave the rows on different clocks, and they are about to be correlated
    sample by sample.
    """
    out = np.empty_like(data)
    orders: list[int] = []
    for i, row in enumerate(data):
        out[i], order = _ar_whiten(row, max_order)
        orders.append(order)
    drop = max(orders) if orders else 0
    return (out[:, drop:] if drop else out), orders


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """Each row to zero mean and unit deviation, a flat row left alone rather than divided by 0."""
    mu  = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mu) / np.where(std < 1e-12, 1.0, std)


def _isc_from_rows(
    data1: np.ndarray, data2: np.ndarray, max_lag: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r of every row of ``data1`` against every row of ``data2``, and at what lag.

    ::

      max_lag 0   ->  r at the same sample, lag 0 everywhere
      max_lag 16  ->  the strongest r within 16 samples either way, and where it was

    With ``max_lag`` the pair is re-correlated at every shift in ``[-max_lag, max_lag]`` and
    the strongest of them kept. Two people's haemodynamic responses do not peak at the same
    instant, so a same-sample correlation reads a coupling a second apart as no coupling; a
    short search either way is what the cross-correlation literature reports instead.

    **Strongest means largest in magnitude, and the sign is kept.** The published form takes
    the largest signed value, which suits a metric built for positive coupling but would turn
    every anticorrelated pairing into a small positive number, and this matrix has both.
    The two agree wherever the coupling is positive.

    Lag is in samples and positive means the second member follows the first. Searching
    inflates the value under no coupling, since it is a maximum over many draws, so a lagged
    matrix belongs with a null searched the same way.

    Each shift is correlated over its own overlap, so both sides are re-standardised per lag
    rather than once over the whole record.
    """
    n = data1.shape[1]
    if max_lag <= 0:
        isc_mat = (_zscore_rows(data1) @ _zscore_rows(data2).T) / n
        np.clip(isc_mat, -1.0, 1.0, out=isc_mat)
        return isc_mat, np.zeros_like(isc_mat)

    best = best_lag = None
    for shift in range(-int(max_lag), int(max_lag) + 1):
        left  = data1[:, max(0, -shift): n - max(0, shift)]
        right = data2[:, max(0, shift): n - max(0, -shift)]
        at_lag = np.clip((_zscore_rows(left) @ _zscore_rows(right).T) / left.shape[1],
                         -1.0, 1.0)
        if best is None:
            best, best_lag = at_lag, np.full(at_lag.shape, float(shift))
            continue
        # NaN compares False, so a blank cell keeps the NaN it started with
        stronger = np.abs(at_lag) > np.abs(best)
        best = np.where(stronger, at_lag, best)
        best_lag = np.where(stronger, float(shift), best_lag)
    return best, np.where(np.isfinite(best), best_lag, np.nan)


def _isc_rows(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    ch_type: str = "hbo",
    sep_bands=None,
    window: "tuple[float, float] | None" = None,
) -> "tuple[np.ndarray, np.ndarray, list[str]] | tuple[None, None, None]":
    """The two members' signals on one montage axis and one clock, ready to correlate.

    One row per axis label per member, NaN where that member has no usable channel
    there, cut to ``window`` if one was asked for. Everything :func:`compute_isc` and
    :func:`compute_isc_pairs` disagree about happens after this.
    """
    if len(subject_ids) < 2:
        return None, None, None
    raw1 = aligned_raws.get(subject_ids[0])
    raw2 = aligned_raws.get(subject_ids[1])
    if raw1 is None or raw2 is None:
        return None, None, None
    # the same refusal WTC makes: alignment equalises duration, not rate, so at two rates
    # sample i of one member and sample i of the other are not the same moment and the
    # correlation between them is a plausible-looking number about nothing
    _shared_sfreq({subject_ids[0]: raw1, subject_ids[1]: raw2})

    def _by_label(raw: mne.io.Raw) -> dict[str, int]:
        """{label: index} over what this member kept, bads dropped: what gets correlated."""
        return {raw.ch_names[p].rsplit(" ", 1)[0]: p
                for p in long_channel_picks(raw, ch_type, sep_bands=sep_bands)}

    # the axis is the montage, the maps are what survived: one shape, blanks where a channel
    # went. Same axis rule as the crossed WTC matrix.
    ch_names = long_axis_over([raw1, raw2], ch_type, sep_bands)
    map1, map2 = _by_label(raw1), _by_label(raw2)
    if not ch_names:
        return None, None, None

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

    if window is not None:
        sfreq = float(raw1.info["sfreq"])
        first = max(0, int(round(float(window[0]) * sfreq)))
        last  = min(n_times, int(round(float(window[1]) * sfreq)))
        # two samples is the least a correlation can be computed from at all; a window this
        # short is a trigger artefact rather than a condition, and returning nothing leaves
        # the panel out instead of printing a coefficient over three points
        if last - first < 2:
            logger.warning("ISC (%s): window %.1f-%.1f s holds %d sample(s) of %d, "
                           "no correlation computed",
                           ch_type, window[0], window[1], max(0, last - first), n_times)
            return None, None, None
        data1, data2 = data1[:, first:last], data2[:, first:last]

    return data1, data2, ch_names


def compute_isc(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    ch_type: str = "hbo",
    sep_bands=None,
    window: "tuple[float, float] | None" = None,
    whiten: int = 0,
    max_lag_s: float = 0.0,
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Compute inter-brain Pearson r matrix (n_ch × n_ch) over long channels.

    matrix[i, j] = Pearson r between sub1_ch_i and sub2_ch_j.
    Diagonal = same-channel ISC.

    ``window`` restricts it to ``(tstart, tstop)`` on the aligned clock, which is how a
    condition gets a correlation of its own. Unlike the wavelet coherence this really is a
    cut and not a slice of a whole-record computation, and it is sound for the reason the
    subject report's own per-condition panels are: a correlation has no frequency axis and
    nothing here filters, so a window carries no edge that the whole record would not have
    had. What it does carry is its own mean and its own standard deviation, and it has to:
    both sides are z-scored inside the window, because the correlation over a stretch is
    against that stretch's mean, not the recording's.

    Cutting the wavelet coherence the same way would be wrong, and that asymmetry is the
    whole of why the two are treated differently here. See
    :func:`~fnirs_pipe.pipeline.synchrony.window_result`.

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

    Raises ValueError if the members were recorded at different sampling rates, which is
    the refusal WTC has always made: alignment equalises duration, not rate.

    ``whiten`` is the largest autoregressive order :func:`_ar_whiten` may spend on each
    channel before the correlation, 0 to correlate the signals themselves. A haemodynamic
    trace is strongly autocorrelated, so a correlation between two of them rests on far
    fewer independent observations than it has samples, and the value it takes under no
    coupling at all is correspondingly large. Whitening puts r back on the scale its sample
    count implies; it also shrinks it, so a whitened matrix and an unwhitened one are not
    comparable and the sidecar records which was written.

    Args:
        ch_type: "hbo" or "hbr".
    """
    data1, data2, ch_names = _isc_rows(aligned_raws, subject_ids, ch_type, sep_bands, window)
    if ch_names is None:
        return None, None
    max_lag = _lag_samples(aligned_raws, subject_ids, max_lag_s)
    return _isc_matrix(data1, data2, whiten, max_lag)[0], ch_names


def _lag_samples(aligned_raws: dict, subject_ids: list[str], max_lag_s: float) -> int:
    """``max_lag_s`` seconds as whole samples of the rate both members share."""
    if not max_lag_s:
        return 0
    sfreq = _shared_sfreq({sid: aligned_raws[sid] for sid in subject_ids[:2]})
    return max(0, int(round(float(max_lag_s) * sfreq)))


def compute_isc_pairs(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    ch_type: str = "hbo",
    sep_bands=None,
    window: "tuple[float, float] | None" = None,
    whiten: int = 0,
    max_lag_s: float = 0.0,
    n_null: int = 0,
    seed: int | None = None,
) -> "tuple[np.ndarray, list[str], pd.DataFrame, np.ndarray | None] | tuple[None, None, None, None]":
    """The ISC matrix and the same numbers as one row per channel pair, ranked against a null.

    ::

      a 3 x 3 matrix  ->  9 rows of sub1, sub2, label, label2, r, r_z, ar_order, ar_order2

    The matrix is what the report draws; the frame is what a group analysis reads, and it is
    the form the extra columns fit in. ``r_z`` is the Fisher r-to-z of ``r``, which is what
    should be averaged across dyads, since r is bounded and its mean is biased toward the
    interior. ``ar_order`` and ``ar_order2`` are what each side's channel was whitened at,
    and are absent when ``whiten`` is 0.

    ``n_null`` phase-scrambles the second member and recomputes, which is the null a
    correlation between two recordings needs: scrambling preserves each signal's own power
    spectrum and so its autocorrelation, and it is the autocorrelation that decides how large
    r gets with no coupling present. It adds ``null_mean``, ``null_sd``, ``null_p95`` and
    ``percentile``, the share of a cell's surrogate draws its real value beat. All four are
    on ``|r|``: the question a null answers here is whether the pair is coupled, not which
    way, and a surrogate is as likely to land either side of zero.

    Whitening and the null answer the same objection by different routes and compose without
    double-counting: whitening moves the estimate onto an honest scale, the null measures the
    scale directly, and with both on the null is drawn through the whitening too.
    """
    data1, data2, ch_names = _isc_rows(aligned_raws, subject_ids, ch_type, sep_bands, window)
    if ch_names is None:
        return None, None, None, None

    max_lag = _lag_samples(aligned_raws, subject_ids, max_lag_s)
    isc_mat, orders1, orders2, lags = _isc_matrix(data1, data2, whiten, max_lag)
    sfreq = _shared_sfreq({sid: aligned_raws[sid] for sid in subject_ids[:2]})
    sub1, sub2 = subject_ids[0], subject_ids[1]

    rows = [{"sub1": sub1, "sub2": sub2, "label": a, "label2": b,
             "r": float(isc_mat[i, j])}
            for i, a in enumerate(ch_names) for j, b in enumerate(ch_names)]
    frame = pd.DataFrame(rows)
    frame.insert(frame.columns.get_loc("r") + 1, "r_z", frame["r"].map(_fisher_z))
    index = {name: i for i, name in enumerate(ch_names)}
    if orders1 is not None:
        frame["ar_order"]  = frame["label"].map(lambda c: orders1[index[c]])
        frame["ar_order2"] = frame["label2"].map(lambda c: orders2[index[c]])
    if max_lag:
        ij = (frame["label"].map(index).to_numpy(), frame["label2"].map(index).to_numpy())
        frame["lag_s"] = lags[ij] / sfreq

    null_level = None
    if n_null > 0:
        draws = _isc_null_draws(data1, data2, whiten, max_lag, n_null, seed)
        null_level = _add_isc_null_columns(frame, isc_mat, ch_names, draws)
    return isc_mat, ch_names, frame, null_level


def _isc_null_draws(
    data1: np.ndarray, data2: np.ndarray, whiten: int, max_lag: int, n_iter: int,
    seed: int | None,
) -> np.ndarray:
    """``n_iter`` ISC matrices against a phase-scrambled second member.

    Only the second member is scrambled, the choice :func:`compute_wtc_phase_null` makes for the
    same reason: scrambling both would test one surrogate against another, which is a weaker
    null than a real recording against a surrogate. A blank row stays blank, having no
    spectrum to preserve.
    """
    rng = np.random.default_rng(seed)
    finite = np.isfinite(data2).all(axis=1)
    draws = np.empty((n_iter, data1.shape[0], data2.shape[0]))
    for i in range(n_iter):
        surrogate = data2.copy()
        for row in np.flatnonzero(finite):
            surrogate[row] = phase_scramble(data2[row], rng)
        draws[i] = _isc_matrix(data1, surrogate, whiten, max_lag)[0]
    return draws


def _add_isc_null_columns(
    frame: pd.DataFrame, isc_mat: np.ndarray, ch_names: list[str], draws: np.ndarray,
) -> np.ndarray:
    """Summarise the surrogate draws onto ``frame`` in place, and return the 95th percentile.

    That percentile is per cell and is what a chord on the connectogram is drawn against, so
    it leaves as a matrix rather than only as a column.

    **Every null column describes the magnitude**, hence ``null_abs_``: a correlation is
    two-sided, so a surrogate that lands at -0.4 is as far from no coupling as one at +0.4,
    and the rank ``percentile`` reports is |r| among |draws|. The ``r`` column beside them is
    signed. Naming these ``null_mean`` next to a signed ``r`` invited reading one against the
    other, which compares a magnitude with a value that can be negative and makes a table look
    self-contradictory.
    """
    absolute = np.abs(draws)
    # a blanked channel makes a cell all-NaN, which every nan-aware reduction warns about and
    # then handles correctly; the blank mask below is what actually decides those cells
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        null_abs_mean = np.nanmean(absolute, axis=0)
        null_abs_sd   = np.nanstd(absolute, axis=0)
        null_abs_p95  = np.nanpercentile(absolute, 95, axis=0)
        beaten        = (absolute < np.abs(isc_mat)[None, :, :]).mean(axis=0) * 100
    # a cell whose draws are all NaN was never ranked against anything, and a count of zero
    # there would read as a real value that lost to every surrogate
    blank = ~np.isfinite(absolute).any(axis=0) | ~np.isfinite(isc_mat)
    beaten = np.where(blank, np.nan, beaten)

    index = {name: i for i, name in enumerate(ch_names)}
    ij = (frame["label"].map(index).to_numpy(), frame["label2"].map(index).to_numpy())
    frame["null_abs_mean"] = null_abs_mean[ij]
    frame["null_abs_sd"]   = null_abs_sd[ij]
    frame["null_abs_p95"]  = null_abs_p95[ij]
    frame["percentile"]    = beaten[ij]
    return null_abs_p95


def _isc_matrix(
    data1: np.ndarray, data2: np.ndarray, whiten: int, max_lag: int = 0,
) -> "tuple[np.ndarray, list[int] | None, list[int] | None, np.ndarray]":
    """The r matrix, the AR order each row was whitened at, and the lag each cell won at.

    Whitening happens after the window has been cut, so each stretch is fitted on its own,
    and before the lag search, so the search runs on the residuals rather than on the
    autocorrelation that would make every shift look alike.
    """
    orders1 = orders2 = None
    if whiten:
        data1, orders1 = _whiten_rows(data1, whiten)
        data2, orders2 = _whiten_rows(data2, whiten)
        n_keep = min(data1.shape[1], data2.shape[1])
        data1, data2 = data1[:, :n_keep], data2[:, :n_keep]
    # a rejected channel contributed a row of NaN above, which the products carry
    isc_mat, lags = _isc_from_rows(data1, data2, max_lag)
    return isc_mat, orders1, orders2, lags


def roi_mean_of_isc(
    isc_mat,
    ch_names: list[str],
    roi_map: dict[str, list[str]],
    min_channels: int = 2,
) -> "tuple[np.ndarray, list[str]] | tuple[None, None]":
    """Average the channel-level ISC inside each ROI pair: the ROI number beside the ROI WTC.

    ::

      4x4 r matrix over S1_D1..S4_D4 + {"L": ["S1_D1", "S2_D2"], "R": [...]}
        -> 2x2 r matrix over ["L", "R"]

    Built from the channel values rather than from an ROI-averaged signal, which is how
    :func:`roi_mean_of_channels` builds the ROI coherence, so the two ROI numbers a page
    prints rest on the same channels.

    Cell (i, j) is the first member's ROI i against the other's ROI j, so the matrix is
    crossed and asymmetric exactly as the channel one is.

    Averaged in Fisher z and returned as r, where the coherence path averages its values
    directly: a correlation is signed and bounded and these are, unlike coherences, spread
    across zero.

    Parameters
    ----------
    isc_mat : array, shape (n_ch, n_ch)
        What :func:`compute_isc` returned, NaN where a member lost a channel.
    ch_names : list of str
        The axis of ``isc_mat``, both sides.
    roi_map : dict
        ``{region: [channel label, ...]}``. Its key order is the returned axis.
    min_channels : int
        Least channel pairs a cell may rest on; thinner cells come back NaN. Counts pairs,
        so a crossed cell needs that many combinations rather than that many channels a side.

    Returns
    -------
    (matrix, labels) or (None, None)
        ``(None, None)`` when there is nothing to average.
    """
    if isc_mat is None or not ch_names or not roi_map:
        return None, None
    mat = np.asarray(isc_mat, dtype=float)
    labels = list(roi_map.keys())
    index = {name: i for i, name in enumerate(ch_names)}
    picks = {roi: [index[ch] for ch in chs if ch in index] for roi, chs in roi_map.items()}

    out = np.full((len(labels), len(labels)), np.nan)
    thin = 0
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if not picks[a] or not picks[b]:
                continue
            block = mat[np.ix_(picks[a], picks[b])]
            z = np.array([_fisher_z(r) for r in block.ravel()])
            z = z[np.isfinite(z)]
            if z.size < max(1, min_channels):
                thin += z.size > 0
                continue
            out[i, j] = float(np.tanh(z.mean()))
    if thin:
        logger.info("ROI ISC: %d cell(s) under %d channel pairs, left blank",
                    thin, min_channels)
    return out, labels
