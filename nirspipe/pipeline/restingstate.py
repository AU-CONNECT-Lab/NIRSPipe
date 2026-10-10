from __future__ import annotations

import numpy as np
import pandas as pd
import mne
from scipy import signal

from nirspipe.pipeline.glm_censored import kept_frames
from nirspipe.utils import fisher_r_to_z
from nirspipe.utils.logging import get_logger

logger = get_logger("post.restingstate")


def lomb_scargle(data: np.ndarray, keep: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Lomb-Scargle power of each row from its kept samples, on the periodogram's own grid.

    The classic periodogram with Scargle's time shift :math:`\tau`, over the kept sample
    times :math:`t_j` of a series demeaned on them:

    .. math::

        P(\omega) = \frac{1}{N_k} \left[
            \frac{\left(\sum_j y_j \cos \omega (t_j - \tau)\right)^2}{\sum_j \cos^2 \omega (t_j - \tau)}
          + \frac{\left(\sum_j y_j \sin \omega (t_j - \tau)\right)^2}{\sum_j \sin^2 \omega (t_j - \tau)}
        \right], \qquad
        \tan 2\omega\tau = \frac{\sum_j \sin 2\omega t_j}{\sum_j \cos 2\omega t_j}

    at the Fourier frequencies of the full length, :math:`N_k` the kept count. This is
    ``scipy.signal.lombscargle(..., normalize=True)`` times the kept samples' variance, which
    puts it on the scale of ``scipy.signal.periodogram(..., scaling="spectrum")``; with every
    sample kept the two are equal.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Full-length series; values at dropped samples are ignored.
    keep : ndarray of bool, shape (n_times,)
        False on dropped samples.
    sfreq : float
        Sampling rate in Hz.

    Returns
    -------
    freqs : ndarray, shape (n_times // 2 + 1,)
        ``np.fft.rfftfreq(n_times, 1 / sfreq)``, DC included.
    power : ndarray, shape (n_channels, n_times // 2 + 1)
        Power per frequency; the DC bin is 0, as after demeaning.

    Notes
    -----
    The kept samples sit on a uniform grid, so every sum above is a DFT of the zero-filled
    series or of the mask, and the 2ω sums are the mask's DFT at bin :math:`2k \bmod n`. The
    result is exact, not the Press-Rybicki approximation: it matches scipy to about 1e-9.
    At the Nyquist bin of an even length every :math:`\sin \omega t_j` is 0, so that term is
    dropped rather than divided by zero.
    """
    n = data.shape[1]
    n_kept = int(keep.sum())
    k = np.arange(1, n // 2 + 1)
    y = data[:, keep]
    z = np.zeros_like(data, dtype=float)
    z[:, keep] = y - y.mean(axis=1, keepdims=True)

    spectrum = np.fft.fft(z, axis=1)[:, k]
    mask_2w = np.fft.fft(keep.astype(float))[(2 * k) % n]
    c2, s2 = mask_2w.real, -mask_2w.imag  # sums of cos 2wt and sin 2wt
    two_w_tau = np.arctan2(s2, c2)
    cos_t, sin_t = np.cos(two_w_tau / 2), np.sin(two_w_tau / 2)
    yc_raw, ys_raw = spectrum.real, -spectrum.imag
    yc = yc_raw * cos_t + ys_raw * sin_t
    ys = ys_raw * cos_t - yc_raw * sin_t
    cc = 0.5 * (n_kept + c2 * np.cos(two_w_tau) + s2 * np.sin(two_w_tau))
    ss = n_kept - cc

    # a sine term whose basis vanishes on every kept sample carries no power
    tiny = 1e-9 * n_kept
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_term = np.where(cc > tiny, yc ** 2 / cc, 0.0)
        sin_term = np.where(ss > tiny, ys ** 2 / ss, 0.0)
    power = np.zeros((data.shape[0], n // 2 + 1))
    power[:, 1:] = (cos_term + sin_term) / n_kept
    return np.fft.rfftfreq(n, 1 / sfreq), power


def compute_alff(raw: mne.io.Raw, low_pass: float, high_pass: float,
                 exclude: "list[str] | None" = None) -> pd.DataFrame:
    r"""ALFF/fALFF per channel plus cross-channel mALFF/zALFF standardization.

    Expects a broadband (detrended, non-lowpassed) residual: fALFF's denominator spans the
    full spectrum, so a bandpassed input collapses fALFF to ~1. ALFF only reads the band, so
    one broadband signal serves both (Zang 2007; Zou 2008):

    .. math::

        \text{ALFF} = \frac{1}{|B|} \sum_{f \in B} \sqrt{P(f)}, \qquad
        \text{fALFF} = \frac{\sum_{f \in B} \sqrt{P(f)}}{\sum_{f > 0} \sqrt{P(f)}},

        \text{mALFF}_c = \frac{\text{ALFF}_c}{\overline{\text{ALFF}}}, \qquad
        \text{zALFF}_c = \frac{\text{ALFF}_c - \overline{\text{ALFF}}}{\operatorname{sd}(\text{ALFF})},

    where :math:`B = [\text{high\_pass}, \text{low\_pass}]` is the low-frequency band and
    :math:`P(f)` is the periodogram power (amplitude is :math:`\sqrt{P}`). fALFF is in
    :math:`[0, 1]`; the DC bin is dropped from the denominator. mALFF/zALFF standardize the
    scale-dependent ALFF across channels so it is comparable at the group level.

    Notes
    -----
    With ``BAD_`` spans on the recording, :math:`P(f)` is the :func:`lomb_scargle` power of
    the kept samples on the same frequency grid, so both sums run over the same bins as
    without spans; the denominator still reaches Nyquist. A channel with no finite sample
    gets NaN for both measures; a constant one keeps 0.

    All four measures are NaN on a rejected channel, and the ``bad`` column says which those
    are. The mALFF/zALFF reference mean and SD already exclude them, so the blanking only
    removes the values themselves.

    ``exclude`` names channels to treat exactly like rejected ones without calling them
    rejected: blanked, and out of the reference. It exists for a channel whose residual is
    identically zero because it was the whole short-channel regressor and was fitted against
    a copy of itself. Blanking such a channel while leaving it in the reference puts a
    numerical zero in the mean and SD that every other channel is scaled by, which shifts
    every other channel's mALFF. The ``bad`` column keeps meaning rejected,
    since that is what the group tables read.

    Rest mode produces two residuals and this function must receive the broadband one. The
    pipeline runs the confound regression a second time without the low-pass and feeds that
    residual here (FC keeps the bandpassed one). The linear detrend before the FFT is
    supplied by the GLM drift regressors.
    """
    data = raw.get_data()  # (n_channels, n_times)
    fs = raw.info["sfreq"]
    keep = kept_frames(raw)
    censored = not keep.all()

    alff_vals  = np.zeros(len(raw.ch_names))
    falff_vals = np.zeros(len(raw.ch_names))

    for i, ch_data in enumerate(data):
        if not np.isfinite(ch_data).any():
            alff_vals[i] = falff_vals[i] = np.nan
            continue
        if np.nanstd(ch_data[keep] if censored else ch_data) == 0:
            continue

        if censored:
            freqs, power = lomb_scargle(ch_data[np.newaxis], keep, fs)
            power = power[0]
        else:
            ch_demeaned = ch_data - np.nanmean(ch_data)
            freqs, power = signal.periodogram(ch_demeaned, fs, scaling="spectrum")
        power_sqrt   = np.sqrt(power)

        # high_pass is the lower bound, low_pass the upper; inclusive mask keeps both edge bins
        band_mask = (freqs >= high_pass) & (freqs <= low_pass)
        band_amp  = power_sqrt[band_mask]

        alff_vals[i] = np.nanmean(band_amp) if band_amp.size else 0.0  # mean band amplitude

        # fALFF: fraction of total spectral amplitude in the low band (Zou 2008), sum/sum ∈ [0,1]
        total_sum = np.nansum(power_sqrt[1:])  # skip DC
        falff_vals[i] = np.nansum(band_amp) / total_sum if total_sum > 0 else 0.0

    # mALFF/zALFF standardize within each chromophore: HbO and HbR sit on different amplitude
    # scales, so a pooled mean/std would distort both; each chromophore is normalized on its own.
    malff_vals = np.zeros_like(alff_vals)
    zalff_vals = np.zeros_like(alff_vals)
    bads = set(raw.info["bads"])
    is_bad = np.array([c in bads for c in raw.ch_names])
    # blanked covers both reasons a channel carries no value; is_bad stays rejection-only
    blanked = np.array([c in bads or c in set(exclude or ()) for c in raw.ch_names])
    for suffix in (" hbo", " hbr"):
        idx = np.array([c.endswith(suffix) for c in raw.ch_names])
        if not idx.any():
            continue
        # blanked channels are still standardized, but they do not define the reference:
        # their ALFF would shift the mean and SD that every good channel is scaled by
        ref = idx & ~blanked
        if not ref.any():
            ref = idx
        grp_mean = np.nanmean(alff_vals[ref])
        # population SD (ddof=0): a rescaling of a fully observed set, not an estimate of anything
        grp_std  = np.nanstd(alff_vals[ref], ddof=0)
        if grp_mean != 0:
            malff_vals[idx] = alff_vals[idx] / grp_mean
        if grp_std != 0 and np.isfinite(grp_std):
            zalff_vals[idx] = (alff_vals[idx] - grp_mean) / grp_std

    # blanked rather than dropped, so the frame keeps one row per channel and a group
    # analysis can stack subjects whose rejections differ
    for vals in (alff_vals, falff_vals, malff_vals, zalff_vals):
        vals[blanked] = np.nan

    return pd.DataFrame({
        "channel": raw.ch_names,
        "alff":    alff_vals,
        "falff":   falff_vals,
        "malff":   malff_vals,
        "zalff":   zalff_vals,
        "bad":     is_bad,
    })


_ALFF_COLUMNS = ("alff", "falff", "malff", "zalff")


def compute_alff_roi(alff_df: pd.DataFrame, raw: mne.io.Raw,
                     roi_map: "dict[str, list[str]]") -> pd.DataFrame:
    r"""ROI-level amplitude: the mean of each ROI's member channels, per chromophore.

    .. math::

        \overline{\mathrm{ALFF}}_A = \frac{1}{|A|} \sum_{c \in A} \mathrm{ALFF}_c

    ::

      alff_df (one row per channel) + {"L": ["S1_D1", ...]}
        -> one row per (ROI, chromophore) with the four measures averaged

    **The averaging happens after the measure, which is the opposite of**
    :func:`compute_fc_roi`. An ROI mean signal carries only the part its channels share, so an
    amplitude measured on it falls as the ROI grows and as its channels agree less.

    Rejected channels are dropped before averaging rather than propagating NaN, the way every
    other ROI-level product here drops them; ``n_channels`` records how many survived, without
    which a thinly covered ROI reads like a well covered one. An ROI with nothing left keeps its
    row, ``n_channels`` 0 and the measures blank, so every subject's table has the same rows.
    """
    if alff_df is None or alff_df.empty or not roi_map:
        return pd.DataFrame()
    by_channel = alff_df.set_index("channel")
    rows = []
    for chromophore in ("hbo", "hbr"):
        members = _roi_members(raw, roi_map, chromophore)
        for roi in roi_map:
            picks = [c for c in members.get(roi, []) if c in by_channel.index]
            values = by_channel.loc[picks, list(_ALFF_COLUMNS)]
            # a channel blanked after the fact (a short channel regressed out of itself) is
            # still a member, so count what actually carried a value rather than len(picks)
            usable = values.dropna(how="all")
            if usable.empty:
                rows.append({"roi": roi, "chromophore": chromophore, "n_channels": 0,
                             **{c: np.nan for c in _ALFF_COLUMNS}})
                continue
            rows.append({"roi": roi, "chromophore": chromophore, "n_channels": len(usable),
                         **usable.mean().to_dict()})
    return pd.DataFrame(rows, columns=["roi", "chromophore", "n_channels", *_ALFF_COLUMNS])


def compute_fc(raw: mne.io.Raw, chromophore: str) -> pd.DataFrame:
    r"""Functional connectivity for one chromophore: channel-by-channel Pearson matrix.

    .. math::

        \rho_{ij} = \operatorname{corr}(x_i, x_j)

    over the channels of a single chromophore (``chromophore`` is "hbo" or "hbr"); HbO and HbR
    anti-correlate, so a mixed matrix has no clean meaning and the two are kept separate. Diagonal
    is 1. Empty frame if the chromophore has < 2 channels. Only frames outside every ``BAD_``
    span enter the correlation, here and in the ROI and seed products.

    A rejected channel's row and column are NaN, not dropped: every subject's matrix keeps the
    same shape and the same channel order, so a group analysis can stack them however their
    rejections differ. It is the package's convention for every channel-by-channel matrix,
    :func:`nirspipe.pipeline.hyper.isc.compute_isc` included.
    ROI-level products do the opposite and drop the rejected channels before averaging (see
    :func:`compute_fc_roi`): a bad channel inside an ROI mean reaches every correlation that
    ROI takes part in, where in a channel matrix it is confined to one row and one column.

    Plain Pearson, not shrinkage: a shrinkage estimator shrinks by an amount that tracks the
    channel-to-sample ratio, so it would differ between subjects. Every FC product here uses
    the same one.
    """
    picks = [c for c in raw.ch_names if c.endswith(f" {chromophore}")]
    if len(picks) < 2:
        return pd.DataFrame()
    fc = pd.DataFrame(np.corrcoef(_kept(raw, raw.get_data(picks=picks))), index=picks, columns=picks)
    bads = [c for c in picks if c in set(raw.info["bads"])]
    fc.loc[bads, :] = np.nan
    fc.loc[:, bads] = np.nan
    return fc


def _kept(raw: mne.io.Raw, data: np.ndarray) -> np.ndarray:
    """The columns of ``data`` outside every ``BAD_`` span; ``data`` itself when there are none."""
    keep = kept_frames(raw)
    return data if keep.all() else data[..., keep]


def fisher_z(fc: pd.DataFrame) -> pd.DataFrame:
    r"""Fisher r-to-z of an FC matrix (diagonal set to 0), for group-level stats.

    .. math::

        z = \operatorname{arctanh}(r) = \tfrac{1}{2} \ln \frac{1 + r}{1 - r}

    Variance-stabilises correlations so they can be averaged / tested across subjects;
    r is clipped just below :math:`\pm 1` to keep perfect correlations from diverging.
    """
    z = fisher_r_to_z(fc.to_numpy())
    # only a square matrix has a self-correlation diagonal; a seed map is ROI x channel, where
    # position (i, i) is an ordinary pair and zeroing it would delete a real value
    if z.ndim == 2 and z.shape[0] == z.shape[1]:
        # a blank row (bad channel, empty ROI) keeps a blank diagonal rather than a 0
        np.fill_diagonal(z, np.where(np.isfinite(np.diag(z)), 0.0, np.nan))
    return pd.DataFrame(z, index=fc.index, columns=fc.columns)


def compute_fc_roi(raw: mne.io.Raw, roi_map: dict[str, list[str]], chromophore: str = "hbo") -> pd.DataFrame:
    r"""ROI-level FC for one chromophore: average each ROI's channels, then Pearson corr between ROIs.

    .. math::

        \rho_{AB} = \operatorname{corr}(\bar{x}_A, \bar{x}_B), \qquad
        \bar{x}_A = \frac{1}{|A|} \sum_{c \in A} x_c

    Averages signals first (higher SNR) rather than averaging channel correlations; roi_map is
    {ROI label: [channel names]}, matched by S-D base ("S1_D1"), with any chromophore suffix on
    the map entry replaced by the requested one so the same map serves hbo and hbr.

    Every ROI the map lists keeps its row and column; one with no good channel is blank, so
    every subject's matrix has the same shape. Empty when fewer than two ROIs resolve, since
    no correlation is left to compute.
    """
    members = _roi_members(raw, roi_map, chromophore)
    if len(members) < 2:
        return pd.DataFrame()
    names = list(members)
    signals = [raw.get_data(picks=picks).mean(axis=0) for picks in members.values()]
    fc = pd.DataFrame(np.corrcoef(_kept(raw, np.vstack(signals))), index=names, columns=names)
    labels = list(roi_map)
    return fc.reindex(index=labels, columns=labels)


def _roi_members(
    raw: mne.io.Raw, roi_map: dict[str, list[str]], chromophore: str,
) -> "dict[str, list[str]]":
    """{ROI: its good channel names} for one chromophore, in roi_map order.

    e.g. roi_map {"L-PFC": ["S1_D1 hbo", "S2_D2 hbo"]} with chromophore "hbr" and S2_D2 bad
    gives {"L-PFC": ["S1_D1 hbr"]}. Map entries are matched by S-D base, so any chromophore
    suffix on them is replaced rather than required, and the same map serves hbo and hbr.
    An ROI with no good channel is left out of the members with a warning; the products built
    from them keep it as a blank row.
    """
    suffix = f" {chromophore}"
    # a bad channel in the average would travel into every correlation this ROI takes part in
    bads = set(raw.info["bads"])
    chan_set = {c for c in raw.ch_names if c.endswith(suffix) and c not in bads}
    members: dict[str, list[str]] = {}
    for roi, chans in roi_map.items():
        picks = []
        for c in chans:
            base = c[:-4] if (c.endswith(" hbo") or c.endswith(" hbr")) else c
            name = f"{base}{suffix}"
            if name in chan_set:
                picks.append(name)
        if picks:
            members[roi] = picks
        else:
            logger.warning("ROI %s has no good %s channel - left blank in the ROI outputs", roi, chromophore)
    return members


def compute_fc_seed(raw: mne.io.Raw, roi_map: dict[str, list[str]], chromophore: str = "hbo") -> pd.DataFrame:
    r"""Seed-to-whole-brain FC: each ROI's mean signal against every channel.

    .. math::

        \rho_{Ac} = \operatorname{corr}(\bar{x}_A, x_c), \qquad
        \bar{x}_A = \frac{1}{|A|} \sum_{k \in A} x_k

    Parameters
    ----------
    raw : mne.io.Raw
        Recording to correlate, one row per channel.
    roi_map : dict[str, list[str]]
        {ROI label: [channel names]}; every ROI becomes one seed, i.e. one row.
    chromophore : str, optional
        "hbo" or "hbr". HbO and HbR anti-correlate, so they never share a map.

    Returns
    -------
    pd.DataFrame
        ROI x channel correlations, one row per listed ROI, blank for one with no good
        channel. Empty frame if no ROI resolves to a good channel.

    Notes
    -----
    Averaging happens on one side only, which makes this a third product rather than a view
    of the other two: it is neither a slice of :func:`compute_fc` (no averaging) nor of
    :func:`compute_fc_roi` (both sides averaged). Averaging first cancels each channel's
    independent noise, so the same pair of regions reads higher through an ROI mean than
    through the mean of its channel-pair correlations.

    A seed's own channels are set to NaN, identified by **membership, not by a correlation
    near 1**: they sit inside the average, so their correlation is inflated by construction,
    and a test by value would also erase a channel that genuinely tracks the seed.

    Membership means the channels actually averaged, so a rejected one listed in ``roi_map``
    is not part of any seed. Columns span every channel, a rejected one's being NaN, as in
    :func:`compute_fc`.
    """
    members = _roi_members(raw, roi_map, chromophore)
    cols = [c for c in raw.ch_names if c.endswith(f" {chromophore}")]
    if not members or len(cols) < 2:
        return pd.DataFrame()

    data = _kept(raw, raw.get_data(picks=cols))
    col_index = {c: i for i, c in enumerate(cols)}
    bad_cols = [col_index[c] for c in cols if c in set(raw.info["bads"])]
    rows = []
    for picks in members.values():
        seed = _kept(raw, raw.get_data(picks=picks).mean(axis=0))
        r = np.corrcoef(np.vstack([seed, data]))[0, 1:]   # row 0 is the seed against every column
        r[[col_index[c] for c in picks]] = np.nan
        r[bad_cols] = np.nan
        rows.append(r)
    seed_map = pd.DataFrame(np.vstack(rows), index=list(members), columns=cols)
    return seed_map.reindex(index=list(roi_map))
