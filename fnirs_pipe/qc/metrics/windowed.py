"""The same quality questions asked per window instead of per recording.

SCI, PSP and GVTD binned onto one shared non-overlapping window grid, so a report can show
where in a recording the quality moved rather than only its average. One grid for all three
on purpose: ``window_s * sfreq`` is rarely an integer, and three independently derived grids
drift apart over a long run.
"""

import mne
import numpy as np

from fnirs_pipe.qc.metrics.gvtd import compute_windowed_filtered_gvtd, compute_windowed_gvtd
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.metrics.windowed")


def compute_windowed_sci(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    window_s: float = 10.0,
) -> "tuple[np.ndarray, np.ndarray]":
    """SCI per channel in each non-overlapping window.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Optical-density recording.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.
    window_s : float, optional
        Non-overlapping window length in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores, times): SCI per channel per window (channel x window), and each
        window's [start, end] times.

    Notes
    -----
    Wrapper over mne_nirs ``scalp_coupling_index_windowed``; used to see how
    coupling drifts over the recording rather than as a single whole-run value.
    """
    from mne_nirs.preprocessing import scalp_coupling_index_windowed
    _, scores, times = scalp_coupling_index_windowed(
        raw_od, time_window=window_s, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq
    )
    return scores, times


def compute_windowed_psp(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    window_s: float = 10.0,
) -> "tuple[np.ndarray, np.ndarray]":
    """Peak spectral power per channel in each non-overlapping window.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Optical-density recording.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.
    window_s : float, optional
        Non-overlapping window length in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores, times): PSP per channel per window, and each window's
        [start, end] times.

    Notes
    -----
    Wrapper over mne_nirs ``peak_power``; the windowed companion to the whole-run
    PSP, showing how the cardiac peak strength varies over time.
    """
    from mne_nirs.preprocessing import peak_power
    _, scores, times = peak_power(
        raw_od, time_window=window_s, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq
    )
    return scores, times


def attach_windowed_series(
    sqm: dict,
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    window_s: float = 10.0,
    *,
    gvtd_od: "mne.io.Raw | None" = None,
) -> dict:
    """Compute sliding-window SCI/PSP/GVTD series, attach summaries to sqm, return raw series.

    Parameters
    ----------
    sqm : dict
        Metric dict; per-window summaries are attached to it in place.
    raw_od : mne.io.Raw
        Optical-density recording the SCI and PSP series are read off.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.
    window_s : float, optional
        Non-overlapping window length in seconds.
    gvtd_od : mne.io.Raw or None, optional
        A second optical-density recording for the GVTD series alone; ``raw_od`` when None.

    Returns
    -------
    dict
        Raw series (sci_matrix/sci_times/psp_matrix/psp_times) for callers that
        also plot them; all None on failure.

    Notes
    -----
    SCI/PSP and GVTD fail independently; whichever survives is still attached. Center
    times collapse the mne-nirs [start, end] window pairs to their midpoint. ``window_s``
    is stored as ``qc_window_s`` so a record says which grid its series were binned on;
    ``psp_mean`` is deliberately not on that grid, see :data:`~fnirs_pipe.qc.metrics.coupling.PSP_WINDOW_S`.

    ``gvtd_od`` exists because the two families want different stages. SCI and PSP measure
    optode coupling, which the motion correction is not supposed to change, so they belong
    on the uncorrected file where the per-channel scores were taken. GVTD measures movement,
    which the correction is entirely about, so it belongs on the corrected one. Both grids
    are derived from ``window_s`` the same way, so the series stay time-aligned across the
    two files as long as neither was resampled.
    """
    def _center_times(t):
        a = np.asarray(t)
        return (a.mean(axis=1) if a.ndim == 2 and a.shape[1] == 2 else a).tolist()

    series = {"sci_matrix": None, "sci_times": None, "psp_matrix": None, "psp_times": None}
    # recorded even when every series below fails: it describes the request, not the result,
    # and without it a stored series cannot be told apart from one binned at another length
    sqm["qc_window_s"] = float(window_s)

    # two try blocks, not one: SCI/PSP filter to the cardiac band and GVTD does not, so a band
    # that the filter rejects must not take the motion series down with it
    sci_matrix = sci_times = psp_matrix = psp_times = None
    try:
        sci_matrix, sci_times = compute_windowed_sci(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
        psp_matrix, psp_times = compute_windowed_psp(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
    except Exception as exc:
        logger.warning("windowed SCI/PSP failed: %s", exc)

    gvtd_per_window = gvtd_p95_per_window = gvtd_t = None
    gvtd_filt_per_window = gvtd_filt_p95_per_window = None
    raw_gvtd = raw_od if gvtd_od is None else gvtd_od
    try:
        gvtd_per_window, gvtd_p95_per_window, gvtd_t = compute_windowed_gvtd(raw_gvtd, window_s)
        gvtd_filt_per_window, gvtd_filt_p95_per_window, _ = compute_windowed_filtered_gvtd(raw_gvtd, window_s)
    except Exception as exc:
        logger.warning("windowed GVTD failed: %s", exc)

    # SCI/PSP matrices are channel × window; collapse to a per-window mean over channels
    if sci_matrix is not None and sci_times is not None:
        sqm["sci_per_window"]      = np.asarray(sci_matrix).mean(axis=0).tolist()
        sqm["sci_window_times_s"]  = _center_times(sci_times)
    if psp_matrix is not None and psp_times is not None:
        sqm["psp_per_window"]      = np.asarray(psp_matrix).mean(axis=0).tolist()
        sqm["psp_window_times_s"]  = _center_times(psp_times)
    if gvtd_per_window is not None and len(gvtd_per_window):
        sqm["gvtd_per_window"]     = np.asarray(gvtd_per_window).tolist()
        sqm["gvtd_p95_per_window"] = np.asarray(gvtd_p95_per_window).tolist()
        sqm["gvtd_window_times_s"] = _center_times(gvtd_t)
    if gvtd_filt_per_window is not None and len(gvtd_filt_per_window):
        sqm["gvtd_filt_per_window"]     = np.asarray(gvtd_filt_per_window).tolist()
        sqm["gvtd_filt_p95_per_window"] = np.asarray(gvtd_filt_p95_per_window).tolist()

    series.update(sci_matrix=sci_matrix, sci_times=sci_times,
                  psp_matrix=psp_matrix, psp_times=psp_times)
    return series


# The window the SCI and PSP pass lines were established at. Deliberately not the report's
# `window_s`, which a user may set freely: PSP is a power and moves with window length (a
# well-coupled channel's rises, an uncoupled channel's falls), so 0.1 selects a different
# set of channels at every length, and a screening line that follows a display setting is a
# screening line that silently stops meaning what it was calibrated to mean. SCI is a
# correlation and is window-free, so only PSP forces this, and one window for both keeps the
# two on one grid.
SCREEN_WINDOW_S = 10.0


def window_centers(times) -> "np.ndarray":
    """Mid-time of each window, from the ``[start, end]`` pairs mne_nirs returns.

    ::

      [[0, 10], [10, 20]]  ->  array([5., 15.])

    A window belongs at its middle, and a scope is decided on that one time rather than on
    overlap, so a window straddling the edge of a task block counts for the side it mostly
    sits in and is never counted twice.
    """
    a = np.asarray(times, dtype=float)
    return a.mean(axis=1) if a.ndim == 2 and a.shape[1] == 2 else a.ravel()


def task_scope_windows(
    raw: mne.io.Raw, min_duration: float = 2 * SCREEN_WINDOW_S,
) -> "list[tuple[str, float, float]]":
    """[(label, tstart, tstop)] over the annotations long enough to hold screening windows.

    ::

      "rest" at 20 s for 300 s, plus a 5 s trigger  ->  [("rest", 20.0, 320.0)]

    The annotation's own duration is the window, so a recording carrying only zero-length
    or short triggers yields nothing here and the caller keeps whatever scope it had. That
    is the case worth knowing about: scoping to five 10 s triggers out of an hour would
    count one minute of the recording and still read like a verdict on the whole thing.

    Clamped to the recording, and on the data axis: a cropped Raw keeps its annotations on
    the original axis while its samples restart at zero.
    """
    origin = float(raw.first_time)
    end = float(raw.times[-1])
    out = []
    for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration,
                                raw.annotations.description):
        start = float(onset) - origin
        stop = min(start + float(dur), end)
        if stop - start >= float(min_duration):
            out.append((str(desc), max(0.0, start), stop))
    return out


def _coupled_mask(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    sci_cutoff: float,
    psp_cutoff: float,
    window_s: float,
):
    """(mask, centers): mask is channel x window, True where the channel is coupled."""
    try:
        sci, times = compute_windowed_sci(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
        psp, _ = compute_windowed_psp(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
    except Exception as exc:
        logger.warning("windowed screening could not be measured (%s); it screens nothing",
                       exc)
        return None, None

    sci, psp = np.asarray(sci, dtype=float), np.asarray(psp, dtype=float)
    if sci.shape != psp.shape or sci.ndim != 2 or sci.shape[1] == 0:
        logger.warning("windowed SCI %s and PSP %s do not share a grid; screening nothing",
                       sci.shape, psp.shape)
        return None, None
    return (sci >= float(sci_cutoff)) & (psp >= float(psp_cutoff)), window_centers(times)


def _in_scope(centers, scope) -> "np.ndarray":
    """Which windows a scope keeps, by window centre. No scope keeps every window."""
    if not scope:
        return np.ones(len(centers), dtype=bool)
    keep = np.zeros(len(centers), dtype=bool)
    for window in scope:
        t0, t1 = float(window[-2]), float(window[-1])
        keep |= (centers >= t0) & (centers <= t1)
    return keep


def good_window_fraction(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    sci_cutoff: float,
    psp_cutoff: float,
    window_s: float = SCREEN_WINDOW_S,
    scope: "list[tuple[str, float, float]] | None" = None,
) -> dict[str, float]:
    """Share of windows in which a channel is coupled, per channel.

    ::

      a channel passing both lines in 300 of 391 windows  ->  0.767

    A window counts when SCI **and** PSP both pass in *that* window. The two are paired
    inside the window rather than judged separately over the recording because they are
    only informative together: movement inflates SCI, and PSP is what catches it, so a
    window with high SCI and near-zero PSP is movement rather than coupling. Comparing a
    whole-run SCI against a whole-run PSP cannot see that, since it no longer knows whether
    the good SCI and the bad PSP happened at the same time.

    Counting windows rather than averaging them is the other half. An average can be carried
    over the line by the part of the recording where the channel was fine, so a channel that
    is excellent for half the run and dead for the other half passes; a count cannot be
    rescued that way.

    ``scope`` restricts the **denominator** to the windows whose centres fall inside those
    stretches. A run usually holds time no analysis reads, a lead-in before the first block
    and the gaps between blocks, and a channel coupled throughout every block should not be
    rejected for what it did while nobody was doing anything. None counts the whole
    recording.

    The scope masks one whole-record pass rather than cutting the recording and measuring
    each piece: SCI and PSP filter to the cardiac band, so a cut piece is filtered against
    its own edges and lands on its own window grid. Masking keeps one grid and one filter.

    Returns an empty dict when either metric cannot be measured, or when the scope keeps no
    window, which screens nothing rather than rejecting everything.
    """
    mask, centers = _coupled_mask(raw_od, cardiac_l_freq, cardiac_h_freq,
                                  sci_cutoff, psp_cutoff, window_s)
    if mask is None:
        return {}
    keep = _in_scope(centers, scope)
    if not keep.any():
        logger.warning("the screening scope keeps none of the %d windows; screening nothing",
                       len(centers))
        return {}
    if scope:
        logger.info("channel screening counts %d of %d windows, %.0f%% of the recording",
                    int(keep.sum()), len(centers), 100 * keep.mean())
    frac = mask[:, keep].mean(axis=1)
    return {ch: float(frac[i]) for i, ch in enumerate(raw_od.ch_names) if i < len(frac)}


def condition_window_fractions(
    raw_od: mne.io.Raw,
    windows: "list[tuple[str, float, float]]",
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    sci_cutoff: float,
    psp_cutoff: float,
    window_s: float = SCREEN_WINDOW_S,
) -> "dict[str, dict[str, float]]":
    """:func:`good_window_fraction` restricted to each named stretch, one share per stretch.

    ::

      [("rest", 0, 300), ("talk", 300, 600)]  ->  {"rest": {ch: 0.98}, "talk": {ch: 0.41}}

    Reported, never screened on. One channel set has to serve every condition or a contrast
    between two conditions is also a contrast between two montages, so this answers "when
    was this channel bad" without changing what is dropped.

    One windowed pass masked per condition, for the reason
    :func:`good_window_fraction` gives: cutting each condition out first would filter each
    piece against its own edges and put each on its own grid, so the conditions would be
    comparable neither with each other nor with the run-wide share.
    """
    mask, centers = _coupled_mask(raw_od, cardiac_l_freq, cardiac_h_freq,
                                  sci_cutoff, psp_cutoff, window_s)
    if mask is None:
        return {}
    names = list(raw_od.ch_names)
    out: dict[str, dict[str, float]] = {}
    for window in windows:
        keep = _in_scope(centers, [window])
        if not keep.any():
            logger.warning("condition %s holds no whole screening window", window[0])
            continue
        frac = mask[:, keep].mean(axis=1)
        out[window[0]] = {ch: float(frac[i]) for i, ch in enumerate(names) if i < len(frac)}
    return out
