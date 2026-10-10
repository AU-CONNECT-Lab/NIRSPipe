"""The same quality questions asked per window instead of per recording.

SCI, PSP and GVTD binned onto one shared non-overlapping window grid, so a report can show
where in a recording the quality moved rather than only its average. One grid for all three
on purpose: ``window_s * sfreq`` is rarely an integer, and three independently derived grids
drift apart over a long run.
"""

import warnings

import mne
import numpy as np

from nirspipe.qc.metrics._helpers import long_short_channels
from nirspipe.qc.metrics.coupling import _window_samples, _windowed_cv, blank_flat_windows
from nirspipe.qc.metrics.gvtd import (
    compute_windowed_filtered_gvtd,
    compute_windowed_gvtd,
)
from nirspipe.utils import is_marker
from nirspipe.utils.logging import get_logger
from nirspipe.utils.spans import bad_spans

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
    return blank_flat_windows(raw_od, scores, window_s), times


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
    return blank_flat_windows(raw_od, scores, window_s), times


def compute_windowed_cv(
    raw_intensity: mne.io.Raw,
    window_s: float = 10.0,
) -> "tuple[np.ndarray, np.ndarray]":
    """CV per channel in each non-overlapping window, and each window's [start, end] times.

    Returned on the same grid and in the same shape as the SCI and PSP series, so the three
    line up column for column. Measured on raw intensity, which is what CV is defined on:
    after the optical-density conversion sigma/mu no longer means relative brightness.
    """
    data = raw_intensity.get_data()
    sfreq = float(raw_intensity.info["sfreq"])
    n = _window_samples(window_s, sfreq)
    if n < 2 or data.shape[1] < n:
        raise ValueError(f"window of {window_s} s does not fit the recording")
    cv = _windowed_cv(data, n)
    starts = np.arange(cv.shape[1]) * (n / sfreq)
    return cv, np.stack([starts, starts + n / sfreq], axis=1)


def attach_windowed_series(
    sqm: dict,
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    window_s: float = 10.0,
    *,
    gvtd_od: "mne.io.Raw | None" = None,
    raw_intensity: "mne.io.Raw | None" = None,
    sep_bands=None,
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
        also plot them, and each matrix's row names (sci_channels etc.); all None on failure.

    Notes
    -----
    SCI/PSP and GVTD fail independently; whichever survives is still attached. Center
    times collapse the mne-nirs [start, end] window pairs to their midpoint. ``window_s``
    is stored as ``qc_window_s`` so a record says which grid its series were binned on;
    ``psp_mean`` is deliberately not on that grid, see :data:`~nirspipe.qc.metrics.coupling.PSP_WINDOW_S`.

    ``gvtd_od`` lets GVTD run on the motion-corrected file while SCI and PSP stay on the
    uncorrected one, where the per-channel scores were taken. Both grids
    are derived from ``window_s`` the same way, so the series stay time-aligned across the
    two files as long as neither was resampled.
    """
    series = {"sci_matrix": None, "sci_times": None, "psp_matrix": None, "psp_times": None,
              "cv_matrix": None, "cv_times": None,
              "sci_channels": None, "psp_channels": None, "cv_channels": None}
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

    # a third try block: CV is the only one measured on intensity, so it fails on its own
    cv_matrix = cv_times = None
    if raw_intensity is not None:
        try:
            cv_matrix, cv_times = compute_windowed_cv(raw_intensity, window_s)
        except Exception as exc:
            logger.warning("windowed CV failed: %s", exc)

    gvtd_per_window = gvtd_p95_per_window = gvtd_t = None
    gvtd_filt_per_window = gvtd_filt_p95_per_window = None
    raw_gvtd = raw_od if gvtd_od is None else gvtd_od
    # One series per separation set, because GVTD is an RMS *across* channels: a long-channel
    # series cannot be recovered from an all-channel one the way an SCI column can be picked
    # out of the stored matrix. The plain keys are the long channels, following
    # `spike_spans_s`, and `_short` / `_all` name the others.
    gvtd_sets: "dict[str, list[str] | None]" = {"": None, "_short": None, "_all": None}
    try:
        long_names, short_names = long_short_channels(raw_gvtd, sep_bands)
        gvtd_sets = {"": long_names or None, "_short": short_names or None, "_all": None}
    except Exception as exc:
        logger.warning("GVTD separation split failed (%s); only the pooled series is stored", exc)

    for suffix, names in gvtd_sets.items():
        if suffix != "_all" and not names:
            continue
        try:
            picked = raw_gvtd.copy().pick(names) if names else raw_gvtd
            means, p95s, times = compute_windowed_gvtd(picked, window_s)
            f_means, f_p95s, _ = compute_windowed_filtered_gvtd(picked, window_s)
        except Exception as exc:
            logger.warning("windowed GVTD%s failed: %s", suffix or " (long)", exc)
            continue
        if suffix == "":
            gvtd_per_window, gvtd_p95_per_window, gvtd_t = means, p95s, times
            gvtd_filt_per_window, gvtd_filt_p95_per_window = f_means, f_p95s
            continue
        if means is not None and len(means):
            sqm[f"gvtd_per_window{suffix}"] = np.asarray(means).tolist()
            sqm[f"gvtd_p95_per_window{suffix}"] = np.asarray(p95s).tolist()
            sqm[f"gvtd_filt_per_window{suffix}"] = np.asarray(f_means).tolist()
            sqm[f"gvtd_filt_p95_per_window{suffix}"] = np.asarray(f_p95s).tolist()

    # SCI/PSP matrices are channel × window; collapse to a per-window mean over channels
    if sci_matrix is not None and sci_times is not None:
        sqm["sci_per_window"]      = np.asarray(sci_matrix).mean(axis=0).tolist()
        sqm["sci_window_times_s"]  = window_centers(sci_times).tolist()
    if psp_matrix is not None and psp_times is not None:
        sqm["psp_per_window"]      = np.asarray(psp_matrix).mean(axis=0).tolist()
        sqm["psp_window_times_s"]  = window_centers(psp_times).tolist()
    if cv_matrix is not None and cv_times is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            cv_mean = np.nanmean(np.asarray(cv_matrix), axis=0)
        sqm["cv_per_window"]      = cv_mean.tolist()
        # 1/CV rather than a second aggregation, so the pair stays exact reciprocals
        sqm["snr_per_window"]     = np.where(cv_mean > 0, 1.0 / cv_mean, np.nan).tolist()
        sqm["cv_window_times_s"]  = window_centers(cv_times).tolist()

    if gvtd_per_window is not None and len(gvtd_per_window):
        sqm["gvtd_per_window"]     = np.asarray(gvtd_per_window).tolist()
        sqm["gvtd_p95_per_window"] = np.asarray(gvtd_p95_per_window).tolist()
        sqm["gvtd_window_times_s"] = window_centers(gvtd_t).tolist()
    if gvtd_filt_per_window is not None and len(gvtd_filt_per_window):
        sqm["gvtd_filt_per_window"]     = np.asarray(gvtd_filt_per_window).tolist()
        sqm["gvtd_filt_p95_per_window"] = np.asarray(gvtd_filt_p95_per_window).tolist()

    series.update(sci_matrix=sci_matrix, sci_times=sci_times,
                  psp_matrix=psp_matrix, psp_times=psp_times,
                  cv_matrix=cv_matrix, cv_times=cv_times)
    # mne-nirs returns one row per fNIRS channel in channel order, bads included
    od_names = [raw_od.ch_names[i] for i in mne.pick_types(raw_od.info, fnirs=True, exclude=())]
    row_names = {"sci": od_names, "psp": od_names,
                 "cv": raw_intensity.ch_names if raw_intensity is not None else None}
    for key, names in row_names.items():
        matrix = series[f"{key}_matrix"]
        if matrix is None:
            continue
        if names is None or len(names) != len(matrix):
            raise ValueError(f"{key}_matrix has {len(matrix)} rows for {len(names or ())} channels")
        series[f"{key}_channels"] = list(names)
    return series


# The window the SCI and PSP pass lines were established at. Not the report's `window_s`:
# PSP moves with window length, so a line that followed it would select a different set of
# channels at every length.
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


def windows_touching(times, spans) -> "np.ndarray":
    """Per window, whether it overlaps any span at all, as mne drops an epoch touching one.

    ::

      windows [0, 10], [10, 20], [20, 30], a span at 18-19 s  ->  [False, True, False]

    ``times`` are the ``[start, end]`` pairs mne_nirs returns; bare centres count as
    zero-width windows.
    """
    a = np.asarray(times, dtype=float)
    start, stop = (a[:, 0], a[:, 1]) if a.ndim == 2 and a.shape[1] == 2 else (a.ravel(), a.ravel())
    hit = np.zeros(len(start), dtype=bool)
    for s0, s1 in spans:
        hit |= (start < s1) & (stop > s0)
    return hit


def task_scope_windows(
    raw: mne.io.Raw, min_duration: float = 2 * SCREEN_WINDOW_S,
) -> "list[tuple[str, float, float]]":
    """[(label, tstart, tstop)] over the annotations long enough to hold screening windows.

    ::

      "rest" at 20 s for 300 s, plus a 5 s trigger  ->  [("rest", 20.0, 320.0)]

    The annotation's own duration is the window, so a recording carrying only zero-length
    or short triggers yields nothing here and the caller keeps whatever scope it had. A
    ``BAD_`` span is never a block, however long.

    Clamped to the recording, and on the data axis: a cropped Raw keeps its annotations on
    the original axis while its samples restart at zero.
    """
    origin = float(raw.first_time)
    end = float(raw.times[-1])
    out = []
    for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration,
                                raw.annotations.description):
        if not is_marker(desc):
            continue
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
    got = _coupled_matrices(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
    if got is None:
        return None, None
    sci, psp, centers = got
    mask = coupled_mask_from_matrices(sci, psp, sci_cutoff, psp_cutoff)
    return (None, None) if mask is None else (mask, centers)


def _coupled_matrices(raw_od, cardiac_l_freq: float, cardiac_h_freq: float, window_s: float):
    """``(sci, psp, centers)`` off one recording, or None when neither could be measured."""
    try:
        sci, times = compute_windowed_sci(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
        psp, _ = compute_windowed_psp(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
    except Exception as exc:
        logger.warning("windowed screening could not be measured (%s); it screens nothing",
                       exc)
        return None
    return sci, psp, window_centers(times)


def coupled_mask_from_matrices(
    sci_matrix, psp_matrix, sci_cutoff: float, psp_cutoff: float,
) -> "np.ndarray | None":
    """Channel x window, True where a channel is coupled in that window. None if unusable.

    ::

      sci 0.85 psp 0.30 against 0.8 / 0.1  ->  True
      sci 0.92 psp 0.04 against 0.8 / 0.1  ->  False   (movement, not coupling)

    The AND is applied inside the window, and it lives here alone so the two callers cannot
    drift apart: the screening
    measures the matrices off a recording, while a per-condition view reads the ones the
    quality record already stored. A second copy of this comparison would let a report
    disagree with the verdict the run was screened by.
    """
    sci = np.asarray(sci_matrix, dtype=float)
    psp = np.asarray(psp_matrix, dtype=float)
    if sci.shape != psp.shape or sci.ndim != 2 or sci.shape[1] == 0:
        logger.warning("windowed SCI %s and PSP %s do not share a grid; screening nothing",
                       sci.shape, psp.shape)
        return None
    return (sci >= float(sci_cutoff)) & (psp >= float(psp_cutoff))


def _in_scope(centers, scope) -> "np.ndarray":
    """Which windows a scope keeps, by window centre. No scope keeps every window."""
    if not scope:
        return np.ones(len(centers), dtype=bool)
    keep = np.zeros(len(centers), dtype=bool)
    for window in scope:
        t0, t1 = float(window[-2]), float(window[-1])
        keep |= (centers >= t0) & (centers <= t1)
    return keep


def _counted(centers, half_width: float, scope, spans) -> "np.ndarray":
    """Which windows the screening counts: centre inside the scope, and touching no span.

    ::

      centres 5, 15, 25 (10 s windows), no scope, a span at 18-19 s  ->  [True, False, True]

    A scope is decided on the centre, but a ``BAD_`` span drops every window it overlaps at
    all, as mne drops an epoch that touches one.
    """
    centers = np.asarray(centers, dtype=float)
    keep = _in_scope(centers, scope)
    for start, stop in spans:
        keep &= ~((centers - half_width < stop) & (centers + half_width > start))
    return keep


def _half_window(raw: mne.io.Raw, window_s: float) -> float:
    sfreq = float(raw.info["sfreq"])
    return int(np.ceil(window_s * sfreq)) / sfreq / 2


def counted_screen_windows(
    raw: mne.io.Raw, scope=None, window_s: float = SCREEN_WINDOW_S,
) -> "tuple[np.ndarray, np.ndarray]":
    """``(centres, counted)`` on the screening grid, before anything is measured.

    ::

      a 100 s recording at 10 Hz, BAD_ over 0-45 s  ->  centres 5..95, counted from 55 s on

    The grid is mne-nirs' windowed SCI grid (``ceil`` samples per window, the last window
    ending one sample early), so this agrees window for window with :func:`coupled_windows`.
    """
    sfreq = float(raw.info["sfreq"])
    win = int(np.ceil(window_s * sfreq))
    starts = np.arange(raw.n_times // win) * win
    ends = np.minimum(starts + win, raw.n_times - 1)
    centers = (starts + ends) / 2 / sfreq
    spans = [(a, b) for a, b, _ in bad_spans(raw)]
    return centers, _counted(centers, _half_window(raw, window_s), scope, spans)


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

      a channel passing both lines in 300 of 400 windows  ->  0.75

    A window counts when SCI **and** PSP both pass in *that* window: a window with high SCI
    and near-zero PSP is movement rather than coupling, which whole-run numbers cannot see.
    Windows are counted rather than averaged, so a channel dead for half the run cannot pass
    on the other half.

    ``scope`` restricts the **denominator** to the windows whose centres fall inside those
    stretches, such as the blocks of a run without its lead-in and gaps. None counts the
    whole recording. A window touching any ``BAD_`` span is left out under either.

    The scope masks one whole-record pass rather than cutting the recording and measuring
    each piece: SCI and PSP filter to the cardiac band, so a cut piece is filtered against
    its own edges and lands on its own window grid. Masking keeps one grid and one filter.

    Returns an empty dict when either metric cannot be measured, or when the scope keeps no
    window, which screens nothing rather than rejecting everything.
    """
    return coupled_windows(raw_od, cardiac_l_freq, cardiac_h_freq,
                           sci_cutoff, psp_cutoff, window_s, scope)["fractions"]


def coupled_windows(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    sci_cutoff: float,
    psp_cutoff: float,
    window_s: float = SCREEN_WINDOW_S,
    scope: "list[tuple[str, float, float]] | None" = None,
) -> dict:
    """:func:`good_window_fraction`, also handing back the grid it counted over.

    ::

      -> {"fractions": {ch: 0.77, ...}, "mask": (n_ch, 390), "centers": (390,),
          "sci": (n_ch, 390), "psp": (n_ch, 390), "channel_order": [...]}

    The share is what screening reads and the grid is what a figure over time draws, and they
    have to be the same measurement: a panel that recomputed its own mask could shade a
    window the verdict had counted the other way. The two matrices come back beside the mask
    for the same reason, since a series drawn over the carpet is explaining the holes in it.

    A dict rather than a tuple because this is the extension point: a metric added to the
    screening has to reach the figures without every caller unpacking one more slot.

    The scope restricts ``fractions`` only, never the grid, because a figure draws the whole
    recording and says which part of it counted. ``fractions`` is empty where
    :func:`good_window_fraction` returns an empty dict; ``mask`` is None when nothing could
    be measured at all.
    """
    got = _coupled_matrices(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
    if got is None:
        return {"fractions": {}, "mask": None, "centers": None,
                "sci": None, "psp": None, "channel_order": list(raw_od.ch_names)}
    sci, psp, centers = got
    mask = coupled_mask_from_matrices(sci, psp, sci_cutoff, psp_cutoff)
    out = {"mask": mask, "centers": centers, "sci": sci, "psp": psp,
           "channel_order": list(raw_od.ch_names), "fractions": {}}
    if mask is None:
        return out

    spans = [(a, b) for a, b, _ in bad_spans(raw_od)]
    keep = _counted(centers, _half_window(raw_od, window_s), scope, spans)
    if not keep.any():
        logger.warning("the screening scope keeps none of the %d windows%s; screening nothing",
                       len(centers), " once BAD_ spans are left out" if spans else "")
        return out
    if scope or spans:
        logger.info("channel screening counts %d of %d windows, %.0f%% of the recording",
                    int(keep.sum()), len(centers), 100 * keep.mean())
    frac = mask[:, keep].mean(axis=1)
    out["fractions"] = {ch: float(frac[i]) for i, ch in enumerate(raw_od.ch_names)
                        if i < len(frac)}
    return out


def condition_window_means(
    matrix: "np.ndarray",
    centers: "np.ndarray",
    windows: "list[tuple[str, float, float]]",
) -> "dict[str, np.ndarray]":
    """Per-condition means of an already-computed windowed metric, by slicing it.

    ::

      matrix (n_ch, 360) on a 10 s grid, a condition running 600 to 1500 s
      -> {"task1": (n_ch,) means over the 90 columns whose centres fall inside it}

    ``matrix`` is reduced over its last axis, so a per-channel metric (channels x windows)
    gives one value per channel and a single series (windows,) gives one scalar.

    A condition keeps the windows whose **centre** falls inside it, the same rule the
    screening scope uses, so every per-condition number in a report comes off one grid and
    one filter.

    Slicing rather than cutting the recording keeps the run's grid: a condition cut into its
    own file is filtered against its own two edges and lands on a grid of its own.

    A condition holding no whole window is left out rather than given an empty mean, which
    is what an annotation shorter than one window looks like.

    ``centers`` may be either the window centres or the mne-nirs ``[start, end]`` pairs that
    :func:`attach_windowed_series` returns alongside each matrix; a pair array is collapsed
    to midpoints here by the same rule that function records. Without that, an (n, 2) array
    would broadcast against the window bounds and mask nothing correctly.
    """
    matrix, centers = np.asarray(matrix), window_centers(centers)
    out: dict[str, np.ndarray] = {}
    for window in windows:
        keep = _in_scope(centers, [window])
        if not keep.any():
            logger.warning("condition %s holds no whole window of the metric grid",
                           window[0])
            continue
        # a window with no value is skipped, as the whole-run mean skips it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[window[0]] = np.nanmean(matrix[..., keep].astype(float), axis=-1)
    return out

