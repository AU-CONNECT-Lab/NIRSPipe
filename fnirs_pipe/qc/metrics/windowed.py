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
