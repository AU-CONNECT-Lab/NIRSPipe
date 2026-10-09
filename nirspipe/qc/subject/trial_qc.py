"""Score each trial window on its own, so one bad trial is visible before it is averaged in.

The recording-level metrics answer "was this run usable". They cannot answer "was trial 12
usable", and a run whose mean SCI is 0.9 can still hold two trials where the cap moved. The
raw QC viewer and the GUI's Data Prep page both show that per-trial view, and they score it
here so the two cannot end up measuring different windows or different metrics.

Nothing is persisted: a trial is not a BIDS entity, so a per-trial record has nowhere to live
in the derivatives tree without colliding on filename.
"""

from __future__ import annotations

import numpy as np

from fnirs_pipe.qc.metrics.coupling import CV_WINDOW_S, PSP_WINDOW_S, _window_samples
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.trial_qc")

# the heatmap's CV, SNR and PSP rows name this window, so a trial is scored only if it holds one
MIN_TRIAL_S = max(CV_WINDOW_S, PSP_WINDOW_S)


def trial_fits(sfreq: float, t0: float, t1: float) -> bool:
    """Whether the crop ``[t0, t1]`` holds one whole :data:`MIN_TRIAL_S` window.

    Counted the way MNE crops, to the nearest sample at each end::

        trial_fits(10.0, 30.0, 40.0)        ->  True   (101 samples, 100 needed)
        trial_fits(7.8125, 33.408, 34.408)  ->  False  (9 samples, 79 needed)
    """
    n = int(np.round(t1 * sfreq)) - int(np.round(t0 * sfreq)) + 1
    return n >= _window_samples(MIN_TRIAL_S, sfreq)


def trial_windows(
    markers: list[dict],
    tmin: float | None,
    tmax: float | None,
    duration: float,
) -> list[tuple[str, float, float, float]]:
    """Turn the event list into (label, t0, t1, onset) windows to score one at a time.

    Two ways to size a window, chosen by whether tmin/tmax were given:
    fixed, `[onset+tmin, onset+tmax]`, which lets a negative tmin pull in a baseline; or the
    event's own duration, `[onset, onset+duration]`, for block designs that record one.

    Example: an event at 30.0 s of 8 s with tmin/tmax unset yields
    ``("trial-001_30s_speak", 30.0, 38.0, 30.0)``.

    The onset travels beside the bounds because it is what says which condition a trial
    belongs to, and with a fixed window it is not recoverable from ``t0``: a negative tmin
    puts ``t0`` before the event and a clamp at either edge of the recording moves it again.

    Events that describe no window are dropped rather than guessed at: a zero duration with
    no tmin/tmax has no extent, and a window starting past the end of the recording has no
    data.
    """
    fixed = tmin is not None and tmax is not None
    windows: list[tuple[str, float, float, float]] = []
    for i, m in enumerate(markers, start=1):
        onset = float(m["onset"])
        if fixed:
            t0, t1 = onset + tmin, onset + tmax
        elif float(m["duration"]) > 0:
            t0, t1 = onset, onset + float(m["duration"])
        else:
            logger.warning("trial %d at %.1fs has no duration and no epoch window; "
                           "skipping", i, onset)
            continue
        t0, t1 = max(0.0, t0), min(duration, t1)
        if t1 - t0 <= 0:
            logger.warning("trial %d at %.1fs falls outside the recording; skipping", i, onset)
            continue
        cond = str(m.get("description", "")).strip()
        windows.append((f"trial-{i:03d}_{onset:.0f}s" + (f"_{cond}" if cond else ""),
                        t0, t1, onset))
    return windows


def trial_sqm(raw, t0: float, t1: float,
              sci_threshold: float, cardiac_l_freq: float, cardiac_h_freq: float,
              psp_threshold: float | None = None,
              min_good_frac: float | None = None) -> dict:
    """SQM scalars for one trial window, scored the way the whole recording was.

    The intensity recording is what gets cropped, not the optical density derived from it,
    so that a trial's CV, SNR and spike count sit on the same scale as the recording-level
    numbers in the same report.

    A trial shorter than one :data:`MIN_TRIAL_S` window is not scored and returns ``{}``,
    drawn blank: its CV and SNR would be the whole crop's under a windowed label, and its SCI
    a correlation over a heartbeat or two. No sliding-window series is attached, and a trial
    shorter than two screening windows gets no coupled-window share, so its status row
    screens nothing rather than rejecting everything.
    """
    from fnirs_pipe.qc.metrics import (
        compute_raw_sqm, compute_sci_scores, resolve_cutoffs, screen_channels,
        screening_scores,
    )

    if not trial_fits(float(raw.info["sfreq"]), t0, t1):
        return {}
    seg = raw.copy().crop(tmin=t0, tmax=t1)
    sci_scores, seg_od = compute_sci_scores(seg, cardiac_l_freq, cardiac_h_freq)
    cutoffs = resolve_cutoffs(sci=sci_threshold, psp=psp_threshold,
                              good_frac=min_good_frac)
    scores = screening_scores(seg_od, cardiac_l_freq, cardiac_h_freq,
                              have={"sci": sci_scores}, cutoffs=cutoffs)
    bad, _ = screen_channels(scores, cutoffs)
    try:
        return compute_raw_sqm(seg, sci_scores, bad, cardiac_l_freq, cardiac_h_freq,
                               scores.get("good_frac"))
    except Exception as exc:
        logger.warning("trial SQM failed: %s", exc)
        return {}


def score_trials(
    raw,
    markers: list[dict],
    sci_threshold: float,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    tmin: float | None = None,
    tmax: float | None = None,
    psp_threshold: float | None = None,
    min_good_frac: float | None = None,
) -> tuple[list[str], list[dict]]:
    """Every trial window scored, as ``(labels, sqms)`` ready for the heatmap.

    Returns two empty lists when no event describes a window that lies inside the recording,
    which is what a block design marking only condition boundaries looks like. The caller
    draws nothing in that case rather than an empty figure. A window too short to score
    (:func:`trial_fits`) keeps its label and comes back as ``{}``.
    """
    windows = trial_windows(markers, tmin, tmax, float(raw.times[-1]))
    labels = [w[0] for w in windows]
    sqms = [trial_sqm(raw, t0, t1, sci_threshold, cardiac_l_freq, cardiac_h_freq,
                      psp_threshold, min_good_frac)
            for _, t0, t1, _ in windows]
    return labels, sqms
