"""Dyad-level quality metrics: the shared screening grid, and the scalars a record holds.

Nothing here imports ``qc.figures``. What lives there is figure data rather than a measured
number: ``motion_series`` builds carpet z-scores, ``head_geometry`` projects optodes.
"""

from __future__ import annotations

import mne
import numpy as np

from fnirs_pipe.qc.metrics.windowed import SCREEN_WINDOW_S
from fnirs_pipe.utils import pair_of
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.metrics.hyper")

def sci_of(sqm_data: dict, sid: str) -> dict:
    """The per-channel windowed SCI a dyad page prints, for one member; never the whole-run one."""
    member = sqm_data.get(sid) or {}
    return member.get("sci_win_per_channel") or {}


def _rejected_pairs(sqm_data: dict, sid: str) -> "set[str] | None":
    """The S-D pairs one member's screening rejected, or None when nothing recorded it.

    ::

      ["S1_D1 760", "S1_D1 850"] -> {"S1_D1"}

    Rejection is stored per wavelength, since that is what prep marks on the recording, and
    every lookup on a dyad page is per pair. The empty set and None are different answers:
    a member that lost no channel has an empty list in its record, a member whose record
    was never written has no list at all, and only the second is unknown.
    """
    bad = sqm_data.get(sid, {}).get("bad_channels")
    if bad is None:
        return None
    return {pair_of(ch) for ch in bad}


def _ch_kept_by_member(
    ch_pair: str,
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    sci_threshold: float,
) -> list[bool | None]:
    """Whether each member kept this channel pair: the screening's verdict, per member.

    ::

      -> [True, False]   # kept by the first member, rejected by the second

    **The verdict, not a threshold on SCI.** These three colours are labelled good / mixed /
    bad on every panel that draws them, and "good" has one meaning in this package: the
    channel survived screening. Screening is `good_frac`, the share of 10 s windows in which
    SCI and PSP both cleared their lines, so a channel can be dropped at a high SCI and kept
    at a lower one; colouring by ``sci_threshold`` would call a rejected channel "good".

    ``sci_threshold`` stays the fallback and nothing else: a record with no rejection list at
    all is the one case with no verdict to draw, and an all-grey montage says less than the
    number that record does carry. The subject report makes the same split and shows both,
    rejection first and SCI as the grading underneath it.
    """
    result: list[bool | None] = []
    for sid in subject_ids:
        rejected = _rejected_pairs(sqm_data, sid)
        if rejected is not None:
            result.append(ch_pair not in rejected)
            continue
        sci_d = sci_of(sqm_data, sid)
        val = sci_d.get(f"{ch_pair} hbo") or sci_d.get(ch_pair)
        result.append(None if val is None else float(val) >= sci_threshold)
    return result


def _nearest_windows(ref: "np.ndarray", t: "np.ndarray") -> "np.ndarray":
    """For each reference centre, the index of the member's window nearest it, or -1.

    ::

      ref [5.0, 15.0, 25.0], t [5.025, 15.075]  ->  [0, 1, -1]

    A window is taken only within half a screening window of the reference centre, so the
    two always overlap by at least half. Members recorded at different rates cut a window
    into a whole number of samples, so each window runs a fraction of a sample long or short
    and their grids drift apart every window; matching by time keeps them paired where index
    would not.
    """
    if not len(t):
        return np.full(len(ref), -1)
    # the nearer of the two neighbours searchsorted lands between
    right = np.clip(np.searchsorted(t, ref), 0, len(t) - 1)
    left = np.clip(right - 1, 0, len(t) - 1)
    idx = np.where(np.abs(t[left] - ref) <= np.abs(t[right] - ref), left, right)
    return np.where(np.abs(t[idx] - ref) <= SCREEN_WINDOW_S / 2, idx, -1)


def _take_windows(matrix: "np.ndarray", idx: "np.ndarray", fill) -> "np.ndarray":
    """``matrix[..., idx]`` with the columns of unmatched windows (-1) set to ``fill``."""
    out = np.asarray(matrix)[..., np.maximum(idx, 0)].copy()
    out[..., idx < 0] = fill
    return out


def coupled_grid(
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    offsets: dict[str, float],
    duration_s: "float | None" = None,
) -> "dict | None":
    """The screening verdict per long pair per window, for every member, on one clock.

    ::

      -> {"t": [...], "pairs": ["S1_D1", ..., "S1_D8", ...],
          "long_pairs": ["S1_D1", ...], "ok": {"sub-01": mask, "sub-02": mask}}

    with each ``mask`` a ``pair x window`` boolean, True where that member's SCI and PSP both
    cleared their lines in that window. The three dyad panels that read it (the usable-time
    carpet, the two head figures) then cannot disagree about which window was good.

    **Every pair, with the long ones named separately.** The carpet is a long-channel picture
    because the dyad measures run on long channels, but a head draws the whole montage, short
    markers included.

    Two alignments are made here. The stored matrices are
    on each member's **own** clock, so the window centres are shifted by that member's crop
    offset, and a dyad with unequal offsets would otherwise compare window *k* of one against
    window *k* of the other. And they cover the **whole** recording while the dyad exists only
    on the aligned span, so windows outside it are dropped rather than drawn past the ends of
    the shared clock.

    The dyad raw pass hands over :func:`shared_screen_windows`, measured on the aligned
    recordings, so the offsets are zero and every member's windows coincide; grids measured on
    each member's own clock stop coinciding as soon as the members' offsets differ.

    The first member's windows are the dyad's; every other member contributes its window
    nearest each of them in time (:func:`_nearest_windows`), and counts as not coupled where
    it has none. ``index`` keeps that choice so :func:`member_series` reads the same columns.

    None when a member has no grid, or shares no window with the first member.
    """
    grids, masks = {}, {}
    for sid in subject_ids:
        member = sqm_data.get(sid) or {}
        grid = member.get("screen_windows") or {}
        mask, centers = grid.get("mask"), grid.get("centers")
        if mask is None or centers is None or not len(centers):
            logger.warning("no screening grid for %s; the dyad grid is empty", sid)
            return None
        grids[sid] = np.asarray(centers, dtype=float) - float(offsets.get(sid, 0.0))
        masks[sid] = np.asarray(mask, dtype=bool)

    ref = grids[subject_ids[0]]
    index = {sid: _nearest_windows(ref, t) for sid, t in grids.items()}
    for sid, idx in index.items():
        if (idx < 0).all():
            logger.warning("%s shares no screening window with %s; the dyad grid is empty",
                           sid, subject_ids[0])
            return None
        if (idx < 0).any():
            logger.info("%s has no window within %g s of %d of the dyad's %d; counted as "
                        "not coupled there", sid, SCREEN_WINDOW_S / 2, int((idx < 0).sum()),
                        len(ref))
    masks = {sid: _take_windows(masks[sid], index[sid], False) for sid in subject_ids}

    keep = ref >= 0
    if duration_s is not None:
        keep &= ref <= float(duration_s)

    pairs: list[str] = []
    long_pairs: list[str] = []
    rows = {sid: [] for sid in subject_ids}
    for sid in subject_ids:
        member = sqm_data.get(sid) or {}
        order = list((member.get("screen_windows") or {}).get("channel_order") or [])
        long_names = {pair_of(k)
                      for k in (member.get("per_channel_long") or {})
                      .get("sci_per_channel", {})}
        if not order:
            return None
        by_pair: dict[str, list[int]] = {}
        for i, name in enumerate(order):
            by_pair.setdefault(pair_of(name), []).append(i)
        if not pairs:
            pairs = list(by_pair)
            long_pairs = [p for p in pairs if p in long_names] or list(pairs)
        for pair in pairs:
            idx = by_pair.get(pair)
            # a pair one member lacks is not usable by the dyad at any moment
            rows[sid].append(masks[sid][idx].all(axis=0) if idx
                             else np.zeros(masks[sid].shape[1], dtype=bool))

    return {"t": ref[keep],
            "pairs": pairs,
            "long_pairs": long_pairs,
            "ok": {sid: np.array(rows[sid])[:, keep] for sid in subject_ids},
            "index": {sid: index[sid][keep] for sid in subject_ids}}


def shared_screen_windows(
    intensity_raws: dict[str, mne.io.Raw],
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> dict[str, dict]:
    """``sqm_data`` with each member's ``screen_windows`` measured again on its aligned recording.

    ::

      -> {"sub-01": {..., "screen_windows": {"mask", "centers", "sci", "psp", "cv", "gvtd",
                                             "channel_order"}}, ...}

    Each member's own screening runs on its own clock from its first sample, so two members
    whose triggers sit at different points of their recordings were screened on windows that
    never coincide. Measured on the aligned copies, every member's windows start at 0 on the
    shared clock and are the same stretches of it, so :func:`coupled_grid` compares like with
    like at zero offset. The verdict a member was rejected by is untouched: that stays the
    one from its own screening, which its own report shows.
    """
    from fnirs_pipe.qc.metrics.gvtd import compute_windowed_filtered_gvtd
    from fnirs_pipe.qc.metrics.windowed import compute_windowed_cv, coupled_windows

    out = {}
    for sid in subject_ids:
        member = dict(sqm_data.get(sid) or {})
        raw = intensity_raws.get(sid)
        cutoffs = member.get("screen_cutoffs") or {}
        if raw is None or "sci" not in cutoffs or "psp" not in cutoffs:
            logger.warning("%s: no aligned recording or screening lines; the dyad grid is "
                           "empty", sid)
            return {}
        od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
        counted = coupled_windows(od, cardiac_l_freq, cardiac_h_freq,
                                  cutoffs["sci"], cutoffs["psp"])
        if counted["mask"] is None:
            return {}
        windows = {k: counted[k] for k in ("mask", "centers", "sci", "psp", "channel_order")}
        # the CV and motion rows are extras on the grid; losing one costs that row alone
        n = len(counted["centers"])
        try:
            cv_m, cv_t = compute_windowed_cv(raw)
            if cv_m is not None and len(cv_t) == n:
                windows["cv"] = cv_m
            gvtd, _p95, gvtd_t = compute_windowed_filtered_gvtd(od)
            if len(gvtd_t) == n:
                windows["gvtd"] = gvtd
        except Exception:
            logger.warning("%s: windowed CV or GVTD could not be measured on the aligned "
                           "recording", sid, exc_info=True)
        member["screen_windows"] = windows
        out[sid] = member
    return out


def dyad_status(grid: dict, subject_ids: list[str]) -> "np.ndarray":
    """``pair x window``: 2 coupled in every member, 1 in some, 0 in none."""
    stack = np.array([grid["ok"][sid] for sid in subject_ids])
    return np.where(stack.all(axis=0), 2, np.where(stack.any(axis=0), 1, 0))


def member_series(sqm_data: dict, sid: str, grid: dict) -> dict:
    """One member's long-channel means per window, on the dyad's clock.

    ::

      -> {"t": (388,), "sci": (388,), "psp": (388,), "cv": (388,), "gvtd": (388,),
          "per_pair_sci": {"S1_D1": (388,), ...}}

    The three series are averaged over the same rows the carpet folds into pairs, off the
    same matrices its mask came from, so a dip in a line and a hole under it are one
    measurement and not two. GVTD is absent unless the member's record carried it; see
    :func:`coupled_grid`. A dyad window the member has no window for is NaN.

    ``per_pair_sci`` keeps every pair separately, short ones included, because the head
    figures colour one marker per channel and draw both separations.
    """
    sw = (sqm_data.get(sid) or {}).get("screen_windows") or {}
    order = list(sw.get("channel_order") or [])
    if not order or sw.get("centers") is None:
        return {}
    long_pairs = set(grid.get("long_pairs") or grid["pairs"])
    rows = [i for i, name in enumerate(order) if pair_of(name) in long_pairs]
    idx = grid["index"][sid]
    out = {"t": np.asarray(grid["t"], dtype=float)}
    for key in ("sci", "psp", "cv"):
        m = sw.get(key)
        if m is not None and rows:
            out[key] = _take_windows(np.asarray(m, dtype=float)[rows], idx,
                                     np.nan).mean(axis=0)
    gvtd = sw.get("gvtd")
    if gvtd is not None and len(gvtd) == len(sw["centers"]):
        out["gvtd"] = _take_windows(np.asarray(gvtd, dtype=float), idx, np.nan)

    sci_m = sw.get("sci")
    if sci_m is not None:
        sci_m = _take_windows(np.asarray(sci_m, dtype=float), idx, np.nan)
        by_pair: dict[str, list[int]] = {}
        for i, name in enumerate(order):
            by_pair.setdefault(pair_of(name), []).append(i)
        # a pair is one SCI, stored once per wavelength; the mean over its rows is that
        # number and not an average of two different ones
        out["per_pair_sci"] = {pair: sci_m[idx].mean(axis=0)
                               for pair, idx in by_pair.items()}
    return out


def motion_summary(motion: dict) -> dict:
    """The dyad's motion scalars, in the same units the figures are drawn in.

    ::

      -> {"unit": "x its own before-median",
          "long": {"before": {"sub-01": 1.61, "sub-02": 1.51, "together": 0.95},
                   "after": {...}, "reduction": 0.33}, ...}

    ``together`` is the mean of the pointwise minimum of the members' traces, which is high
    only where both were high. A minimum needs no cutoff chosen, and being a mean of the same
    normalised trace it is comparable between stages. ``reduction`` is the share of simultaneous movement the correction removed.
    """
    if not motion:
        return {}
    out: dict = {"unit": "x its own before-median", "stages": list(motion["stages"])}
    for name in motion["sets"]:
        per_set: dict = {}
        for stage in motion["stages"]:
            entries = motion["series"][stage].get(name) or []
            if not entries:
                continue
            row = {sid: round(float(np.nanmean(y)), 3) for sid, y in entries}
            if len(entries) > 1:
                mins = np.minimum.reduce([y for _, y in entries])
                row["together"] = round(float(np.nanmean(mins)), 3)
            per_set[stage] = row
        before, after = per_set.get("before"), per_set.get("after")
        if before and after and before.get("together"):
            per_set["reduction"] = round(
                1.0 - after["together"] / before["together"], 3)
        out[name] = per_set
    # not nested under a set: the spike test runs on the canonical set alone, so a count
    # printed under "short" would be a claim about channels it never looked at
    out["spike_spans_both"] = {stage: len(motion["spikes_both"].get(stage) or [])
                               for stage in motion["stages"]}
    return out


def compute_hyper_sqm(
    sqm_data: dict[str, dict],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    subject_ids: list[str],
    sci_threshold: float,
) -> dict:
    ch_set: set[str] = set()
    for sid in subject_ids:
        for k in sci_of(sqm_data, sid):
            ch_set.add(pair_of(k))

    n_all_good = n_mixed = n_all_bad = n_unknown = 0
    for ch in ch_set:
        statuses = _ch_kept_by_member(ch, sqm_data, subject_ids, sci_threshold)
        known = [s for s in statuses if s is not None]
        if not known:
            n_unknown += 1
        elif all(known):
            n_all_good += 1
        elif not any(known):
            n_all_bad += 1
        else:
            n_mixed += 1

    n_total  = len(ch_set)
    pct_good = round(n_all_good / n_total * 100, 1) if n_total > 0 else None

    aligned_duration_s = None
    if aligned_raws:
        aligned_duration_s = round(float(next(iter(aligned_raws.values())).times[-1]), 1)

    max_offset_s = round(max(offsets.values()), 3) if offsets else None

    return dict(
        n_all_good=n_all_good, n_mixed=n_mixed,
        n_all_bad=n_all_bad, n_unknown=n_unknown, n_total=n_total,
        pct_all_good=pct_good,
        aligned_duration_s=aligned_duration_s, max_offset_s=max_offset_s,
    )
