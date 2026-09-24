"""The re-paired null: one member against people they never interacted with.

Split from :mod:`fnirs_pipe.pipeline.hyper.wtc_null` rather than folded into it because the two
nulls sit at different levels. Phase randomisation needs one dyad and can run inside the
per-dyad pass; re-pairing needs the rest of the cohort, so it reads the pairs table and the
finished derivatives tree and runs after them.

Its parameters are read back off the real table's sidecar instead of being taken from the
command line. A null is only meaningful against the table it is subtracted from, so a band
or a mask the caller could type differently is a way for the two to disagree silently. Same
reason ``alignment_params`` is read off the lineage stamps rather than passed in.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.io.derivatives import group_output_path
from fnirs_pipe.pipeline.hyper.surrogate import compute_wtc_pair_null, _average_iterations
from fnirs_pipe.pipeline.hyper.wtc import cone_margin_s
from fnirs_pipe.pipeline.hyper.wtc_null import _for_chroma, _real_table, write_tsv
from fnirs_pipe.qc.common.windows import condition_windows, split_windows

logger = logging.getLogger(__name__)

POOLS = ("position", "any")

# a partner whose recording cannot reach the real dyad's analysed length is refused rather
# than shortened: coherence rises as a record shortens, so a null averaged over draws of
# unequal length is a mixture the real table cannot be compared against
_DURATION_TOL_S = 1e-3


def partner_pool(
    groups: dict,
    group_id: str,
    task: str,
    pool: str = "position",
    position: int = 1,
) -> list:
    """The entries that can stand in for one member, taken from the other groups.

    ``groups`` is :func:`parse_group_csv`'s mapping. ``position`` is the index in the target
    group of the member being replaced.

    ::

        pairs with groups d01..d03, two members each
        partner_pool(groups, "d01", "full")  -> [d02's member 1, d03's member 1]

    ``position`` keeps a member's role: where the two members of a group are not
    interchangeable, only the other groups' member at the same index is a valid stand-in.
    It is also the only safe pool where one person appears in several groups, which is what
    a cohort of the same pair recorded over many days looks like: ``any`` would there pair
    somebody with themselves on another day, which is not a null at all. ``any`` doubles the
    pool and is refused unless the table shows nobody is repeated.
    """
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")

    same_task = {gid: members for (gid, tsk), members in groups.items() if tsk == task}
    if group_id not in same_task:
        raise StageError(f"group {group_id!r} has no task {task!r} in the pairs table")
    if pool == "any":
        _refuse_repeated_subjects(same_task, task)
        _warn_unverifiable_pool(same_task)
        own = {entry.subject_id for entry in same_task[group_id]}
        return [entry for gid, members in same_task.items() if gid != group_id
                for entry in members if entry.subject_id not in own]

    # only the members this draw keeps are barred, not the whole group. Where the cohort is
    # the same people recorded repeatedly, every stand-in shares a subject id with the member
    # it replaces, and barring the group would empty the pool on exactly the cohort this
    # null was asked for. Pairing somebody with themselves is what has to be barred, and
    # that is the member being held fixed
    held = {e.subject_id for i, e in enumerate(same_task[group_id]) if i != position}
    return [members[position] for gid, members in same_task.items()
            if gid != group_id and len(members) > position
            and members[position].subject_id not in held]


def _refuse_repeated_subjects(same_task: dict, task: str) -> None:
    """Refuse an `any` pool where one person is in several groups, or twice in one."""
    for gid, members in same_task.items():
        seen = [e.subject_id for e in members]
        if len(set(seen)) != len(seen):
            raise StageError(
                f"group {gid!r} lists the same subject twice for task {task!r}, so "
                "'--wtc-pair-null-pool any' cannot tell a stand-in from the member it "
                "replaces. Use the default 'position' pool.")

    groups_by_subject: dict[str, list[str]] = {}
    for gid, members in same_task.items():
        for entry in members:
            groups_by_subject.setdefault(entry.subject_id, []).append(gid)
    repeated = {sub: gids for sub, gids in groups_by_subject.items() if len(gids) > 1}
    if repeated:
        shown = ", ".join(f"{sub} in {sorted(gids)}" for sub, gids in sorted(repeated.items()))
        raise StageError(
            f"'--wtc-pair-null-pool any' would pair somebody with themselves: {shown}. "
            "A cohort of the same people recorded repeatedly has to use the default "
            "'position' pool, where a member is only ever replaced by another group's "
            "member at the same index.")


def _warn_unverifiable_pool(same_task: dict) -> None:
    """Say out loud what `any` rests on, when the table cannot be read to check it.

    The refusal above catches a cohort of repeated people only where the same person keeps
    one ``subject_id`` across groups, which is how BIDS encodes it: a stable ``sub-`` label
    and a ``ses-`` label per visit, and ``parse_group_csv`` takes a ``session`` column for
    exactly that. A table that instead bakes the visit into the subject id, ``sub-p1d01``
    and ``sub-p1d03`` for one person, looks identical to a table of strangers, and nothing
    on disk distinguishes them. So where every id is unique the check has not passed, it has
    had nothing to test, and the difference matters: under `any` a repeated person would be
    ranked against another recording of themselves.
    """
    ids = [e.subject_id for members in same_task.values() for e in members]
    if len(set(ids)) != len(ids):
        return          # the refusal above already had something to test
    logger.warning(
        "'--wtc-pair-null-pool any' assumes the members of a group are interchangeable and "
        "that no person appears in more than one group. Every subject id here is unique, so "
        "that could not be checked: a cohort of the same people recorded repeatedly looks "
        "the same as a cohort of strangers when the visit is baked into the subject id. If "
        "these %d groups are repeat visits by the same people, use the default 'position' "
        "pool, or give the pairs table a stable subject_id and a session column.",
        len(same_task))


def real_table_params(real_tsv: Path) -> dict:
    """The real WTC table's own record of how it was computed, off its sidecar.

    Raises rather than falling back on defaults: a null built on a guess about the band is
    worse than no null, because nothing downstream can tell the two apart.
    """
    sidecar = real_tsv.with_suffix(".json")
    if not sidecar.exists():
        raise StageError(
            f"no real WTC table to rank against: {sidecar} is missing. Run "
            "`fnirs-hyper` for this group first; the null reads its band, its mask and "
            "its clock off that table rather than taking them again from the command line.")
    try:
        params = json.loads(sidecar.read_text()).get("parameters", {})
    except (OSError, json.JSONDecodeError) as exc:
        raise StageError(f"unreadable sidecar {sidecar}: {exc}") from exc

    missing = [k for k in ("band_fmin", "band_fmax", "wtc_fmin", "wtc_fmax", "mask_coi")
               if params.get(k) is None]
    if missing:
        raise StageError(
            f"{sidecar} does not record {', '.join(missing)}, so the null cannot be built to "
            "match it. Rerun `fnirs-hyper` for this group on current code.")
    return params


def _partner_condition_onsets(partner_raw, labels) -> "dict[str, float]":
    """Where each named condition starts in the stand-in's own recording.

    ``condition_windows`` numbers a repeated description ``desc#1``, ``desc#2`` in time
    order and the annotation carries the bare one, so ``desc#k`` is the k-th ``desc`` by
    onset. A label the stand-in never reached is absent from the result rather than
    defaulted, and the caller refuses it.

    ::

        stand-in marks game1 at 100 s and 900 s, labels ["game1#2"] -> {"game1#2": 900.0}
    """
    from fnirs_pipe.qc.common.windows import markers_on_data_axis

    markers = sorted(markers_on_data_axis(partner_raw), key=lambda m: float(m["onset"]))
    descriptions = {str(m["description"]) for m in markers}
    out: dict[str, float] = {}
    for label in labels:
        bare, sep, number = str(label).rpartition("#")
        # a description that itself carries a "#" is its own label, not a numbered repeat
        if not (sep and number.isdigit()) or str(label) in descriptions:
            bare, number = str(label), "1"
        onsets = [float(m["onset"]) for m in markers if str(m["description"]) == bare]
        if len(onsets) >= int(number):
            out[label] = onsets[int(number) - 1]
    return out


def _draw_condition_pairs(
    derivatives_dir: Path,
    task: str,
    fixed_id: str,
    fixed_raw,
    candidates: list,
    *,
    desc: str,
    bads_scope: str,
    scope_tasks: "list[str] | None",
    windows: "list[tuple[str, float, float]]",
    band_fmin: float,
    n_max: "int | None",
    refused: dict,
    window_sources: "dict[str, tuple[str, float]] | None" = None,
):
    """Yield ``(partner_id, label, {fixed, partner} segments, the condition's place in them)``.

    Each condition is taken from **the stand-in's own onset**, not from where it sat in the
    real dyad's clock. Sessions that run to one timetable drift: the first trigger lines up
    by construction and the gaps between blocks do not, so by the last condition the two
    recordings are a quarter of a block apart. Cutting the stand-in at the real dyad's window
    would then correlate one person's conversation against another's silence and call it a
    null, which reads as "no coupling" for a reason that has nothing to do with coupling.

    Taking each side from its own marker makes every draw the same length as the condition it
    stands in for, so the whole-record duration stops mattering: it used to have to reach the
    real dyad's, which on a cohort with a 338 s spread left the longest sessions with no
    stand-ins at all.
    """
    from fnirs_pipe.pipeline.hyper.group_io import load_group_haemo
    from fnirs_pipe.pipeline.hyper.group_quality import apply_group_bads, load_group_sqm

    margin = cone_margin_s(band_fmin)
    window_sources = window_sources or {}
    drawn = 0
    for entry in candidates:
        if n_max is not None and drawn >= n_max:
            logger.info("stopping at %d stand-ins, the limit asked for", drawn)
            break
        pid = entry.subject_id
        try:
            partner = load_group_haemo(derivatives_dir, [entry], desc=desc)
        except Exception as exc:
            refused.setdefault("unreadable", []).append(pid)
            logger.debug("%s refused as a stand-in: %s", pid, exc)
            continue

        partner_raw = next(iter(partner.values()))
        if abs(float(partner_raw.info["sfreq"]) - float(fixed_raw.info["sfreq"])) > 1e-6:
            refused.setdefault("sampling_rate", []).append(pid)
            continue

        try:
            apply_group_bads(partner, load_group_sqm(derivatives_dir, [entry], bads_scope=bads_scope,
                                                     scope_tasks=scope_tasks))
        except Exception as exc:
            refused.setdefault("no_quality_record", []).append(pid)
            logger.warning("%s refused as a stand-in, no quality record: %s", pid, exc)
            continue

        # asked for the conditions, not the window labels: the annotation carries a condition
        # description and a window is an offset into one
        wanted = [window_sources.get(w[0], (w[0], 0.0))[0] for w in windows]
        onsets = _partner_condition_onsets(partner_raw, wanted)
        end = float(partner_raw.times[-1])
        fixed_end = float(fixed_raw.times[-1])
        segments = []
        for label, t0, t1 in windows:
            span = float(t1) - float(t0)
            cond, offset = window_sources.get(label, (label, 0.0))
            base = onsets.get(cond)
            # the stand-in's own condition marker plus the window's offset into it, so each
            # side is cut at the same place in its own session rather than on one clock
            start = None if base is None else base + offset
            if start is None:
                refused.setdefault(f"no_{cond}", []).append(pid)
                continue
            if start + span > end + _DURATION_TOL_S:
                # the stand-in's own block is shorter than the real one, so there is no
                # equal-length stretch of it to stand in
                refused.setdefault(f"short_{label}", []).append(pid)
                continue
            # the alignment crop leaves the anchor marker at 0 give or take a rounding
            # error, and crop refuses a tmin of -1e-14; the tolerance above likewise lets
            # an end through by up to a millisecond, which crop refuses on the far side
            start, real_t0 = max(0.0, start), max(0.0, float(t0))
            span = min(span, end - start, fixed_end - real_t0)
            # context either side, so the cone of influence reaches into the pad and not
            # into the condition: the real table is windowed out of a whole-record
            # transform and a draw cut to the bare block is not, which is the same length
            # bias the two routes were measured to differ by. Whatever the recordings can
            # spare, equal on both sides so the pair stays aligned and the same length.
            lead = min(margin, start, real_t0)
            trail = min(margin, end - start - span, fixed_end - real_t0 - span)
            segments.append((label, start, span, real_t0, lead, max(0.0, trail)))

        if not segments:
            continue
        drawn += 1
        for label, start, span, real_t0, lead, trail in segments:
            pair = {fixed_id: fixed_raw.copy().crop(tmin=real_t0 - lead,
                                                    tmax=real_t0 + span + trail),
                    pid: partner_raw.copy().crop(tmin=start - lead,
                                                 tmax=start + span + trail)}
            yield pid, label, pair, (lead, lead + span)


def run_pair_null(
    group_id: str,
    task: str,
    members: list,
    groups: dict,
    derivatives_dir: Path,
    output_dir: Path,
    *,
    pool: str = "position",
    n_max: "int | None" = None,
    desc: str = "preproc",
    bads_scope: str = "run",
    scope_tasks: "list[str] | None" = None,
    chroma: "tuple[str, ...]" = ("hbo", "hbr"),
    cross: bool = False,
    limit_scales: bool = True,
    roi_map: "dict[str, list[str]] | None" = None,
    roi_map_name: str = "custom",
    roi_min_channels: int = 2,
    isc_whiten: int = 0,
    isc_max_lag_s: float = 0.0,
    isc_band: "tuple[float | None, float | None] | None" = None,
) -> Path:
    """Draw, rank and write one group's re-paired null.

    Writes ``..._cond-all_null-pair_stat-wtc_relmat.tsv`` with the columns the
    phase-scrambled table has, plus the per-condition and homologous-ROI tables where the
    real side has them. Returns the whole-run path.

    The re-paired ISC rides along on the same draws: making one is two recordings read,
    aligned and cropped, and a correlation over that costs nothing beside it. It is the only
    null the correlation has, the phase-scrambled one having the same defect here as it does
    for the coherence and a worse one, a whole-record correlation carrying more of the shared
    task than a band mean does.

    Unlike the phase-scrambled null this runs after the real table rather than around it,
    and takes its band, its mask, its frequency range and its window off that sidecar. Draw
    and write are one step for the same reason: there is nothing to write between them, the
    table being ranked against is already on disk.
    """
    from fnirs_pipe.pipeline.hyper.group_io import load_group_haemo
    from fnirs_pipe.pipeline.hyper.group_quality import (apply_group_bads, load_group_sqm,
                                                   resolve_group_bands)
    from fnirs_pipe.pipeline.hyper import (_hyper_sidecar, align_recordings,
                                                   alignment_params)
    from fnirs_pipe.pipeline.hyper._helpers import long_axis_over
    from fnirs_pipe.pipeline.hyper.wtc import wtc_grid_params
    from fnirs_pipe.utils.lineage import path_from

    roi_entities = {"segmentation": roi_map_name, "aggregation": "homologous"}

    def _path(entities: dict) -> Path:
        return group_output_path(output_dir, group_id, {"task": task, **entities},
                                 "relmat", ".tsv")

    real_wtc = _path({"statistic": "wtc"})
    real_by_cond_path = _path({"condition": "all", "statistic": "wtc"})
    real_params = real_table_params(real_wtc)
    band_fmin, band_fmax = real_params["band_fmin"], real_params["band_fmax"]
    wtc_fmin, wtc_fmax = real_params["wtc_fmin"], real_params["wtc_fmax"]
    mask_coi = bool(real_params["mask_coi"])
    window_s = real_params.get("analysis_window_s")
    analysis_window = tuple(window_s) if window_s else None

    isc_whiten, isc_max_lag_s, isc_band = _isc_settings_of(
        _path({"statistic": "isc"}).with_suffix(".json"),
        isc_whiten, isc_max_lag_s, isc_band)

    raws = load_group_haemo(derivatives_dir, members, desc=desc)
    group_sqm = load_group_sqm(derivatives_dir, members, bads_scope=bads_scope,
                               scope_tasks=scope_tasks)
    apply_group_bads(raws, group_sqm)
    sep_bands = resolve_group_bands(members, group_sqm)

    aligned_real, offsets = align_recordings(raws, task)
    fixed_id = members[0].subject_id
    true_pair = (members[0].subject_id, members[1].subject_id)
    aligned_duration = min(float(r.times[-1]) for r in aligned_real.values())
    # A stand-in no longer has to match the whole recording's length: each condition is cut
    # from its own marker on both sides, so what has to agree is the block, and the blocks
    # are what the trigger defines. That is what lets a cohort whose sessions differ by
    # minutes draw from its whole pool rather than only from the sessions that ran longer.
    recorded = real_params.get("aligned_duration_s")
    if recorded is not None and abs(float(recorded) - aligned_duration) > _DURATION_TOL_S:
        # the tree moved under the table: the null would describe a different stretch
        raise StageError(
            f"the real table was written on {float(recorded):.3f} s of aligned recording but "
            f"the tree now aligns to {aligned_duration:.3f} s. Rerun `fnirs-hyper` for "
            f"group {group_id!r} before drawing its null.")

    candidates = partner_pool(groups, group_id, task, pool=pool)
    if not candidates:
        raise StageError(
            f"no other group runs task {task!r}, so there is nobody to re-pair "
            f"group {group_id!r} with. This null needs a cohort, not one dyad.")
    logger.info("re-paired null for %s/%s: %d candidate stand-in(s) in the %r pool",
                group_id, task, len(candidates), pool)

    windows: list = []
    # {window label: (condition label, seconds into that condition)}, empty unless the real
    # table was written on a window grid. A draw cuts each side at its own marker, so for a
    # window it needs the condition it belongs to and how far into it the window starts
    window_sources: dict = {}
    if real_by_cond_path.exists():
        windows = condition_windows(aligned_real[fixed_id], min_duration=1.0 / wtc_fmin)
        if analysis_window is not None:
            lo, hi = analysis_window
            windows = [w for w in windows if w[1] >= lo and w[2] <= hi]
        # taken off the sidecar rather than the command line, for the same reason the band
        # and the mask are: a null resolved on a different grid than the table it is
        # subtracted from measures the grid, not the pairing
        if real_params.get("wtc_window_s"):
            windows, window_sources = split_windows(
                windows, float(real_params["wtc_window_s"]))
            logger.info("null on the real table's window grid: %d window(s) of %.1f s",
                        len(windows), float(real_params["wtc_window_s"]))

    # the whole-run tables are not read: this null has no whole-run half to rank against
    real_by_cond = _real_table(real_by_cond_path)
    real_roi_by_cond = _real_table(
        _path({**roi_entities, "condition": "all", "statistic": "wtc"}))

    cond_frames, roi_cond_frames = [], []
    draw_frames: list = []
    isc_draw_frames: list = []
    isc_frames: list = []
    isc_cond_frames: list = []
    refused: dict[str, list[str]] = {}

    def _isc_collector(ch_type: str):
        """A callback that correlates each drawn pair, whole run and per condition."""
        from fnirs_pipe.pipeline.hyper.isc import compute_isc_pairs

        def _collect(partner_id: str, label: str, pair: dict, inner: tuple) -> None:
            ids = [fixed_id, partner_id]
            # the segment carries context either side of the condition so the coherence's
            # cone lands in the pad; the correlation reads the condition out of it, filtered
            # on the whole segment, which is the order `_isc_rows` documents
            for _ in (0,):
                try:
                    _, _, pairs, _ = compute_isc_pairs(
                        pair, ids, ch_type, sep_bands, window=inner,
                        whiten=isc_whiten, max_lag_s=isc_max_lag_s, band=isc_band)
                except Exception:
                    logger.debug("re-paired ISC failed against %s (%s)", partner_id, label)
                    continue
                if pairs is None or pairs.empty:
                    continue
                # relabelled to the true pair for the reason the coherence is: every draw
                # has a different partner, and the summary groups by sub1/sub2
                pairs = pairs.rename(columns={"r": "coherence"})
                pairs["sub1"], pairs["sub2"] = true_pair
                pairs.insert(0, "chromophore", ch_type)
                pairs["n_valid_frac"] = 1.0
                pairs.insert(1, "condition", label)
                isc_cond_frames.append(pairs)
                # one row per draw as well as the summary, so the correlation can be read
                # above the cell the way the coherence can
                isc_draw_frames.append(pairs.assign(draw=partner_id))

        return _collect

    partners: list[str] = []
    for ch_type in chroma:
        draws = _draw_condition_pairs(
            derivatives_dir, task, fixed_id, aligned_real[fixed_id], candidates, desc=desc,
            bads_scope=bads_scope, scope_tasks=scope_tasks, windows=windows,
            band_fmin=band_fmin, n_max=n_max, refused=refused,
            window_sources=window_sources)
        null = compute_wtc_pair_null(
            draws, true_pair, long_axis_over(aligned_real.values(), ch_type, sep_bands),
            band_fmin, band_fmax, fmin=wtc_fmin, fmax=wtc_fmax, cross=cross,
            limit_scales=limit_scales, mask_coi=mask_coi, ch_type=ch_type,
            sep_bands=sep_bands, windows=windows, analysis_window=analysis_window,
            on_draw=_isc_collector(ch_type))
        partners = null.partners or []
        for draw_id, frame in zip(null.cond_draw_ids or [], null.cond_draws):
            draw_frames.append(frame.assign(chromophore=ch_type, draw=draw_id))

        _, by_cond = null.summarise(real=None,
                                    real_by_cond=_for_chroma(real_by_cond, ch_type))
        # no whole-run row: the two recordings are aligned one condition at a time, so there
        # is no stretch of them that stands in for the whole session
        for part, bucket in ((by_cond, cond_frames),):
            if part is not None:
                part = part.copy()
                # labelled after the grouping, never before: the summary groups by the
                # columns it knows and would drop this one
                part.insert(0, "chromophore", ch_type)
                bucket.append(part)

        if roi_map:
            _, roi_cond = null.summarise_roi(
                roi_map, real=None,
                real_by_cond=_for_chroma(real_roi_by_cond, ch_type),
                min_channels=roi_min_channels)
            for part, bucket in ((roi_cond, roi_cond_frames),):
                if part is not None:
                    part = part.copy()
                    part.insert(0, "chromophore", ch_type)
                    bucket.append(part)

    _log_draw_quality(partners, refused, len(candidates))

    sources = [p for p in (path_from(r) for r in aligned_real.values()) if p]
    params = dict(
        band_fmin=band_fmin, band_fmax=band_fmax, mask_coi=mask_coi,
        **({"analysis_window_s": [round(t, 3) for t in analysis_window]}
           if analysis_window is not None else {}),
        wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax, n_iter=len(partners), cross=cross,
        chroma=list(chroma), null_kind="repaired", pair_pool=pool,
        pair_partners=sorted(partners), pair_candidates=len(candidates),
        pair_refused={reason: sorted(set(subs)) for reason, subs in sorted(refused.items())},
        # no cond_overlap: each side is cut at its own marker, so the overlap is 1 by
        # construction rather than something the run has to report
        pair_align="per-condition-marker",
        **({"wtc_window_s": float(real_params["wtc_window_s"])}
           if real_params.get("wtc_window_s") else {}),
        pair_cond_pad_s=round(cone_margin_s(band_fmin), 3),
        **wtc_grid_params(aligned_real),
        # the null is subtracted from the real table row by row, so the two have to say
        # they were built on the same clock for that subtraction to mean anything
        **alignment_params(aligned_real),
    )

    # Per condition only. A whole-run re-paired null would mean cutting the stand-in at the
    # real dyad's clock, and the sessions drift apart between blocks, so that table used to
    # correlate one person's conversation against another's game and rank a real value
    # against it.
    out_path = None
    cond_null = {"condition": "all", "nulldist": "pair", "statistic": "wtc"}
    for bucket, entities, step, extra in (
            (draw_frames, {**cond_null, "desc": "draws"},
             "hyper_wtc_bycondition_pairnull_draws",
             {"conditions": [w[0] for w in windows]}),
            (cond_frames, cond_null, "hyper_wtc_bycondition_pairnull",
             {"conditions": [w[0] for w in windows]}),
            (roi_cond_frames, {**roi_entities, **cond_null},
             "hyper_wtc_bycondition_roihom_pairnull",
             {"conditions": [w[0] for w in windows]})):
        if not bucket:
            continue
        path = write_tsv(pd.concat(bucket, ignore_index=True), _path(entities))
        _hyper_sidecar(path, step, sources, **extra, **params)
        logger.info("re-paired WTC table saved: %s", path)
        # the draws are the same null at full detail, so they are not what to return
        if entities.get("desc") != "draws":
            out_path = out_path or path

    if out_path is None:
        raise ValueError(
            "no usable stand-in was drawn for any condition, so there is no null. The log "
            "says which test each candidate failed.")

    _write_isc_null(isc_frames, isc_cond_frames, isc_draw_frames, _path, sources,
                    params, windows, isc_whiten, isc_max_lag_s, isc_band)
    return out_path


def _isc_settings_of(sidecar: Path, whiten: int, max_lag_s: float, band):
    """The band, whitening and lag the real ISC used, read off its sidecar.

    Taken from the file rather than the command line for the reason the coherence's band is:
    a null computed on other settings than the table it is subtracted from is not a null of
    anything. A tree written before those fields existed has none, and then what the caller
    passed stands, with a line in the log saying the two were not checked against each other.
    """
    try:
        params = json.loads(sidecar.read_text(encoding="utf-8")).get("parameters", {})
    except Exception:
        params = {}
    if "isc_whiten_max_order" not in params:
        logger.warning("%s records no ISC settings, so the re-paired ISC cannot be checked "
                       "against the real one; rerun `fnirs-hyper` to stamp them",
                       sidecar.name)
        return whiten, max_lag_s, band
    stored_band = params.get("isc_band_hz")
    settings = (int(params.get("isc_whiten_max_order", whiten)),
                float(params.get("isc_max_lag_s", max_lag_s)),
                tuple(stored_band) if stored_band else None)
    logger.info("re-paired ISC follows the real table: band %s, whitening up to AR(%d), "
                "lag search %gs", settings[2] or "none", settings[0], settings[1])
    return settings


def _isc_real(path: Path, by_condition: bool) -> "pd.DataFrame | None":
    """The true dyad's ISC in the shape the null summary ranks against, or None.

    The real table carries `r`; the null's summary code is the coherence null's and keys on
    `coherence`, so the column is renamed rather than the code duplicated. A whole-run row
    has no condition, which is how the two halves are told apart in one file.
    """
    if not path.exists():
        return None
    real = pd.read_csv(path, sep="\t").rename(columns={"r": "coherence"})
    if "condition" not in real.columns:
        return None if by_condition else real
    has_cond = real["condition"].notna()
    part = real[has_cond] if by_condition else real[~has_cond]
    return part.drop(columns=[] if by_condition else ["condition"]) if not part.empty else None


def _write_isc_null(frames, cond_frames, draw_frames, path_of, sources, params,
                    windows, isc_whiten, isc_max_lag_s, isc_band) -> None:
    """Summarise the re-paired correlations and write them beside the coherence tables.

    ``path_of`` is the caller's namer, so these land under the same group and task its
    coherence tables do without this function knowing either.
    """
    from fnirs_pipe.pipeline.hyper import _hyper_sidecar

    keys = ["chromophore", "sub1", "sub2", "label", "label2"]
    isc_params = {k: v for k, v in params.items()
                  if k not in ("band_fmin", "band_fmax", "wtc_fmin", "wtc_fmax", "mask_coi")}
    isc_params.update(isc_whiten_max_order=isc_whiten, isc_max_lag_s=isc_max_lag_s,
                      isc_band_hz=list(isc_band) if isc_band else None)

    for bucket, entities, step, cond in (
            (frames, {"nulldist": "pair", "statistic": "isc"},
             "hyper_isc_pairnull", False),
            (cond_frames, {"condition": "all", "nulldist": "pair", "statistic": "isc"},
             "hyper_isc_bycondition_pairnull", True)):
        if not bucket:
            continue
        real = _isc_real(path_of({"statistic": "isc"}), by_condition=cond)
        table = _average_iterations(bucket, (["condition"] if cond else []) + keys, real=real)
        path = write_tsv(table, path_of(entities))
        _hyper_sidecar(path, step, sources,
                       **({"conditions": [w[0] for w in windows]} if cond and windows else {}),
                       **isc_params)
        logger.info("re-paired ISC saved: %s", path)

    if draw_frames:
        path = write_tsv(pd.concat(draw_frames, ignore_index=True),
                         path_of({"condition": "all", "nulldist": "pair",
                                  "statistic": "isc", "desc": "draws"}))
        _hyper_sidecar(path, "hyper_isc_bycondition_pairnull_draws", sources,
                       **({"conditions": [w[0] for w in windows]} if windows else {}),
                       **isc_params)
        logger.info("re-paired ISC draws saved: %s", path)


def _log_draw_quality(partners: list, refused: dict, n_candidates: int) -> None:
    """Say what the pool gave and what it cost, since the pool is what limits the ranking."""
    n = len(partners)
    logger.info("re-paired null built from %d of %d candidates; a percentile off %d draws "
                "resolves to about %.1f points", n, n_candidates, n, 100.0 / max(n, 1))
    if n < 20:
        logger.warning("only %d stand-in(s) were available, so this null ranks coarsely and "
                       "its 95th percentile rests on %d values. That is a property of the "
                       "cohort, not of any setting.", n, n)
    for reason, subs in sorted(refused.items()):
        logger.info("  %d refused, %s: %s", len(set(subs)), reason, ", ".join(sorted(set(subs))))
