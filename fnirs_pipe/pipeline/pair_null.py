"""The re-paired null: one member against people they never interacted with.

Split from :mod:`fnirs_pipe.pipeline.wtc_null` rather than folded into it because the two
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


def real_table_params(data_dir: Path, stem: str) -> dict:
    """The real WTC table's own record of how it was computed.

    Raises rather than falling back on defaults: a null built on a guess about the band is
    worse than no null, because nothing downstream can tell the two apart.
    """
    sidecar = data_dir / f"{stem}-wtc.json"
    if not sidecar.exists():
        raise StageError(
            f"no real WTC table to rank against: {sidecar} is missing. Run "
            "`fnirs-hyper run` for this group first; the null reads its band, its mask and "
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
            "match it. Rerun `fnirs-hyper run` for this group on current code.")
    return params


def condition_coverage(
    partner_raw,
    windows: list[tuple[str, float, float]],
) -> dict[str, float]:
    """How much of each condition window the partner spent in the same-named condition.

    ::

        windows [("game1", 100.0, 400.0)], partner annotated "game1" over 150-400 s
          -> {"game1": 0.833}

    The windows come from the real dyad, so on a cohort whose sessions ran to the same
    timetable every draw scores 1.0. A draw that scores low is still counted: dropping it
    would make the number of draws depend on the partner's timing, and the table would
    quietly hold a different null per condition. It is reported instead, here, in the log
    and on the sidecar, so a cohort whose sessions drifted is visible rather than silent.
    """
    from fnirs_pipe.qc.common.windows import markers_on_data_axis

    markers = markers_on_data_axis(partner_raw)
    end = float(partner_raw.times[-1])
    out: dict[str, float] = {}
    for label, tstart, tstop in windows:
        span = max(float(tstop) - float(tstart), 0.0)
        if span <= 0:
            out[label] = 0.0
            continue
        covered = 0.0
        for m in markers:
            # condition_windows numbers a repeated description "desc#1", "desc#2"; the
            # annotation it came from carries the bare description
            if str(label).split("#")[0] != str(m["description"]):
                continue
            m_start = float(m["onset"])
            m_stop = m_start + float(m["duration"] or 0.0)
            if m_stop <= m_start:
                m_stop = end
            covered += max(0.0, min(m_stop, float(tstop)) - max(m_start, float(tstart)))
        out[label] = round(min(covered / span, 1.0), 4)
    return out


def _draw_pairs(
    output_dir: Path,
    task: str,
    fixed_id: str,
    fixed_raw,
    candidates: list,
    *,
    desc: str,
    bads_scope: str,
    scope_tasks: "list[str] | None",
    real_duration: float,
    real_offset: float,
    n_max: "int | None",
    refused: dict,
    coverage: dict,
    windows: "list[tuple[str, float, float]] | None",
):
    """Yield ``(partner_id, aligned_pair)`` for every candidate that can stand in.

    Each pair is aligned the way the real dyad was and then cut to the real dyad's analysed
    length, so every draw describes a stretch of the same length as the table it will be
    subtracted from. Three things disqualify a candidate, and each is counted in ``refused``
    rather than raised: coherence is not comparable across sampling rates, a recording too
    short to reach the real length would make the draws a mixture of lengths, and a candidate
    that moves the fixed member's own crop would put the draw on a different clock than the
    real table and invalidate the cached transforms this reuses.
    """
    from fnirs_pipe.exceptions import AlignmentError
    from fnirs_pipe.pipeline.group_io import load_group_haemo
    from fnirs_pipe.pipeline.group_quality import apply_group_bads, load_group_sqm
    from fnirs_pipe.pipeline.hyperscanning import align_recordings

    drawn = 0
    for entry in candidates:
        if n_max is not None and drawn >= n_max:
            logger.info("stopping at %d draws, the limit asked for; %d candidates unused",
                        drawn, len(candidates) - drawn)
            break
        pid = entry.subject_id
        try:
            partner = load_group_haemo(output_dir, [entry], desc=desc)
        except Exception as exc:
            refused.setdefault("unreadable", []).append(pid)
            # one line each would be the whole log on a cohort only partly preprocessed;
            # _log_draw_quality names them together
            logger.debug("%s refused as a stand-in: %s", pid, exc)
            continue

        partner_raw = next(iter(partner.values()))
        if abs(float(partner_raw.info["sfreq"]) - float(fixed_raw.info["sfreq"])) > 1e-6:
            refused.setdefault("sampling_rate", []).append(pid)
            continue

        try:
            apply_group_bads(partner, load_group_sqm(output_dir, [entry], bads_scope=bads_scope,
                                                     scope_tasks=scope_tasks))
        except Exception as exc:
            # a stand-in with no quality record keeps every channel, which would let a
            # channel the real dyad rejected into the null and nowhere else
            refused.setdefault("no_quality_record", []).append(pid)
            logger.warning("%s refused as a stand-in, no quality record: %s", pid, exc)
            continue

        try:
            aligned, offsets = align_recordings({fixed_id: fixed_raw.copy(), pid: partner_raw},
                                                task)
        except AlignmentError:
            refused.setdefault("no_shared_trigger", []).append(pid)
            continue

        if abs(float(offsets[fixed_id]) - float(real_offset)) > _DURATION_TOL_S:
            # the earliest trigger this pair shares is not the one the real dyad was cut
            # from, so the fixed member sits on a different clock than the real table
            refused.setdefault("moves_the_clock", []).append(pid)
            continue

        drawn_duration = min(float(r.times[-1]) for r in aligned.values())
        if drawn_duration < float(real_duration) - _DURATION_TOL_S:
            refused.setdefault("too_short", []).append(pid)
            continue

        for raw in aligned.values():
            raw.crop(tmax=float(real_duration))
        if windows:
            coverage[pid] = condition_coverage(aligned[pid], windows)
        drawn += 1
        yield pid, aligned


def run_pair_null(
    group_id: str,
    task: str,
    members: list,
    groups: dict,
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
    roi_min_channels: int = 2,
    isc_whiten: int = 0,
    isc_max_lag_s: float = 0.0,
    isc_band: "tuple[float | None, float | None] | None" = None,
) -> Path:
    """Draw, rank and write one group's re-paired null.

    Writes ``group-<id>_task-<task>_hyper-wtc-pairnull.tsv`` with the columns the
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
    from fnirs_pipe.io.derivatives import group_data_dir
    from fnirs_pipe.pipeline.group_io import load_group_haemo
    from fnirs_pipe.pipeline.group_quality import (apply_group_bads, load_group_sqm,
                                                   resolve_group_bands)
    from fnirs_pipe.pipeline.hyperscanning import (_hyper_sidecar, align_recordings,
                                                   alignment_params)
    from fnirs_pipe.pipeline.synchrony import (compute_wtc_pair_null, long_axis_over,
                                               wtc_grid_params)
    from fnirs_pipe.pipeline.wtc_null import _for_chroma, _real_table, write_tsv
    from fnirs_pipe.qc.common.windows import condition_windows
    from fnirs_pipe.utils.lineage import path_from

    data_dir = group_data_dir(output_dir, group_id)
    stem = f"group-{group_id}_task-{task}_hyper"
    real_params = real_table_params(data_dir, stem)
    band_fmin, band_fmax = real_params["band_fmin"], real_params["band_fmax"]
    wtc_fmin, wtc_fmax = real_params["wtc_fmin"], real_params["wtc_fmax"]
    mask_coi = bool(real_params["mask_coi"])
    window_s = real_params.get("analysis_window_s")
    analysis_window = tuple(window_s) if window_s else None

    isc_whiten, isc_max_lag_s, isc_band = _isc_settings_of(
        data_dir / f"{stem}-iscpairs.json", isc_whiten, isc_max_lag_s, isc_band)

    raws = load_group_haemo(output_dir, members, desc=desc)
    group_sqm = load_group_sqm(output_dir, members, bads_scope=bads_scope,
                               scope_tasks=scope_tasks)
    apply_group_bads(raws, group_sqm)
    sep_bands = resolve_group_bands(members, group_sqm)

    aligned_real, offsets = align_recordings(raws, task)
    fixed_id = members[0].subject_id
    true_pair = (members[0].subject_id, members[1].subject_id)
    aligned_duration = min(float(r.times[-1]) for r in aligned_real.values())
    # what a stand-in has to cover is what the real table describes, which is the analysis
    # window when there is one rather than the whole aligned recording. Demanding the whole
    # recording refuses partners whose recording reaches every sample actually compared, and
    # it takes away the one route out of a cohort whose usable lengths differ: pinning every
    # dyad to a common --tstart/--tend
    real_duration = float(analysis_window[1]) if analysis_window else aligned_duration
    recorded = real_params.get("aligned_duration_s")
    if recorded is not None and abs(float(recorded) - aligned_duration) > _DURATION_TOL_S:
        # the tree moved under the table: the null would describe a different stretch
        raise StageError(
            f"the real table was written on {float(recorded):.3f} s of aligned recording but "
            f"the tree now aligns to {aligned_duration:.3f} s. Rerun `fnirs-hyper run` for "
            f"group {group_id!r} before drawing its null.")

    candidates = partner_pool(groups, group_id, task, pool=pool)
    if not candidates:
        raise StageError(
            f"no other group runs task {task!r}, so there is nobody to re-pair "
            f"group {group_id!r} with. This null needs a cohort, not one dyad.")
    logger.info("re-paired null for %s/%s: %d candidate stand-in(s) in the %r pool",
                group_id, task, len(candidates), pool)

    windows: list = []
    if (data_dir / f"{stem}-wtcbycond.tsv").exists():
        windows = condition_windows(aligned_real[fixed_id], min_duration=1.0 / wtc_fmin)
        if analysis_window is not None:
            lo, hi = analysis_window
            windows = [w for w in windows if w[1] >= lo and w[2] <= hi]

    real = _real_table(data_dir / f"{stem}-wtc.tsv")
    real_by_cond = _real_table(data_dir / f"{stem}-wtcbycond.tsv")
    real_roi = _real_table(data_dir / f"{stem}-wtc-roihom.tsv")
    real_roi_by_cond = _real_table(data_dir / f"{stem}-wtcbycond-roihom.tsv")

    frames, cond_frames, roi_frames, roi_cond_frames = [], [], [], []
    isc_frames: list = []
    isc_cond_frames: list = []
    refused: dict[str, list[str]] = {}

    def _isc_collector(ch_type: str):
        """A callback that correlates each drawn pair, whole run and per condition."""
        from fnirs_pipe.pipeline.synchrony import compute_isc_pairs

        def _collect(partner_id: str, aligned: dict) -> None:
            ids = [fixed_id, partner_id]
            scopes = [(None, None)] + [(w[0], (w[1], w[2])) for w in (windows or [])]
            for label, window in scopes:
                try:
                    _, _, pairs, _ = compute_isc_pairs(
                        aligned, ids, ch_type, sep_bands, window=window,
                        whiten=isc_whiten, max_lag_s=isc_max_lag_s, band=isc_band)
                except Exception:
                    logger.debug("re-paired ISC failed against %s (%s)",
                                 partner_id, label or "whole run")
                    continue
                if pairs is None or pairs.empty:
                    continue
                # relabelled to the true pair for the reason the coherence is: every draw
                # has a different partner, and the summary groups by sub1/sub2
                pairs = pairs.rename(columns={"r": "coherence"})
                pairs["sub1"], pairs["sub2"] = true_pair
                pairs.insert(0, "chromophore", ch_type)
                pairs["n_valid_frac"] = 1.0
                if label is None:
                    isc_frames.append(pairs)
                else:
                    pairs.insert(1, "condition", label)
                    isc_cond_frames.append(pairs)

        return _collect

    coverage: dict[str, dict[str, float]] = {}
    partners: list[str] = []
    for ch_type in chroma:
        draws = _draw_pairs(
            output_dir, task, fixed_id, raws[fixed_id], candidates, desc=desc,
            bads_scope=bads_scope, scope_tasks=scope_tasks, real_duration=real_duration,
            real_offset=float(offsets[fixed_id]), n_max=n_max, refused=refused,
            coverage=coverage, windows=windows)
        null = compute_wtc_pair_null(
            draws, true_pair, long_axis_over(aligned_real.values(), ch_type, sep_bands),
            band_fmin, band_fmax, fmin=wtc_fmin, fmax=wtc_fmax, cross=cross,
            limit_scales=limit_scales, mask_coi=mask_coi, ch_type=ch_type,
            sep_bands=sep_bands, windows=windows, analysis_window=analysis_window,
            on_draw=_isc_collector(ch_type))
        partners = null.partners or []

        whole, by_cond = null.summarise(real=_for_chroma(real, ch_type),
                                        real_by_cond=_for_chroma(real_by_cond, ch_type))
        for part, bucket in ((whole, frames), (by_cond, cond_frames)):
            if part is not None:
                part = part.copy()
                # labelled after the grouping, never before: the summary groups by the
                # columns it knows and would drop this one
                part.insert(0, "chromophore", ch_type)
                bucket.append(part)

        if roi_map:
            roi_whole, roi_cond = null.summarise_roi(
                roi_map, real=_for_chroma(real_roi, ch_type),
                real_by_cond=_for_chroma(real_roi_by_cond, ch_type),
                min_channels=roi_min_channels)
            for part, bucket in ((roi_whole, roi_frames), (roi_cond, roi_cond_frames)):
                if part is not None:
                    part = part.copy()
                    part.insert(0, "chromophore", ch_type)
                    bucket.append(part)

    _log_draw_quality(partners, refused, coverage, len(candidates))

    sources = [p for p in (path_from(r) for r in aligned_real.values()) if p]
    params = dict(
        band_fmin=band_fmin, band_fmax=band_fmax, mask_coi=mask_coi,
        **({"analysis_window_s": [round(t, 3) for t in analysis_window]}
           if analysis_window is not None else {}),
        wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax, n_iter=len(partners), cross=cross,
        chroma=list(chroma), null_kind="repaired", pair_pool=pool,
        pair_partners=sorted(partners), pair_candidates=len(candidates),
        pair_refused={reason: sorted(set(subs)) for reason, subs in sorted(refused.items())},
        **({"cond_overlap": _median_coverage(coverage)} if coverage else {}),
        **wtc_grid_params(aligned_real),
        # the null is subtracted from the real table row by row, so the two have to say
        # they were built on the same clock for that subtraction to mean anything
        **alignment_params(aligned_real),
    )

    out_path = write_tsv(pd.concat(frames, ignore_index=True),
                         data_dir / f"{stem}-wtc-pairnull.tsv")
    _hyper_sidecar(out_path, "hyper_wtc_pairnull", sources, **params)
    logger.info("re-paired WTC band means saved: %s", out_path)

    for bucket, suffix, step, extra in (
            (cond_frames, "-wtcbycond-pairnull", "hyper_wtc_bycondition_pairnull",
             {"conditions": [w[0] for w in windows]}),
            (roi_frames, "-wtc-roihom-pairnull", "hyper_wtc_roihom_pairnull", {}),
            (roi_cond_frames, "-wtcbycond-roihom-pairnull",
             "hyper_wtc_bycondition_roihom_pairnull",
             {"conditions": [w[0] for w in windows]})):
        if not bucket:
            continue
        path = write_tsv(pd.concat(bucket, ignore_index=True), data_dir / f"{stem}{suffix}.tsv")
        _hyper_sidecar(path, step, sources, **extra, **params)
        logger.info("re-paired WTC table saved: %s", path)

    _write_isc_null(isc_frames, isc_cond_frames, data_dir, stem, sources, params, windows,
                    isc_whiten, isc_max_lag_s, isc_band)
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
                       "against the real one; rerun `fnirs-hyper run` to stamp them",
                       sidecar.name)
        return whiten, max_lag_s, band
    stored_band = params.get("isc_band_hz")
    return (int(params.get("isc_whiten_max_order", whiten)),
            float(params.get("isc_max_lag_s", max_lag_s)),
            tuple(stored_band) if stored_band else None)


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


def _write_isc_null(frames, cond_frames, data_dir, stem, sources, params, windows,
                    isc_whiten, isc_max_lag_s, isc_band) -> None:
    """Summarise the re-paired correlations and write them beside the coherence tables."""
    from fnirs_pipe.pipeline.hyperscanning import _hyper_sidecar
    from fnirs_pipe.pipeline.synchrony import _average_iterations
    from fnirs_pipe.pipeline.wtc_null import write_tsv

    keys = ["chromophore", "sub1", "sub2", "label", "label2"]
    isc_params = {k: v for k, v in params.items()
                  if k not in ("band_fmin", "band_fmax", "wtc_fmin", "wtc_fmax", "mask_coi")}
    isc_params.update(isc_whiten_max_order=isc_whiten, isc_max_lag_s=isc_max_lag_s,
                      isc_band_hz=list(isc_band) if isc_band else None)

    for bucket, suffix, step, cond in (
            (frames, "-isc-pairnull", "hyper_isc_pairnull", False),
            (cond_frames, "-iscbycond-pairnull", "hyper_isc_bycondition_pairnull", True)):
        if not bucket:
            continue
        real = _isc_real(data_dir / f"{stem}-iscpairs.tsv", by_condition=cond)
        table = _average_iterations(bucket, (["condition"] if cond else []) + keys, real=real)
        path = write_tsv(table, data_dir / f"{stem}{suffix}.tsv")
        _hyper_sidecar(path, step, sources,
                       **({"conditions": [w[0] for w in windows]} if cond and windows else {}),
                       **isc_params)
        logger.info("re-paired ISC saved: %s", path)


def _median_coverage(coverage: dict) -> dict:
    """Per condition, the median of how much of it each stand-in spent in that condition."""
    labels = {label for per_draw in coverage.values() for label in per_draw}
    out = {}
    for label in sorted(labels):
        vals = sorted(per_draw[label] for per_draw in coverage.values() if label in per_draw)
        mid = len(vals) // 2
        out[label] = round(vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2, 4)
    return out


def _log_draw_quality(partners: list, refused: dict, coverage: dict, n_candidates: int) -> None:
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
    if coverage:
        for label, med in sorted(_median_coverage(coverage).items()):
            if med < 0.8:
                logger.warning("condition %r: the stand-ins were in that condition for a "
                               "median %.0f%% of the window the real table used, so this "
                               "condition null is only partly task-matched", label, med * 100)
