"""nirspipe-hyper CLI (argparse): dyad analysis over a derivatives tree.

Hyperscanning is its own domain: its input is a pairs table, its unit is a dyad, and it
reads derivatives rather than BIDS raw.

Each command is its own console script. `nirspipe-hyper` and `nirspipe-hyper-pairnull` read the
subject tree and write a separate dyad tree (`<source> <output> group`); the rest re-read
only the dyad tree (`<output> group`).
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

from threadpoolctl import threadpool_limits

from nirspipe.utils import ROI_MIN_CHANNELS, pair_of
from nirspipe.cli import _app_options
from nirspipe.cli import _shared
from nirspipe.io.naming import roi_map_name
from nirspipe.pipeline.hyper.pair_null_group import ISC_TESTS, P_CORRECTIONS
from nirspipe.qc.metrics import SCI_WINDOW_S
from nirspipe.utils.logging import get_logger, setup_logging, thread_log_file
from nirspipe import __version__
from nirspipe.cli._app_options import separation_bands_from_args
from nirspipe.exceptions import GroupCSVError, AlignmentError, MissingDerivativesError, StageError
from nirspipe.io.derivatives import (
    LINK_PREPROCESSED, entity_of, group_label, group_report_dir, read_json, write_bidsignore,
    write_dataset_description,
)
from nirspipe.io.groups import parse_group_csv
from nirspipe.pipeline.hyper.alignment import resolve_analysis_window
from nirspipe.pipeline.hyper.group_quality import (resolve_group_bands, write_group_bads,
                                                   write_group_channel_coords)
from nirspipe.qc.metrics._helpers import bands_to_record, long_channel_picks

setup_logging()

logger = get_logger("cli.hyper")

_BADS_SCOPE_CHOICES = ["run", "subject"]


def _select_groups(pairs_csv: Path, group_id: str | None, task_label: list[str] | None,
                   participant_label: list[str] | None = None) -> dict:
    """Parse the group CSV and filter by group_id / task_label / participant_label. Exits non-zero on empty selection."""
    try:
        groups = parse_group_csv(pairs_csv)
    except GroupCSVError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)

    if group_id is not None:
        groups = {k: v for k, v in groups.items() if k[0] == group_id}
        if not groups:
            print(f"[error] group_id '{group_id}' not found in CSV", file=sys.stderr)
            raise SystemExit(1)

    if task_label is not None:
        groups = {k: v for k, v in groups.items() if k[1] in task_label}
        if not groups:
            print(f"[error] task_label {task_label} not found in CSV", file=sys.stderr)
            raise SystemExit(1)

    if participant_label is not None:
        # the CSV may spell a subject with or without sub-, the flag never carries it
        wanted = set(participant_label)
        groups = {k: v for k, v in groups.items()
                  if any(_shared.BidsLabel(e.subject_id) in wanted for e in v)}
        if not groups:
            print(f"[error] no group in the CSV has participant(s) {participant_label}",
                  file=sys.stderr)
            raise SystemExit(1)

    return groups


def _group_outcome(process, key: tuple, members: list,
                   log_file: "Path | None") -> "tuple[bool, str | None]":
    """One group's run as (succeeded, its output path or the line naming the failure)."""
    gid, task, ses = key
    with thread_log_file(log_file) if log_file is not None else nullcontext():
        try:
            path = process(gid, task, ses, members)
            return True, (str(path) if path is not None else None)
        except MissingDerivativesError as exc:
            return False, f"[skip] {exc}"
        except AlignmentError as exc:
            # never a silent fallback: trimming is only right when the members started together
            return False, (f"[skip] alignment failed: {exc} If the members started recording "
                           "at the same moment (a task without triggers), --no-align trims "
                           "every recording to the shortest instead; it assumes simultaneous "
                           "starts.")
        except StageError as exc:
            return False, f"[error] {exc}"
        except Exception as exc:
            logger.exception("group %s task %s failed", gid, task)
            return False, f"[error] unexpected error: {exc}"


# held for the worker's life: threadpoolctl lifts its limits when the object goes
_WORKER_BLAS = None


def _init_group_worker(blas: "int | None", verbose: bool) -> None:
    global _WORKER_BLAS
    setup_logging(verbose=verbose)
    if blas:
        _WORKER_BLAS = threadpool_limits(limits=blas)


def _run_groups(groups: dict, process, nprocs: "int | None" = None, verbose: bool = False,
                log_file=None, what: str = "report") -> None:
    """Run process(gid, task, session, members) -> output path per group, logging each to log_file(key); exit non-zero on failure."""
    jobs, blas = _shared.split_cpus(nprocs, len(groups))
    print(f"Processing {len(groups)} group session(s)"
          + (f", {jobs} at once..." if jobs > 1 else "..."))

    def _name(key, members):
        gid, task, ses = key
        return f"{gid}/{f'ses-{ses}/' if ses else ''}{task} ({len(members)} subjects)"

    def _report(ok, text, prefix):
        if ok:
            if text is not None:
                print(f"{prefix}{what} -> {text}", flush=True)
        else:
            print(f"{prefix}{text}", file=sys.stderr, flush=True)
        return 0 if ok else 1

    logs = {key: (log_file(key) if log_file else None) for key in groups}
    n_fail = 0
    if jobs == 1:
        with threadpool_limits(limits=blas) if blas else nullcontext():
            for key, members in groups.items():
                print(f"  -> {_name(key, members)}", flush=True)
                n_fail += _report(*_group_outcome(process, key, members, logs[key]), "     ")
    else:
        # processes, not threads: matplotlib holds the GIL while it rasterises. So `process`
        # has to pickle, a module-level function or a partial of one
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_group_worker,
                                 initargs=(blas, verbose)) as pool:
            futures = {pool.submit(_group_outcome, process, key, members, logs[key]): key
                       for key, members in groups.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    # the worker itself died, or its result would not pickle
                    outcome = (False, f"[error] worker failed: {exc!r}")
                n_fail += _report(*outcome, f"  {_name(key, groups[key])}: ")

    print(f"\nDone: {len(groups) - n_fail} succeeded, {n_fail} failed.")
    if n_fail > 0:
        raise SystemExit(1)


def _load_aligned_group(derivatives_dir, members, task, desc, no_align, normalize, bads_scope,
                        passband_check=None, scope_tasks=None):
    """Load one dyad, put both recordings on one time axis, and mark the rejected channels.

    Returns (aligned_raws, offsets, group_sqm). The rejections are applied here rather than
    in each metric because --bads-scope decides them: one file's sidecar holds only that
    run's own rejections.

    --tstart/--tend are not applied here. They name a window of the analysis, and the report
    takes it out of the transform of the whole recording rather than cutting the recording
    to it; see `resolve_analysis_window`.

    `passband_check` is the (fmin, fmax) a metric is about to ask for, checked against the
    bandpass the files record while the sidecars are still in hand.
    """
    from nirspipe.pipeline.hyper.alignment import (
        align_recordings,
        normalize_raws,
        trim_to_shortest,
    )
    from nirspipe.pipeline.hyper.group_quality import (
        apply_group_bads,
        load_group_sqm,
    )
    from nirspipe.pipeline.hyper.group_io import (
        load_group_haemo,
        warn_outside_passband,
    )

    raws = load_group_haemo(derivatives_dir, members, desc=desc)
    if passband_check is not None:
        warn_outside_passband(raws, *passband_check)
    group_sqm = load_group_sqm(derivatives_dir, members, bads_scope=bads_scope,
                               scope_tasks=scope_tasks)
    apply_group_bads(raws, group_sqm)
    if no_align:
        aligned_raws, offsets = trim_to_shortest(raws)
    else:
        aligned_raws, offsets = align_recordings(raws, task)
    if normalize:
        aligned_raws = normalize_raws(aligned_raws)
    return aligned_raws, offsets, group_sqm


def _quality_summary(aligned_raws: dict, group_sqm: dict, sep_bands=None) -> None:
    """Print what the metrics are about to be computed on, one line per subject.

    ``sub-01  long 18/20  bad 2  mean SCI (10 s) 0.86  from: tapping``

    Counted off the aligned Raw after the rejections are applied, so it describes the channel
    set the coherence actually uses rather than what the montage holds.
    """
    for subject_id, raw in aligned_raws.items():
        sqm = group_sqm.get(subject_id, {})
        pairs = {pair_of(ch) for ch in raw.ch_names
                 if ch.endswith(" hbo") or ch.endswith(" hbr")}
        bad = {pair_of(ch) for ch in raw.info["bads"]}
        kept = len(long_channel_picks(raw, "hbo", sep_bands=sep_bands))  # bads already dropped

        scores = [v for v in (sqm.get("sci_win_per_channel") or {}).values()
                  if v is not None and v == v]
        label = f"mean SCI ({SCI_WINDOW_S:g} s)"
        sci = f"{label} {sum(scores) / len(scores):.2f}" if scores else f"{label} n/a"

        tasks = sorted({t for ts in (sqm.get("bad_channel_sources") or {}).values()
                        for t in ts})
        origin = f"  from: {', '.join(tasks)}" if tasks else ""

        print(f"     {subject_id}  long {kept}/{len(pairs)}  bad {len(bad)}  {sci}{origin}")
        if pairs and len(bad) > len(pairs) / 2:
            print(f"     [warn] {subject_id} loses {len(bad)} of {len(pairs)} channel pairs",
                  file=sys.stderr)


def _merge_reminder(output_dir: Path) -> None:
    """Say so when the merged tables are missing or older than the per-dyad ones.

    Merging is not done here: merging the whole tree after a one-dyad run would fail on bands
    a later run changed, and would race a parallel run for the same files.

    Driven off the aggregator's own discovery, so the counts are the ones `merge` would
    use and a kind added later cannot be left out.
    """
    from nirspipe.pipeline.hyper.wtc_aggregate import _merged_path, merge_kinds

    lines = []
    for parts in merge_kinds(output_dir).values():
        merged = _merged_path(output_dir, parts[0])
        if not merged.exists():
            lines.append(f"  {len(parts)} table(s) for {merged.name}, never merged")
            continue
        stale = sum(p.stat().st_mtime > merged.stat().st_mtime for p in parts)
        if stale:
            lines.append(f"  {len(parts)} table(s) for {merged.name}, {stale} newer than it")

    if lines:
        print("\n".join(["", *lines, f"Run `nirspipe-hyper-merge {output_dir} group` for one table per kind."]))


def _warn_band_mismatch(isc_band, wtc_band_fmin, wtc_band_fmax) -> None:
    """Say so when ISC and the coherence are about to describe different frequencies.

    The two are averages of one complex coherency, so they compare only on one band: zero-lag
    Pearson r is the power-weighted mean of ``|gamma| cos phi`` and the WTC band mean is the
    unweighted mean of ``|gamma|^2``.

    Reported rather than enforced, so a run that computes no ISC need not name a band for it.
    """
    wtc_band = (wtc_band_fmin, wtc_band_fmax)
    if wtc_band == (None, None):
        return
    if isc_band is None:
        logger.warning(
            "ISC reads the whole passband while the coherence band is %s-%s Hz, so the two "
            "describe different frequencies and comparing them is not meaningful. Pass "
            "--band-fmin/--band-fmax to put them on one band.",
            wtc_band_fmin, wtc_band_fmax)
    elif isc_band != wtc_band:
        logger.warning(
            "ISC band %s-%s Hz against coherence band %s-%s Hz: deliberate if you meant it, "
            "but the two are then not comparable.",
            isc_band[0], isc_band[1], wtc_band_fmin, wtc_band_fmax)


def _run_one_group(s: SimpleNamespace, gid, task, ses, members):
    """One dyad of `nirspipe-hyper`: align, null, report, run record. ``s`` is the run's settings."""
    from nirspipe.cli.run_record import write_group_run_record
    from nirspipe.pipeline.hyper.wtc_null import run_wtc_null
    from nirspipe.qc.common.windows import condition_windows, split_windows
    from nirspipe.qc.hyper.hyper_report import build_hyper_post_report

    aligned_raws, offsets, group_sqm = _load_aligned_group(
        s.derivatives_dir, members, task, s.desc, s.no_align, s.normalize, s.bads_scope,
        passband_check=(s.wtc_fmin, s.wtc_fmax), scope_tasks=s.scope_tasks)
    analysis_window = resolve_analysis_window(aligned_raws, s.tstart, s.tend)
    sep_bands = resolve_group_bands(members, group_sqm, s.bands_override)
    _quality_summary(aligned_raws, group_sqm, sep_bands)
    if s.check_only:
        return None
    bad_channels = {sid: sqm.get("bad_channels", []) for sid, sqm in group_sqm.items()}
    write_group_bads(s.output_dir, members, group_sqm, s.bads_scope, session=ses)
    write_group_channel_coords(s.derivatives_dir, s.output_dir, members, s.desc, session=ses)
    # resolved here rather than twice downstream: the report and the null have to agree
    # on the windows or their tables cannot be subtracted row by row
    ref = next(iter(aligned_raws.values()), None)
    cond_windows = (condition_windows(ref, min_duration=1.0 / s.wtc_fmin)
                    if s.wtc_by_condition and ref is not None else [])
    if analysis_window is not None:
        # a condition outside the analysis window has nothing in it to report, and one
        # straddling an edge would be reported as a full block while only part of it
        # was read. Dropped rather than clipped, so no page claims a span it did not get
        lo, hi = analysis_window
        inside = [w for w in cond_windows if w[1] >= lo and w[2] <= hi]
        if len(inside) != len(cond_windows):
            dropped = [w[0] for w in cond_windows if w not in inside]
            print(f"     [info] {len(dropped)} condition(s) outside "
                  f"--tstart/--tend: {', '.join(dropped)}")
        cond_windows = inside
    if s.wtc_window_s and cond_windows:
        # equal-length windows replace the conditions as the unit of analysis; the null
        # resolves the same grid off the same reference
        cond_windows, _ = split_windows(cond_windows, s.wtc_window_s)
        if not cond_windows:
            print(f"     [warn] --wtc-window-s {s.wtc_window_s}: no condition is long "
                  "enough for one window, so there is nothing per condition to report",
                  file=sys.stderr)
    # Before the report, not after: the level the phase arrows are drawn against comes
    # out of the surrogates, and the figures are built inside the report. The report
    # writes the table this returns, once the real band means it is ranked against exist.
    nulls = run_wtc_null(
        group_id=gid,
        task=task,
        aligned_raws=aligned_raws,
        output_dir=s.output_dir,
        n_iter=s.wtc_phase_null,
        wtc_fmin=s.wtc_fmin,
        wtc_fmax=s.wtc_fmax,
        band_fmin=s.wtc_band_fmin,
        band_fmax=s.wtc_band_fmax,
        seed=s.wtc_seed,
        cross=s.wtc_phase_null_cross,
        limit_scales=s.wtc_limit_scales,
        mask_coi=s.wtc_mask_coi,
        chroma=s.chroma,
        sep_bands=sep_bands,
        windows=cond_windows,
        analysis_window=analysis_window,
        whiten_s=s.wtc_whiten,
        session=ses,
        # an ROI level only where the null crosses as the real maps do: a crossed (R, R)
        # map and an uncrossed R average different pairings
        roi_map=s.roi_map if s.wtc_phase_null_cross == s.wtc_channel_cross else None,
        roi_map_name=s.roi_name,
        roi_min_channels=s.wtc_roi_min_channels,
    ) if s.wtc_phase_null else None

    report_path = build_hyper_post_report(
        group_id=gid,
        task=task,
        session=ses,
        group=members,
        aligned_raws=aligned_raws,
        offsets=offsets,
        output_dir=s.output_dir,
        roi_map=s.roi_map,
        roi_map_name=s.roi_name,
        bad_channels=bad_channels,
        subject_sqm=group_sqm,
        wtc_fmin=s.wtc_fmin,
        wtc_fmax=s.wtc_fmax,
        wtc_band_fmin=s.wtc_band_fmin,
        wtc_band_fmax=s.wtc_band_fmax,
        wtc_significance=s.wtc_significance,
        wtc_seed=s.wtc_seed,
        wtc_mc_count=s.wtc_mc_count,
        wtc_channel_cross=s.wtc_channel_cross,
        wtc_by_condition=s.wtc_by_condition, wtc_window_s=s.wtc_window_s,
        wtc_cond_pad_s=s.cond_pad,
        cond_windows=cond_windows,
        wtc_limit_scales=s.wtc_limit_scales,
        wtc_save_maps=s.wtc_save_maps,
        wtc_mask_coi=s.wtc_mask_coi,
        wtc_roi_min_channels=s.wtc_roi_min_channels,
        wtc_arrow_min=s.wtc_arrow_min,
        wtc_chroma=s.chroma,
        wtc_whiten_s=s.wtc_whiten,
        wtc_nulls=nulls,
        wtc_phase_null=s.wtc_phase_null,
        wtc_phase_null_cross=s.wtc_phase_null_cross,
        isc_threshold=s.isc_threshold,
        isc_whiten_s=s.isc_whiten_s,
        isc_max_lag_s=s.isc_max_lag,
        isc_phase_null=s.isc_phase_null,
        isc_band=s.isc_band,
        no_report=s.no_report,
        sep_bands=sep_bands,
        analysis_window=analysis_window,
        desc=s.desc,
        bads_scope=s.bads_scope,
    )
    try:
        # the resolved bands under the keys the SQM record stamps them with, so the two
        # can be compared directly; run_args holds only the flags as typed. An absent
        # upper bound drops out, TOML having no null and the writer skipping None
        # throughout, so absent here means off rather than unrecorded
        record = write_group_run_record(
            {**s.run_args, **bands_to_record(sep_bands)},
            gid, task, s.timestamp, s.output_dir,
            group_report_dir(s.output_dir, gid),
            members=[e.subject_id for e in members], session=ses,
        )
        logger.info("group-%s | run record -> %s", gid, record)
    except Exception:
        logger.warning("group-%s | run record failed", gid, exc_info=True)

    return report_path


def cmd_run(
    derivatives_dir: Path, output_dir: Path,
    pairs_csv: Path, group_id: str | None, task_label: list[str] | None,
    desc: str, roi_mapping: Path | None,
    wtc_fmin: float, wtc_fmax: float,
    wtc_band_fmin: float | None, wtc_band_fmax: float | None,
    wtc_significance: bool, wtc_mc_count: int, wtc_seed: int | None,
    wtc_mask_coi: bool, wtc_roi_min_channels: int, wtc_arrow_min: float,
    wtc_channel_cross: bool,
    wtc_by_condition: bool, wtc_chroma: str, wtc_window_s: "float | None",
    wtc_cond_transform: bool, wtc_cond_pad_s: "float | None",
    wtc_limit_scales: bool, wtc_save_maps: bool, wtc_whiten: float,
    wtc_phase_null: int | None, wtc_phase_null_cross: bool | None,
    bads_scope: str, isc_threshold: "float | None", isc_whiten_s: float,
    isc_max_lag: float, isc_phase_null: int,
    isc_fmin: "float | None", isc_fmax: "float | None",
    no_report: bool,
    normalize: bool, no_align: bool, tstart: float | None, tend: float | None,
    short_max_dist: float | None, long_min_dist: float | None,
    long_max_dist: float | None,
    check_only: bool, verbose: bool,
    participant_label: list[str] | None = None,
    band_fmin: "float | None" = None, band_fmax: "float | None" = None,
    config: "Path | None" = None, nprocs: "int | None" = None,
) -> None:
    """Dyad WTC + ISC report per group, plus the phase-scrambled null when --wtc-phase-null is given.

    The null reuses this run's aligned recordings and every band parameter, so it cannot be
    computed over a different band than the table it sits beside.
    """
    # unset, the null is crossed exactly when the real table is
    if wtc_phase_null_cross is None:
        wtc_phase_null_cross = wtc_channel_cross
    # the shared band fills whichever metric was not given its own
    wtc_band_fmin = band_fmin if wtc_band_fmin is None else wtc_band_fmin
    wtc_band_fmax = band_fmax if wtc_band_fmax is None else wtc_band_fmax
    isc_fmin = band_fmin if isc_fmin is None else isc_fmin
    isc_fmax = band_fmax if isc_fmax is None else isc_fmax
    # every parameter as resolved, for the run record. Read off locals() before anything
    # else runs, so a new option lands in the record without being listed here as well.
    run_args = dict(locals())

    # each metric gets the band it was given, never the other's; a mismatch is only warned about
    isc_band = (isc_fmin, isc_fmax) if (isc_fmin is not None or isc_fmax is not None) else None
    _warn_band_mismatch(isc_band, wtc_band_fmin, wtc_band_fmax)

    # validated here so a bad triple fails before any dyad is loaded; which of the three the
    # caller actually named is what resolve_group_bands needs, so the dict is what is kept
    bands_override = separation_bands_from_args({
        "short_max_dist": short_max_dist, "long_min_dist": long_min_dist,
        "long_max_dist": long_max_dist,
    })

    from nirspipe.cli.run_record import RUN_TIMESTAMP_FORMAT

    setup_logging(verbose=verbose)
    write_dataset_description(output_dir, name="nirspipe-hyper output",
                              generated_by="nirspipe-hyper", source=derivatives_dir,
                              link=LINK_PREPROCESSED)
    write_bidsignore(output_dir)
    timestamp = datetime.now().strftime(RUN_TIMESTAMP_FORMAT)

    if wtc_window_s is not None:
        if not wtc_by_condition:
            print("[error] --wtc-window-s needs --by-condition; there are no conditions to "
                  "cut into windows without it.", file=sys.stderr)
            raise SystemExit(1)
        if wtc_window_s <= 0:
            print(f"[error] --wtc-window-s must be positive, got {wtc_window_s}",
                  file=sys.stderr)
            raise SystemExit(1)

    if wtc_channel_cross and roi_mapping is None:
        print("[warn] --channel-cross without --roi-mapping: the crossed channel table "
              "is written but no ROI x ROI matrix is built from it.", file=sys.stderr)

    chroma = ("hbo", "hbr") if wtc_chroma == "both" else (wtc_chroma,)
    if len(chroma) > 1:
        print("[info] --wtc-chroma both: two full WTC passes per dyad, so roughly twice "
              "the runtime. Pass hbo or hbr for one.", file=sys.stderr)
    if wtc_phase_null and wtc_phase_null_cross:
        print("[info] the phase-scrambled null is crossed: every iteration covers every "
              "channel pairing, the squared pair count, and dominates the runtime. "
              "--no-wtc-phase-null-cross draws it over the homologous pairings only.",
              file=sys.stderr)

    groups = _select_groups(pairs_csv, group_id, task_label, participant_label)

    roi_map = _app_options.load_roi_mapping(roi_mapping)
    # the seg- entity every ROI table takes, so one tree can hold two ROI definitions
    roi_name = roi_map_name(roi_mapping)

    scope_tasks = sorted({key[1] for key in groups})

    # ---- how each condition is read: windowed out of the run, or transformed on its own ----
    # Resolved once, not per dyad: it depends only on the band, and a run whose conditions
    # were read two different ways would put incomparable rows in one table.
    cond_pad = None
    if wtc_cond_transform:
        if wtc_phase_null:
            print("[error] --wtc-cond-transform cannot be used with --wtc-phase-null. The "
                  "flag puts the real per-condition tables on the cut-then-transform route "
                  "and the null has no way to follow: it reads its conditions out of one "
                  "whole-run transform per iteration, and matching that would cost a "
                  "transform per condition per iteration. A null read off a different route "
                  "than the table it is subtracted from measures the difference between the "
                  "routes. Drop one of the two.", file=sys.stderr)
            raise SystemExit(1)
        if not wtc_by_condition:
            print("[error] --wtc-cond-transform needs --wtc-by-condition; there are no "
                  "conditions to transform without it.", file=sys.stderr)
            raise SystemExit(1)
        if str(wtc_cond_pad_s).lower() == "auto":
            from nirspipe.utils import cone_margin_s
            # the band the means are taken over, which is what the cone has to clear;
            # --wtc-band-fmin falls back to --wtc-fmin exactly as the report resolves it
            cond_pad = cone_margin_s(wtc_band_fmin or wtc_fmin)
        else:
            cond_pad = float(wtc_cond_pad_s)
        logger.info("each condition transformed on its own over a cut padded by %.1f s "
                    "per side%s", cond_pad,
                    "" if cond_pad else " (none: cut to its own boundaries, which inflates "
                                        "a short condition)")
    elif str(wtc_cond_pad_s).lower() != "auto":
        logger.warning("--wtc-cond-pad-s is ignored without --wtc-cond-transform")

    if wtc_phase_null:
        logger.info("phase-scrambled null requested: %d full WTC runs per dyad, on top of the real one",
                    wtc_phase_null)
    settings = SimpleNamespace(
        **run_args, run_args=run_args, chroma=chroma, cond_pad=cond_pad,
        bands_override=bands_override, roi_map=roi_map, roi_name=roi_name,
        scope_tasks=scope_tasks, isc_band=isc_band, timestamp=timestamp)
    _run_groups(groups, partial(_run_one_group, settings), nprocs, verbose,
                log_file=None if check_only else (
                    lambda key: group_report_dir(output_dir, key[0]) / "logs"
                    / f"{group_label(*key)}.log"))
    if check_only:
        return

    # The index is rebuilt for every dyad this run touched, reading the tables rather than
    # anything held in memory, so a dyad whose other tasks were analysed in an earlier run
    # still lists them. Not built inside `_run_one_group`, which would rewrite it once per task
    # from a tree missing the tasks still to come.
    from nirspipe.qc.hyper.hyper_index import write_hyper_index

    for gid in dict.fromkeys(key[0] for key in groups):
        folder = output_dir / f"group-{gid}"
        try:
            path = write_hyper_index(folder, gid,
                                     [e.subject_id for e in groups[next(
                                         k for k in groups if k[0] == gid)]],
                                     run_command=" ".join(sys.argv))
            if path is not None:
                print(f"index  -> {path}")
        except Exception:
            logger.warning("group-%s | dyad index failed", gid, exc_info=True)

    try:
        _merge_reminder(output_dir)      # a hint, never a reason to fail the run
    except Exception as exc:
        logger.debug("merge reminder skipped: %s", exc)


def cmd_band(
    output_dir: Path, wtc_band_fmin: float, wtc_band_fmax: float,
    wtc_mask_coi: bool, wtc_suffix: str | None, verbose: bool,
) -> None:
    """Re-average every saved WTC map over a new band, without recomputing the transform."""
    from nirspipe.pipeline.hyper.wtc_store import reband_tree

    setup_logging(verbose=verbose)

    written = reband_tree(output_dir, wtc_band_fmin, wtc_band_fmax,
                          suffix=wtc_suffix, mask_coi=wtc_mask_coi)
    for path in written:
        print(f"reband -> {path}")
    if not written:
        print(f"no *_stat-wtc_relmat.npz under {output_dir}; rerun `nirspipe-hyper "
              "--wtc-save-maps` to write them", file=sys.stderr)


def cmd_group_null(
    output_dir: Path, task: str, chroma: "list[str] | None", null: "list[str] | None",
    roi_mapping: "Path | None", n_resample: int, seed: int | None, verbose: bool,
    p_correction: str = "none", isc_test: str = "signed",
) -> None:
    """Read each null's draws above the cell: one verdict per occasion, one per cohort."""
    from nirspipe.pipeline.hyper.pair_null_group import (
        chromas_on_disk, nulls_on_disk, write_group_null)
    from nirspipe.qc.hyper.groupnull_report import write_groupnull_report

    setup_logging(verbose=verbose)
    roi_map = _app_options.load_roi_mapping(roi_mapping)
    nulls = null or nulls_on_disk(output_dir, task)
    if not nulls:
        print(f"Error: no null draws for task {task} under {output_dir}. Run "
              "nirspipe-hyper-pairnull, or nirspipe-hyper with --wtc-phase-null or "
              "--isc-phase-null, first.", file=sys.stderr)
        raise SystemExit(1)
    # each combination seeded alike, so it matches a run that asked for it alone
    for one_null in nulls:
        for one_chroma in chroma or chromas_on_disk(output_dir, task, one_null):
            written = write_group_null(output_dir, task=task, chroma=one_chroma,
                                       null=one_null, roi_map=roi_map,
                                       n_resample=n_resample, seed=seed,
                                       p_correction=p_correction, isc_test=isc_test)
            for path in written:
                print(f"group-null -> {path}")
            # one per statistic read, coherence and correlation each having its own test
            for cohort in (p for p in written if entity_of(p.name, "desc") == "cohort"):
                print(f"group-null methods -> {_write_group_null_methods(cohort)}")
    # every table on disk for the task, so a second chromophore or null adds a section
    page = write_groupnull_report(output_dir, task)
    if page is not None:
        print(f"group-null page -> {page}")


def _write_group_null_methods(cohort: Path) -> Path:
    """The Methods paragraph for a cohort test, in logs/ beside its tables, as the BIDS apps
    leave theirs: no report carries it, the tables being cross-dyad."""
    from nirspipe.qc.boilerplate import collect_software_versions, generate_methods_text
    from nirspipe.qc.boilerplate.vocabulary import boilerplate_key, template_slots

    side = read_json(cohort.with_suffix(".json"))
    params = side.get("parameters") or {}
    key = boilerplate_key(side.get("step"), params)
    text = generate_methods_text(versions=collect_software_versions(),
                                 steps=[(key, template_slots(key, params))])
    out = cohort.parent / "logs" / f"{cohort.stem}_methods.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text["markdown"] + "\n", encoding="utf-8")
    return out


def cmd_index(output_dir: Path, group_id: str | None, verbose: bool) -> None:
    """Write one dyad index per group-* directory, from the tables already on disk."""
    from nirspipe.qc.hyper.hyper_index import write_hyper_index

    setup_logging(verbose=verbose)

    folders = sorted(d for d in output_dir.glob("group-*") if d.is_dir())
    if group_id is not None:
        folders = [d for d in folders if d.name == f"group-{group_id}"]
        if not folders:
            print(f"[error] no group-{group_id} directory under {output_dir}",
                  file=sys.stderr)
            raise SystemExit(1)

    wrote = 0
    for folder in folders:
        path = write_hyper_index(folder, folder.name.removeprefix("group-"),
                                 run_command=" ".join(sys.argv))
        if path is not None:
            print(f"{folder.name} -> {path}")
            wrote += 1
    if not wrote:
        print(f"no coherence tables under {output_dir}; run `nirspipe-hyper` first")


def _pair_null_one_group(s: SimpleNamespace, gid, task, ses, members):
    """One target of `nirspipe-hyper-pairnull`, its stand-ins drawn from every group of the table."""
    from nirspipe.pipeline.hyper.pair_null import run_pair_null

    return run_pair_null(
        gid, task, members, s.all_groups, s.derivatives_dir, s.output_dir,
        pool=s.wtc_pair_pool, n_max=s.wtc_pair_max, desc=s.desc,
        bads_scope=s.bads_scope, scope_tasks=s.scope_tasks, chroma=s.chroma,
        cross=s.wtc_pair_cross, limit_scales=s.wtc_limit_scales,
        roi_map=s.roi_map, roi_map_name=s.roi_name,
        roi_min_channels=s.wtc_roi_min_channels, session=ses)


def cmd_pair_null(
    derivatives_dir: Path,
    output_dir: Path,
    pairs_csv: Path,
    group_id: str | None,
    task_label: list[str] | None,
    desc: str | None,
    roi_mapping: str | None,
    bads_scope: str | None,
    chroma: str,
    wtc_pair_pool: str,
    wtc_pair_max: int | None,
    wtc_pair_cross: bool | None,
    wtc_roi_min_channels: int | None,
    wtc_limit_scales: bool,
    verbose: bool,
    participant_label: list[str] | None = None,
    nprocs: int | None = None,
) -> None:
    """Draw the re-paired null for dyads whose real tables are already on disk."""
    setup_logging(verbose=verbose)
    write_dataset_description(output_dir, name="nirspipe-hyper output",
                              generated_by="nirspipe-hyper", source=derivatives_dir,
                              link=LINK_PREPROCESSED)
    write_bidsignore(output_dir)

    # the pool comes from every group in the table, the targets from the selection: a null
    # drawn only from the dyads the caller happened to name would be a different null
    targets = _select_groups(pairs_csv, group_id, task_label, participant_label)
    all_groups = parse_group_csv(pairs_csv)

    roi_map = _app_options.load_roi_mapping(roi_mapping)

    chroma = ("hbo", "hbr") if chroma == "both" else (chroma,)
    scope_tasks = sorted({key[1] for key in all_groups})

    settings = SimpleNamespace(
        all_groups=all_groups, derivatives_dir=derivatives_dir, output_dir=output_dir,
        wtc_pair_pool=wtc_pair_pool, wtc_pair_max=wtc_pair_max, desc=desc,
        bads_scope=bads_scope, scope_tasks=scope_tasks, chroma=chroma,
        wtc_pair_cross=wtc_pair_cross, wtc_limit_scales=wtc_limit_scales, roi_map=roi_map,
        roi_name=roi_map_name(roi_mapping), wtc_roi_min_channels=wtc_roi_min_channels)
    _run_groups(targets, partial(_pair_null_one_group, settings), nprocs, verbose,
                log_file=lambda key: group_report_dir(output_dir, key[0]) / "logs"
                / f"{group_label(*key)}_null-pair.log", what="pair null")
    # a closing hint must not decide the exit code
    try:
        _merge_reminder(output_dir)
    except Exception as exc:
        logger.debug("merge reminder skipped: %s", exc)

def cmd_merge(output_dir: Path, verbose: bool) -> None:
    """Merge every per-dyad coherence table into one long table per kind."""
    from nirspipe.pipeline.hyper.wtc_aggregate import write_all_aggregates

    setup_logging(verbose=verbose)

    written = write_all_aggregates(output_dir)
    for path in written:
        print(f"{path.name}")
    if not written:
        print(f"no dyad coherence tables under {output_dir}; run `nirspipe-hyper` first")


# `group` is the only level these commands have: a dyad is two subjects, so nothing here
# can run one participant at a time. Spelled out anyway, because the positional is what
# makes the command the shape a BIDS App runner expects.
_LEVELS = ["group"]


def _command_parser(prog: str, description: str, *, reads_subjects: bool,
                    parents: "list[argparse.ArgumentParser]") -> argparse.ArgumentParser:
    """One command's own parser, positionals included.

    ``reads_subjects`` is what separates the two kinds of command here. `run` and
    `pair-null` read each member's own recording, so they take the tree that holds it and
    write to a second one. The rest re-read tables this package already wrote and have no
    subject data to open, so a source tree would be a positional they ignore.
    """
    p = argparse.ArgumentParser(prog=prog, description=description, parents=parents)
    p.add_argument("--version", action="version", version=f"{prog} {__version__}")
    if reads_subjects:
        p.add_argument("derivatives_dir", type=Path,
                       help="BIDS derivatives directory nirspipe wrote, holding each "
                            "member's sub-<id>/nirs/ recordings.")
    p.add_argument("output_dir", type=Path,
                   help="Where the dyad results go. Keep it apart from the source tree so "
                        "each carries its own dataset_description.json.")
    p.add_argument("analysis_level", choices=_LEVELS,
                   help="Always `group`: every metric here needs both members present.")
    return p


def _parsers() -> dict[str, argparse.ArgumentParser]:
    # --wtc-band-fmin/fmax and --wtc-mask-coi mean the same thing to `run` and to `band`, so
    # they are declared once rather than spelled twice
    band_opts = argparse.ArgumentParser(add_help=False)
    band_opts.add_argument("--wtc-band-fmin", type=float, default=None,
                           help="Lower bound (Hz) of the band the per-channel WTC TSV "
                                "averages over. For nirspipe-hyper it defaults to --band-fmin, "
                                "else --wtc-fmin, i.e. the whole computed axis; for "
                                "nirspipe-hyper-band it is the new band and is required.")
    band_opts.add_argument("--wtc-band-fmax", type=float, default=None,
                           help="Upper bound (Hz) of that band. Defaults to --band-fmax, "
                                "else --wtc-fmax, for nirspipe-hyper; required for "
                                "nirspipe-hyper-band.")
    band_opts.add_argument("--wtc-mask-coi", action=argparse.BooleanOptionalAction,
                           default=True,
                           help="Average each band mean only over cells outside the cone "
                                "of influence, the region the edges of the record reach "
                                "(default on). --no-wtc-mask-coi averages the whole band "
                                "instead; the share outside the cone is reported as "
                                "n_valid_frac either way.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verbose", action="store_true")

    # the dyad selection and the alignment window are the same parameters `nirspipe-qc
    # hyper-raw` takes, so they are declared once for both scripts
    pairs = _app_options.pairs_selection()
    window = _app_options.alignment_window()

    run = _command_parser(
        "nirspipe-hyper",
        "Hyperscanning analysis: wavelet coherence, inter-subject correlation and the "
        "phase-scrambled null, over a derivatives tree nirspipe has already written.",
        reads_subjects=True, parents=[common, pairs, window, band_opts])
    run.add_argument("--desc", default="preproc",
                     help="desc entity of the per-subject stage the inter-brain metrics read, "
                          "e.g. 'preproc' (Beer-Lambert output) or 'errts' (confound-regression "
                          "residual, which is what short-channel regression leaves behind). "
                          "Must be a haemoglobin stage, not an optical-density one.")
    run.add_argument("--roi-mapping", type=Path, default=None,
                     help="JSON file mapping ROI labels to lists of channel names, for "
                          "ROI-level WTC. Optional.")
    run.add_argument("--wtc-fmin", type=float, default=0.004,
                     help="Lower bound (Hz) for the WTC frequency axis.")
    run.add_argument("--wtc-fmax", type=float, default=0.20,
                     help="Upper bound (Hz) for the WTC frequency axis.")
    run.add_argument("--wtc-significance", action="store_true",
                     help="Overlay a Monte Carlo significance contour on WTC "
                          "(slow: see --wtc-mc-count for how slow). This asks whether a "
                          "time-frequency cell beats red noise, which is a different question "
                          "from the phase-scrambled null of --wtc-phase-null.")
    run.add_argument("--wtc-mc-count", type=int, default=300,
                     help="Surrogate series behind each --wtc-significance contour "
                          "(default 300). This is what the runtime is spent on and it "
                          "scales linearly; lower it to preview a run, raise it to settle "
                          "a contour. Ignored without --wtc-significance.")
    run.add_argument("--wtc-seed", type=int, default=None,
                     help="Seed the Monte Carlo surrogates behind --wtc-significance and the "
                          "phase randomisation behind --wtc-phase-null. Also bypasses pycwt's "
                          "on-disk cache, which is not keyed on the seed.")
    run.add_argument("--wtc-chroma", choices=("hbo", "hbr", "both"), default="both",
                     help="Chromophore(s) the coherence and the correlation run on (default "
                          "both). HbO and HbR are separate passes, never mixed, so 'both' costs "
                          "twice as much; the tables gain a chromophore column and the report a "
                          "switch.")
    run.add_argument("--wtc-roi-min-channels", type=int, default=ROI_MIN_CHANNELS, metavar="N",
                     help="Drop an ROI cell where either member contributes fewer than N "
                          f"channels (default {ROI_MIN_CHANNELS}). The default keeps a region "
                          "one surviving channel stands in for; the n_ch column says how many "
                          "pairings each value rests on. Each side of a crossed cell is counted "
                          "on its own.")
    run.add_argument("--wtc-arrow-min", type=float, default=0.5, metavar="R",
                     help="Coherence a cell needs before its phase arrow is drawn, when no null "
                          "was computed (default 0.5). Display only; --wtc-phase-null or "
                          "--wtc-significance replace it with a level per frequency.")
    run.add_argument("--channel-cross", dest="wtc_channel_cross",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="Cross every long channel with every channel of the other brain "
                          "(default on), so n channels give n^2 values; the channel TSV gains a "
                          "label2 column. --no-channel-cross pairs each channel with its "
                          "counterpart only, for the ISC as for the coherence.")
    run.add_argument("--wtc-window-s", type=float, default=None, metavar="SECONDS",
                     help="Cut every condition into non-overlapping windows of this length and "
                          "make the window the unit. Needs --by-condition; the remainder, and "
                          "any condition shorter than one window, is dropped.")
    run.add_argument("--by-condition", "--wtc-by-condition", dest="wtc_by_condition",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="Read the coherence out of each task annotation's window as well as "
                          "over the whole recording (default on; --no-by-condition turns it "
                          "off), as rows of stat-wtc_relmat.tsv with a condition column. A "
                          "trigger without a duration runs to the next; windows shorter than "
                          "one cycle of --wtc-fmin are skipped.")
    run.add_argument("--wtc-cond-transform", action="store_true",
                     help="Transform each condition on its own instead of reading it out of "
                          "the whole-run transform, over a cut padded by --wtc-cond-pad-s. "
                          "Costs one extra transform per condition per chromophore. Requires "
                          "--wtc-by-condition; refused with --wtc-phase-null.")
    run.add_argument("--wtc-cond-pad-s", type=str, default="auto", metavar="SEC|auto",
                     help="Seconds kept on each side of a condition under "
                          "--wtc-cond-transform, then windowed back off. 'auto' (default) is "
                          "2*sqrt(2)/--wtc-band-fmin, e.g. 47 s at 0.06 Hz. 0 cuts each "
                          "condition to its own boundaries, which biases a short condition's "
                          "band mean upward.")
    run.add_argument("--wtc-limit-scales", action=argparse.BooleanOptionalAction, default=True,
                     help="Compute only the wavelet scales inside --wtc-fmin/--wtc-fmax "
                          "plus margin, instead of every scale the record length allows "
                          "(default on). The coherences match the unrestricted ones. "
                          "--no-wtc-limit-scales computes every scale.")
    run.add_argument("--wtc-save-maps", action="store_true",
                     help="Save the full time-frequency coherence maps beside each TSV as "
                          "npz, so a different band can be averaged later with "
                          "`nirspipe-hyper-band` instead of a second wavelet transform. Large: one array "
                          "per pair per dyad per task.")
    run.add_argument("--wtc-phase-null", type=int, default=None, metavar="N",
                     help="Also write the phase-scrambled null over N iterations, each as "
                          "costly as a full WTC run; omitted, no null is computed. It follows "
                          "the --by-condition windows, and the maps then draw phase arrows "
                          "against its level instead of --wtc-arrow-min.")
    run.add_argument("--wtc-phase-null-cross", action=argparse.BooleanOptionalAction,
                     default=None,
                     help="Cross the channels for the phase-scrambled null too. Unset, it "
                          "follows --channel-cross. Crossing squares the pair count on "
                          "every iteration. --no-wtc-phase-null-cross draws the null over the "
                          "homologous pairings only, which covers the homologous rows of a "
                          "crossed real table (label equal to label2).")
    run.add_argument("--bads-scope", choices=_BADS_SCOPE_CHOICES, default="run",
                     help="Which rejected channels are excluded: 'run' (default) this task's "
                          "own, 'subject' the union over the subject's runs in the pairs table, "
                          "so all conditions share one channel set. They differ only when "
                          "conditions were cropped to separate tasks before preprocessing.")
    run.add_argument("--isc-threshold", type=float, default=None,
                     help="Absolute |ISC| a pairing must clear for a chord in the connectivity "
                          "circle. By default chords mark pairings above their own null (with "
                          "--isc-phase-null) or the strongest tenth (without). Its scale moves "
                          "with --desc, the passband, whitening and --isc-max-lag.")
    run.add_argument("--no-report", action="store_true",
                     help="Write the tables and skip the HTML report and its figures. "
                          "Everything else is unchanged: the same numbers, the same files, the same npz when "
                          "--wtc-save-maps is given.")
    run.add_argument("--band-fmin", type=float, default=None, metavar="HZ",
                     help="Low edge of the band both metrics read, the coherence averaging "
                          "over it and the correlation filtering to it. --wtc-band-fmin and "
                          "--isc-fmin override it for one metric.")
    run.add_argument("--band-fmax", type=float, default=None, metavar="HZ",
                     help="High edge of that band. --wtc-band-fmax and --isc-fmax override "
                          "it for one metric.")
    run.add_argument("--isc-fmin", type=float, default=None, metavar="HZ",
                     help="Band-limit each member before the correlation, low edge "
                          "(default --band-fmin). Without either, ISC reads whatever the "
                          "preprocessing passband left. The run warns when the ISC band "
                          "differs from the coherence band. Filtering happens on the whole "
                          "recording, before a condition window is cut.")
    run.add_argument("--isc-fmax", type=float, default=None, metavar="HZ",
                     help="The high edge of that band (default --band-fmax).")
    run.add_argument("--wtc-whiten", type=float, default=0.0, metavar="SECONDS",
                     help="Prewhiten each long channel with an autoregressive model of this "
                          "many seconds of order before the wavelet coherence, fitted on the "
                          "whole aligned record; 0, the default, transforms the signals "
                          "themselves. Phase and lag then describe the whitened signals, and "
                          "the nulls follow it.")
    run.add_argument("--isc-whiten-s", type=float, default=0.0, metavar="SECONDS",
                     help="Prewhiten each long channel with an autoregressive model of this "
                          "many seconds of order before the correlation, fitted on the whole "
                          "aligned record; 0, the default, correlates the signals themselves. "
                          "Whitening shrinks r, so whitened and unwhitened matrices do not "
                          "compare; the nulls follow it.")
    run.add_argument("--isc-max-lag", type=float, default=0.0, metavar="SECONDS",
                     help="Correlate at every shift within this many seconds either way and "
                          "keep the strongest (default 0, no search); the shift reaches "
                          "stat-isc_relmat.tsv as lag_s. A maximum over shifts is inflated "
                          "under no coupling, so pair it with --isc-phase-null, which searches "
                          "the same way.")
    run.add_argument("--isc-phase-null", type=int, default=0, metavar="N",
                     help="Also rank each correlation against N phase-scrambled surrogates "
                          "of the second member, which keep its spectrum. Adds null_abs_mean, "
                          "null_abs_sd, null_abs_p95 and percentile to stat-isc_relmat.tsv, "
                          "all of them magnitudes, since a correlation is two-sided. Off by default; "
                          "it costs N extra correlations per chromophore per window.")
    run.add_argument("--check-only", action="store_true",
                     help="Load and align each dyad, print what the metrics would be "
                          "computed on, and stop. Nothing is written. Use it to look over a "
                          "cohort's channel budget before committing to a long run, such as "
                          "one with --wtc-phase-null.")
    run.add_argument("--config", type=Path, default=None,
                     help="A dyad's run record (logs/group-<id>_task-<task>.toml) or a file "
                          "with its [hyper] section. Fills every setting the command line "
                          "leaves out; the directories and --pairs-csv still come from the "
                          "command line.")
    _shared.add_nprocs(run, unit="group")
    _app_options.add_separation_bands(run, note="An override: the bands come from each member's "
                                 "record, and members prepped with different bands are "
                                 "refused. Use it only for records that carry none.")

    band = _command_parser(
        "nirspipe-hyper-band", reads_subjects=False, parents=[common, band_opts],
        description="Re-averages the maps a `nirspipe-hyper --wtc-save-maps` saved, so no wavelet "
                    "transform runs a second time. Writes tables, not a report. "
                    "--wtc-band-fmin and --wtc-band-fmax are both required here.")
    band.add_argument("--wtc-suffix", default=None,
                      help="Value of the band- entity on each output TSV, letters and "
                           "digits only. Defaults to the band, e.g. 0p05to0p2 for "
                           "band-0p05to0p2, so the new tables sit beside the originals "
                           "rather than replacing them.")

    group_null = _command_parser(
        "nirspipe-hyper-groupnull", reads_subjects=False, parents=[common],
        description="Averages each dyad's channels first and ranks that mean inside a null's "
                    "draws, once per occasion and once over the cohort. It says whether the "
                    "pairing beats its null, not which channel does; nirspipe-hyper-pairnull "
                    "ranks each channel inside its own draws, where against n stand-ins no "
                    "cell can reach a p under 1/(n+1). It reads draws already on disk and "
                    "runs no transform.")
    # one value, unlike the other commands' --task-label: the cohort test reads one task
    group_null.add_argument("--task-label", "--task_label", "--task", dest="task",
                            required=True, type=_shared.BidsLabel,
                            help="Task whose tables to read, one at a time.")
    group_null.add_argument("--chroma", nargs="+", choices=("hbo", "hbr"), default=None,
                            help="Chromophores to read, each into its own tables (default "
                                 "every chromophore the null's draws hold). One the draws do "
                                 "not hold is refused.")
    group_null.add_argument("--null", nargs="+", choices=("repaired", "phase"), default=None,
                            help="Which nulls' draws to read, each into its own tables "
                                 "(default every null whose draws are on disk for the task). "
                                 "Both are read the same way above the cell; they differ in "
                                 "what a draw is, a stand-in against a scrambled partner.")
    group_null.add_argument("--roi-mapping", type=Path, default=None,
                            help="Region to channel map, as in the other stages. Given it, each "
                                 "region is a level of its own beside the whole brain (its "
                                 "homologous pairings averaged), and with a crossed null so is "
                                 "every ordered region pair. A region needs the real tables' "
                                 "--wtc-roi-min-channels channels per member.")
    group_null.add_argument("--n-resample", type=int, default=20000,
                            help="Resamples behind the cohort null (default 20000). Each "
                                 "picks one stand-in per occasion, so the finest p it can "
                                 "express is 1/(n+1).")
    group_null.add_argument("--seed", type=int, default=None,
                            help="Seed the resampling, so the cohort p is reproducible.")
    group_null.add_argument("--p-correction", default="none",
                            choices=P_CORRECTIONS,
                            help="Multiple-comparison correction, within one condition at one "
                                 "level (default none). The uncorrected p column is always "
                                 "written; a method adds a p_<method> column beside it, and "
                                 "the family column says how many tests it ran over.")
    group_null.add_argument("--isc-test", default="signed", choices=ISC_TESTS,
                            help="How the correlation is tested (default signed): 'signed' "
                                 "averages Fisher z and tests either direction; 'magnitude' "
                                 "averages |Fisher z| and tests it above the null, so opposite "
                                 "signs do not cancel. Coherence is always tested above its "
                                 "null.")

    index = _command_parser(
        "nirspipe-hyper-index", reads_subjects=False, parents=[common],
        description="Writes group-<id>_desc-index_report.html, one row per analysed window, linking to "
                    "that window's report. nirspipe-hyper writes it too; this rebuilds it for a tree "
                    "produced earlier, or after the pages were regenerated by hand. It "
                    "reads the tables and nothing else, so it serves both orders the "
                    "pipeline can be driven in: a recording preprocessed whole lists its "
                    "conditions as windows of one task, and a tree cropped per condition "
                    "first lists them as separate tasks.")
    index.add_argument("--group-id", default=None,
                       help="Only this dyad. Every group-* directory by default.")

    pair = _command_parser(
        "nirspipe-hyper-pairnull", reads_subjects=True, parents=[common, pairs],
        description="Recomputes the coherence of one member against people they never "
                    "interacted with, drawn from the other groups of the same task, and "
                    "writes the null-pair tables beside the real ones. "
                    "Unlike --wtc-phase-null, a re-paired partner did the same task, so the "
                    "null keeps the shared task response. Needs a cohort: the number of "
                    "draws is the number of other groups, which is what limits how finely "
                    "the percentile can rank. Run it after `nirspipe-hyper`, whose tables "
                    "it reads its band, its mask, its frequency range and its window off, "
                    "and by default its stage, its rejection scope, its crossing and its "
                    "ROI minimum.")
    pair.add_argument("--desc", default=None,
                      help="desc entity of the per-subject stage the null reads. Unset, it "
                           "is the one the real tables recorded; a different one is refused.")
    pair.add_argument("--roi-mapping", type=Path, default=None,
                      help="JSON file mapping ROI labels to channel names, to also write the "
                           "null of the homologous ROI means and, when the null and the real "
                           "tables are both crossed, of the ROI x ROI matrix. Optional.")
    pair.add_argument("--bads-scope", choices=_BADS_SCOPE_CHOICES, default=None,
                      help="Which rejected channels are excluded, as in nirspipe-hyper. Unset, "
                           "it is the scope the real tables recorded; a different one is "
                           "refused. A stand-in with no quality record is refused rather than "
                           "kept whole.")
    pair.add_argument("--chroma", choices=("hbo", "hbr", "both"), default="both",
                      help="Chromophore(s) to draw the null on, coherence and correlation alike (default both).")
    pair.add_argument("--wtc-pair-pool", choices=("position", "any"), default="position",
                      help="Who may stand in: 'position' (default) only another group's member "
                           "at the same index, which keeps roles apart and is safe when one "
                           "person is in several groups; 'any' every other group's member, "
                           "refused when anyone appears twice.")
    pair.add_argument("--wtc-pair-max", type=int, default=None, metavar="N",
                      help="Stop after N draws. The pool is finite, so this is a ceiling "
                           "rather than a count: without it every eligible stand-in is "
                           "used, which is what gives the percentile its best resolution.")
    pair.add_argument("--wtc-pair-cross", action=argparse.BooleanOptionalAction, default=None,
                      help="Draw the null over every channel pair rather than homologous "
                           "ones only. Unset, it follows the real table: crossed when that "
                           "table is. A crossed null costs one full run per channel pair, so "
                           "--no-wtc-pair-cross over a crossed table draws the homologous "
                           "pairings only and ranks that table's diagonal.")
    pair.add_argument("--wtc-roi-min-channels", type=int, default=None, metavar="N",
                      help="Drop an ROI cell where either member contributes fewer than N "
                           "channels. Unset, it is the value the real tables recorded; a "
                           "different one is refused.")
    pair.add_argument("--wtc-limit-scales", action=argparse.BooleanOptionalAction, default=True,
                      help="Compute only the scales inside the frequency range plus margin "
                           "(default on), as in nirspipe-hyper.")
    _shared.add_nprocs(pair, unit="group")

    merge = _command_parser(
        "nirspipe-hyper-merge", reads_subjects=False, parents=[common],
        description="Concatenates every group-*_task-*_*_relmat.tsv under the tree into "
                    "one table per kind at its root, adding group_id and task columns, so a "
                    "cohort analysis reads one file. Refuses to merge tables that disagree "
                    "on the band, on mask_coi, on which null they are or on the stand-in "
                    "pool, crossed with homologous tables, and matrices over different "
                    "channels. A differing null iteration count only warns: n_iter is kept "
                    "per row.")

    return {p.prog: p for p in (run, band, group_null, index, pair, merge)}


# each console script and the function it hands off to. One table, so a command cannot be
# registered in pyproject.toml without something here saying what it runs
COMMANDS = {
    "nirspipe-hyper":           cmd_run,
    "nirspipe-hyper-pairnull":  cmd_pair_null,
    "nirspipe-hyper-groupnull": cmd_group_null,
    "nirspipe-hyper-band":      cmd_band,
    "nirspipe-hyper-index":     cmd_index,
    "nirspipe-hyper-merge":     cmd_merge,
}


# [hyper] keys that describe the record's own run: inputs come from the command line, and the
# separation bands (sep_*) are resolved from the members again
_RECORD_ONLY = ("derivatives_dir", "pairs_csv", "check_only", "no_report")
# a value the run resolved from another key, dropped when equal to it so that a source typed
# over the record reaches it, as on a fresh command line
_DERIVED_FROM = {"wtc_band_fmin": "band_fmin", "wtc_band_fmax": "band_fmax",
                 "isc_fmin": "band_fmin", "isc_fmax": "band_fmax",
                 "wtc_phase_null_cross": "wtc_channel_cross"}


def read_hyper_config(path: Path) -> dict:
    """--config as the [hyper] settings to fill in, plus the version that wrote it.

    ::

      [hyper] wtc_fmin = 0.02, band_fmin = 0.03, wtc_band_fmin = 0.03, pairs_csv = "..."
      -> {"version": "0.68.0", "hyper": {"wtc_fmin": 0.02, "band_fmin": 0.03}}

    A record's [environment] and [execution] describe its own run and are skipped. An
    unknown key, an unknown section or a key outside any section is refused, since a
    misspelt setting would otherwise drop out without a word.
    """
    from nirspipe.cli.run import _coerce
    from nirspipe.utils import load_toml

    raw = load_toml(path)
    stray = [k for k, v in raw.items() if not isinstance(v, dict)]
    if stray:
        print(f"Error: {path}: {stray[0]} sits outside any section; --config takes a [hyper] "
              "section keyed by flag name, as a run record does.", file=sys.stderr)
        raise SystemExit(1)
    unknown = sorted(set(raw) - {"hyper", "environment", "execution"})
    if unknown:
        print(f"Error: {path}: no section [{unknown[0]}]; expected [hyper].", file=sys.stderr)
        raise SystemExit(1)
    actions = {a.dest: a for a in _parsers()["nirspipe-hyper"]._actions if a.option_strings}
    allowed = set(actions) - {"help", "version", "config", "verbose"}
    settings = raw.get("hyper", {})
    bad = [k for k in settings
           if k not in allowed and k not in _RECORD_ONLY and not k.startswith("sep_")]
    if bad:
        print(f"Error: {path}: [hyper] has no setting {bad[0]!r}.", file=sys.stderr)
        raise SystemExit(1)
    taken = {k: _coerce(actions[k], k, v, path) for k, v in settings.items()
             if k in allowed and k not in _RECORD_ONLY}
    for key, source in _DERIVED_FROM.items():
        if key in taken and source in taken and taken[key] == taken[source]:
            del taken[key]
    return {"version": raw.get("environment", {}).get("nirspipe_version"), "hyper": taken}


def _apply_config(args: argparse.Namespace, argv: "list[str] | None") -> None:
    """Fill what the command line left out from --config."""
    config = read_hyper_config(args.config)
    if config["version"] and config["version"] != __version__:
        print(f"Warning: {args.config} was written by nirspipe {config['version']} and this is "
              f"{__version__}; a setting whose meaning or default changed in between runs as "
              "this version reads it.", file=sys.stderr)
    # the flags actually typed, which argparse's own defaults would otherwise hide
    quiet = _parsers()["nirspipe-hyper"]
    for action in quiet._actions:
        action.default = argparse.SUPPRESS
    given = set(vars(quiet.parse_args(argv)))
    for key, value in config["hyper"].items():
        if key not in given:
            setattr(args, key, value)


def _dispatch(prog: str, func, argv, *, require: "tuple[str, ...]" = ()) -> None:
    """Parse one command's own argv and hand the rest to its implementation.

    ``analysis_level`` is dropped: it is there to make the command the shape a BIDS App
    runner expects, and `group` is its only value, so nothing downstream reads it.
    """
    args = _parsers()[prog].parse_args(argv)
    if getattr(args, "config", None) is not None:
        _apply_config(args, argv)

    for dest in require:
        if getattr(args, dest) is None:
            print(f"Error: Missing option '--{dest.replace('_', '-')}'.", file=sys.stderr)
            raise SystemExit(1)

    kw = {k: v for k, v in vars(args).items() if k != "analysis_level"}
    if kw.get("derivatives_dir") is not None:
        _shared.refuse_output_is_source(kw["derivatives_dir"], kw["output_dir"])
    func(**kw)


def main(argv: list[str] | None = None) -> None:
    _dispatch("nirspipe-hyper", COMMANDS["nirspipe-hyper"], argv)


def main_pair_null(argv: list[str] | None = None) -> None:
    _dispatch("nirspipe-hyper-pairnull", COMMANDS["nirspipe-hyper-pairnull"], argv)


def main_group_null(argv: list[str] | None = None) -> None:
    _dispatch("nirspipe-hyper-groupnull", COMMANDS["nirspipe-hyper-groupnull"], argv)


def main_index(argv: list[str] | None = None) -> None:
    _dispatch("nirspipe-hyper-index", COMMANDS["nirspipe-hyper-index"], argv)


def main_merge(argv: list[str] | None = None) -> None:
    _dispatch("nirspipe-hyper-merge", COMMANDS["nirspipe-hyper-merge"], argv)


def main_band(argv: list[str] | None = None) -> None:
    # the band is the whole point of this one, so it is required here and optional on `run`
    _dispatch("nirspipe-hyper-band", COMMANDS["nirspipe-hyper-band"], argv,
              require=("wtc_band_fmin", "wtc_band_fmax"))
