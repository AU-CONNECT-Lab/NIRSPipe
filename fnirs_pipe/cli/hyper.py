"""fnirs-hyper CLI (argparse) — dyad analysis over a derivatives tree.

Hyperscanning is its own domain: its input is a pairs table, its unit is a dyad, and it
reads derivatives rather than BIDS raw. It was `fnirs-qc hyper-post` until 0.26.0, which
put a wavelet-coherence analysis inside the quality-control tool.

Every command here takes one derivatives directory and nothing else positional.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fnirs_pipe.cli import _shared
from fnirs_pipe.pipeline.synchrony import ISC_MAX_AR_ORDER
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.utils.logging import get_logger, setup_logging

setup_logging()

logger = get_logger("cli.hyper")

_BADS_SCOPE_CHOICES = ["run", "subject"]


def _select_groups(pairs_csv: Path, group_id: str | None, task_label: list[str] | None) -> dict:
    """Parse the group CSV and filter by group_id / task_label. Exits non-zero on empty selection."""
    from fnirs_pipe.exceptions import GroupCSVError
    from fnirs_pipe.pipeline.hyperscanning import parse_group_csv

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

    return groups


def _run_groups(groups: dict, process) -> None:
    """Run process(gid, task, members) -> report_path per group, tally ok/fail, exit non-zero on failure."""
    from fnirs_pipe.exceptions import AlignmentError, MissingDerivativesError

    print(f"Processing {len(groups)} group session(s)...")
    n_ok = n_fail = 0
    for (gid, task), members in groups.items():
        print(f"  -> {gid}/{task} ({len(members)} subjects)")
        try:
            report_path = process(gid, task, members)
            if report_path is not None:
                print(f"     report -> {report_path}")
            n_ok += 1
        except MissingDerivativesError as exc:
            print(f"     [skip] {exc}", file=sys.stderr)
            n_fail += 1
        except AlignmentError as exc:
            print(f"     [skip] alignment failed: {exc}", file=sys.stderr)
            n_fail += 1
        except Exception as exc:
            logger.exception("group %s task %s failed", gid, task)
            print(f"     [error] unexpected error: {exc}", file=sys.stderr)
            n_fail += 1

    print(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    if n_fail > 0:
        raise SystemExit(1)


def _load_aligned_group(output_dir, members, task, desc, no_align, normalize, bads_scope,
                        passband_check=None, scope_tasks=None):
    """Load one dyad, put both recordings on one time axis, and mark the rejected channels.

    Returns (aligned_raws, offsets, group_sqm). The rejections are applied here rather than
    in each metric because --bads-scope decides them: a metric reading the Raw alone gets
    whatever that one file's sidecar recorded, which is the run's own rejections and not
    the union over the subject's runs that `subject` scope asks for.

    --tstart/--tend are not applied here. They name a window of the analysis, and the report
    takes it out of the transform of the whole recording rather than cutting the recording
    to it; see `resolve_analysis_window`.

    `passband_check` is the (fmin, fmax) a metric is about to ask for, checked against the
    bandpass the files record while the sidecars are still in hand.
    """
    from fnirs_pipe.pipeline.hyperscanning import (
        align_recordings,
        apply_group_bads,
        load_group_haemo,
        load_group_sqm,
        warn_outside_passband,
        normalize_raws,
        trim_to_shortest,
    )

    raws = load_group_haemo(output_dir, members, desc=desc)
    if passband_check is not None:
        warn_outside_passband(raws, *passband_check)
    group_sqm = load_group_sqm(output_dir, members, bads_scope=bads_scope,
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

    ``sub-01  long 12/14  bad 2  mean SCI 0.86  from: tapping``

    Counted off the aligned Raw after the rejections are applied, so it describes the channel
    set the coherence actually uses rather than what the montage holds. The report says the
    same thing, but only once the run has finished, which with --wtc-phase-null is hours later.
    """
    from fnirs_pipe.io.snirf import long_channel_picks

    for subject_id, raw in aligned_raws.items():
        sqm = group_sqm.get(subject_id, {})
        pairs = {ch.rsplit(" ", 1)[0] for ch in raw.ch_names
                 if ch.endswith(" hbo") or ch.endswith(" hbr")}
        bad = {ch.rsplit(" ", 1)[0] for ch in raw.info["bads"]}
        kept = len(long_channel_picks(raw, "hbo", sep_bands=sep_bands))  # bads already dropped

        scores = [v for v in (sqm.get("sci_per_channel") or {}).values()
                  if v is not None and v == v]
        sci = f"mean SCI {sum(scores) / len(scores):.2f}" if scores else "mean SCI n/a"

        tasks = sorted({t for ts in (sqm.get("bad_channel_sources") or {}).values()
                        for t in ts})
        origin = f"  from: {', '.join(tasks)}" if tasks else ""

        print(f"     {subject_id}  long {kept}/{len(pairs)}  bad {len(bad)}  {sci}{origin}")
        if pairs and len(bad) > len(pairs) / 2:
            print(f"     [warn] {subject_id} loses {len(bad)} of {len(pairs)} channel pairs",
                  file=sys.stderr)


def _merge_reminder(output_dir: Path) -> None:
    """Say so when the merged tables are missing or older than the per-dyad ones.

    A stale merged table is worse than none: it reads like a finished result. Merging is not
    done here because a run often covers one dyad, and merging the whole tree after it would
    fail on bands that a later run legitimately changed, and would race a parallel run for
    the same three files.

    Driven off the aggregator's own kinds and its own glob, so the counts are the ones
    `merge` would use and a new kind cannot be left out.
    """
    from fnirs_pipe.pipeline.wtc_aggregate import _KINDS

    lines = []
    for kind, stem in _KINDS.items():
        parts = list(output_dir.rglob(f"*_hyper-{kind}.tsv"))
        if not parts:
            continue
        merged = output_dir / f"{stem}.tsv"
        if not merged.exists():
            lines.append(f"  {len(parts)} {kind} table(s) on disk, never merged")
            continue
        stale = sum(p.stat().st_mtime > merged.stat().st_mtime for p in parts)
        if stale:
            lines.append(f"  {len(parts)} {kind} table(s) on disk, {stale} newer than {merged.name}")

    if lines:
        print("\n".join(["", *lines, f"Run `fnirs-hyper merge {output_dir}` for one table per kind."]))


def _warn_band_mismatch(isc_band, wtc_band_fmin, wtc_band_fmax) -> None:
    """Say so when ISC and the coherence are about to describe different frequencies.

    The two are averages of one complex coherency, so comparing them across dyads is only
    meaningful on one band: zero-lag Pearson r is the power-weighted mean of
    ``|gamma| cos phi`` and the WTC band mean is the unweighted mean of ``|gamma|^2``. Under
    a 1/f spectrum an unbanded ISC puts most of its weight below a coherence band that
    starts at 0.06 Hz, so the two can be uncorrelated with nothing wrong in either.

    Reported rather than enforced: a run may want them apart, and a run that computes no ISC
    at all should not be made to name a band for it.
    """
    wtc_band = (wtc_band_fmin, wtc_band_fmax)
    if wtc_band == (None, None):
        return
    if isc_band is None:
        logger.warning(
            "ISC reads the whole passband while the coherence band is %s-%s Hz, so the two "
            "describe different frequencies and comparing them is not meaningful. Pass "
            "--isc-fmin/--isc-fmax to put them on one band.",
            wtc_band_fmin, wtc_band_fmax)
    elif isc_band != wtc_band:
        logger.warning(
            "ISC band %s-%s Hz against coherence band %s-%s Hz: deliberate if you meant it, "
            "but the two are then not comparable.",
            isc_band[0], isc_band[1], wtc_band_fmin, wtc_band_fmax)


def cmd_run(
    output_dir: Path, pairs_csv: Path, group_id: str | None, task_label: list[str] | None,
    desc: str, roi_mapping: Path | None,
    wtc_fmin: float, wtc_fmax: float,
    wtc_band_fmin: float | None, wtc_band_fmax: float | None,
    wtc_significance: bool, wtc_mc_count: int, wtc_seed: int | None,
    wtc_mask_coi: bool, wtc_roi_min_channels: int, wtc_arrow_min: float,
    wtc_channel_cross: bool,
    wtc_by_condition: bool, wtc_chroma: str,
    wtc_cond_transform: bool, wtc_cond_pad_s: "float | None",
    wtc_limit_scales: bool, wtc_save_maps: bool,
    wtc_phase_null: int | None, wtc_phase_null_cross: bool,
    bads_scope: str, isc_threshold: "float | None", isc_whiten: int,
    isc_max_lag: float, isc_phase_null: int,
    isc_fmin: "float | None", isc_fmax: "float | None",
    no_report: bool,
    sci_threshold: float,
    normalize: bool, no_align: bool, tstart: float | None, tend: float | None,
    short_max_dist: float | None, long_min_dist: float | None,
    long_max_dist: float | None,
    check_only: bool, verbose: bool,
) -> None:
    """Dyad WTC + ISC report per group, plus the phase-scrambled null when --wtc-phase-null is given.

    The null reuses this run's aligned recordings and every band parameter, so it cannot be
    computed over a different band than the table it sits beside.
    """
    # every parameter as resolved, for the run record. Read off locals() before anything
    # else runs, so a new option lands in the record without being listed here as well.
    run_args = dict(locals())

    from fnirs_pipe.cli._shared import separation_bands_from_args

    # Each metric gets the band it was given, never the other's: a flag that silently moves
    # a second metric cannot be read off the command line it is absent from, and a reader of
    # a methods section has no way to recover it. What the two bands are is checked instead.
    isc_band = (isc_fmin, isc_fmax) if (isc_fmin is not None or isc_fmax is not None) else None
    _warn_band_mismatch(isc_band, wtc_band_fmin, wtc_band_fmax)

    # validated here so a bad triple fails before any dyad is loaded; which of the three the
    # caller actually named is what resolve_group_bands needs, so the dict is what is kept
    bands_override = separation_bands_from_args({
        "short_max_dist": short_max_dist, "long_min_dist": long_min_dist,
        "long_max_dist": long_max_dist,
    })

    import json
    from datetime import datetime

    from fnirs_pipe.io.derivatives import group_report_dir
    from fnirs_pipe.pipeline.hyperscanning import (
        resolve_analysis_window, resolve_group_bands, write_group_bads,
    )
    from fnirs_pipe.qc.hyper.hyper_report import build_hyper_post_report
    from fnirs_pipe.qc.common.windows import condition_windows
    from fnirs_pipe.qc.metrics._helpers import bands_to_record
    from fnirs_pipe.pipeline.wtc_null import run_wtc_null, write_wtc_null
    from fnirs_pipe.utils.run_record import write_group_run_record

    setup_logging(verbose=verbose)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if wtc_channel_cross and roi_mapping is None:
        print("[warn] --wtc-channel-cross without --roi-mapping: the crossed channel table "
              "is written but no ROI x ROI matrix is built from it.", file=sys.stderr)

    chroma = ("hbo", "hbr") if wtc_chroma == "both" else (wtc_chroma,)
    if len(chroma) > 1:
        print("[info] --wtc-chroma both: two full WTC passes per dyad, so roughly twice "
              "the runtime. Pass hbo or hbr for one.", file=sys.stderr)

    groups = _select_groups(pairs_csv, group_id, task_label)

    roi_map: dict[str, list[str]] | None = None
    if roi_mapping is not None:
        try:
            roi_map = json.loads(Path(roi_mapping).read_text())
        except Exception as exc:
            print(f"[error] failed to load ROI mapping: {exc}", file=sys.stderr)
            raise SystemExit(1)

    scope_tasks = sorted({key[1] for key in groups})

    # ---- how each condition is read: windowed out of the run, or transformed on its own ----
    # Resolved once, not per dyad: it depends only on the band, and a run whose conditions
    # were read two different ways would put incomparable rows in one table.
    cond_pad = None
    if wtc_cond_transform:
        if not wtc_by_condition:
            print("[error] --wtc-cond-transform needs --wtc-by-condition; there are no "
                  "conditions to transform without it.", file=sys.stderr)
            raise SystemExit(1)
        if str(wtc_cond_pad_s).lower() == "auto":
            from fnirs_pipe.pipeline.synchrony import cone_margin_s
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

    def _process(gid, task, members):
        aligned_raws, offsets, group_sqm = _load_aligned_group(
            output_dir, members, task, desc, no_align, normalize, bads_scope,
            passband_check=(wtc_fmin, wtc_fmax), scope_tasks=scope_tasks)
        analysis_window = resolve_analysis_window(aligned_raws, tstart, tend)
        sep_bands = resolve_group_bands(members, group_sqm, bands_override)
        _quality_summary(aligned_raws, group_sqm, sep_bands)
        if check_only:
            return None
        bad_channels = {sid: sqm.get("bad_channels", []) for sid, sqm in group_sqm.items()}
        write_group_bads(output_dir, members, group_sqm, bads_scope)
        # resolved here rather than twice downstream: the report and the null have to agree
        # on the windows or their tables cannot be subtracted row by row
        ref = next(iter(aligned_raws.values()), None)
        cond_windows = (condition_windows(ref, min_duration=1.0 / wtc_fmin)
                        if wtc_by_condition and ref is not None else [])
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
        # Before the report, not after: the level the phase arrows are drawn against comes
        # out of the surrogates, and the figures are built inside the report. The table this
        # returns is written after it instead, its percentile column being a rank against
        # the real band means the same report writes.
        nulls = run_wtc_null(
            group_id=gid,
            task=task,
            aligned_raws=aligned_raws,
            output_dir=output_dir,
            n_iter=wtc_phase_null,
            wtc_fmin=wtc_fmin,
            wtc_fmax=wtc_fmax,
            band_fmin=wtc_band_fmin,
            band_fmax=wtc_band_fmax,
            seed=wtc_seed,
            cross=wtc_phase_null_cross,
            limit_scales=wtc_limit_scales,
            mask_coi=wtc_mask_coi,
            chroma=chroma,
            sep_bands=sep_bands,
            windows=cond_windows,
            analysis_window=analysis_window,
        ) if wtc_phase_null else None

        report_path = build_hyper_post_report(
            group_id=gid,
            task=task,
            group=members,
            aligned_raws=aligned_raws,
            offsets=offsets,
            output_dir=output_dir,
            roi_map=roi_map,
            bad_channels=bad_channels,
            subject_sqm=group_sqm,
            wtc_fmin=wtc_fmin,
            wtc_fmax=wtc_fmax,
            wtc_band_fmin=wtc_band_fmin,
            wtc_band_fmax=wtc_band_fmax,
            wtc_significance=wtc_significance,
            wtc_seed=wtc_seed,
            wtc_mc_count=wtc_mc_count,
            wtc_channel_cross=wtc_channel_cross,
            wtc_by_condition=wtc_by_condition,
            wtc_cond_pad_s=cond_pad,
            cond_windows=cond_windows,
            wtc_limit_scales=wtc_limit_scales,
            wtc_save_maps=wtc_save_maps,
            wtc_mask_coi=wtc_mask_coi,
            wtc_roi_min_channels=wtc_roi_min_channels,
            wtc_arrow_min=wtc_arrow_min,
            wtc_chroma=chroma,
            isc_threshold=isc_threshold,
            isc_whiten=isc_whiten,
            isc_max_lag_s=isc_max_lag,
            isc_phase_null=isc_phase_null,
            isc_band=isc_band,
            no_report=no_report,
            sci_threshold=sci_threshold,
            sep_bands=sep_bands,
            analysis_window=analysis_window,
        )
        if nulls:
            null_path = write_wtc_null(
                nulls,
                group_id=gid,
                task=task,
                aligned_raws=aligned_raws,
                output_dir=output_dir,
                n_iter=wtc_phase_null,
                wtc_fmin=wtc_fmin,
                wtc_fmax=wtc_fmax,
                band_fmin=wtc_band_fmin,
                band_fmax=wtc_band_fmax,
                seed=wtc_seed,
                cross=wtc_phase_null_cross,
                mask_coi=wtc_mask_coi,
                windows=cond_windows,
                analysis_window=analysis_window,
                roi_map=roi_map,
            )
            print(f"     null   -> {null_path}")

        try:
            # the resolved bands under the keys the SQM record stamps them with, so the two
            # can be compared directly; run_args holds only the flags as typed. An absent
            # upper bound drops out, TOML having no null and the writer skipping None
            # throughout, so absent here means off rather than unrecorded
            record = write_group_run_record(
                {**run_args, **bands_to_record(sep_bands)},
                gid, task, timestamp, output_dir,
                group_report_dir(output_dir, gid),
                members=[e.subject_id for e in members],
            )
            logger.info("group-%s | run record -> %s", gid, record)
        except Exception:
            logger.warning("group-%s | run record failed", gid, exc_info=True)

        return report_path

    if wtc_phase_null:
        logger.info("phase-scrambled null requested: %d full WTC runs per dyad, on top of the real one",
                    wtc_phase_null)
    _run_groups(groups, _process)
    if check_only:
        return

    # The index is rebuilt for every dyad this run touched, reading the tables rather than
    # anything held in memory, so a dyad whose other tasks were analysed in an earlier run
    # still lists them. It cannot be built inside `_process`: a dyad with several tasks
    # would then have its index rewritten once per task, each time from a tree missing the
    # tasks still to come.
    from fnirs_pipe.qc.hyper.hyper_index import write_hyper_index

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

    _merge_reminder(output_dir)


def cmd_band(
    output_dir: Path, wtc_band_fmin: float, wtc_band_fmax: float,
    wtc_mask_coi: bool, wtc_suffix: str | None, verbose: bool,
) -> None:
    """Re-average every saved WTC map over a new band, without recomputing the transform."""
    from fnirs_pipe.pipeline.wtc_store import reband_tree

    setup_logging(verbose=verbose)

    written = reband_tree(output_dir, wtc_band_fmin, wtc_band_fmax,
                          suffix=wtc_suffix, mask_coi=wtc_mask_coi)
    for path in written:
        print(f"reband -> {path}")
    if not written:
        print(f"no *_hyper-wtc*.npz under {output_dir}; rerun `fnirs-hyper run "
              "--wtc-save-maps` to write them", file=sys.stderr)


def cmd_group_null(
    output_dir: Path, task: str, wtc_chroma: str, n_resample: int,
    seed: int | None, verbose: bool,
) -> None:
    """Read the re-paired draws above the cell: one verdict per occasion, one per cohort."""
    from fnirs_pipe.pipeline.pair_null_group import write_group_null

    setup_logging(verbose=verbose)
    for path in write_group_null(output_dir, task=task, chroma=wtc_chroma,
                                 n_resample=n_resample, seed=seed):
        print(f"group-null -> {path}")


def cmd_index(output_dir: Path, group_id: str | None, verbose: bool) -> None:
    """Write one dyad index per group-* directory, from the tables already on disk."""
    from fnirs_pipe.qc.hyper.hyper_index import write_hyper_index

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
        print(f"no coherence tables under {output_dir}; run `fnirs-hyper run` first")


def cmd_pair_null(
    output_dir: Path,
    pairs_csv: Path,
    group_id: str | None,
    task_label: list[str] | None,
    desc: str,
    roi_mapping: str | None,
    bads_scope: str,
    wtc_chroma: str,
    wtc_pair_pool: str,
    wtc_pair_max: int | None,
    wtc_pair_cross: bool,
    wtc_roi_min_channels: int,
    wtc_limit_scales: bool,
    verbose: bool,
) -> None:
    """Draw the re-paired null for dyads whose real tables are already on disk."""
    import json

    from fnirs_pipe.pipeline.hyperscanning import parse_group_csv
    from fnirs_pipe.pipeline.pair_null import run_pair_null

    setup_logging(verbose=verbose)

    # the pool comes from every group in the table, the targets from the selection: a null
    # drawn only from the dyads the caller happened to name would be a different null
    targets = _select_groups(pairs_csv, group_id, task_label)
    all_groups = parse_group_csv(pairs_csv)

    roi_map = None
    if roi_mapping:
        try:
            roi_map = json.loads(Path(roi_mapping).read_text())
        except Exception as exc:
            print(f"[error] failed to load ROI mapping: {exc}", file=sys.stderr)
            raise SystemExit(1)

    chroma = ("hbo", "hbr") if wtc_chroma == "both" else (wtc_chroma,)
    scope_tasks = sorted({key[1] for key in all_groups})

    failures = 0
    for (gid, task), members in targets.items():
        # flushed, or it interleaves with the stderr line naming the failure it belongs to
        print(f"  -> {gid}/{task}", flush=True)
        try:
            path = run_pair_null(
                gid, task, members, all_groups, output_dir,
                pool=wtc_pair_pool, n_max=wtc_pair_max, desc=desc,
                bads_scope=bads_scope, scope_tasks=scope_tasks, chroma=chroma,
                cross=wtc_pair_cross, limit_scales=wtc_limit_scales,
                roi_map=roi_map, roi_min_channels=wtc_roi_min_channels)
            print(f"     pair null -> {path}")
        except Exception as exc:
            print(f"     [error] {exc}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"\n{failures} group(s) failed", file=sys.stderr)
        raise SystemExit(1)
    _merge_reminder(output_dir)

def cmd_merge(output_dir: Path, verbose: bool) -> None:
    """Merge every per-dyad WTC band-mean table into one long table per kind."""
    from fnirs_pipe.pipeline.wtc_aggregate import _KINDS, write_aggregate_wtc

    setup_logging(verbose=verbose)

    # driven off the aggregator's own kinds, so removing or adding one cannot leave this
    # list behind
    wrote = False
    for kind in _KINDS:
        path = write_aggregate_wtc(output_dir, kind=kind)
        if path is not None:
            print(f"{kind} -> {path}")
            wrote = True
    if not wrote:
        print(f"no hyper-wtc tables under {output_dir}; run `fnirs-hyper run` first")


def _build_parser() -> argparse.ArgumentParser:
    from fnirs_pipe import __version__

    # --wtc-band-fmin/fmax and --wtc-mask-coi mean the same thing to `run` and to `band`, so
    # they are declared once rather than spelled twice
    band_opts = argparse.ArgumentParser(add_help=False)
    band_opts.add_argument("--wtc-band-fmin", type=float, default=None,
                           help="Lower bound (Hz) of the band the per-channel WTC TSV "
                                "averages over. For `run` it defaults to --wtc-fmin, i.e. "
                                "the whole computed axis; for `band` it is the new band and "
                                "is required.")
    band_opts.add_argument("--wtc-band-fmax", type=float, default=None,
                           help="Upper bound (Hz) of that band. Defaults to --wtc-fmax for "
                                "`run`; required for `band`.")
    band_opts.add_argument("--wtc-mask-coi", action=argparse.BooleanOptionalAction,
                           default=True,
                           help="Average each band mean only over cells inside the cone of "
                                "influence. On by default: cells outside it are wavelet "
                                "coefficients padded against the edges of the record, near "
                                "1 whatever the data did. --no-wtc-mask-coi averages the "
                                "whole band instead; the share inside the cone is reported "
                                "as n_valid_frac either way.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("output_dir", type=Path,
                        help="fnirs-pipe derivatives directory. No BIDS input is read.")
    common.add_argument("--verbose", action="store_true")

    # the dyad selection and the alignment window are the same parameters `fnirs-qc
    # hyper-raw` takes, so they are declared once for both scripts
    pairs = _shared.pairs_selection()
    window = _shared.alignment_window()

    p = argparse.ArgumentParser(
        prog="fnirs-hyper",
        description="Hyperscanning analysis: wavelet coherence, inter-subject correlation "
                    "and the phase-scrambled null, computed over a derivatives tree that "
                    "fnirs-pipe has already written.",
    )
    p.add_argument("--version", action="version", version=f"fnirs-hyper {__version__}")
    sub = p.add_subparsers(required=True, metavar="COMMAND")

    run = sub.add_parser("run", parents=[common, pairs, window, band_opts],
                         help="WTC + ISC report per dyad, and the null with --wtc-phase-null.")
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
                     help="Chromophore(s) the coherence runs on (default both). HbO and HbR "
                          "are two parallel passes over the same code: a member's HbO pairs "
                          "only with the other member's HbO, they are never mixed and never "
                          "averaged, so 'both' costs exactly twice as much. The reason to "
                          "run both is a consistency check rather than two results -- HbO "
                          "has the larger amplitude and the better SNR, HbR is the less "
                          "contaminated by scalp and systemic circulation, so a coupling in "
                          "HbO with nothing in HbR is a caution flag. Every band-mean table "
                          "gains a chromophore column, and the report gains a switch "
                          "that moves every coherence panel between the chromophores at "
                          "once. The null of --wtc-phase-null follows, since a null on one "
                          "chromophore says nothing about the other.")
    run.add_argument("--wtc-roi-min-channels", type=int, default=2, metavar="N",
                     help="Drop an ROI cell resting on fewer than N channel pairs, so one "
                          "surviving optode does not stand in for a region (default 2).")
    run.add_argument("--wtc-arrow-min", type=float, default=0.5, metavar="R",
                     help="Coherence a cell has to reach before its phase arrow is drawn on "
                          "the WTC maps, when neither null was computed (default 0.5). "
                          "Display only: no table or figure value changes with it. Both "
                          "--wtc-phase-null and --wtc-significance override it with a level per "
                          "frequency, the phase-scrambled one winning where both ran, and that "
                          "is the form to prefer: surrogate coherence rises at both ends of "
                          "the computed range, so one number over the whole map marks the "
                          "band edges first. The flat threshold is what is left when nothing "
                          "was drawn to compare against, and the relative phase of two "
                          "uncorrelated series being a uniformly random direction, a map "
                          "drawn with no threshold at all fills with arrows that read as "
                          "structure.")
    run.add_argument("--wtc-channel-cross", action="store_true",
                     help="Cross every long channel with every other across the two brains "
                          "instead of pairing each channel with its counterpart, so n "
                          "channels give n^2 coherence values rather than n. The extra "
                          "pairs reach the channel TSV with a label2 column; the "
                          "time-frequency heatmaps stay on the homologous pairs. Single "
                          "channels are noisier than ROI averages, so treat the off-diagonal "
                          "as exploratory and correct for the number of tests. Does not "
                          "affect the null: see --wtc-phase-null-cross.")
    run.add_argument("--by-condition", "--wtc-by-condition", dest="wtc_by_condition",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="Read the coherence out of each task annotation's own window, "
                          "so a block design gets one result per block as well as the one "
                          "over the whole recording (default on; --no-by-condition turns it "
                          "off). Band means land in hyper-wtcbycond.tsv with "
                          "a condition column, and each window gets its own figures. A "
                          "trigger with a duration uses it; one without runs to the next "
                          "trigger, and the last to the end. Windows shorter than one cycle "
                          "of --wtc-fmin are skipped. Each window is read off the whole-run "
                          "transform rather than transformed on its own, so it costs almost "
                          "nothing and a short condition is not inflated by its own edges.")
    run.add_argument("--wtc-cond-transform", action="store_true",
                     help="Transform each condition on its own instead of reading it out of "
                          "the whole-run transform. With the margin below this gives the "
                          "same numbers as the default route, so it is a form a methods "
                          "section can describe rather than a different result; it costs one "
                          "extra transform per condition per chromophore. Requires "
                          "--wtc-by-condition.")
    run.add_argument("--wtc-cond-pad-s", type=str, default="auto", metavar="SEC|auto",
                     help="Seconds kept on each side of a condition under "
                          "--wtc-cond-transform, then windowed back off. 'auto' is "
                          "2*sqrt(2)/--wtc-band-fmin, the width at which a condition's band "
                          "mean stops moving (47 s at 0.06 Hz). 0 cuts each condition to its "
                          "own boundaries, which is what most published per-condition "
                          "pipelines do and is biased upward by an amount that grows as the "
                          "condition shortens; it is here to reproduce such a result.")
    run.add_argument("--wtc-limit-scales", action=argparse.BooleanOptionalAction, default=True,
                     help="Compute only the wavelet scales inside --wtc-fmin/--wtc-fmax "
                          "plus margin, instead of every scale the record length allows "
                          "(default on). A narrow band over a long record leaves most of the "
                          "default scale range unused, and the saving is proportional. The "
                          "kept scales land on pycwt's own grid and the margin is wider than "
                          "the scale-smoothing window, so the coherences match the "
                          "unrestricted ones bit for bit. --no-wtc-limit-scales restores the "
                          "old behaviour.")
    run.add_argument("--wtc-save-maps", action="store_true",
                     help="Save the full time-frequency coherence maps beside each TSV as "
                          "npz, so a different band can be averaged later with `fnirs-hyper "
                          "band` instead of a second wavelet transform. Large: one array "
                          "per pair per dyad per task.")
    run.add_argument("--wtc-phase-null", type=int, default=None, metavar="N",
                     help="Also write the phase-scrambled null: the same band means against a "
                          "phase-scrambled partner, averaged over N iterations (100 is what "
                          "published work uses). Coherence between two unrelated recordings "
                          "is not zero, so this is what a real value is read against. Omit "
                          "it and no null is computed: each iteration costs a full WTC run, "
                          "so this is the expensive half of a hyper run. With "
                          "--wtc-by-condition the null follows the same windows and lands in "
                          "a second table, at no extra transform: a short condition tested "
                          "against a whole-record null looks further above chance than it "
                          "is. Each table carries the spread the mean came out of and each "
                          "cell's percentile inside its own draws, and the maps draw their "
                          "phase arrows against the null's level rather than "
                          "--wtc-arrow-min.")
    run.add_argument("--wtc-phase-null-cross", action="store_true",
                     help="Cross the channels for the null too. Deliberately separate from "
                          "--wtc-channel-cross: crossing squares the pair count, and the null "
                          "pays that on every iteration. The homologous null is still the "
                          "null for the homologous cells of a crossed real table, which are "
                          "the rows where label and label2 agree.")
    run.add_argument("--bads-scope", choices=_BADS_SCOPE_CHOICES, default="run",
                     help="Which rejected channels are excluded from the inter-brain "
                          "metrics. 'run' (default) uses this task's own rejections. "
                          "'subject' unions them over the subject's runs that the pairs "
                          "table names, so all conditions rest on the same channel set. "
                          "The two differ only where the conditions were cropped to "
                          "separate tasks before preprocessing: a recording preprocessed "
                          "whole is screened once, so its conditions already rest on one "
                          "channel set and the union is that one run's own rejections. A "
                          "line in the log says which case a given run is.")
    run.add_argument("--isc-threshold", type=float, default=None,
                     help="Absolute |ISC| a pairing has to clear to get a chord in the "
                          "connectivity circle. Left alone the rule is chosen instead of the "
                          "number: with --isc-phase-null a chord is drawn where the pairing beats "
                          "the 95th percentile of its own surrogate draws, and without one "
                          "the strongest tenth are drawn and the subtitle says they are a "
                          "display cut rather than a test. Naming a number here forces the "
                          "absolute cut, which is what reproducing a fixed threshold needs; "
                          "it is scale-dependent, and the scale moves with --desc, the "
                          "passband, --isc-whiten and --isc-max-lag.")
    run.add_argument("--no-report", action="store_true",
                     help="Write the tables and skip the HTML report and its figures. The "
                          "figures are most of what this step puts on disk, and an analysis "
                          "that reads the tables never opens them. Everything else is "
                          "unchanged: the same numbers, the same files, the same npz when "
                          "--wtc-save-maps is given.")
    run.add_argument("--isc-fmin", type=float, default=None, metavar="HZ",
                     help="Band-limit each member before the correlation, low edge. Without "
                          "it ISC reads whatever the preprocessing passband left, which on a "
                          "stage with no low-pass is dominated by the cardiac component; the "
                          "run warns when that leaves it on different frequencies than "
                          "--wtc-band-fmin/fmax, since the two metrics are averages of one "
                          "complex coherency and comparing them needs one band. Set it equal "
                          "to --wtc-band-fmin for that. Filtering happens before a condition "
                          "window is cut, so a short condition carries no edge the whole "
                          "recording did not have.")
    run.add_argument("--isc-fmax", type=float, default=None, metavar="HZ",
                     help="The high edge of that band. No default: see --isc-fmin.")
    run.add_argument("--isc-whiten", type=int, default=0, metavar="ORDER",
                     help="Fit an autoregressive model of at most this order to each channel "
                          "before the inter-subject correlation and correlate the residuals; "
                          "0, the default, correlates the signals themselves. A haemoglobin "
                          "trace is strongly autocorrelated, so a correlation between two of "
                          "them rests on far fewer independent observations than it has "
                          "samples and the value it reaches with nothing coupled is "
                          "correspondingly large; whitening puts r back on the scale its "
                          "sample count implies. It also shrinks r by roughly a factor of "
                          f"six, so a whitened matrix is not comparable with an unwhitened "
                          f"one. The published work that whitens uses {ISC_MAX_AR_ORDER} as "
                          "the ceiling and picks the order per channel by BIC, which is what "
                          "passing that number does. The order each channel used reaches "
                          "hyper-iscpairs.tsv as ar_order.")
    run.add_argument("--isc-max-lag", type=float, default=0.0, metavar="SECONDS",
                     help="Re-correlate the pair at every shift within this many seconds "
                          "either way and keep the strongest, instead of correlating sample "
                          "against sample (default 0, no search). Two people's haemodynamic "
                          "responses do not peak at the same instant, so a same-sample "
                          "correlation reads a coupling a second apart as no coupling; the "
                          "cross-correlation strand of the literature searches 2 s either "
                          "way for that reason. Strongest means largest in magnitude with "
                          "the sign kept, which differs from the published largest-signed "
                          "form only where a pairing is anticorrelated. The winning shift "
                          "reaches hyper-iscpairs.tsv as lag_s, positive where the second "
                          "member follows the first. A maximum over many shifts is larger "
                          "than any one of them under no coupling, so pair this with "
                          "--isc-phase-null, whose surrogates are searched the same way.")
    run.add_argument("--isc-phase-null", type=int, default=0, metavar="N",
                     help="Also rank each correlation against N phase-scrambled surrogates "
                          "of the second member, which is the null a correlation between "
                          "two recordings needs: scrambling preserves each signal's own "
                          "spectrum and so its autocorrelation. Adds null_abs_mean, "
                          "null_abs_sd, null_abs_p95 and percentile to hyper-iscpairs.tsv, "
                          "all of them magnitudes, since a correlation is two-sided. Off by default; "
                          "it costs N extra correlations per chromophore per window.")
    run.add_argument("--check-only", action="store_true",
                     help="Load and align each dyad, print what the metrics would be "
                          "computed on, and stop. Nothing is written. Use it to look over a "
                          "cohort's channel budget before committing to a run, which with "
                          "--wtc-phase-null is hours.")
    # not the shared screening block: nothing is screened here, the rejections were decided
    # upstream and are read off the sidecars, so a --psp-threshold would do nothing at all
    run.add_argument("--sci-threshold", type=float, default=SCI_PASS,
                     help=f"The SCI line the per-subject quality table is coloured against "
                          f"(default {SCI_PASS}). Detects nothing here: screening happened "
                          f"in fnirs-pipe. Pass what the run was prepped with.")
    _shared.add_separation_bands(run, note="An override, not the source: the bands are "
                                 "read back from what fnirs-pipe stamped in each member's "
                                 "record, and members prepped with different bands are "
                                 "refused. Pass this only for a tree prepped before the "
                                 "stamp existed. A band left off keeps the records' value.")
    run.set_defaults(func=cmd_run)

    band = sub.add_parser(
        "band", parents=[common, band_opts],
        help="Re-average saved WTC maps over another band, with no second transform.",
        description="Re-averages the maps a `run --wtc-save-maps` saved, so no wavelet "
                    "transform runs a second time. Writes tables, not a report. "
                    "--wtc-band-fmin and --wtc-band-fmax are both required here.")
    band.add_argument("--wtc-suffix", default=None,
                      help="Name added to each output TSV. Defaults to the band, e.g. "
                           "'band0p05-0p2', so the new tables sit beside the originals "
                           "rather than replacing them.")
    band.set_defaults(func=cmd_band)

    group_null = sub.add_parser(
        "group-null", parents=[common],
        help="Read the re-paired draws above the cell: per occasion, and per cohort.",
        description="`pair-null` ranks each channel of each dyad inside its own draws, which "
                    "says where a channel stands and spends the pool's resolution on saying "
                    "it: against 22 stand-ins no cell can reach a p under 1/23, so a test "
                    "corrected over a thousand cells rejects almost nothing whatever the "
                    "data does. This averages the channels first and ranks that, once per "
                    "occasion and once over the cohort, which cannot say which channel and "
                    "can say whether the pairing beats its null at all. It reads the draws "
                    "`pair-null` wrote and runs no transform.")
    group_null.add_argument("--task", default="full",
                            help="Task whose tables to read (default full).")
    group_null.add_argument("--wtc-chroma", choices=("hbo", "hbr"), default="hbo",
                            help="Chromophore to read (default hbo). One at a time: the two "
                                 "are separate measurements and averaging across them means "
                                 "nothing.")
    group_null.add_argument("--n-resample", type=int, default=20000,
                            help="Resamples behind the cohort null (default 20000). Each "
                                 "picks one stand-in per occasion, so the finest p it can "
                                 "express is 1/(n+1).")
    group_null.add_argument("--seed", type=int, default=None,
                            help="Seed the resampling, so the cohort p is reproducible.")
    group_null.set_defaults(func=cmd_group_null)

    index = sub.add_parser(
        "index", parents=[common],
        help="Rebuild the dyad landing page from the tables already on disk.",
        description="Writes group-<id>_index.html, one row per analysed window, linking to "
                    "that window's report. `run` writes it too; this rebuilds it for a tree "
                    "produced earlier, or after the pages were regenerated by hand. It "
                    "reads the tables and nothing else, so it serves both orders the "
                    "pipeline can be driven in: a recording preprocessed whole lists its "
                    "conditions as windows of one task, and a tree cropped per condition "
                    "first lists them as separate tasks.")
    index.add_argument("--group-id", default=None,
                       help="Only this dyad. Every group-* directory by default.")
    index.set_defaults(func=cmd_index)

    pair = sub.add_parser(
        "pair-null", parents=[common, pairs],
        help="Draw the re-paired null: each dyad against members of the other dyads.",
        description="Recomputes the coherence of one member against people they never "
                    "interacted with, drawn from the other groups of the same task, and "
                    "writes group-*_task-*_hyper-wtc-pairnull.tsv beside the real tables. "
                    "Unlike --wtc-phase-null, which destroys every temporal structure "
                    "including each member's own time-locked response to the task, a "
                    "re-paired partner did the same task, so what survives is coupling "
                    "beyond what the shared task explains. Needs a cohort: the number of "
                    "draws is the number of other groups, which is what limits how finely "
                    "the percentile can rank. Run it after `fnirs-hyper run`, whose tables "
                    "it reads its band, its mask, its frequency range and its window off.")
    pair.add_argument("--desc", default="preproc",
                      help="desc entity of the per-subject stage the null reads. Must match "
                           "the one the real tables were computed from.")
    pair.add_argument("--roi-mapping", type=Path, default=None,
                      help="JSON file mapping ROI labels to channel names, to also write the "
                           "null of the homologous ROI means. Optional.")
    pair.add_argument("--bads-scope", choices=_BADS_SCOPE_CHOICES, default="run",
                      help="Which rejected channels are excluded, as in `run`. A stand-in "
                           "with no quality record is refused rather than kept whole.")
    pair.add_argument("--wtc-chroma", choices=("hbo", "hbr", "both"), default="both",
                      help="Chromophore(s) to draw the null on (default both). A null drawn "
                           "on HbO says nothing about an HbR coupling.")
    pair.add_argument("--wtc-pair-pool", choices=("position", "any"), default="position",
                      help="Who may stand in. 'position' (default) replaces a member only "
                           "with another group's member at the same index, which keeps a "
                           "role where the two members are not interchangeable and is the "
                           "only safe pool where one person appears in several groups, as "
                           "in a cohort of the same pair recorded over many days. 'any' "
                           "draws from every other group's members, doubling the pool, and "
                           "is refused where the table shows anybody repeated: there it "
                           "would rank a person against themselves.")
    pair.add_argument("--wtc-pair-max", type=int, default=None, metavar="N",
                      help="Stop after N draws. The pool is finite, so this is a ceiling "
                           "rather than a count: without it every eligible stand-in is "
                           "used, which is what gives the percentile its best resolution.")
    pair.add_argument("--wtc-pair-cross", action="store_true",
                      help="Draw the null over every channel pair rather than homologous "
                           "ones only. Kept separate from the real run's --wtc-channel-cross "
                           "for the same reason --wtc-phase-null-cross is: a crossed null "
                           "costs one full run per channel pair.")
    pair.add_argument("--wtc-roi-min-channels", type=int, default=2, metavar="N",
                      help="Drop an ROI cell resting on fewer than N channel pairs "
                           "(default 2). Match the value the real tables used.")
    pair.add_argument("--wtc-limit-scales", action=argparse.BooleanOptionalAction, default=True,
                      help="Compute only the scales inside the frequency range plus margin "
                           "(default on), as in `run`.")
    pair.set_defaults(func=cmd_pair_null)

    merge = sub.add_parser(
        "merge", parents=[common],
        help="Merge the per-dyad band-mean tables into one long table per kind.",
        description="Concatenates every group-*_task-*_hyper-wtc*.tsv under the tree into "
                    "one table per kind at its root, adding group_id and task columns, so a "
                    "cohort analysis reads one file. Refuses to merge tables that disagree "
                    "on the band, on mask_coi, on the null's iteration count or on which null "
                    "they are.")
    merge.set_defaults(func=cmd_merge)

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.func is cmd_band:
        for flag, value in (("--wtc-band-fmin", args.wtc_band_fmin),
                            ("--wtc-band-fmax", args.wtc_band_fmax)):
            if value is None:
                print(f"Error: Missing option '{flag}'.", file=sys.stderr)
                raise SystemExit(1)

    kw = {k: v for k, v in vars(args).items() if k != "func"}
    args.func(**kw)
