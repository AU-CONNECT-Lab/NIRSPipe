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

from fnirs_pipe.utils.logging import get_logger, setup_logging

setup_logging()

logger = get_logger("cli.hyper")

_BADS_SCOPE_CHOICES = ["run", "subject"]

# the hyper report colours its quality table by SCI but detects no bad channels of its own
_SCI_DEFAULT = 0.8


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
                        tstart=None, tend=None):
    """Load one dyad, put both recordings on one time axis, and mark the rejected channels.

    Returns (aligned_raws, offsets, group_sqm). The rejections have to be applied here rather
    than in each metric: `desc-errts` carries an empty ``info["bads"]``, so a metric that
    reads the Raw alone sees the full montage whatever --bads-scope was asked for.
    """
    from fnirs_pipe.pipeline.hyperscanning import (
        align_recordings,
        apply_group_bads,
        crop_aligned_window,
        load_group_haemo,
        load_group_sqm,
        normalize_raws,
        trim_to_shortest,
    )

    raws = load_group_haemo(output_dir, members, desc=desc)
    group_sqm = load_group_sqm(output_dir, members, bads_scope=bads_scope)
    apply_group_bads(raws, group_sqm)
    if no_align:
        aligned_raws, offsets = trim_to_shortest(raws)
    else:
        aligned_raws, offsets = align_recordings(raws, task)
    aligned_raws = crop_aligned_window(aligned_raws, tstart, tend)
    if normalize:
        aligned_raws = normalize_raws(aligned_raws)
    return aligned_raws, offsets, group_sqm


def _quality_summary(aligned_raws: dict, group_sqm: dict) -> None:
    """Print what the metrics are about to be computed on, one line per subject.

    ``sub-01  long 12/14  bad 2  mean SCI 0.86  from: tapping``

    Counted off the aligned Raw after the rejections are applied, so it describes the channel
    set the coherence actually uses rather than what the montage holds. The report says the
    same thing, but only once the run has finished, which with --wtc-pseudo is hours later.
    """
    from fnirs_pipe.io.snirf import long_channel_picks

    for subject_id, raw in aligned_raws.items():
        sqm = group_sqm.get(subject_id, {})
        pairs = {ch.rsplit(" ", 1)[0] for ch in raw.ch_names
                 if ch.endswith(" hbo") or ch.endswith(" hbr")}
        bad = {ch.rsplit(" ", 1)[0] for ch in raw.info["bads"]}
        kept = len(long_channel_picks(raw, "hbo"))     # pick_types drops bads already

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
    from fnirs_pipe.qc.wtc_aggregate import _KINDS

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


def cmd_run(
    output_dir: Path, pairs_csv: Path, group_id: str | None, task_label: list[str] | None,
    desc: str, roi_mapping: Path | None,
    wtc_fmin: float, wtc_fmax: float,
    wtc_band_fmin: float | None, wtc_band_fmax: float | None,
    wtc_significance: bool, wtc_mc_count: int, wtc_seed: int | None,
    wtc_mask_coi: bool, wtc_roi_min_channels: int, wtc_channel_cross: bool,
    wtc_limit_scales: bool, wtc_save_maps: bool,
    wtc_pseudo: int | None, wtc_pseudo_cross: bool,
    bads_scope: str, isc_threshold: float, sci_threshold: float | None,
    normalize: bool, no_align: bool, tstart: float | None, tend: float | None,
    check_only: bool, verbose: bool,
) -> None:
    """Dyad WTC + ISC report per group, plus the pseudo-dyad null when --wtc-pseudo is given.

    The null reuses this run's aligned recordings and every band parameter, so it cannot be
    computed over a different band than the table it sits beside.
    """
    import json

    from fnirs_pipe.pipeline.hyperscanning import write_group_bads
    from fnirs_pipe.qc.hyper_report import build_hyper_post_report
    from fnirs_pipe.qc.wtc_null import write_wtc_null

    setup_logging(verbose=verbose)

    if sci_threshold is None:
        sci_threshold = _SCI_DEFAULT

    if wtc_channel_cross and roi_mapping is None:
        print("[warn] --wtc-channel-cross without --roi-mapping: the crossed channel table "
              "is written but no ROI x ROI matrix is built from it.", file=sys.stderr)

    groups = _select_groups(pairs_csv, group_id, task_label)

    roi_map: dict[str, list[str]] | None = None
    if roi_mapping is not None:
        try:
            roi_map = json.loads(Path(roi_mapping).read_text())
        except Exception as exc:
            print(f"[error] failed to load ROI mapping: {exc}", file=sys.stderr)
            raise SystemExit(1)

    def _process(gid, task, members):
        aligned_raws, offsets, group_sqm = _load_aligned_group(
            output_dir, members, task, desc, no_align, normalize, bads_scope, tstart, tend)
        _quality_summary(aligned_raws, group_sqm)
        if check_only:
            return None
        bad_channels = {sid: sqm.get("bad_channels", []) for sid, sqm in group_sqm.items()}
        write_group_bads(output_dir, members, group_sqm, bads_scope)
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
            wtc_limit_scales=wtc_limit_scales,
            wtc_save_maps=wtc_save_maps,
            wtc_mask_coi=wtc_mask_coi,
            wtc_roi_min_channels=wtc_roi_min_channels,
            isc_threshold=isc_threshold,
            sci_threshold=sci_threshold,
        )
        if wtc_pseudo:
            null_path = write_wtc_null(
                group_id=gid,
                task=task,
                aligned_raws=aligned_raws,
                output_dir=output_dir,
                n_iter=wtc_pseudo,
                wtc_fmin=wtc_fmin,
                wtc_fmax=wtc_fmax,
                band_fmin=wtc_band_fmin,
                band_fmax=wtc_band_fmax,
                seed=wtc_seed,
                cross=wtc_pseudo_cross,
                limit_scales=wtc_limit_scales,
                mask_coi=wtc_mask_coi,
            )
            print(f"     null   -> {null_path}")
        return report_path

    if wtc_pseudo:
        logger.info("pseudo-dyad null requested: %d full WTC runs per dyad, on top of the real one",
                    wtc_pseudo)
    _run_groups(groups, _process)
    if not check_only:
        _merge_reminder(output_dir)


def cmd_band(
    output_dir: Path, wtc_band_fmin: float, wtc_band_fmax: float,
    wtc_mask_coi: bool, wtc_suffix: str | None, verbose: bool,
) -> None:
    """Re-average every saved WTC map over a new band, without recomputing the transform."""
    from fnirs_pipe.qc.wtc_store import reband_tree

    setup_logging(verbose=verbose)

    written = reband_tree(output_dir, wtc_band_fmin, wtc_band_fmax,
                          suffix=wtc_suffix, mask_coi=wtc_mask_coi)
    for path in written:
        print(f"reband -> {path}")
    if not written:
        print(f"no *_hyper-wtc*.npz under {output_dir}; rerun `fnirs-hyper run "
              "--wtc-save-maps` to write them", file=sys.stderr)


def cmd_merge(output_dir: Path, verbose: bool) -> None:
    """Merge every per-dyad WTC band-mean table into one long table per kind."""
    from fnirs_pipe.qc.wtc_aggregate import _KINDS, write_aggregate_wtc

    setup_logging(verbose=verbose)

    # driven off the aggregator's own kinds, so removing or adding one cannot leave this
    # list behind: it named wtc-roi, gone since 0.24.0, and never named wtc-pseudo at all
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
    # they are declared once: two spellings of one parameter is what the band flags used to be
    band_opts = argparse.ArgumentParser(add_help=False)
    band_opts.add_argument("--wtc-band-fmin", type=float, default=None,
                           help="Lower bound (Hz) of the band the per-channel WTC TSV "
                                "averages over. For `run` it defaults to --wtc-fmin, i.e. "
                                "the whole computed axis; for `band` it is the new band and "
                                "is required.")
    band_opts.add_argument("--wtc-band-fmax", type=float, default=None,
                           help="Upper bound (Hz) of that band. Defaults to --wtc-fmax for "
                                "`run`; required for `band`.")
    band_opts.add_argument("--wtc-mask-coi", action="store_true",
                           help="Average each band mean only over cells inside the cone of "
                                "influence. Off by default, which is what the field does; "
                                "the share inside the cone is reported as n_valid_frac "
                                "either way. Masking discards more of a short segment than "
                                "of a long one, so it moves conditions of different length "
                                "by different amounts.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("output_dir", type=Path,
                        help="fnirs-pipe derivatives directory. No BIDS input is read.")
    common.add_argument("--verbose", action="store_true")

    p = argparse.ArgumentParser(
        prog="fnirs-hyper",
        description="Hyperscanning analysis: wavelet coherence, inter-subject correlation "
                    "and the pseudo-dyad null, computed over a derivatives tree that "
                    "fnirs-pipe has already written.",
    )
    p.add_argument("--version", action="version", version=f"fnirs-hyper {__version__}")
    sub = p.add_subparsers(required=True, metavar="COMMAND")

    run = sub.add_parser("run", parents=[common, band_opts],
                         help="WTC + ISC report per dyad, and the null with --wtc-pseudo.")
    run.add_argument("--pairs-csv", type=Path, required=True,
                     help="CSV with columns: group_id, subject_id, task. Each unique "
                          "(group_id, task) pair is processed as one session.")
    run.add_argument("--group-id", default=None,
                     help="Process only this group_id. Omit to process all groups.")
    run.add_argument("--task-label", nargs="+", action="extend",
                     help="Task label(s) to include, filtering the pairs table.")
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
                          "from the pseudo-dyad null of --wtc-pseudo.")
    run.add_argument("--wtc-mc-count", type=int, default=300,
                     help="Surrogate series behind each --wtc-significance contour "
                          "(default 300). This is what the runtime is spent on and it "
                          "scales linearly; lower it to preview a run, raise it to settle "
                          "a contour. Ignored without --wtc-significance.")
    run.add_argument("--wtc-seed", type=int, default=None,
                     help="Seed the Monte Carlo surrogates behind --wtc-significance and the "
                          "phase randomisation behind --wtc-pseudo. Also bypasses pycwt's "
                          "on-disk cache, which is not keyed on the seed.")
    run.add_argument("--wtc-roi-min-channels", type=int, default=2, metavar="N",
                     help="Drop an ROI cell resting on fewer than N channel pairs, so one "
                          "surviving optode does not stand in for a region (default 2).")
    run.add_argument("--wtc-channel-cross", action="store_true",
                     help="Cross every long channel with every other across the two brains "
                          "instead of pairing each channel with its counterpart, so n "
                          "channels give n^2 coherence values rather than n. The extra "
                          "pairs reach the channel TSV with a label2 column; the "
                          "time-frequency heatmaps stay on the homologous pairs. Single "
                          "channels are noisier than ROI averages, so treat the off-diagonal "
                          "as exploratory and correct for the number of tests. Does not "
                          "affect the null: see --wtc-pseudo-cross.")
    run.add_argument("--wtc-limit-scales", action=argparse.BooleanOptionalAction, default=True,
                     help="Compute only the wavelet scales inside --wtc-fmin/--wtc-fmax "
                          "plus margin, instead of every scale the record length allows "
                          "(default on). A narrow band over a long record leaves most of the "
                          "default scale range unused, and the saving is proportional. The "
                          "kept scales land on pycwt's own grid and the margin is wider than "
                          "its scale-smoothing window, so the coherences match the "
                          "unrestricted ones bit for bit. --no-wtc-limit-scales restores the "
                          "old behaviour.")
    run.add_argument("--wtc-save-maps", action="store_true",
                     help="Save the full time-frequency coherence maps beside each TSV as "
                          "npz, so a different band can be averaged later with `fnirs-hyper "
                          "band` instead of a second wavelet transform. Large: one array "
                          "per pair per dyad per task.")
    run.add_argument("--wtc-pseudo", type=int, default=None, metavar="N",
                     help="Also write the pseudo-dyad null: the same band means against a "
                          "phase-scrambled partner, averaged over N iterations (100 is what "
                          "published work uses). Coherence between two unrelated recordings "
                          "is not zero, so this is what a real value is read against. Omit "
                          "it and no null is computed: each iteration costs a full WTC run, "
                          "so this is the expensive half of a hyper run.")
    run.add_argument("--wtc-pseudo-cross", action="store_true",
                     help="Cross the channels for the null too. Deliberately separate from "
                          "--wtc-channel-cross: crossing squares the pair count, and the null "
                          "pays that on every iteration. The homologous null is still the "
                          "null for the homologous cells of a crossed real table, which are "
                          "the rows where label and label2 agree.")
    run.add_argument("--bads-scope", choices=_BADS_SCOPE_CHOICES, default="run",
                     help="Which rejected channels are excluded from the inter-brain "
                          "metrics. 'run' (default) uses this task's own rejections. "
                          "'subject' unions them over every run of the subject, so all "
                          "conditions rest on the same channel set, which is what comparing "
                          "conditions needs; the cost is losing a channel everywhere because "
                          "one segment was bad.")
    run.add_argument("--isc-threshold", type=float, default=0.3,
                     help="Minimum mean ISC to draw an arc in the connectivity circle.")
    run.add_argument("--check-only", action="store_true",
                     help="Load and align each dyad, print what the metrics would be "
                          "computed on, and stop. Nothing is written. Use it to look over a "
                          "cohort's channel budget before committing to a run, which with "
                          "--wtc-pseudo is hours.")
    run.add_argument("--sci-threshold", type=float, default=None,
                     help="SCI below which a channel counts as badly coupled. Detects "
                          "nothing here; it only colours the per-subject quality table. "
                          "Pass what the run was prepped with (default 0.8).")
    run.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                     help="Z-score each channel per subject after alignment.")
    run.add_argument("--no-align", action="store_true",
                     help="Skip trigger-based alignment; trim all recordings to the shortest duration.")
    run.add_argument("--tstart", type=float, default=None,
                     help="Keep only from this time (s) on the aligned clock, where 0 is the "
                          "shared trigger. Omit to start at the alignment point.")
    run.add_argument("--tend", type=float, default=None,
                     help="Keep only up to this time (s) on the aligned clock. Omit to run to "
                          "the end; a value past the end is clipped. The window narrows the "
                          "synchrony metrics only: the per-subject quality record describes "
                          "the whole recording either way.")
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

    merge = sub.add_parser(
        "merge", parents=[common],
        help="Merge the per-dyad band-mean tables into one long table per kind.",
        description="Concatenates every group-*_task-*_hyper-wtc*.tsv under the tree into "
                    "one table per kind at its root, adding group_id and task columns, so a "
                    "cohort analysis reads one file. Refuses to merge tables that disagree "
                    "on the band, on mask_coi or on the null's iteration count.")
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
