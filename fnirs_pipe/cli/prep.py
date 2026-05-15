"""fnirs-prep CLI — data preparation utilities (marker editing, crop, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from fnirs_pipe.utils.logging import get_logger

app = typer.Typer(
    name="fnirs-prep",
    help="fNIRS data preparation: marker editing and related utilities.",
    pretty_exceptions_show_locals=False,
)

markers_app = typer.Typer(help="Edit markers in SNIRF files.")
app.add_typer(markers_app, name="edit-markers")

logger = get_logger("cli.prep")


def _run_parallel(fn, subjects: list[str], n_jobs: int, **kwargs) -> list[tuple]:
    """Run fn(sub, **kwargs) for each subject, return list of (sub, ok, msg)."""
    from joblib import Parallel, delayed

    def _one(sub):
        try:
            result = fn(sub, **kwargs)
            return sub, True, str(result)
        except Exception as exc:
            return sub, False, str(exc)

    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one)(sub) for sub in subjects
    )


@app.command("crop")
def crop(
    bids_dir:        Annotated[Path, typer.Argument(help="BIDS dataset root.")],
    derivatives_dir: Annotated[Path, typer.Argument(help="Derivatives output directory.")],
    participant_label: Annotated[list[str],     typer.Option("--participant-label", help="Subject ID(s) to process.")],
    ses:  Annotated[Optional[str], typer.Option("--ses",  help="Session label.")] = None,
    task: Annotated[Optional[str], typer.Option("--task", help="Task label.")] = None,
    run:  Annotated[Optional[str], typer.Option("--run",  help="Run label.")] = None,
    tmin: Annotated[Optional[float], typer.Option("--tmin", help="Start time in seconds (single segment).")] = None,
    tmax: Annotated[Optional[float], typer.Option("--tmax", help="End time in seconds (single segment).")] = None,
    segments_path: Annotated[Optional[Path], typer.Option("--segments-path", help="TSV with onset/duration columns defining segments to keep.")] = None,
    combine: Annotated[bool, typer.Option("--combine/--no-combine", help="Concatenate multi-segment output into one file.")] = False,
    n_jobs: Annotated[int, typer.Option("--n-jobs", help="Parallel subject jobs.")] = 1,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
) -> None:
    """Crop raw SNIRFs and write to derivatives/cropped/.

    Single segment: --tmin / --tmax (either or both).
    Multi-segment:  --segments-path TSV with onset/duration columns.
    Use --combine to concatenate multi-segment output into one file.
    """
    if segments_path is not None and (tmin is not None or tmax is not None):
        typer.echo("[error] --segments-path and --tmin/--tmax are mutually exclusive.", err=True)
        raise typer.Exit(1)
    if segments_path is None and tmin is None and tmax is None:
        typer.echo("[error] Specify --segments-path or at least one of --tmin / --tmax.", err=True)
        raise typer.Exit(1)
    if combine and segments_path is None:
        typer.echo("[error] --combine requires --segments-path.", err=True)
        raise typer.Exit(1)

    from fnirs_pipe.pipeline.crop import crop_snirf

    def _crop_one(sub):
        return crop_snirf(
            bids_dir, derivatives_dir, sub,
            ses=ses, task=task, run=run,
            tmin=tmin, tmax=tmax,
            segments_path=segments_path,
            combine=combine,
            validate=not skip_bids_validation,
        )

    results = _run_parallel(_crop_one, participant_label, n_jobs)
    n_fail = 0
    for sub, ok, msg in results:
        if ok:
            typer.echo(f"  sub-{sub}: {msg}")
        else:
            typer.echo(f"  sub-{sub}: [error] {msg}", err=True)
            n_fail += 1
    if n_fail:
        raise typer.Exit(1)


@markers_app.command("export")
def markers_export(
    bids_dir: Annotated[Path, typer.Argument(help="BIDS dataset root.")],
    out_dir:  Annotated[Path, typer.Argument(help="Directory to write the exported events TSV(s).")],
    participant_label: Annotated[list[str],     typer.Option("--participant-label", help="Subject ID(s) to process.")],
    ses:  Annotated[Optional[str], typer.Option("--ses",  help="Session label.")] = None,
    task: Annotated[Optional[str], typer.Option("--task", help="Task label.")] = None,
    run:  Annotated[Optional[str], typer.Option("--run",  help="Run label.")] = None,
    n_jobs: Annotated[int, typer.Option("--n-jobs", help="Parallel subject jobs.")] = 1,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
) -> None:
    """Export events.tsv(s) to out_dir for manual editing."""
    from fnirs_pipe.pipeline.edit_markers import export_markers

    def _export_one(sub):
        return export_markers(
            bids_dir, sub, out_dir,
            ses=ses, task=task, run=run,
            validate=not skip_bids_validation,
        )

    results = _run_parallel(_export_one, participant_label, n_jobs)
    n_fail = 0
    for sub, ok, msg in results:
        if ok:
            typer.echo(f"  sub-{sub}: {msg}")
        else:
            typer.echo(f"  sub-{sub}: [error] {msg}", err=True)
            n_fail += 1
    if n_fail:
        raise typer.Exit(1)


@markers_app.command("apply")
def markers_apply(
    bids_dir:        Annotated[Path, typer.Argument(help="BIDS dataset root.")],
    derivatives_dir: Annotated[Path, typer.Argument(help="Derivatives output directory.")],
    participant_label: Annotated[list[str],      typer.Option("--participant-label", help="Subject ID(s) to process.")],
    ses:  Annotated[Optional[str], typer.Option("--ses",  help="Session label.")] = None,
    task: Annotated[Optional[str], typer.Option("--task", help="Task label.")] = None,
    run:  Annotated[Optional[str], typer.Option("--run",  help="Run label.")] = None,
    tsv:          Annotated[Optional[Path],      typer.Option("--tsv",          help="Edited events TSV to apply (same file applied to all subjects).")] = None,
    shift:        Annotated[Optional[float],     typer.Option("--shift",        help="Shift all onsets by this many seconds (negative = earlier). Clipped to 0.")] = None,
    set_duration: Annotated[Optional[float],     typer.Option("--set-duration", help="Set all marker durations to this value (seconds).")] = None,
    rename:       Annotated[Optional[list[str]], typer.Option("--rename",       help="Rename marker: 'old:new'. Repeatable.")] = None,
    n_jobs: Annotated[int, typer.Option("--n-jobs", help="Parallel subject jobs.")] = 1,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
) -> None:
    """Apply marker edits to runs and write to derivatives/marker_edited/.

    Exactly one operation per call: --tsv, --shift, --set-duration, or --rename.
    All operations support batch via multiple --participant-label values.
    --tsv applies the same file to every subject (useful for shared event structure).
    """
    ops = [x for x in (tsv, shift, set_duration, rename) if x is not None]
    if not ops:
        typer.echo("[error] Specify one of: --tsv, --shift, --set-duration, --rename", err=True)
        raise typer.Exit(1)
    if len(ops) > 1:
        typer.echo("[error] Only one of --tsv, --shift, --set-duration, --rename can be used at a time.", err=True)
        raise typer.Exit(1)
    from fnirs_pipe.pipeline.edit_markers import apply_markers

    def _apply_one(sub):
        return apply_markers(
            bids_dir, derivatives_dir, sub,
            ses=ses, task=task, run=run,
            tsv=tsv, shift=shift, set_duration=set_duration, rename=rename,
            validate=not skip_bids_validation,
        )

    results = _run_parallel(_apply_one, participant_label, n_jobs)
    n_fail = 0
    for sub, ok, msg in results:
        if ok:
            typer.echo(f"  sub-{sub}: {msg}")
        else:
            typer.echo(f"  sub-{sub}: [error] {msg}", err=True)
            n_fail += 1
    if n_fail:
        raise typer.Exit(1)
