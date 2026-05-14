"""fnirs-pipe CLI entry point (Typer)."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from fnirs_pipe import __version__

app = typer.Typer(
    name="fnirs-pipe",
    help="BIDS-compatible fNIRS preprocessing and postprocessing pipeline.",
    pretty_exceptions_show_locals=False,
)


class AnalysisLevel(str, Enum):
    participant = "participant"
    group = "group"

# TODO: wavelet and spline motion correction are not yet implemented
class MotionCorrection(str, Enum):
    tddr = "tddr"
    wavelet = "wavelet"
    spline = "spline"
    none = "none"

# Only for mode
class PostMode(str, Enum):
    denoise = "denoise"
    glm = "glm"
    rest = "rest"

class HRFModel(str, Enum):
    spm = "spm"
    spm_derivative = "spm + derivative"
    spm_full = "spm + derivative + dispersion"
    glover = "glover"
    glover_derivative = "glover + derivative"
    glover_full = "glover + derivative + dispersion"
    fir = "fir"

class NoiseModel(str, Enum):
    ols = "ols"
    ar1 = "ar1"
    ar2 = "ar2"
    ar3 = "ar3"
    ar4 = "ar4"
    ar5 = "ar5"

class DriftModel(str, Enum):
    cosine = "cosine"
    polynomial = "polynomial"
    none = "none"

class ShortChannel(str, Enum):
    none = "none"
    mean = "mean"
    pca = "pca"

class IgnoreAspect(str, Enum):
    events = "events"
    bids_validation = "bids-validation"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"fnirs-pipe {__version__}")
        raise typer.Exit()


@app.command()
def main(
    bids_dir:       Annotated[Path,          typer.Argument(help="BIDS dataset directory.")],
    output_dir:     Annotated[Path,          typer.Argument(help="Output directory (BIDS Derivatives).")],
    analysis_level: Annotated[AnalysisLevel, typer.Argument(help="Processing level.")],

    # ---- prep (required unless --qc-raw) ----
    dpf:           Annotated[Optional[list[float]], typer.Option("--dpf",          help="Differential pathlength factor. One value or one per wavelength.")] = None,
    sci_threshold: Annotated[Optional[float],       typer.Option("--sci-threshold", help="SCI threshold for bad channel detection, e.g. 0.8.")] = None,

    # ---- subject / session / task selection ----
    participant_label: Annotated[Optional[list[str]], typer.Option("--participant-label", help="Subject ID(s) to process.")] = None,
    session_label:     Annotated[Optional[list[str]], typer.Option("--session-label",     help="Session label(s) to process.")] = None,
    task_label:        Annotated[Optional[list[str]], typer.Option("--task-label",        help="Task label(s) to process.")] = None,
    bids_filter_file:  Annotated[Optional[Path],      typer.Option("--bids-filter-file",  help="JSON file with extra pybids query filters.")] = None,

    # ---- prep (optional) ----
    motion_correction:  Annotated[MotionCorrection,      typer.Option("--motion-correction",  help="Motion correction method.")] = MotionCorrection.tddr,
    exclude_channels:   Annotated[Optional[str],         typer.Option("--exclude-channels",   help="Comma-separated channel names to manually exclude, e.g. 'S1_D1 hbo,S1_D1 hbr'.")] = None,
    cardiac_l_freq:     Annotated[float, typer.Option("--cardiac-l-freq", help="Lower bound of cardiac band in Hz. Increase for children (e.g. 1.0).")] = 0.7,
    cardiac_h_freq:     Annotated[float, typer.Option("--cardiac-h-freq", help="Upper bound of cardiac band in Hz. Increase for children (e.g. 2.5).")] = 1.5,

    # ---- post: mode ----
    mode:   Annotated[Optional[PostMode], typer.Option("--mode",   help="Postprocessing mode: denoise or glm.")] = None,
    config: Annotated[Optional[Path],     typer.Option("--config", help="TOML file providing post parameter values. CLI flags override TOML.")] = None,

    # ---- post: filtering ----
    high_pass: Annotated[Optional[float], typer.Option("--high-pass", help="High-pass filter cutoff in Hz, e.g. 0.01.")] = None,
    low_pass:  Annotated[Optional[float], typer.Option("--low-pass",  help="Low-pass filter cutoff in Hz, e.g. 0.5.")] = None,

    # ---- post: resample ----
    resample_sfreq: Annotated[Optional[float], typer.Option("--resample-sfreq", help="Target sampling rate in Hz after filtering, e.g. 2.0.")] = None,

    # ---- post: crop ----
    segments_path: Annotated[Optional[Path],  typer.Option("--segments-path", help="TSV file with onset/duration columns defining segments to keep.")] = None,
    crop_tmin:     Annotated[Optional[float], typer.Option("--crop-tmin",     help="Start time in seconds to crop to (single segment).")] = None,
    crop_tmax:     Annotated[Optional[float], typer.Option("--crop-tmax",     help="End time in seconds to crop to (single segment).")] = None,

    # ---- post: GLM (glm mode) ----
    stim_dur:        Annotated[Optional[float],       typer.Option("--stim-dur",        help="Stimulus duration (s) for annotation-based events. Mutually exclusive with --events-path.")] = None,
    hrf_model:       Annotated[Optional[HRFModel],   typer.Option("--hrf-model",       help="HRF basis. spm_derivative adds temporal derivative column.")] = None,
    noise_model:     Annotated[Optional[NoiseModel],  typer.Option("--noise-model",     help="Residual autocorrelation model.")] = None,
    drift_model:     Annotated[Optional[DriftModel],  typer.Option("--drift-model",     help="Low-frequency drift regressors in design matrix.")] = None,
    drift_high_pass: Annotated[Optional[float],       typer.Option("--drift-high-pass", help="High-pass cutoff for cosine drift in Hz.")] = None,
    drift_order:     Annotated[int,                   typer.Option("--drift-order",     help="Polynomial drift order (polynomial drift model only).")] = 1,
    fir_delays:      Annotated[Optional[str],         typer.Option("--fir-delays",      help="FIR delay bins in scans, comma-separated, e.g. '0,1,2,3,4,5' (only used when --hrf-model fir).")] = None,
    short_channel:   Annotated[Optional[ShortChannel],typer.Option("--short-channel",   help="Short-channel confound regressor strategy.")] = None,
    events_path:     Annotated[Optional[Path],        typer.Option("--events-path",     help="Path to *_events.tsv. If omitted, extracted from snirf annotations.")] = None,
    contrast_file:   Annotated[Optional[Path],        typer.Option("--contrast-file",   help="TOML file defining GLM contrasts.")] = None,

    # ---- post: other ----
    combine_runs: Annotated[bool, typer.Option("--combine-runs/--no-combine-runs", help="Concatenate multiple runs before postprocessing.")] = False,

    # ---- output ----
    no_report:    Annotated[bool,        typer.Option("--no-report/--report")] = False,
    n_jobs:       Annotated[int,         typer.Option("--n-jobs", help="Parallel subject jobs.")] = 1,
    work_dir:     Annotated[Optional[Path], typer.Option("--work-dir", help="Hash cache directory.")] = None,

    # ---- escape hatches ----
    ignore:               Annotated[Optional[list[IgnoreAspect]], typer.Option("--ignore")] = None,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
    dry_run:              Annotated[bool, typer.Option("--dry-run/--no-dry-run", help="Write pipeline script without executing.")] = False,

    verbose: Annotated[bool, typer.Option("--verbose/--no-verbose")] = False,
    version: Annotated[Optional[bool], typer.Option("--version", callback=_version_callback, is_eager=True)] = None,
):
    """Run fNIRS preprocessing (and optionally postprocessing).

    Prep always runs. Add --mode to run postprocessing.
    Parameters come from CLI flags or --config TOML (CLI overrides TOML).

    Step order within each mode:
      bandpass → resample → [GLM if mode=glm]
    """
    if dpf is None:
        typer.echo("Error: Missing option '--dpf'.", err=True)
        raise typer.Exit(1)
    if sci_threshold is None:
        typer.echo("Error: Missing option '--sci-threshold'.", err=True)
        raise typer.Exit(1)

    from fnirs_pipe.cli.workflows import run_participant_level, run_group_level

    if analysis_level == AnalysisLevel.participant:
        run_participant_level(locals())
    else:
        run_group_level(locals())


