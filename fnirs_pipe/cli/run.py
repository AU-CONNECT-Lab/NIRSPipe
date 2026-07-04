"""fnirs-pipe CLI entry point (argparse, BIDS App convention)."""

import argparse
import sys
from pathlib import Path

from fnirs_pipe import __version__

_MOTION_CHOICES        = ["tddr", "wavelet", "spline", "none"]
_MODE_CHOICES          = ["denoise", "glm", "rest"]
_HRF_CHOICES           = [
    "spm", "spm + derivative", "spm + derivative + dispersion",
    "glover", "glover + derivative", "glover + derivative + dispersion", "fir",
]
_NOISE_CHOICES         = ["ols", "ar1", "ar2", "ar3", "ar4", "ar5"]
_DRIFT_CHOICES         = ["cosine", "polynomial", "none"]
_SHORT_CHANNEL_CHOICES = ["none", "mean", "pca"]
_IGNORE_CHOICES        = ["events", "bids-validation"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-pipe",
        description="BIDS-compatible fNIRS preprocessing and postprocessing pipeline. "
                    "Prep always runs; add --mode to run postprocessing.",
    )
    p.add_argument("--version", action="version", version=f"fnirs-pipe {__version__}")

    p.add_argument("bids_dir",   type=Path, help="BIDS dataset directory.")
    p.add_argument("output_dir", type=Path, help="Output directory (BIDS Derivatives).")
    p.add_argument("analysis_level", choices=["participant", "group"], help="Processing level.")

    prep = p.add_argument_group("preprocessing (required for participant level)")
    # extend: accepts space-separated (--dpf 6 6) and repeated (--dpf 6 --dpf 6) forms.
    prep.add_argument("--dpf", nargs="+", type=float, action="extend",
                      help="Differential pathlength factor. One value or one per wavelength.")
    prep.add_argument("--sci-threshold", type=float,
                      help="SCI threshold for bad channel detection, e.g. 0.8.")

    sel = p.add_argument_group("subject / session / task selection")
    sel.add_argument("--participant-label", nargs="+", action="extend", help="Subject ID(s) to process.")
    sel.add_argument("--session-label",     nargs="+", action="extend", help="Session label(s) to process.")
    sel.add_argument("--task-label",        nargs="+", action="extend", help="Task label(s) to process.")
    sel.add_argument("--bids-filter-file",  type=Path, help="JSON file with extra pybids query filters.")

    prep_opt = p.add_argument_group("preprocessing (optional)")
    prep_opt.add_argument("--motion-correction", choices=_MOTION_CHOICES, default="tddr",
                          help="Motion correction method.")
    prep_opt.add_argument("--exclude-channels",
                          help="Comma-separated channel names to manually exclude, e.g. 'S1_D1 hbo,S1_D1 hbr'.")
    prep_opt.add_argument("--cardiac-l-freq", type=float, default=0.7,
                          help="Lower bound of cardiac band in Hz. Increase for children (e.g. 1.0).")
    prep_opt.add_argument("--cardiac-h-freq", type=float, default=1.5,
                          help="Upper bound of cardiac band in Hz. Increase for children (e.g. 2.5).")

    post = p.add_argument_group("postprocessing (requires --mode)")
    post.add_argument("--mode", choices=_MODE_CHOICES, help="Postprocessing mode: denoise, glm or rest.")
    post.add_argument("--config", type=Path,
                      help="TOML file providing post parameter values. CLI flags override TOML.")
    post.add_argument("--high-pass", type=float, help="High-pass filter cutoff in Hz, e.g. 0.01.")
    post.add_argument("--low-pass",  type=float, help="Low-pass filter cutoff in Hz, e.g. 0.5.")
    post.add_argument("--resample-sfreq", type=float,
                      help="Target sampling rate in Hz after filtering, e.g. 2.0.")
    post.add_argument("--combine-runs", action=argparse.BooleanOptionalAction, default=False,
                      help="Concatenate multiple runs before postprocessing.")

    glm = p.add_argument_group("postprocessing: GLM (--mode glm)")
    glm.add_argument("--stim-dur", type=float,
                     help="Stimulus duration (s) for annotation-based events. Mutually exclusive with --events-path.")
    glm.add_argument("--hrf-model",   choices=_HRF_CHOICES,
                     help="HRF basis. 'spm + derivative' adds temporal derivative column.")
    glm.add_argument("--noise-model", choices=_NOISE_CHOICES, help="Residual autocorrelation model.")
    glm.add_argument("--drift-model", choices=_DRIFT_CHOICES,
                     help="Low-frequency drift regressors in design matrix.")
    glm.add_argument("--drift-high-pass", type=float, help="High-pass cutoff for cosine drift in Hz.")
    glm.add_argument("--drift-order", type=int, default=1,
                     help="Polynomial drift order (polynomial drift model only).")
    glm.add_argument("--fir-delays",
                     help="FIR delay bins in scans, comma-separated, e.g. '0,1,2,3,4,5' (only used when --hrf-model fir).")
    glm.add_argument("--short-channel", choices=_SHORT_CHANNEL_CHOICES,
                     help="Short-channel confound regressor strategy.")
    glm.add_argument("--events-path", type=Path,
                     help="Path to *_events.tsv. If omitted, extracted from snirf annotations.")
    glm.add_argument("--contrast-file", type=Path, help="TOML file defining GLM contrasts.")

    out = p.add_argument_group("output")
    out.add_argument("--no-report", action="store_true", help="Skip the QC HTML report.")
    out.add_argument("--n-jobs", type=int, default=1, help="Parallel subject jobs.")
    out.add_argument("--work-dir", type=Path, help="Hash cache directory.")

    esc = p.add_argument_group("escape hatches")
    esc.add_argument("--ignore", nargs="+", action="extend", choices=_IGNORE_CHOICES,
                     help="Processing aspects to skip.")
    esc.add_argument("--skip-bids-validation", action="store_true", help="Skip BIDS validation.")
    esc.add_argument("--dry-run", action="store_true", help="Write pipeline script without executing.")
    esc.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.dpf is None:
        print("Error: Missing option '--dpf'.", file=sys.stderr)
        raise SystemExit(1)
    if args.sci_threshold is None:
        print("Error: Missing option '--sci-threshold'.", file=sys.stderr)
        raise SystemExit(1)

    from fnirs_pipe.cli.workflows import run_group_level, run_participant_level

    opts = vars(args)
    if args.analysis_level == "participant":
        run_participant_level(opts)
    else:
        run_group_level(opts)
