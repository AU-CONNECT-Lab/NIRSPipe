"""fnirs-pipe CLI entry point (argparse, BIDS App convention)."""

import argparse
import sys
from pathlib import Path

from fnirs_pipe import __version__
from fnirs_pipe.cli import _shared
from fnirs_pipe.pipeline.denoise import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_FILTER_ORDER,
    FILTER_METHODS,
)

_MOTION_CHOICES        = ["tddr", "wavelet", "spline", "none"]
_MODE_CHOICES          = ["denoise", "glm", "rest"]
_HRF_CHOICES           = [
    "spm", "spm + derivative", "spm + derivative + dispersion",
    "glover", "glover + derivative", "glover + derivative + dispersion", "fir",
]
_NOISE_CHOICES         = ["ols", "ar1", "ar2", "ar3", "ar4", "ar5"]
_DRIFT_CHOICES         = ["cosine", "polynomial", "none"]
_SHORT_CHANNEL_CHOICES = ["none", "mean"]
_IGNORE_CHOICES        = ["events", "bids-validation"]
# analysis levels and the flags each one cannot run without. One table, so a level added
# here cannot reach the parser without also declaring what it needs
_LEVEL_REQUIRES = {
    "participant": ("--dpf", "--sci-threshold",
                    "--cardiac-l-freq", "--cardiac-h-freq", "--resp-l-freq", "--resp-h-freq"),
    "group":       (),
}
_LEVEL_CHOICES         = list(_LEVEL_REQUIRES)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-pipe",
        description="BIDS-compatible fNIRS preprocessing and postprocessing pipeline. "
                    "Prep always runs; add --mode to run postprocessing. Dyad analysis "
                    "lives in fnirs-hyper, which reads what this writes.",
    )
    p.add_argument("--version", action="version", version=f"fnirs-pipe {__version__}")

    p.add_argument("bids_dir",   type=Path, help="BIDS dataset directory.")
    p.add_argument("output_dir", type=Path, help="Output directory (BIDS Derivatives).")
    p.add_argument("analysis_level", choices=_LEVEL_CHOICES, help="Processing level.")

    prep = p.add_argument_group("preprocessing (required for participant level)")
    # extend: accepts space-separated (--dpf 6 6) and repeated (--dpf 6 --dpf 6) forms.
    prep.add_argument("--dpf", nargs="+", type=float, action="extend",
                      help="Differential pathlength factor. One value or one per wavelength.")
    # no default: _LEVEL_REQUIRES makes it required at participant level
    _shared.add_sci_threshold(prep)

    sel = p.add_argument_group("subject / session / task selection")
    sel.add_argument("--participant-label", nargs="+", action="extend", help="Subject ID(s) to process.")
    sel.add_argument("--session-label",     nargs="+", action="extend", help="Session label(s) to process.")
    sel.add_argument("--task-label",        nargs="+", action="extend", help="Task label(s) to process.")
    sel.add_argument("--bids-filter-file",  type=Path, help="JSON file with extra pybids query filters.")

    prep_opt = p.add_argument_group("preprocessing (optional)")
    prep_opt.add_argument("--motion-correction", choices=_MOTION_CHOICES, default="tddr",
                          help="Motion correction method.")
    prep_opt.add_argument("--bad-channels",
                          help="Source-detector labels to mark as bad, e.g. 'S1_D1,S2_D3', applied to "
                               "every subject. Or a path to a table with participant_id and "
                               "bad_channels columns, one row per subject. Naming either wavelength "
                               "of a pair marks both. Kept in the data, unioned with SCI-detected "
                               "bad channels.")
    prep_opt.add_argument("--cardiac-l-freq", type=float,
                          help="Lower bound of cardiac band in Hz (required at participant level; "
                               "population-dependent). Adult resting ~0.7; children/infants higher "
                               "(e.g. 1.0-1.7).")
    prep_opt.add_argument("--cardiac-h-freq", type=float,
                          help="Upper bound of cardiac band in Hz (required at participant level; "
                               "population-dependent). Adult resting ~1.5; children/infants higher "
                               "(e.g. 2.5-3.0).")
    prep_opt.add_argument("--resp-l-freq", type=float,
                          help="Lower bound of respiration band in Hz (required at participant level; "
                               "population-dependent). Adult ~0.1; infants higher.")
    prep_opt.add_argument("--resp-h-freq", type=float,
                          help="Upper bound of respiration band in Hz (required at participant level; "
                               "population-dependent). Adult ~0.5; infants higher.")
    # the other screening line. Optional, unlike --sci-threshold: PSP has a published
    # default that holds across populations, so a run that does not name it is not guessing
    _shared.add_psp_threshold(prep_opt)
    _shared.add_min_good_frac(prep_opt)
    _shared.add_screen_scope(prep_opt)
    _shared.add_separation_bands(prep_opt)
    prep_opt.add_argument("--window-length", type=float, default=10.0,
                          help="Sliding-window length (s) for windowed SCI/PSP/GVTD QC series.")
    prep_opt.add_argument("--epoch-tmin", type=float, default=None,
                          help="Trial window start relative to event onset in s, for the "
                               "report's epoch figures and per-trial scoring; negative pulls "
                               "in a baseline. Omit for -5 to 25 s, which suits a single "
                               "trial and not a 60 s block.")
    prep_opt.add_argument("--epoch-tmax", type=float, default=None,
                          help="Trial window end relative to event onset in s. Given together "
                               "with --epoch-tmin, or neither.")
    prep_opt.add_argument("--epoch-chunk-duration", type=float, default=None, metavar="SECONDS",
                          help="Cut each task annotation into trials this long before any "
                               "epoching, so a block design gets one trial per piece instead "
                               "of one per block. A 240 s block at 25 s gives 9 trials and "
                               "drops the remainder, which is what MNE does. Nothing can "
                               "average a single 240 s trial, so without this the epoch "
                               "figures describe the start of each block.")
    prep_opt.add_argument("--by-condition", action="store_true",
                          help="Also write one QC report page per annotated condition, "
                               "beside the run's own, as desc-<condition>. Their numbers "
                               "are sliced out of the windowed pass already in the quality "
                               "record, so every condition sits on the same window grid and "
                               "the same filter as the run; nothing is cut and nothing is "
                               "measured again. Each page carries the scalar panel and the "
                               "channel table for that condition; the epoch, topography and "
                               "GLM panels are left blank, since the epoch window is set for "
                               "a trial and would describe the start of a block. Each page "
                               "screens on its own stretch, so its verdict is that "
                               "condition's; the run was processed under the run's.")
    prep_opt.add_argument("--gvtd-censor", nargs="?", const="long", default=None,
                          choices=("long", "short", "all"), metavar="SET",
                          help="Mark the frames GVTD flags as BAD_gvtd. The data is annotated, "
                               "never cut, so epoching drops the trials they overlap and a "
                               "threshold set too strictly is undone by rerunning. Off by "
                               "default: on a high-motion recording this can flag everything. "
                               "Takes the channel set to flag on, defaulting to long, the set "
                               "the analysis uses; `all` is the conservative choice, since a "
                               "movement seen only on the scalp channels still marks the "
                               "frame. It changes which frames are censored and nothing else: "
                               "the QC panels still draw every set, and the metrics that "
                               "decide whether a run is usable stay on the long channels.")
    prep_opt.add_argument("--gvtd-censor-n-std", type=float, default=10.0,
                          help="Threshold for --gvtd-censor, in left-tail SDs above the GVTD "
                               "mode. 10 is the lenient value used for censoring; the reports "
                               "score at 3, which censors far more.")
    prep_opt.add_argument("--gvtd-min-epoch-s", type=float, default=30.0,
                          help="Shortest surviving stretch --gvtd-censor keeps (s). Anything "
                               "shorter is censored with the artifacts around it, since a few "
                               "seconds between two of them cannot carry an analysis.")

    post = p.add_argument_group("postprocessing (requires --mode)")
    post.add_argument("--mode", choices=_MODE_CHOICES,
                      help="Postprocessing mode: denoise (bandpass, plus confound regression "
                           "when --short-channel or --drift-model is given), glm or rest.")
    post.add_argument("--config", type=Path,
                      help="TOML file providing post parameter values. CLI flags override TOML.")
    post.add_argument("--high-pass", type=float, help="High-pass filter cutoff in Hz, e.g. 0.01.")
    post.add_argument("--low-pass",  type=float, help="Low-pass filter cutoff in Hz, e.g. 0.5.")
    post.add_argument("--filter-method", choices=FILTER_METHODS,
                      help=f"Bandpass design, default {DEFAULT_FILTER_METHOD!r}. 'iir' is a zero-phase "
                           "Butterworth, what the fNIRS toolboxes use. 'fir' is a hamming-windowed "
                           "linear-phase filter, which needs 3.3 * sfreq / transition samples and "
                           "is refused when that is longer than the recording.")
    post.add_argument("--filter-order", type=int,
                      help=f"Butterworth order, default {DEFAULT_FILTER_ORDER}, ignored by "
                           "--filter-method fir. Applied with filtfilt, so the effective rolloff "
                           "is twice this and the cutoff sits at -6 dB.")
    post.add_argument("--resample-sfreq", type=float,
                      help="Target sampling rate in Hz after filtering, e.g. 2.0.")
    post.add_argument("--combine-runs", action=argparse.BooleanOptionalAction, default=False,
                      help="Concatenate multiple runs before postprocessing.")

    glm = p.add_argument_group("postprocessing: GLM and confound regression")
    glm.add_argument("--stim-dur", type=float,
                     help="Stimulus duration (s) for annotation-based events. Mutually exclusive with --events-path.")
    glm.add_argument("--hrf-model",   choices=_HRF_CHOICES,
                     help="HRF basis. 'spm + derivative' adds temporal derivative column.")
    glm.add_argument("--noise-model", choices=_NOISE_CHOICES, help="Residual autocorrelation model.")
    glm.add_argument("--drift-model", choices=_DRIFT_CHOICES,
                     help="Low-frequency drift regressors in design matrix. Required by "
                          "--mode glm and rest; optional in denoise, where the bandpass detrends.")
    glm.add_argument("--drift-high-pass", type=float, help="High-pass cutoff for cosine drift in Hz.")
    glm.add_argument("--drift-order", type=int, default=1,
                     help="Polynomial drift order (polynomial drift model only).")
    glm.add_argument("--fir-delays",
                     help="FIR delay bins in scans, comma-separated, e.g. '0,1,2,3,4,5' (only used when --hrf-model fir).")
    glm.add_argument("--short-channel", choices=_SHORT_CHANNEL_CHOICES,
                     help="Short-channel confound regressor strategy. Honoured by every mode: "
                          "glm fits it alongside the task, denoise and rest on its own.")
    glm.add_argument("--aux-regressors", action="store_true", default=None,
                     help="Add the recording's auxiliary channels (accelerometers, "
                          "gyroscopes, pulse and whatever else the device wrote to the "
                          "snirf aux group) to the confound regression, in every mode that "
                          "regresses. Preprocessing extracts them to desc-aux_timeseries.tsv.gz; "
                          "this reads that table, resamples it onto the data's time axis "
                          "with an anti-alias filter, and band-limits it to --high-pass / "
                          "--low-pass so regressors and data sit in one frequency band.")
    glm.add_argument("--aux-channels", nargs="+", action="extend",
                     help="Which aux channels to use, by the name the recording gives them. "
                          "Default is all of them. Name them when the aux group holds "
                          "channels that are not confounds, such as an event or trigger line.")
    glm.add_argument("--fc", action="store_true", default=None,
                     help="Also write the connectivity products rest mode writes, from "
                          "whatever the mode produced: glm correlates the task residual, so "
                          "what correlates is what the model did not explain; denoise "
                          "correlates its confound residual, or the bandpassed data itself "
                          "when no regression was asked for. Ignored by --mode rest, which "
                          "writes them anyway.")
    glm.add_argument("--events-path", type=Path,
                     help="Path to *_events.tsv. If omitted, extracted from snirf annotations.")
    glm.add_argument("--contrast-file", type=Path, help="TOML file defining GLM contrasts.")

    out = p.add_argument_group("output")
    out.add_argument("--no-report", action="store_true", help="Skip the QC HTML report.")
    out.add_argument("--roi-mapping", type=Path, default=None,
                     help="JSON file mapping ROI labels to lists of channel names, for ROI grouping in the report denoising carpet. Optional.")
    out.add_argument("--n-jobs", type=int, default=1, help="Parallel subject jobs.")
    out.add_argument("--work-dir", type=Path, help="Hash cache directory.")

    esc = p.add_argument_group("escape hatches")
    esc.add_argument("--ignore", nargs="+", action="extend", choices=_IGNORE_CHOICES,
                     help="Processing aspects to skip.")
    esc.add_argument("--skip-bids-validation", action="store_true", help="Skip BIDS validation.")
    esc.add_argument("--allow-cropped-input", action="store_true",
                     help="Run on a `fnirs-prep crop` tree, which is otherwise refused. "
                          "Every condition is then preprocessed on its own, and motion "
                          "correction and the bandpass each see one segment, which moves "
                          "both. Preprocess the uncut recording and crop the result instead "
                          "(`fnirs-prep crop --input-desc`).")
    esc.add_argument("--dry-run", action="store_true", help="Write pipeline script without executing.")
    esc.add_argument("--verbose", action="store_true")
    return p


def _check_epoch_window(args: argparse.Namespace) -> None:
    """Both edges or neither: one alone has no window to describe."""
    if (args.epoch_tmin is None) != (args.epoch_tmax is None):
        print("Error: --epoch-tmin and --epoch-tmax must be given together.", file=sys.stderr)
        raise SystemExit(1)


def _require(args: argparse.Namespace, *flags: str) -> None:
    """Exit non-zero naming the first flag the chosen level needs and did not get."""
    for flag in flags:
        if getattr(args, flag.lstrip("-").replace("-", "_")) is None:
            print(f"Error: Missing option '{flag}'.", file=sys.stderr)
            raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    level = args.analysis_level

    # checked before the workflow import: a bad command line should not first pay for mne
    _require(args, *_LEVEL_REQUIRES[level])
    _check_epoch_window(args)

    from fnirs_pipe.cli.workflows import run_group_level, run_participant_level

    runner = {
        "participant": run_participant_level,
        "group":       run_group_level,
    }[level]
    runner(vars(args))
