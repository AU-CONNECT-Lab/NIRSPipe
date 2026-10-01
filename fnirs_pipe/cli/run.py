"""fnirs-pipe CLI entry point (argparse, BIDS App convention)."""

import argparse
import re
import sys
from pathlib import Path

from fnirs_pipe import __version__
from fnirs_pipe.cli import _shared
from fnirs_pipe.pipeline.denoise import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_FILTER_ORDER,
    FILTER_METHODS,
)

_MOTION_CHOICES        = ["tddr", "wavelet", "none"]
_MODE_CHOICES          = ["denoise", "glm", "rest"]
_PRESETS_DIR           = Path(__file__).resolve().parent.parent / "data" / "presets"


def mode_defaults(mode: str | None) -> dict:
    """The post settings ``--mode`` fills in below ``--config`` and the command line."""
    if mode in (None, "none"):
        return {}
    from fnirs_pipe.utils import load_toml
    return load_toml(_PRESETS_DIR / f"{mode}.toml")


_HRF_CHOICES           = [
    "spm", "spm + derivative", "spm + derivative + dispersion",
    "glover", "glover + derivative", "glover + derivative + dispersion", "fir",
]
# what the GUI dropdown offers and the help lists; "auto" is an AR order of 4x the sampling rate
_NOISE_CHOICES        = ["auto", "ols", "ar1", "ar2", "ar3", "ar4", "ar5", "ar_irls"]
# the one rule, shared with the GUI's `pattern` so the browser refuses what argparse would
NOISE_MODEL_PATTERN    = r"ols|auto|ar[1-9][0-9]*|ar_irls(?:[1-9][0-9]*)?"
_DRIFT_CHOICES         = ["cosine", "polynomial", "none"]

def _noise_model(value: str) -> str:
    """``ols``, ``auto``, ``arN`` for any order the library will take, or ``ar_irls``.

    A pattern rather than a closed list; the dropdown and the help name the common ones.
    """
    if re.fullmatch(NOISE_MODEL_PATTERN, value):
        return value
    raise argparse.ArgumentTypeError(
        f"{value!r}: expected 'ols', 'auto', 'arN' (e.g. ar1 or ar16), or 'ar_irls'")


_SHORT_CHANNEL_CHOICES = ["none", "mean", "pca"]
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
    p.add_argument("analysis_level", choices=_LEVEL_CHOICES,
                   help="Processing level. `group` is the BIDS Apps name for the cohort pass; it writes desc-subjects_qc.tsv and desc-subjects_report.html and, where the tree holds groups, desc-groups_qc.tsv and desc-groups_report.html.")

    prep = p.add_argument_group("preprocessing (required for participant level)")
    # extend: accepts space-separated (--dpf 6 6) and repeated (--dpf 6 --dpf 6) forms.
    prep.add_argument("--dpf", nargs="+", type=float, action="extend",
                      help="Differential pathlength factor. One value or one per wavelength.")
    # no default: _LEVEL_REQUIRES makes it required at participant level
    _shared.add_sci_threshold(prep)

    sel = p.add_argument_group("subject / session / task selection")
    sel.add_argument("--participant-label", "--participant_label", nargs="+", action="extend",
                     type=_shared.BidsLabel, help="Subject ID(s) to process.")
    sel.add_argument("--session-label", "--session_label", nargs="+", action="extend",
                     type=_shared.BidsLabel, help="Session label(s) to process.")
    sel.add_argument("--task-label", "--task_label", nargs="+", action="extend",
                     type=_shared.BidsLabel, help="Task label(s) to process.")
    sel.add_argument("--bids-filter-file", "--bids_filter_file", type=Path,
                     help="JSON file with extra pybids query filters.")

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
    # the other screening line, optional unlike --sci-threshold
    _shared.add_psp_threshold(prep_opt)
    _shared.add_min_good_frac(prep_opt)
    _shared.add_screen_scope(prep_opt)
    _shared.add_separation_bands(prep_opt)
    prep_opt.add_argument("--window-length", type=float, default=10.0,
                          help="Sliding-window length (s) for windowed SCI/PSP/GVTD QC series.")
    prep_opt.add_argument("--epoch-tmin", type=float, default=None,
                          help="Trial window start relative to event onset in s, for the "
                               "report's epoch figures and per-trial scoring; negative pulls "
                               "in a baseline. Omitted, the epoch figures use -5 to 25 s, "
                               "which suits a single trial and not a 60 s block, and each "
                               "trial is scored over its event's own duration.")
    prep_opt.add_argument("--epoch-tmax", type=float, default=None,
                          help="Trial window end relative to event onset in s. Given together "
                               "with --epoch-tmin, or neither.")
    prep_opt.add_argument("--epoch-chunk-duration", type=float, default=None, metavar="SECONDS",
                          help="Cut each task annotation into trials this long before any "
                               "epoching, so a block design gets one trial per piece instead "
                               "of one per block. A 240 s block at 25 s gives 9 trials and "
                               "drops the remainder. Without this, the epoch figures of a "
                               "block design describe the start of each block.")
    prep_opt.add_argument("--epoch-single-trial", action="store_true",
                          help="Draw the epoch section even when no condition repeats, where "
                               "it is skipped by default because one trial per condition "
                               "leaves nothing to average. Pass this when the single trial is "
                               "the thing to look at, such as a block-onset transient. Has no "
                               "effect when the run carries no events or no event leaves room "
                               "for the window.")
    prep_opt.add_argument("--by-condition", action="store_true",
                          help="Also write one QC report page per annotated condition, "
                               "beside the run's own, as cond-<condition>. Their numbers "
                               "are sliced out of the windowed pass already in the quality "
                               "record, so every condition sits on the same window grid and "
                               "the same filter as the run; nothing is measured again. The "
                               "spectra, epoch preview and topography are redrawn on a copy "
                               "cropped to the condition, and the GLM panel shows that "
                               "condition's activation. Each page screens on its own "
                               "stretch, so its verdict is that condition's; the run was "
                               "processed under the run's.")
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
                               "shorter is censored with the artifacts around it.")

    post = p.add_argument_group("postprocessing (requires --mode)")
    post.add_argument("--mode", choices=_MODE_CHOICES,
                      help="Postprocessing mode: denoise (bandpass, plus confound regression "
                           "when --short-channel or --drift-model is given), glm or rest. "
                           "Each mode fills in its own defaults for the post settings, which "
                           "--config and then the command line override. The run record "
                           "names where each setting came from.")
    post.add_argument("--config", type=Path,
                      help="TOML file of post settings, over the mode's defaults. CLI flags "
                           "override both.")
    post.add_argument("--high-pass", type=float, help="High-pass filter cutoff in Hz, e.g. 0.01.")
    post.add_argument("--low-pass",  type=float, help="Low-pass filter cutoff in Hz, e.g. 0.5.")
    post.add_argument("--filter-method", choices=FILTER_METHODS,
                      help=f"Bandpass design, default {DEFAULT_FILTER_METHOD!r}. 'iir' is a zero-phase "
                           "Butterworth. 'fir' is a hamming-windowed "
                           "linear-phase filter, which needs 3.3 * sfreq / transition samples and "
                           "is refused when that is longer than the recording.")
    post.add_argument("--filter-order", type=int,
                      help=f"Butterworth order, default {DEFAULT_FILTER_ORDER}, ignored by "
                           "--filter-method fir. Applied with filtfilt, so the effective rolloff "
                           "is twice this and the cutoff sits at -6 dB.")
    post.add_argument("--resample-sfreq", type=float,
                      help="Target sampling rate in Hz after filtering, e.g. 2.0.")
    # None rather than False, here and on --drift-order, so a --config or mode value can show
    post.add_argument("--combine-runs", action=argparse.BooleanOptionalAction, default=None,
                      help="Concatenate multiple runs before postprocessing.")

    glm = p.add_argument_group("postprocessing: GLM and confound regression")
    glm.add_argument("--stim-dur", type=float,
                     help="Stimulus duration (s) for annotation-based events. Mutually exclusive with --events-path.")
    glm.add_argument("--hrf-model",   choices=_HRF_CHOICES,
                     help="HRF basis. 'spm + derivative' adds temporal derivative column.")
    glm.add_argument("--noise-model", type=_noise_model, metavar="MODEL",
                     help="Residual autocorrelation model, default 'auto'. 'ols', 'auto', "
                          "'arN' for any order, or 'ar_irls'. 'auto' is an AR order of 4x the "
                          "sampling rate. An order too low for the sampling rate leaves a "
                          "task contrast's t values too large. 'ar_irls' adds a robust norm "
                          "on top of the whitening, so "
                          "residual motion is down-weighted instead of fitted; 'ar_irlsN' "
                          "pins the largest order it may choose, which otherwise follows the "
                          "same 4x rule.")
    glm.add_argument("--drift-model", choices=_DRIFT_CHOICES,
                     help="Low-frequency drift regressors in design matrix. glm and rest "
                          "default to cosine; optional in denoise, where the bandpass detrends.")
    glm.add_argument("--drift-high-pass", type=float,
                     help="High-pass cutoff for cosine drift in Hz. Required by --mode glm, "
                          "since it depends on the design: at most 1/(2 x the slowest "
                          "condition repeat).")
    glm.add_argument("--drift-order", type=int,
                     help="Polynomial drift order (polynomial drift model only), default 1.")
    glm.add_argument("--fir-delays",
                     help="FIR delay bins in scans, comma-separated, e.g. '0,1,2,3,4,5' (only used when --hrf-model fir).")
    glm.add_argument("--short-channel", choices=_SHORT_CHANNEL_CHOICES,
                     help="Short-channel confound regressor strategy. Honoured by every mode: "
                          "glm fits it alongside the task, denoise and rest on its own. "
                          "'mean' gives one column per chromophore; 'pca' gives one column "
                          "per short channel, orthogonalised, which fits the same as entering "
                          "every short channel.")
    # withheld from --help pending evaluation; both still work when named explicitly
    glm.add_argument("--aux-regressors", action="store_true", default=None,
                     help=argparse.SUPPRESS)
    glm.add_argument("--aux-channels", nargs="+", action="extend",
                     help=argparse.SUPPRESS)
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
                     help="JSON file mapping ROI labels to lists of channel names. Groups the report denoising carpet, and in rest mode adds the ROI correlation matrix and one seed topography per ROI. Optional.")
    # --nprocs / --n_cpus are what the BIDS Apps interface calls this one
    out.add_argument("--n-jobs", "--n_jobs", "--nprocs", "--n_cpus",
                     dest="n_jobs", type=int, default=1, help="Parallel subject jobs.")
    out.add_argument("--work-dir", "--work_dir", type=Path, help="Hash cache directory.")

    esc = p.add_argument_group("escape hatches")
    esc.add_argument("--ignore", nargs="+", action="extend", choices=_IGNORE_CHOICES,
                     help="Processing aspects to skip.")
    esc.add_argument("--skip-bids-validation", "--skip_bids_validation", "--skip_bids_validator",
                     dest="skip_bids_validation", action="store_true",
                     help="Do not check the input with bids-validator. Files BIDS does not "
                          "recognise are left out either way.")
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


def _check_dirs(args: argparse.Namespace) -> None:
    """Refuse to write into the input dataset: the run would stamp it as a derivative."""
    _shared.refuse_output_in_input(args.bids_dir, args.output_dir, "fnirs-pipe")
    bids_dir = args.bids_dir.resolve()
    work_dir = args.work_dir.resolve() if args.work_dir else None
    if work_dir is not None and (work_dir == bids_dir or bids_dir in work_dir.parents):
        print("Error: the work directory is inside the input BIDS directory; choose one "
              "outside it.", file=sys.stderr)
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
    _check_dirs(args)

    from fnirs_pipe.cli.workflows import run_group_level, run_participant_level

    runner = {
        "participant": run_participant_level,
        "group":       run_group_level,
    }[level]
    runner(vars(args))
