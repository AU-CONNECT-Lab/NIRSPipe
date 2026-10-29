"""nirspipe CLI entry point (argparse, BIDS App convention)."""

import argparse
import re
import sys
from pathlib import Path

from nirspipe import __version__
from nirspipe.cli import _app_options
from nirspipe.cli import _shared
from nirspipe.pipeline.censor_fill import (
    DEFAULT_CENSOR_FILL,
    FILL_METHODS,
    LOMB_FMAX_HZ,
)
from nirspipe.pipeline.chromophores import CHROMOPHORES, DEFAULT_CHROMOPHORES
from nirspipe.pipeline.denoise import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_FILTER_ORDER,
    FILTER_METHODS,
)
from nirspipe.pipeline.population import BAND_KEYS
from nirspipe.utils import P_CORRECTIONS

_MODE_CHOICES          = ["denoise", "glm", "rest"]
_PRESETS_DIR           = Path(__file__).resolve().parent.parent / "data" / "presets"


def mode_defaults(mode: str | None) -> dict:
    """The post settings ``--mode`` fills in below ``--config`` and the command line."""
    if mode in (None, "none"):
        return {}
    from nirspipe.utils import load_toml
    return load_toml(_PRESETS_DIR / f"{mode}.toml")


_HRF_CHOICES           = [
    "spm", "spm + derivative", "spm + derivative + dispersion",
    "glover", "glover + derivative", "glover + derivative + dispersion", "fir",
]
# what the GUI dropdown offers and the help lists; "auto" is an AR order of 4x the sampling rate
_NOISE_CHOICES        = ["auto", "ols", "ar1", "ar2", "ar3", "ar4", "ar5", "ar_irls", "worsley"]
# the one rule the validator applies; the GUI leaves the check to the CLI it calls
NOISE_MODEL_PATTERN    = r"ols|auto|ar[1-9][0-9]*|ar_irls(?:[1-9][0-9]*)?|worsley(?:[1-9][0-9]*)?"
_DRIFT_CHOICES         = ["cosine", "polynomial", "none"]
# a plain non-negative decimal or none
CUTOFF_PATTERN         = r"[Nn][Oo][Nn][Ee]|[0-9]+(?:\.[0-9]*)?|\.[0-9]+"

def _cutoff(value: str) -> "float | str":
    """A frequency in Hz, or ``none``, kept as the string so it outranks a file layer's value."""
    value = value.strip()
    if not re.fullmatch(CUTOFF_PATTERN, value):
        raise argparse.ArgumentTypeError(f"{value!r}: expected a frequency in Hz or 'none'")
    return "none" if value.lower() == "none" else float(value)


def _drift_cutoff(value: str) -> "float | str":
    """A cosine drift cutoff in Hz, or ``auto`` to set it per run from the run's events."""
    if value.strip().lower() == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: expected a cutoff in Hz or 'auto'") from None


def _noise_model(value: str) -> str:
    """``ols``, ``auto``, ``arN`` for any order the library will take, ``ar_irls`` or ``worsley``.

    A pattern rather than a closed list; the dropdown and the help name the common ones.
    """
    if re.fullmatch(NOISE_MODEL_PATTERN, value):
        return value
    raise argparse.ArgumentTypeError(
        f"{value!r}: expected 'ols', 'auto', 'arN' (e.g. ar1 or ar16), 'ar_irls' or 'worsley'")


_SHORT_CHANNEL_CHOICES = ["auto", "none", "mean", "pca"]
_IGNORE_CHOICES        = ["events", "bids-validation", "channel-status"]
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
        prog="nirspipe",
        description="BIDS-compatible fNIRS preprocessing and postprocessing pipeline. "
                    "Prep always runs; add --mode to run postprocessing. Dyad analysis "
                    "lives in nirspipe-hyper, which reads what this writes.",
    )
    p.add_argument("--version", action="version", version=f"nirspipe {__version__}")

    p.add_argument("bids_dir",   type=Path, help="BIDS dataset directory.")
    p.add_argument("output_dir", type=Path, help="Output directory (BIDS Derivatives).")
    p.add_argument("analysis_level", choices=_LEVEL_CHOICES,
                   help="Processing level. `group` is the BIDS Apps name for the cohort pass; it writes desc-subjects_qc.tsv and desc-subjects_report.html and, where the tree holds groups, desc-groups_qc.tsv and desc-groups_report.html.")

    prep = p.add_argument_group("preprocessing (required for participant level)")
    _app_options.add_dpf(prep)
    # no default: _LEVEL_REQUIRES makes it required at participant level
    _app_options.add_sci_threshold(prep)

    sel = p.add_argument_group("subject / session / task selection")
    sel.add_argument("--participant-label", "--participant_label", nargs="+", action="extend",
                     type=_shared.BidsLabel, help="Subject ID(s) to process.")
    sel.add_argument("--session-label", "--session_label", nargs="+", action="extend",
                     type=_shared.BidsLabel, help="Session label(s) to process.")
    sel.add_argument("--task-label", "--task_label", nargs="+", action="extend",
                     type=_shared.BidsLabel, help="Task label(s) to process.")
    _app_options.add_bids_filter_file(sel)

    prep_opt = p.add_argument_group("preprocessing (optional)")
    prep_opt.add_argument("--motion-correction", choices=_app_options.MOTION_CHOICES,
                          default="tddr",
                          help="Motion correction method.")
    _app_options.add_template(prep_opt)
    _app_options.add_population(prep_opt, "the four cardiac and respiration band edges")
    _app_options.add_motion_settings(prep_opt)
    _app_options.add_reviewer_marks(
        prep_opt, bad_spans_note=" Filled by --censor-fill before any filter.")
    prep_opt.add_argument("--cardiac-l-freq", type=float,
                          help="Lower bound of cardiac band in Hz (required at participant level "
                               "unless --population sets it). Adult resting ~0.7; children and "
                               "infants higher (e.g. 1.0-1.7).")
    prep_opt.add_argument("--cardiac-h-freq", type=float,
                          help="Upper bound of cardiac band in Hz (required at participant level "
                               "unless --population sets it). Adult resting ~1.5; children and "
                               "infants higher (e.g. 2.5-3.0).")
    prep_opt.add_argument("--resp-l-freq", type=float,
                          help="Lower bound of respiration band in Hz (required at participant "
                               "level unless --population sets it). Adult ~0.1; infants higher.")
    prep_opt.add_argument("--resp-h-freq", type=float,
                          help="Upper bound of respiration band in Hz (required at participant "
                               "level unless --population sets it). Adult ~0.5; infants higher.")
    # the other screening line, optional unlike --sci-threshold
    _app_options.add_psp_threshold(prep_opt)
    _app_options.add_min_good_frac(prep_opt)
    _app_options.add_screen_scope(prep_opt)
    _app_options.add_separation_bands(prep_opt)
    prep_opt.add_argument("--window-length", type=float, default=10.0,
                          help="Sliding-window length (s) for windowed SCI/PSP/GVTD QC series.")
    prep_opt.add_argument("--epoch-tmin", type=float, default=None,
                          help="Trial window start relative to event onset in s, for the epoch "
                               "figures, the glm block average and per-trial scoring; negative "
                               "pulls in a baseline. Omitted, the figures and the block average "
                               "use -5 to 25 s and each trial is scored over its own duration.")
    prep_opt.add_argument("--epoch-tmax", type=float, default=None,
                          help="Trial window end relative to event onset in s. Given together "
                               "with --epoch-tmin, or neither.")
    prep_opt.add_argument("--epoch-chunk-duration", type=float, default=None, metavar="SECONDS",
                          help="Cut each task annotation into trials this long before epoching, "
                               "so a block design gets one trial per piece: a 240 s block at 25 "
                               "s gives 9, the remainder dropped.")
    prep_opt.add_argument("--by-condition", action="store_true",
                          help="Also write one QC report page per annotated condition, as "
                               "cond-<condition>. Its numbers are sliced from the run's "
                               "windowed pass, its spectra and epochs are redrawn on the "
                               "condition, and its verdict is the condition's; the data were "
                               "processed under the run's.")
    prep_opt.add_argument("--gvtd-censor", nargs="?", const="long", default=None,
                          choices=("long", "short", "all"), metavar="SET",
                          help="Mark the frames GVTD flags as BAD_gvtd on the given channel "
                               "set: long if none is named, or `all` to also catch movement "
                               "seen only on short channels. Annotated, never cut, so epoching "
                               "drops the trials they overlap. Off by default: on a high-motion "
                               "recording it can flag everything.")
    prep_opt.add_argument("--gvtd-censor-n-std", type=float, default=10.0,
                          help="Threshold for --gvtd-censor, in left-tail SDs above the GVTD "
                               "mode. 10 is the lenient value used for censoring, and the "
                               "reports score at the same 10.")
    prep_opt.add_argument("--gvtd-min-epoch-s", type=float, default=30.0,
                          help="Shortest surviving stretch --gvtd-censor keeps (s). Anything "
                               "shorter is censored with the artifacts around it.")

    post = p.add_argument_group("postprocessing (requires --mode)")
    post.add_argument("--mode", choices=_MODE_CHOICES,
                      help="Postprocessing mode: denoise (bandpass, plus confound regression "
                           "when short-channel regression runs or --drift-model is given), "
                           "glm or rest. "
                           "Each mode fills in its own defaults for the post settings, which "
                           "--config and then the command line override. The run record "
                           "names where each setting came from.")
    post.add_argument("--config", type=Path,
                      help="A run record (sub-<label>/logs/sub-<label>*.toml) or a TOML of the same shape: "
                           "[prep] and [post] keyed by flag name. Reruns with its settings; "
                           "flags given here override it, and its post settings override the "
                           "mode's defaults.")
    post.add_argument("--high-pass", type=_cutoff,
                      help="High-pass filter cutoff in Hz, e.g. 0.01, or none to switch off the "
                           "mode's default.")
    post.add_argument("--low-pass",  type=_cutoff,
                      help="Low-pass filter cutoff in Hz, e.g. 0.5, or none to switch off the "
                           "mode's default.")
    post.add_argument("--filter-method", choices=FILTER_METHODS,
                      help=f"Bandpass design, default {DEFAULT_FILTER_METHOD!r}. 'iir' is a zero-phase "
                           "Butterworth. 'fir' is a hamming-windowed "
                           "linear-phase filter, which needs 3.3 * sfreq / transition samples and "
                           "is refused when that is longer than the recording.")
    post.add_argument("--filter-order", type=int,
                      help=f"Butterworth order, default {DEFAULT_FILTER_ORDER}, ignored by "
                           "--filter-method fir. Applied with filtfilt, so the effective rolloff "
                           "is twice this and the cutoff sits at -6 dB.")
    post.add_argument("--export-mean-filled", action="store_true", default=None,
                      help="Also write desc-meanfilled beside the final residual: every frame "
                           "inside a BAD_ span set to its channel's kept mean, annotations "
                           "kept, for tools that ignore annotations. Not for ALFF, fALFF or a "
                           "GLM: the filled frames are placeholders.")
    post.add_argument("--censor-fill", choices=FILL_METHODS,
                      help=f"How corrupted BAD_ spans (all but BAD_unselected) are filled "
                           f"before the bandpass, default {DEFAULT_CENSOR_FILL!r}: 'linear' "
                           f"joins the kept samples either side, 'spline' fits one cubic spline "
                           f"through them, 'lomb' rebuilds the span from a Lomb-Scargle fit up "
                           f"to {LOMB_FMAX_HZ:g} Hz. The spans stay marked.")
    post.add_argument("--resample-sfreq", type=float,
                      help="Target sampling rate in Hz after filtering, e.g. 2.0.")
    # None rather than False, here and on --drift-order, so a --config or mode value can show
    post.add_argument("--combine-runs", action=argparse.BooleanOptionalAction, default=None,
                      help="After postprocessing, join each task's runs, each z-scored per "
                           "channel over its kept frames, into one series written without the "
                           "run entity, and compute FC on it. A channel joins only from the runs "
                           "where it is good. ALFF and GLM tables stay per run.")
    post.add_argument("--chromophores", nargs="+", action="extend", choices=CHROMOPHORES,
                      help=f"Chromophores the GLM tables, block average, FC and ALFF carry, "
                           f"default {' '.join(DEFAULT_CHROMOPHORES)}. hbt, total haemoglobin, "
                           f"is HbO + HbR of each channel; the stage files keep HbO and HbR.")

    glm = p.add_argument_group("postprocessing: GLM and confound regression")
    glm.add_argument("--stim-dur", type=float,
                     help="Stimulus duration (s) for annotation-based events, above 0. Mutually "
                          "exclusive with --events-path.")
    glm.add_argument("--hrf-model",   choices=_HRF_CHOICES,
                     help="HRF basis. 'spm + derivative' adds temporal derivative column.")
    glm.add_argument("--noise-model", type=_noise_model, metavar="MODEL",
                     help="Residual autocorrelation model, default 'auto' in glm mode and "
                          "'ols' in rest and denoise: 'ols', 'auto' (an AR "
                          "order of 4x the sampling rate), 'arN' for order N, or 'ar_irls', "
                          "which adds a robust norm so residual motion is down-weighted "
                          "('ar_irlsN' caps its order), or 'worsley', auto's order with the "
                          "coefficients corrected for the bias the fit leaves in the "
                          "residual ('worsleyN' for order N). Too low an order inflates task "
                          "t values.")
    glm.add_argument("--drift-model", choices=_DRIFT_CHOICES,
                     help="Low-frequency drift regressors in design matrix. glm and rest "
                          "default to cosine; optional in denoise, where the bandpass detrends.")
    glm.add_argument("--drift-high-pass", type=_drift_cutoff, metavar="HZ|auto",
                     help="High-pass cutoff for cosine drift in Hz. Required by --mode glm, "
                          "since it depends on the design: at most 1/(2 x the slowest "
                          "condition repeat). 'auto' (glm only) computes that per run from its "
                          "events, capped at 0.01 Hz and never below --high-pass.")
    glm.add_argument("--drift-order", type=int,
                     help="Polynomial drift order (polynomial drift model only), default 1.")
    glm.add_argument("--fir-delays",
                     help="FIR delay bins in scans, comma-separated, e.g. '0,1,2,3,4,5' (only used when --hrf-model fir).")
    glm.add_argument("--short-channel", choices=_SHORT_CHANNEL_CHOICES,
                     help="Short-channel confound regressor strategy, default auto: pca where "
                          "the montage has a channel at or under --short-max-dist, none where "
                          "it has none. Honoured by every mode: "
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
                     help="Also write the connectivity products rest mode writes, from the "
                          "residual this mode leaves: the task residual in glm, the confound "
                          "residual in denoise. Ignored by --mode rest, which writes them "
                          "anyway.")
    glm.add_argument("--min-time", type=float,
                     help="Seconds outside every BAD_ span a run needs before FC and ALFF are "
                          "computed; default 0, off. A run under it gets no FC or ALFF file, "
                          "and its residual's sidecar says why. The GLM fit is not affected.")
    glm.add_argument("--events-path", type=Path,
                     help="Path to *_events.tsv. If omitted, extracted from snirf annotations.")
    glm.add_argument("--contrast-file", type=Path,
                     help="GLM contrasts: a TSV or CSV with one row per contrast, its name in a "
                          "contrast column and a weight under each condition's column (an "
                          "empty cell leaves it out), or a TOML of contrast = {condition = "
                          "weight}.")
    glm.add_argument("--p-correction", choices=P_CORRECTIONS,
                     help="Multiple-comparison correction of the GLM and contrast p values "
                          "(default none), within one regressor or contrast and one "
                          "chromophore, over the long channels not marked bad. p_value, "
                          "two-sided, is always written; a method adds a p_<method> column "
                          "beside it, and the family column says how many tests it ran over.")

    out = p.add_argument_group("output")
    out.add_argument("--no-report", action="store_true", help="Skip the QC HTML report.")
    out.add_argument("--roi-mapping", type=Path, default=None,
                     help="JSON file mapping ROI labels to lists of channel names. Groups the report denoising carpet and lists each region's long channels kept, and in rest mode adds the ROI correlation matrix and one seed topography per ROI; at group level the cohort page counts the runs per region that kept none. Optional.")
    _shared.add_nprocs(out)

    esc = p.add_argument_group("escape hatches")
    esc.add_argument("--ignore", nargs="+", action="extend", choices=_IGNORE_CHOICES,
                     help="Processing aspects to skip. 'channel-status': do not mark bad "
                          "the channels each run's BIDS channels.tsv marks bad in its status "
                          "column, which are otherwise unioned with --bad-channels.")
    esc.add_argument("--skip-bids-validation", "--skip_bids_validation", "--skip_bids_validator",
                     dest="skip_bids_validation", action="store_true",
                     help="Do not check the input with bids-validator. Files BIDS does not "
                          "recognise are left out either way.")
    esc.add_argument("--allow-cropped-input", action="store_true",
                     help="Run on a `nirspipe-prep crop` tree, which is otherwise refused. "
                          "Every condition is then preprocessed on its own, and motion "
                          "correction and the bandpass each see one segment, which moves "
                          "both. Preprocess the uncut recording and crop the result instead "
                          "(`nirspipe-prep crop --input-desc`).")
    esc.add_argument("--dry-run", action="store_true",
                     help="Write each subject's run record, then stop without processing.")
    esc.add_argument("--verbose", action="store_true")
    return p


def _check_epoch_window(args: argparse.Namespace) -> None:
    """Both edges or neither: one alone has no window to describe."""
    if (args.epoch_tmin is None) != (args.epoch_tmax is None):
        print("Error: --epoch-tmin and --epoch-tmax must be given together.", file=sys.stderr)
        raise SystemExit(1)


def _check_dirs(args: argparse.Namespace) -> None:
    """Refuse to write into the input dataset: the run would stamp it as a derivative."""
    _shared.refuse_output_in_input(args.bids_dir, args.output_dir, "nirspipe")


# ---- --config: a run record, or any file of its shape ----

# a section sets the flags of the parser groups whose title starts with these; roi_mapping
# sits under output but changes the ROI results, so it travels with the post settings
_CONFIG_GROUPS = {"prep": "preprocessing", "post": "postprocessing"}
_CONFIG_EXTRA = {"prep": (), "post": ("roi_mapping",)}
_EXECUTION_KEYS = ("participant_label", "session_label", "task_label", "bids_filter_file",
                   "ignore", "allow_cropped_input")
# describe the run that wrote the record rather than set the next one
_RECORD_ONLY = ("environment", "prep_sources", "post_sources", "outcome")


def config_keys() -> dict[str, tuple[str, ...]]:
    """The flags each section of a --config file may set, by section."""
    parser = _build_parser()
    keys = {section: tuple(a.dest for g in parser._action_groups if g.title.startswith(prefix)
                           for a in g._group_actions if a.dest != "config") + _CONFIG_EXTRA[section]
            for section, prefix in _CONFIG_GROUPS.items()}
    keys["execution"] = _EXECUTION_KEYS
    return keys


def read_config(path: Path) -> dict:
    """--config as the settings of each section, plus the version that wrote it.

    The file is a run record or anything of its shape: [prep] and [post] keyed by flag
    name, [execution] for the selection. A record's other fields describe its own run and
    are skipped; an unknown key in [prep] or [post], or a key outside any section, is
    refused, since a misspelt setting would otherwise drop out without a word.
    """
    from nirspipe.utils import load_toml

    raw = load_toml(path)
    stray = [k for k, v in raw.items() if not isinstance(v, dict)]
    if stray:
        print(f"Error: {path}: {stray[0]} sits outside any section; --config takes [prep] and "
              "[post] sections keyed by flag name, as a run record does.", file=sys.stderr)
        raise SystemExit(1)
    keys = config_keys()
    unknown = sorted(set(raw) - set(keys) - set(_RECORD_ONLY))
    if unknown:
        print(f"Error: {path}: no section [{unknown[0]}]; expected [prep], [post] or "
              "[execution].", file=sys.stderr)
        raise SystemExit(1)
    actions = {a.dest: a for a in _build_parser()._actions}
    out: dict = {"version": raw.get("environment", {}).get("nirspipe_version")}
    for section, allowed in keys.items():
        settings = raw.get(section, {})
        if section != "execution":
            bad = [k for k in settings if k not in allowed]
            if bad:
                print(f"Error: {path}: [{section}] has no setting {bad[0]!r}.", file=sys.stderr)
                raise SystemExit(1)
        out[section] = {k: _shared.coerce_setting(actions[k], k, v, path)
                        for k, v in settings.items() if k in allowed}
    return out


def _apply_config(args: argparse.Namespace, argv: list[str] | None) -> None:
    """Fill what the command line left out from --config; the post settings layer in later."""
    config = read_config(args.config)
    if config["version"] and config["version"] != __version__:
        print(f"Warning: {args.config} was written by nirspipe {config['version']} and this is "
              f"{__version__}; a setting whose meaning or default changed in between runs as "
              "this version reads it.", file=sys.stderr)
    given = _shared.typed_flags(_build_parser(), argv)
    taken = {**config["prep"], **config["execution"],
             **{k: v for k, v in config["post"].items() if k in ("mode", "roi_mapping")}}
    for key, value in taken.items():
        if key not in given:
            setattr(args, key, value)
    # the record's post_sources reads this to tell a mode from the file from a typed one
    args.config_applied = sorted(k for k in taken if k not in given)


def _require(args: argparse.Namespace, *flags: str) -> None:
    """Exit non-zero naming the first flag the chosen level needs and did not get."""
    for flag in flags:
        dest = flag.lstrip("-").replace("-", "_")
        if getattr(args, dest) is None:
            hint = " (or --population)" if dest in (*BAND_KEYS, "dpf") else ""
            print(f"Error: Missing option '{flag}'{hint}.", file=sys.stderr)
            raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.config is not None:
        _apply_config(args, argv)
    level = args.analysis_level
    # below the command line and --config, so either overrides any value it fills
    args.from_population = _app_options.fill_population(args, BAND_KEYS)

    # checked before the workflow import: a bad command line should not first pay for mne
    _require(args, *_LEVEL_REQUIRES[level])
    _check_epoch_window(args)
    _app_options.check_keep_spans_from(args.keep_spans, args.keep_spans_from)
    _app_options.check_motion_settings(args.motion_correction, args.motion_std_thresh, args.wavelet_iqr)
    _check_dirs(args)

    from nirspipe.cli.workflows import run_group_level, run_participant_level

    runner = {
        "participant": run_participant_level,
        "group":       run_group_level,
    }[level]
    runner(vars(args))
