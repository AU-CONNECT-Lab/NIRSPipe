"""Translate a page's form into an argv for the CLI that owns it.

The batch editing is `nirspipe-prep`, the dyad analysis is `nirspipe-hyper`, the aggregate reports
are `nirspipe-qc`, the BIDS conversion is `nirspipe-recon`. Kept beside the pages rather than derived from the parsers, because argparse
can say a flag exists but not which widget should fill it. `tests/gui/test_gui_cli_surface.py` is
what stops the two surfaces drifting apart.
"""

from __future__ import annotations

from nirspipe.pipeline.motion import IQR_FACTOR_DEFAULTS, STD_THRESH_DEFAULTS

# nirspipe-qc commands whose whole argument list is one output_dir
_AGGREGATE = ("cohort", "cohort-hyper", "provenance")

# Each dyad command is its own console script, because they do not take the same
# positionals: `run` and `pair-null` read each member's recording out of a source tree, the
# rest only re-read tables this package already wrote. The page keeps the short names.
_HYPER_PROG = {
    "run":        "nirspipe-hyper",
    "pair-null":  "nirspipe-hyper-pairnull",
    "group-null": "nirspipe-hyper-groupnull",
    "band":       "nirspipe-hyper-band",
    "merge":      "nirspipe-hyper-merge",
    "index":      "nirspipe-hyper-index",
}
_HYPER = tuple(_HYPER_PROG)

# the two that open subject data, and so need the source tree as well as the output one
_HYPER_READS_SUBJECTS = ("run", "pair-null")


def _num(flag: str, value) -> list[str]:
    return [flag, str(value)] if value is not None else []


def _text(flag: str, value) -> list[str]:
    return [flag, str(value)] if value else []


def _split(flag: str, value) -> list[str]:
    """A space-separated box feeds an nargs="+" flag: the flag once, then every value."""
    items = (value or "").split()
    return [flag, *items] if items else []


def build_qc_args(command: str, opts: dict) -> list[str]:
    if command in _AGGREGATE:
        return ["nirspipe-qc", command, opts["output_dir"]]

    args = [_HYPER_PROG[command]]
    if command in _HYPER_READS_SUBJECTS:
        args.append(opts["derivatives_dir"])
    args += [opts["output_dir"], "group"]

    # the band and the COI switch are shared with `run`; only the suffix is this one's own
    if command == "band":
        args += _num("--wtc-band-fmin", opts.get("band_fmin"))
        args += _num("--wtc-band-fmax", opts.get("band_fmax"))
        # the switch turns masking off, masking being the default
        if "band_no_mask_coi" in (opts.get("band_flags") or []):
            args.append("--no-wtc-mask-coi")
        args += _text("--wtc-suffix", opts.get("band_suffix"))
        return args

    # the re-paired null runs over the finished tree, so it reuses the run form's selection
    # and reads everything else off the real tables rather than off this page
    if command == "pair-null":
        args += _text("--pairs-csv", opts.get("pairs_csv"))
        args += _text("--group-id", opts.get("group_id"))
        args += _split("--task-label", opts.get("hyper_task"))
        args += _text("--roi-mapping", opts.get("roi_mapping"))
        args += _text("--chroma", opts.get("wtc_chroma"))
        args += _text("--wtc-pair-pool", opts.get("wtc_pair_pool"))
        args += _num("--wtc-pair-max", opts.get("wtc_pair_max"))
        # unticked, the crossing follows the real table
        if "wtc_pair_homologous" in (opts.get("pair_flags") or []):
            args.append("--no-wtc-pair-cross")
        return args

    # reads the draws `pair-null` wrote, so the band, the mask and the window are whatever
    # those tables carry. Only what to read and how finely to resample is this form's
    if command == "group-null":
        args += _text("--task-label", opts.get("gn_task"))
        # nargs="+" without extend: one flag carries every value; none sent reads all on disk
        for flag, key in (("--chroma", "gn_chroma"), ("--null", "gn_null")):
            if opts.get(key):
                args += [flag, *opts[key]]
        args += _text("--roi-mapping", opts.get("gn_roi_mapping"))
        args += _num("--n-resample", opts.get("gn_resample"))
        args += _num("--seed", opts.get("gn_seed"))
        args += _text("--p-correction", opts.get("gn_p_correction"))
        args += _text("--isc-test", opts.get("gn_isc_test"))
        return args

    if command == "run":
        args += _text("--pairs-csv", opts.get("pairs_csv"))
        args += _text("--group-id", opts.get("group_id"))
        args += _text("--desc", opts.get("desc"))
        args += _text("--roi-mapping", opts.get("roi_mapping"))
        args += _num("--wtc-fmin", opts.get("wtc_fmin"))
        args += _num("--wtc-fmax", opts.get("wtc_fmax"))
        args += _num("--band-fmin", opts.get("shared_band_fmin"))
        args += _num("--band-fmax", opts.get("shared_band_fmax"))
        args += _num("--wtc-band-fmin", opts.get("wtc_band_fmin"))
        args += _num("--wtc-band-fmax", opts.get("wtc_band_fmax"))
        args += _num("--wtc-mc-count", opts.get("wtc_mc_count"))
        args += _num("--wtc-seed", opts.get("wtc_seed"))
        args += _num("--wtc-phase-null", opts.get("wtc_phase_null"))
        args += _num("--isc-threshold", opts.get("isc_threshold"))
        args += _num("--isc-fmin", opts.get("isc_fmin"))
        args += _num("--isc-fmax", opts.get("isc_fmax"))
        args += _num("--wtc-roi-min-channels", opts.get("wtc_roi_min_channels"))
        args += _num("--wtc-window-s", opts.get("wtc_window_s"))
        args += _num("--wtc-whiten", opts.get("wtc_whiten"))
        args += _num("--isc-whiten-s", opts.get("isc_whiten"))
        args += _num("--isc-max-lag", opts.get("isc_max_lag"))
        args += _num("--isc-phase-null", opts.get("isc_phase_null"))
        args += _text("--wtc-chroma", opts.get("wtc_chroma"))
        args += _split("--task-label", opts.get("hyper_task"))
        flags = opts.get("hyper_flags") or []
        if "wtc_significance" in flags:
            args.append("--wtc-significance")
        if "wtc_no_mask_coi" in flags:
            args.append("--no-wtc-mask-coi")
        # crossing is the default, so an unticked switch has to say so
        args.append("--channel-cross" if "wtc_channel_cross" in flags
                    else "--no-channel-cross")
        args.append("--wtc-phase-null-cross" if "wtc_phase_null_cross" in flags
                    else "--no-wtc-phase-null-cross")
        # the switch turns the per-condition pass off, that pass being the default
        if "no_by_condition" in flags:
            args.append("--no-by-condition")
        if "check_only" in flags:
            args.append("--check-only")
        if "bads_subject" in flags:
            args += ["--bads-scope", "subject"]
        if "wtc_save_maps" in flags:
            args.append("--wtc-save-maps")
        if "no_align" in flags:
            args.append("--no-align")
        if "normalize" in flags:
            args.append("--normalize")
        args += _num("--tstart", opts.get("tstart"))
        args += _num("--tend", opts.get("tend"))

    return args


def missing(command: str, opts: dict) -> str | None:
    """The CLI would fail on these anyway; saying so here costs no subprocess."""
    if not opts.get("output_dir"):
        return "Output directory is required."
    if command == "band":
        if opts.get("band_fmin") is None or opts.get("band_fmax") is None:
            return "band needs both a band start and end."
        return None
    if command in _HYPER_READS_SUBJECTS and not opts.get("derivatives_dir"):
        return f"{command} reads each member's recording, so it needs a source derivatives directory."
    if command == "run" and not opts.get("pairs_csv"):
        return "The dyad analysis needs a pairs CSV."
    return None


# ---- nirspipe-prep ----

# the four batch operations, named as the page's radio names them
_PREP_OPERATIONS = ("markers", "crop", "keep_spans", "hyper_align")


def _selection(opts: dict) -> list[str]:
    args = ["--participant-label", *[str(s) for s in (opts.get("subjects") or [])]]
    args += _text("--session-label", opts.get("ses"))
    args += _text("--task-label", opts.get("task"))
    args += _text("--run-label", opts.get("run"))
    args += _num("--n-jobs", opts.get("n_jobs"))
    return args + _skip_validation(opts)


def _skip_validation(opts: dict) -> list[str]:
    return ["--skip-bids-validation"] if opts.get("skip_bids_validation") else []


def build_prep_args(operation: str, opts: dict) -> list[str]:
    bids, deriv = opts.get("bids_dir"), opts.get("deriv_dir")

    # align selects its subjects through the group CSV, so it takes no selection flags
    if operation == "hyper_align":
        return ["nirspipe-prep", "align", bids, deriv,
                *_text("--group-csv", opts.get("group_csv")), *_skip_validation(opts)]

    if operation == "markers":
        args = ["nirspipe-prep", "edit-markers", "apply", bids, deriv, *_selection(opts)]
        marker_op = opts.get("marker_op")
        if marker_op == "shift":
            args += _num("--shift", opts.get("shift"))
        elif marker_op == "set_duration":
            args += _num("--set-duration", opts.get("set_duration"))
        elif marker_op == "rename":
            pairs = opts.get("rename") or []
            if pairs:
                args += ["--rename", *pairs]
        return args

    if operation == "keep_spans":
        args = ["nirspipe-prep", "keep-spans", bids, opts.get("keep_spans_path"),
                *_selection(opts)]
        names = opts.get("conditions") or []
        if names:
            args += ["--conditions", *names]
        return args + _num("--duration", opts.get("keep_duration"))

    args = ["nirspipe-prep", "crop", bids, deriv, *_selection(opts)]
    if opts.get("crop_mode") == "multi":
        args += _text("--segments-path", opts.get("segments_path"))
        # the negative half is the default, so only the positive is ever sent
        if opts.get("combine"):
            args.append("--combine")
    else:
        args += _num("--tmin", opts.get("crop_tmin"))
        args += _num("--tmax", opts.get("crop_tmax"))
    return args


def missing_prep(operation: str, opts: dict) -> str | None:
    if not opts.get("bids_dir") or not opts.get("deriv_dir"):
        return "Set BIDS and derivatives directories."

    if operation == "hyper_align":
        return None if opts.get("group_csv") else "Set Group CSV path."

    if not (opts.get("subjects") or []):
        return "Select at least one subject."

    if operation == "markers":
        marker_op = opts.get("marker_op")
        if marker_op == "shift" and opts.get("shift") is None:
            return "Enter a shift value."
        if marker_op == "set_duration" and opts.get("set_duration") is None:
            return "Enter a duration."
        if marker_op == "rename" and not (opts.get("rename") or []):
            return "Add at least one rename pair."
        return None

    if operation == "keep_spans":
        return None if opts.get("conditions") else "Enter at least one condition name."

    if opts.get("crop_mode") == "multi":
        return None if opts.get("segments") else "Add at least one segment."
    if opts.get("crop_tmin") is None and opts.get("crop_tmax") is None:
        return "Enter tmin or tmax."
    return None


# ---- nirspipe-qc raw reports ----

# the Raw QC page's two commands
_RAW_QC = ("prep-raw", "hyper-raw")


def _screening(opts: dict) -> list[str]:
    args = _num("--sci-threshold", opts.get("sci_threshold"))
    args += _num("--psp-threshold", opts.get("psp_threshold"))
    args += _num("--min-good-frac", opts.get("min_good_frac"))
    args += _text("--screen-scope", opts.get("screen_scope"))
    args += _num("--dpf", opts.get("dpf"))
    args += _num("--cardiac-l-freq", opts.get("cardiac_l"))
    args += _num("--cardiac-h-freq", opts.get("cardiac_h"))
    args += _num("--short-max-dist", opts.get("short_max_dist"))
    args += _num("--long-min-dist", opts.get("long_min_dist"))
    args += _num("--long-max-dist", opts.get("long_max_dist"))
    args += _split("--participant-label", opts.get("subjects"))
    args += _split("--session-label", opts.get("ses"))
    args += _split("--task-label", opts.get("task"))
    args += _text("--roi-mapping", opts.get("roi_mapping"))
    if opts.get("ignore_channel_status"):
        args += ["--ignore", "channel-status"]
    if opts.get("skip_bids_validation"):
        args.append("--skip-bids-validation")
    return args


def _motion(opts: dict) -> list[str]:
    method = opts.get("motion_correction")
    if not method or method == "none":
        return []
    args = ["--motion-correction", method]
    # only to a method that reads it: the CLI refuses the setting anywhere else
    if opts.get("motion_std_thresh") is not None and method in STD_THRESH_DEFAULTS:
        args += ["--motion-std-thresh", str(opts["motion_std_thresh"])]
    if opts.get("wavelet_iqr") is not None and method in IQR_FACTOR_DEFAULTS:
        args += ["--wavelet-iqr", str(opts["wavelet_iqr"])]
    return args


def build_raw_qc_args(command: str, opts: dict) -> list[str]:
    args = ["nirspipe-qc", command, opts.get("bids_dir"), opts.get("output_dir")]

    if command == "prep-raw":
        args += _screening(opts)
        args += _num("--window-length", opts.get("window_length"))
        # both edges or neither: the CLI refuses half a window
        if opts.get("epoch_tmin") is not None and opts.get("epoch_tmax") is not None:
            args += _num("--epoch-tmin", opts["epoch_tmin"])
            args += _num("--epoch-tmax", opts["epoch_tmax"])
        if opts.get("epoch_qc"):
            args.append("--epoch-qc")
        if opts.get("by_condition"):
            args.append("--by-condition")
        return args + _motion(opts)

    args.append("group")
    args += _text("--derivatives-dir", opts.get("derivatives_dir"))
    args += _text("--pairs-csv", opts.get("pairs_csv"))
    args += _text("--group-id", opts.get("group_id"))
    args += _screening(opts)
    if opts.get("normalize"):
        args.append("--normalize")
    if opts.get("no_align"):
        args.append("--no-align")
    args += _num("--tstart", opts.get("tstart"))
    args += _num("--tend", opts.get("tend"))
    return args


def missing_raw_qc(command: str, opts: dict) -> str | None:
    if not opts.get("bids_dir") or not opts.get("output_dir"):
        return "Set the BIDS and output directories."
    # these four have no defaults anywhere, by design
    if opts.get("sci_threshold") is None:
        return "SCI threshold is required."
    if opts.get("dpf") is None:
        return "DPF is required."
    if opts.get("cardiac_l") is None or opts.get("cardiac_h") is None:
        return "Both ends of the cardiac band are required."
    if command == "prep-raw":
        return None if (opts.get("subjects") or "").split() else "Enter at least one subject."
    return None if opts.get("pairs_csv") else "Set the group CSV."


# ---- nirspipe-recon ----

def build_recon_args(row: dict, bids_dir: str, overwrite: bool,
                     optode_frame: str | None) -> list[str] | None:
    """One file's nirspipe-recon argv, or None while its row lacks a subject or a task."""
    subject, task = row.get("subject"), row.get("task")
    if not subject or not task:
        return None
    args = ["nirspipe-recon", row["path"], bids_dir,
            "--participant-label", subject, "--task-label", task]
    args += _text("--session-label", row.get("session"))
    args += _text("--run-label", row.get("run"))
    # unknown is the CLI's default, so only a frame someone chose is sent
    if optode_frame and optode_frame != "unknown":
        args += ["--optode-frame", optode_frame]
    if overwrite:
        args.append("--overwrite")
    return args
