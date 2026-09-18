"""Translate a page's form into an argv for the CLI that owns it.

The batch editing is `fnirs-prep`, the dyad analysis is `fnirs-hyper`, the aggregate reports
are `fnirs-qc`. Kept beside the pages rather than derived from the parsers, because argparse
can say a flag exists but not which widget should fill it. `tests/test_gui_cli_surface.py` is
what stops the two surfaces drifting apart.
"""

from __future__ import annotations

# fnirs-qc commands whose whole argument list is one output_dir
_AGGREGATE = ("cohort", "cohort-hyper", "provenance")

# fnirs-hyper subcommands, which take one output_dir and their own flags
_HYPER = ("run", "band", "merge", "index")


def _num(flag: str, value) -> list[str]:
    return [flag, str(value)] if value is not None else []


def _text(flag: str, value) -> list[str]:
    return [flag, str(value)] if value else []


def _split(flag: str, value) -> list[str]:
    """A space-separated box feeds an nargs="+" flag, which repeats rather than joins."""
    return [x for item in (value or "").split() for x in (flag, item)]


def build_qc_args(command: str, opts: dict) -> list[str]:
    if command in _AGGREGATE:
        return ["fnirs-qc", command, opts["output_dir"]]

    args = ["fnirs-hyper", command, opts["output_dir"]]

    # the band and the COI switch are shared with `run`; only the suffix is this one's own
    if command == "band":
        args += _num("--wtc-band-fmin", opts.get("band_fmin"))
        args += _num("--wtc-band-fmax", opts.get("band_fmax"))
        # the switch turns masking off, masking being the default
        if "band_no_mask_coi" in (opts.get("band_flags") or []):
            args.append("--no-wtc-mask-coi")
        args += _text("--wtc-suffix", opts.get("band_suffix"))
        return args

    if command == "run":
        args += _text("--pairs-csv", opts.get("pairs_csv"))
        args += _text("--group-id", opts.get("group_id"))
        args += _text("--desc", opts.get("desc"))
        args += _text("--roi-mapping", opts.get("roi_mapping"))
        args += _num("--wtc-fmin", opts.get("wtc_fmin"))
        args += _num("--wtc-fmax", opts.get("wtc_fmax"))
        args += _num("--wtc-band-fmin", opts.get("wtc_band_fmin"))
        args += _num("--wtc-band-fmax", opts.get("wtc_band_fmax"))
        args += _num("--wtc-mc-count", opts.get("wtc_mc_count"))
        args += _num("--wtc-seed", opts.get("wtc_seed"))
        args += _num("--wtc-phase-null", opts.get("wtc_phase_null"))
        args += _num("--isc-threshold", opts.get("isc_threshold"))
        args += _num("--wtc-roi-min-channels", opts.get("wtc_roi_min_channels"))
        args += _num("--isc-whiten", opts.get("isc_whiten"))
        args += _num("--isc-max-lag", opts.get("isc_max_lag"))
        args += _num("--isc-phase-null", opts.get("isc_phase_null"))
        args += _text("--wtc-chroma", opts.get("wtc_chroma"))
        args += _split("--task-label", opts.get("hyper_task"))
        flags = opts.get("hyper_flags") or []
        if "wtc_significance" in flags:
            args.append("--wtc-significance")
        if "wtc_no_mask_coi" in flags:
            args.append("--no-wtc-mask-coi")
        if "wtc_channel_cross" in flags:
            args.append("--wtc-channel-cross")
        if "wtc_phase_null_cross" in flags:
            args.append("--wtc-phase-null-cross")
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
    if command == "run" and not opts.get("pairs_csv"):
        return "The dyad analysis needs a pairs CSV."
    return None


# ---- fnirs-prep ----

# the three batch operations, named as the page's radio names them
_PREP_OPERATIONS = ("markers", "crop", "hyper_align")


def _selection(opts: dict) -> list[str]:
    args = ["--participant-label", *[str(s) for s in (opts.get("subjects") or [])]]
    args += _text("--session-label", opts.get("ses"))
    args += _text("--task-label", opts.get("task"))
    args += _text("--run-label", opts.get("run"))
    args += _num("--n-jobs", opts.get("n_jobs"))
    return args


def build_prep_args(operation: str, opts: dict) -> list[str]:
    bids, deriv = opts.get("bids_dir"), opts.get("deriv_dir")

    # align selects its subjects through the group CSV, so it takes no selection flags
    if operation == "hyper_align":
        return ["fnirs-prep", "align", bids, deriv,
                *_text("--group-csv", opts.get("group_csv"))]

    if operation == "markers":
        args = ["fnirs-prep", "edit-markers", "apply", bids, deriv, *_selection(opts)]
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

    args = ["fnirs-prep", "crop", bids, deriv, *_selection(opts)]
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

    if opts.get("crop_mode") == "multi":
        return None if opts.get("segments") else "Add at least one segment."
    if opts.get("crop_tmin") is None and opts.get("crop_tmax") is None:
        return "Enter tmin or tmax."
    return None


# ---- fnirs-qc raw reports ----

# the static counterparts of what the two QC pages show interactively
_RAW_QC = ("prep-raw", "hyper-raw")


def _screening(opts: dict) -> list[str]:
    args = _num("--sci-threshold", opts.get("sci_threshold"))
    args += _num("--dpf", opts.get("dpf"))
    args += _num("--cardiac-l-freq", opts.get("cardiac_l"))
    args += _num("--cardiac-h-freq", opts.get("cardiac_h"))
    args += _num("--short-max-dist", opts.get("short_max_dist"))
    args += _num("--long-min-dist", opts.get("long_min_dist"))
    args += _num("--long-max-dist", opts.get("long_max_dist"))
    args += _text("--session-label", opts.get("ses"))
    args += _text("--task-label", opts.get("task"))
    return args


def build_raw_qc_args(command: str, opts: dict) -> list[str]:
    args = ["fnirs-qc", command, opts.get("bids_dir"), opts.get("output_dir")]

    if command == "prep-raw":
        args += _text("--participant-label", opts.get("subject"))
        args += _screening(opts)
        args += _num("--window-length", opts.get("window_length"))
        args += _num("--epoch-tmin", opts.get("epoch_tmin"))
        args += _num("--epoch-tmax", opts.get("epoch_tmax"))
        if opts.get("epoch_qc"):
            args.append("--epoch-qc")
        return args

    args += _text("--pairs-csv", opts.get("pairs_csv"))
    args += _text("--group-id", opts.get("group_id"))
    args += _screening(opts)
    return args


def missing_raw_qc(command: str, opts: dict) -> str | None:
    if not opts.get("bids_dir") or not opts.get("output_dir"):
        return "Set the BIDS and output directories."
    # these three have no defaults anywhere, by design
    if opts.get("dpf") is None:
        return "DPF is required."
    if opts.get("cardiac_l") is None or opts.get("cardiac_h") is None:
        return "Both ends of the cardiac band are required."
    if command == "prep-raw":
        return None if opts.get("subject") else "Load a run first."
    return None if opts.get("pairs_csv") else "Set the group CSV."
