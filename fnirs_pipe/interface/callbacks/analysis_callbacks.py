"""Callbacks for the analysis page."""

from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, no_update

from fnirs_pipe.interface import process_stream
from fnirs_pipe.interface.callbacks._cli_run import log_panel
from fnirs_pipe.interface.callbacks._sections import rng, summary, value
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("interface.analysis_callbacks")


# ── Directory sync: store → page (one-way; page 1 writes, page 2 reads) ──────

@callback(
    Output("an-bids-dir",   "value"),
    Output("an-output-dir", "value"),
    Input("app-bids-dir",   "data"),
    Input("app-output-dir", "data"),
)
def _prefill_an_from_store(bids_dir, output_dir):
    return bids_dir or no_update, output_dir or no_update


@callback(
    Output("an-subjects-result",    "children"),
    Output("an-subjects-container", "children"),
    Output("an-subjects-store",     "data"),
    Input("an-detect-btn", "n_clicks"),
    State("an-bids-dir",   "value"),
    prevent_initial_call=True,
)
def detect_subjects(n_clicks, bids_dir):
    if not bids_dir:
        return dbc.Alert("BIDS directory is empty.", color="warning", className="mb-0"), "", []
    try:
        from fnirs_pipe.io.bids import get_layout
        layout = get_layout(bids_dir, validate=False)
        subjects = sorted(layout.get_subjects())
    except Exception as exc:
        return dbc.Alert(f"Failed to read BIDS dir: {exc}", color="danger", className="mb-0"), "", []

    if not subjects:
        return dbc.Alert("No subjects found.", color="warning", className="mb-0"), "", []

    alert = dbc.Alert(f"Found {len(subjects)} subject(s).", color="success", className="mb-0")
    checklist = dbc.Checklist(
        id="an-subjects-checklist",
        options=[{"label": s, "value": s} for s in subjects],
        value=subjects,
        inline=True,
        className="mt-2",
    )
    return alert, checklist, subjects


@callback(
    Output("an-subjects-store", "data", allow_duplicate=True),
    Input("an-subjects-checklist", "value"),
    prevent_initial_call=True,
)
def _sync_subjects_store(selected):
    return selected or []


@callback(
    Output("an-glm-section", "style"),
    Output("an-confound-section", "style"),
    Input("an-post-mode", "value"),
)
def toggle_glm_section(mode):
    shown, hidden = {"display": "block"}, {"display": "none"}
    # every mode honours the confound options; only the task model is GLM-only
    return (shown if mode == "glm" else hidden,
            shown if mode in ("denoise", "glm", "rest") else hidden)


def _provenance_elements(output_dir: str | None) -> list | None:
    """The real graph a finished run left on disk, or None when there is nothing to read.

    `provenance.scan` reads the sidecars, so this reflects the files that exist rather than
    the pipeline that was configured. The first subject or group directory carrying sidecars
    wins; one run's chain is the shape worth showing, and the others repeat it.
    """
    if not output_dir:
        return None
    try:
        from fnirs_pipe.qc.common.provenance import scan

        root = Path(output_dir)
        for nirs_dir in [*sorted(root.glob("sub-*/nirs")), *sorted(root.glob("group-*/nirs"))]:
            nodes = scan(nirs_dir)
            if not nodes:
                continue
            # step and state ride along as data rather than in the label: the stylesheet
            # sizes nodes to a single line, and a wrapped label would overlap its neighbours
            elements = [
                {"data": {"id": key, "label": n.label,
                          "step": n.step or "", "detail": n.detail, "state": n.state},
                 "classes": "prep" if n.domain in ("input", "od") else "post"}
                for key, n in nodes.items()
            ]
            elements += [
                {"data": {"source": src, "target": key}, "classes": "post"}
                for key, n in nodes.items() for src in n.sources if src in nodes
            ]
            return elements
    except Exception:
        logger.warning("provenance graph unavailable, falling back to the schematic",
                       exc_info=True)
    return None


@callback(
    Output("an-dag", "elements"),
    Input("an-post-mode", "value"),
    Input("an-run-status", "children"),
    State("an-output-dir", "value"),
)
def update_dag(post_mode, _run_status, output_dir):
    # after a run there is a real graph on disk; before one, the schematic is all we have
    real = _provenance_elements(output_dir)
    if real:
        return real

    def node(nid, label, cls):
        return {"data": {"id": nid, "label": label}, "classes": cls}

    def edge(src, tgt, cls):
        return {"data": {"source": src, "target": tgt}, "classes": cls}

    elements = [
        node("load",   "Load SNIRF",         "prep"),
        node("od",     "Optical Density",     "prep"),
        node("sci",    "SCI / Bad Channels",  "prep"),
        node("bll",    "Beer-Lambert Law",    "prep"),
        node("motion", "Motion Correction",   "prep"),
        edge("load",   "od",     "prep"),
        edge("od",     "sci",    "prep"),
        edge("sci",    "bll",    "prep"),
        edge("bll",    "motion", "prep"),
    ]

    last = "motion"

    if post_mode and post_mode != "none":
        elements += [
            node("filter", "Bandpass Filter", "post"),
            edge(last, "filter", "post"),
        ]
        last = "filter"

        elements += [
            node("resample", "Resample", "post"),
            edge(last, "resample", "post"),
        ]
        last = "resample"

        if post_mode == "glm":
            elements += [
                node("glm", "GLM", "glm"),
                edge(last, "glm", "glm"),
            ]
            last = "glm"

    return elements


def _build_cli_args(opts: dict) -> list[str]:
    """Translate widget values into a fnirs-pipe argv list."""
    args: list[str] = [
        "fnirs-pipe",
        opts["bids_dir"],
        opts["output_dir"],
        "participant",
    ]
    # list-typed options need the flag repeated per value (Typer/Click)
    for sub in opts.get("subjects") or []:
        args += ["--participant-label", sub]
    for ses in (opts.get("session_label") or "").split():
        args += ["--session-label", ses]
    for task in (opts.get("task_label") or "").split():
        args += ["--task-label", task]
    if opts.get("dpf") is not None:
        args += ["--dpf", str(opts["dpf"])]
    if opts.get("sci_thresh") is not None:
        args += ["--sci-threshold", str(opts["sci_thresh"])]
    if opts.get("psp_thresh") is not None:
        args += ["--psp-threshold", str(opts["psp_thresh"])]
    # the two lines above only say what a coupled window is; this is what rejects
    if opts.get("min_good_frac") is not None:
        args += ["--min-good-frac", str(opts["min_good_frac"])]
    if opts.get("screen_scope"):
        args += ["--screen-scope", opts["screen_scope"]]
    if opts.get("window_length") is not None:
        args += ["--window-length", str(opts["window_length"])]
    # each independently optional: omitting --long-max-dist is how "no upper bound" is said
    for opt_key, flag in (("short_max_dist", "--short-max-dist"),
                          ("long_min_dist", "--long-min-dist"),
                          ("long_max_dist", "--long-max-dist")):
        if opts.get(opt_key) is not None:
            args += [flag, str(opts[opt_key])]
    # both edges or neither: the CLI refuses half a window, and so does the report
    if opts.get("epoch_tmin") is not None and opts.get("epoch_tmax") is not None:
        args += ["--epoch-tmin", str(opts["epoch_tmin"]),
                 "--epoch-tmax", str(opts["epoch_tmax"])]
    if opts.get("epoch_chunk") is not None:
        args += ["--epoch-chunk-duration", str(opts["epoch_chunk"])]
    if opts.get("by_condition"):
        args.append("--by-condition")
    # nargs="?": the flag carries the channel set, and off means not sending it at all
    if opts.get("gvtd_censor") and opts["gvtd_censor"] != "off":
        args += ["--gvtd-censor", opts["gvtd_censor"]]
        if opts.get("gvtd_n_std") is not None:
            args += ["--gvtd-censor-n-std", str(opts["gvtd_n_std"])]
    if opts.get("motion_correction"):
        args += ["--motion-correction", opts["motion_correction"]]
    if opts.get("cardiac_l") is not None:
        args += ["--cardiac-l-freq", str(opts["cardiac_l"])]
    if opts.get("cardiac_h") is not None:
        args += ["--cardiac-h-freq", str(opts["cardiac_h"])]
    if opts.get("resp_l") is not None:
        args += ["--resp-l-freq", str(opts["resp_l"])]
    if opts.get("resp_h") is not None:
        args += ["--resp-h-freq", str(opts["resp_h"])]

    mode = opts.get("post_mode")
    if mode and mode != "none":
        args += ["--mode", mode]
        if opts.get("high_pass") is not None:
            args += ["--high-pass", str(opts["high_pass"])]
        if opts.get("low_pass") is not None:
            args += ["--low-pass", str(opts["low_pass"])]
        if opts.get("filter_method"):
            args += ["--filter-method", str(opts["filter_method"])]
        if opts.get("filter_order") is not None:
            args += ["--filter-order", str(int(opts["filter_order"]))]
        if opts.get("resample") is not None:
            args += ["--resample-sfreq", str(opts["resample"])]
        if opts.get("roi_mapping"):
            args += ["--roi-mapping", opts["roi_mapping"]]

        # confound regression: honoured by every mode, and glm and rest refuse to run
        # without a drift model
        if opts.get("drift_model"):
            args += ["--drift-model", opts["drift_model"]]
            if opts["drift_model"] == "cosine" and opts.get("drift_high_pass") is not None:
                args += ["--drift-high-pass", str(opts["drift_high_pass"])]
            if opts["drift_model"] == "polynomial" and opts.get("drift_order") is not None:
                args += ["--drift-order", str(opts["drift_order"])]
        if opts.get("short_channel") and opts["short_channel"] != "none":
            args += ["--short-channel", opts["short_channel"]]
        if mode == "glm":
            if opts.get("hrf_model"):   args += ["--hrf-model",   opts["hrf_model"]]
            if opts.get("noise_model"): args += ["--noise-model", opts["noise_model"]]
            if opts.get("stim_dur") is not None:
                args += ["--stim-dur", str(opts["stim_dur"])]

        # rest writes the connectivity products regardless, so the flag would be noise there
        if mode in ("denoise", "glm") and opts.get("fc"):
            args.append("--fc")

        if "combine_runs" in (opts.get("flags") or []):
            args.append("--combine-runs")

    if opts.get("n_jobs"):
        args += ["--n-jobs", str(opts["n_jobs"])]

    flags = opts.get("flags") or []
    if "dry_run"              in flags: args.append("--dry-run")
    if "skip_bids_validation" in flags: args.append("--skip-bids-validation")
    if "no_report"            in flags: args.append("--no-report")
    return args


@callback(
    Output("an-command-preview", "children"),
    Output("an-command-store",   "data"),
    Input("an-generate-btn",     "n_clicks"),
    State("an-bids-dir",          "value"),
    State("an-output-dir",        "value"),
    State("an-subjects-store",    "data"),
    State("an-dpf",               "value"),
    State("an-sci-thresh",        "value"),
    State("an-psp-thresh",        "value"),
    State("an-min-good-frac",     "value"),
    State("an-screen-scope",      "value"),
    State("an-window-length",     "value"),
    State("an-short-max-dist",    "value"),
    State("an-long-min-dist",     "value"),
    State("an-long-max-dist",     "value"),
    State("an-epoch-tmin",        "value"),
    State("an-epoch-tmax",        "value"),
    State("an-epoch-chunk",       "value"),
    State("an-by-condition",      "value"),
    State("an-gvtd-censor",       "value"),
    State("an-gvtd-n-std",        "value"),
    State("an-motion-correction", "value"),
    State("an-cardiac-l",         "value"),
    State("an-cardiac-h",         "value"),
    State("an-resp-l",            "value"),
    State("an-resp-h",            "value"),
    State("an-post-mode",         "value"),
    State("an-high-pass",         "value"),
    State("an-low-pass",          "value"),
    State("an-filter-method",     "value"),
    State("an-filter-order",      "value"),
    State("an-resample",          "value"),
    State("an-n-jobs",            "value"),
    State("an-hrf-model",         "value"),
    State("an-noise-model",       "value"),
    State("an-short-channel",     "value"),
    State("an-drift-model",       "value"),
    State("an-drift-high-pass",   "value"),
    State("an-drift-order",       "value"),
    State("an-stim-dur",          "value"),
    State("an-roi-mapping",       "value"),
    State("an-fc",                "value"),
    State("an-flags",             "value"),
    State("an-session-label",     "value"),
    State("an-task-label",        "value"),
    State("an-shell-select",      "value"),
    prevent_initial_call=True,
)
def generate_command(n_clicks, bids_dir, output_dir, subjects, dpf, sci_thresh, psp_thresh,
                     min_good_frac, screen_scope, window_length,
                     short_max_dist, long_min_dist, long_max_dist,
                     epoch_tmin, epoch_tmax, epoch_chunk,
                     by_condition, gvtd_censor, gvtd_n_std,
                     motion_correction, cardiac_l, cardiac_h, resp_l, resp_h,
                     post_mode, high_pass, low_pass, filter_method, filter_order,
                     resample, n_jobs,
                     hrf_model, noise_model, short_channel,
                     drift_model, drift_high_pass, drift_order, stim_dur,
                     roi_mapping, fc, flags,
                     session_label, task_label, shell):
    if not bids_dir or not output_dir:
        return "Error: BIDS and output directories are required.", {}
    if not subjects:
        return "Error: detect and select subjects first.", {}
    if dpf is None or sci_thresh is None:
        return "Error: DPF and SCI threshold are required.", {}
    if cardiac_l is None or cardiac_h is None:
        return "Error: cardiac band lower/upper frequency is required (population-dependent).", {}
    if resp_l is None or resp_h is None:
        return "Error: respiration band lower/upper frequency is required (population-dependent).", {}

    opts = dict(
        bids_dir=bids_dir, output_dir=output_dir, subjects=subjects,
        session_label=session_label, task_label=task_label,
        dpf=dpf, sci_thresh=sci_thresh, psp_thresh=psp_thresh,
        min_good_frac=min_good_frac, screen_scope=screen_scope,
        window_length=window_length,
        short_max_dist=short_max_dist, long_min_dist=long_min_dist,
        long_max_dist=long_max_dist,
        epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax, epoch_chunk=epoch_chunk,
        by_condition=bool(by_condition), gvtd_censor=gvtd_censor, gvtd_n_std=gvtd_n_std,
        motion_correction=motion_correction, cardiac_l=cardiac_l, cardiac_h=cardiac_h,
        resp_l=resp_l, resp_h=resp_h,
        post_mode=post_mode, high_pass=high_pass, low_pass=low_pass,
        filter_method=filter_method, filter_order=filter_order,
        resample=resample, n_jobs=n_jobs,
        hrf_model=hrf_model, noise_model=noise_model, short_channel=short_channel,
        drift_model=drift_model, drift_high_pass=drift_high_pass,
        drift_order=drift_order, stim_dur=stim_dur,
        roi_mapping=roi_mapping, fc=bool(fc),
        flags=flags,
    )
    argv = _build_cli_args(opts)
    # Pretty-print: shell-specific line-continuation per arg group for readability.
    cont = {"bash": "\\", "cmd": "^", "powershell": "`"}.get(shell, "\\")
    lines = [argv[0]] + [f"  {a}" for a in argv[1:]]
    preview = f" {cont}\n".join(lines)
    return preview, {"argv": argv}


@callback(
    Output("an-run-status", "children"),
    Output("an-run-store",  "data"),
    Output("an-run-tick",   "disabled"),
    Output("an-stop-btn",   "disabled"),
    Output("an-run-btn",    "disabled"),
    Input("an-run-btn",     "n_clicks"),
    State("an-command-store", "data"),
    prevent_initial_call=True,
)
def run_pipeline(n_clicks, cmd_data):
    if not cmd_data or not cmd_data.get("argv"):
        return (dbc.Alert("Click 'Generate Command' first.", color="warning",
                          className="mb-0"), None, True, True, False)

    argv = cmd_data["argv"]
    try:
        run_id = process_stream.start(argv)
    except FileNotFoundError:
        return (dbc.Alert(
            f"`{argv[0]}` not found on PATH - make sure fnirs-pipe is installed.",
            color="danger", className="mb-0"), None, True, True, False)
    except Exception as exc:
        return (dbc.Alert(f"Failed to launch: {exc}", color="danger", className="mb-0"),
                None, True, True, False)

    logger.info("started %s as run %s", argv[0], run_id)
    return log_panel("Running...", [], 0, "info"), run_id, False, False, True


@callback(
    Output("an-run-status", "children", allow_duplicate=True),
    Output("an-run-tick",   "disabled", allow_duplicate=True),
    Output("an-stop-btn",   "disabled", allow_duplicate=True),
    Output("an-run-btn",    "disabled", allow_duplicate=True),
    Input("an-run-tick",    "n_intervals"),
    State("an-run-store",   "data"),
    prevent_initial_call=True,
)
def stream_run_output(_n, run_id):
    if not run_id:
        return no_update, True, True, False

    lines, returncode, dropped = process_stream.poll(run_id)
    if returncode is None:
        return log_panel("Running...", lines, dropped, "info"), False, False, True

    if returncode == 0:
        header, color = "Pipeline finished.", "success"
    elif returncode < 0:
        header, color = f"Pipeline stopped (signal {-returncode}).", "warning"
    else:
        header, color = f"Pipeline failed (exit {returncode}).", "danger"
    process_stream.forget(run_id)
    return log_panel(header, lines, dropped, color), True, True, False


@callback(
    Output("an-stop-btn", "disabled", allow_duplicate=True),
    Input("an-stop-btn",  "n_clicks"),
    State("an-run-store", "data"),
    prevent_initial_call=True,
)
def stop_pipeline(_n, run_id):
    # the tick reports the outcome; this only asks the process to end
    if run_id:
        process_stream.stop(run_id)
    return True


# ── What each collapsed group says in its header ─────────────────────────────

_SHOWN = {"display": ""}
_GONE  = {"display": "none"}


@callback(
    Output("an-od-summary",       "children"),
    Output("an-screen-summary",   "children"),
    Output("an-bands-summary",    "children"),
    Output("an-epochs-summary",   "children"),
    Output("an-gvtd-summary",     "children"),
    Output("an-confound-summary", "children"),
    Output("an-glm-summary",      "children"),
    Input("an-dpf", "value"), Input("an-motion-correction", "value"),
    Input("an-short-max-dist", "value"), Input("an-long-min-dist", "value"),
    Input("an-sci-thresh", "value"), Input("an-psp-thresh", "value"),
    Input("an-min-good-frac", "value"), Input("an-screen-scope", "value"),
    Input("an-window-length", "value"),
    Input("an-cardiac-l", "value"), Input("an-cardiac-h", "value"),
    Input("an-resp-l", "value"), Input("an-resp-h", "value"),
    Input("an-epoch-tmin", "value"), Input("an-epoch-tmax", "value"),
    Input("an-epoch-chunk", "value"), Input("an-by-condition", "value"),
    Input("an-gvtd-censor", "value"), Input("an-gvtd-n-std", "value"),
    Input("an-drift-model", "value"), Input("an-drift-high-pass", "value"),
    Input("an-short-channel", "value"), Input("an-fc", "value"),
    Input("an-hrf-model", "value"), Input("an-noise-model", "value"),
    Input("an-stim-dur", "value"),
)
def section_summaries(dpf, motion, short_max, long_min, sci, psp, frac, scope, window,
                      card_l, card_h, resp_l, resp_h, tmin, tmax, chunk, by_cond,
                      censor, n_std, drift, drift_hp, short_ch, fc,
                      hrf, noise, stim_dur):
    return (
        summary(value("DPF", dpf), motion,
                rng("separations", short_max, long_min, " mm")),
        summary(value("SCI", sci), value("PSP", psp), value("coupled", frac),
                {"run": "whole run", "task": "task blocks"}.get(scope),
                value("window", window, " s")),
        summary(rng("cardiac", card_l, card_h, " Hz"),
                rng("respiration", resp_l, resp_h, " Hz")) or "not set",
        summary(rng("epoch", tmin, tmax, " s"),
                value("chunk", chunk, " s"),
                "per-condition QC" if by_cond else None) or "not set",
        summary("off" if censor in (None, "off") else f"{censor} channels",
                value("at", n_std, " SD") if censor not in (None, "off") else None),
        summary(value("drift", drift), value("high-pass", drift_hp, " Hz"),
                value("short channel", short_ch), "FC products" if fc else None),
        summary(value("HRF", hrf), value("noise", noise), value("stim", stim_dur, " s")),
    )


@callback(
    Output("an-gvtd-n-std-wrap",  "style"),
    Output("an-drift-hp-wrap",    "style"),
    Output("an-drift-order-wrap", "style"),
    Input("an-gvtd-censor", "value"),
    Input("an-drift-model", "value"),
)
def hide_what_does_not_apply(censor, drift):
    # each of these is read by exactly one setting of the control above it
    return (
        _SHOWN if censor not in (None, "off") else _GONE,
        _SHOWN if drift == "cosine" else _GONE,
        _SHOWN if drift == "polynomial" else _GONE,
    )


# ── The drift cutoff is bracketed, so say so while the form is being filled ──

@callback(
    Output("an-band-note", "children"),
    Input("an-post-mode",      "value"),
    Input("an-high-pass",      "value"),
    Input("an-drift-model",    "value"),
    Input("an-drift-high-pass", "value"),
)
def band_note(mode, high_pass, drift_model, drift_high_pass):
    if mode in (None, "none"):
        return None

    # cosine without a cutoff stops the run in PostConfig, so say it here instead
    if drift_model == "cosine" and drift_high_pass is None:
        return dbc.Alert(
            "A cosine drift model needs a cutoff. Load a run on Data Preparation: it reads "
            "the slowest repeat of each condition off the markers and gives the number to "
            "put here.",
            color="warning", className="mb-0")

    if mode != "glm" or high_pass is None:
        return None

    if drift_model != "cosine":
        return dbc.Alert(
            f"The data is high-passed at {high_pass} Hz but the task regressors are not, "
            f"and a {drift_model or 'none'} drift model does not span that band. "
            "Task betas will come out too small. Use a cosine drift model at "
            f"{high_pass} Hz, or leave the high-pass empty and let the drift basis do it.",
            color="warning", className="mb-0")

    if drift_high_pass is None or drift_high_pass < high_pass:
        return dbc.Alert(
            f"Drift high-pass {drift_high_pass} Hz is below the {high_pass} Hz the data is "
            "filtered at, so the drift basis does not cover what the filter removed. "
            f"Set it to {high_pass} Hz or higher.",
            color="warning", className="mb-0")

    return dbc.Alert(
        f"The high-pass and the cosine drift basis are both at {high_pass} Hz and doing the "
        "same job. The high-pass can be left empty.",
        color="info", className="mb-0")
