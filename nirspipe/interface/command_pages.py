"""Pages that fill a command's flags from a form generated off its parser, run it, and stream its output."""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from pathlib import Path

from flask import Blueprint, Response, abort, jsonify, redirect, render_template, request, url_for

from nirspipe.interface import process_stream, recon_batch
from nirspipe.interface.commands import (
    COMMANDS, VIEWERS, executable, parser_for, preview_text, subcommands,
)
from nirspipe.interface.forms import check, fields, form_values, to_argv
from nirspipe.interface.report_serve import REPORT_PATTERNS, find_report, report_url
from nirspipe.interface.sidebar import PAGES
from nirspipe.interface.stages import pipeline_stages

DEFAULT_SHELL = "powershell" if os.name == "nt" else "bash"

# (prog, sub) -> the REPORT_PATTERNS entry for what it writes; nirspipe's depends on the level
_REPORTS = {
    ("nirspipe-qc", "prep-raw"): "prep-raw",
    ("nirspipe-qc", "hyper-raw"): "hyper-raw",
    ("nirspipe-qc", "cohort"): "cohort",
    ("nirspipe-qc", "cohort-hyper"): "cohort-hyper",
    ("nirspipe-hyper", None): "run",
    ("nirspipe-hyper-index", None): "index",
}

# run id -> (what the run writes, where), read when the page asks for the report
_RUNS: dict[str, tuple["str | None", "Path | None"]] = {}


def _resolve(prog: str, sub: "str | None"):
    """(the parser the form is built from, the argv's first tokens, the command's own parser)."""
    if prog not in COMMANDS:
        abort(404)
    top = parser_for(prog)
    if sub is None:
        return top, [prog], top
    parser = subcommands(top).get(sub)
    if parser is None:
        abort(404)
    return parser, [prog, sub], top


def _form(prog: str, sub: "str | None") -> dict:
    """What the form template needs for one command."""
    parser, prefix, _top = _resolve(prog, sub)
    return dict(prefix=" ".join(prefix), description=parser.description or "",
                groups=fields(parser), opens_viewer=(prog, sub) in VIEWERS, shell=DEFAULT_SHELL,
                preview_url=url_for("commands.preview", prog=prog, sub=sub),
                run_url=url_for("commands.run", prog=prog, sub=sub), values={})


def _record(path: str) -> "tuple[dict, list[str]]":
    """The Analysis form's values from a run record, read as ``--config`` reads it, and what to say."""
    from nirspipe import __version__
    from nirspipe.cli.run import read_config

    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            config = read_config(Path(path))
    except SystemExit:
        return {}, [err.getvalue().strip()]
    except Exception as exc:
        return {}, [f"Could not read {path}: {exc}"]
    notes = [f"Loaded the settings of {Path(path).name}. The folders are not in a record: fill them."]
    if config["version"] and config["version"] != __version__:
        notes.append(f"Written by nirspipe {config['version']}; this is {__version__}, so a setting "
                     "whose meaning or default changed runs as this version reads it.")
    settings = {**config["prep"], **config["post"], **config["execution"]}
    return form_values(parser_for("nirspipe"), settings), notes


def _report_key(prog: str, sub: "str | None", args) -> "str | None":
    if prog == "nirspipe":
        return "participant" if args.analysis_level == "participant" else "cohort"
    return _REPORTS.get((prog, sub))


def _open_viewer(web, top, argv: list[str]):
    """Mount the rating viewer an ``nirspipe-rate`` subcommand would serve, and link its pages."""
    from nirspipe.cli import rate

    args = vars(top.parse_args(argv[1:]))
    opener = {"rate": rate.open_rate, "raw": rate.open_raw, "hyper": rate.open_hyper}[args["command"]]
    for key in ("command", "func", "port"):
        args.pop(key, None)
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            viewer, label, entries = opener(**args)
    except SystemExit:
        return jsonify(error=err.getvalue().strip() or "the viewer could not be opened"), 400
    paths = web.mount(viewer.app, label, entries)
    return jsonify(opened=[{"label": f"{label} {entry}".strip(), "href": "/" + path}
                           for entry, path in zip(entries, paths)])


def blueprint(web) -> Blueprint:
    bp = Blueprint("commands", __name__)

    @bp.app_context_processor
    def sidebar():
        return {"pages": PAGES}

    @bp.get("/commands/")
    def index():
        listed = []
        for prog in COMMANDS:
            parser = parser_for(prog)
            listed.append((prog, parser.description or "", list(subcommands(parser))))
        return render_template("commands.html", commands=listed)

    @bp.get("/commands/<prog>/")
    @bp.get("/commands/<prog>/<sub>/")
    def form(prog, sub=None):
        parser, _prefix, _top = _resolve(prog, sub)
        subs = list(subcommands(parser)) if sub is None else []
        if subs:
            return render_template("command.html", prog=prog, prefix=prog, subs=subs,
                                   description=parser.description or "")
        return render_template("command.html", prog=prog, subs=[], **_form(prog, sub))

    def page_view(page, tab=None):
        chosen = next((t for t in page.tabs if t.key == tab), None) if tab else None
        if chosen is None:
            return redirect(url_for(f"commands.{page.slug}", tab=page.tabs[0].key))
        context = _form(chosen.prog, chosen.sub)
        if chosen.template:
            folder = request.args.get("folder", "").strip().strip('"')
            rows, found = recon_batch.detect(folder) if folder else ([], "")
            context.update(groups=recon_batch.shared_groups(), folder=folder, rows=rows,
                           found=found, preview_url=url_for("commands.recon_batch_preview"),
                           run_url=url_for("commands.recon_batch_run"))
            return render_template(chosen.template, page=page, tab=chosen, **context)
        record = request.args.get("record", "").strip().strip('"')
        notes = []
        if record and chosen.prog == "nirspipe":
            context["values"], notes = _record(record)
        return render_template("page.html", page=page, tab=chosen, record=record, notes=notes,
                               **context)

    for page in PAGES:
        def view(tab=None, page=page):
            return page_view(page, tab)
        bp.add_url_rule(f"/{page.slug}/", page.slug, view)
        bp.add_url_rule(f"/{page.slug}/<tab>/", page.slug, view)
    recon_batch.register(bp, DEFAULT_SHELL)

    @bp.post("/preview/<prog>/")
    @bp.post("/preview/<prog>/<sub>/")
    def preview(prog, sub=None):
        parser, prefix, top = _resolve(prog, sub)
        argv = to_argv(prefix, parser, request.form)
        return jsonify(preview=preview_text(argv, request.form.get("_shell", DEFAULT_SHELL)),
                       error=check(top, argv[1:]),
                       stages=pipeline_stages(request.form) if prog == "nirspipe" else None)

    @bp.post("/run/<prog>/")
    @bp.post("/run/<prog>/<sub>/")
    def run(prog, sub=None):
        parser, prefix, top = _resolve(prog, sub)
        argv = to_argv(prefix, parser, request.form)
        error = check(top, argv[1:])
        if error:
            return jsonify(error=error), 400
        if (prog, sub) in VIEWERS:
            return _open_viewer(web, top, argv)
        try:
            run_id = process_stream.start([executable(prog), *argv[1:]])
        except OSError as exc:
            return jsonify(error=f"could not launch {prog}: {exc}"), 500
        args = top.parse_args(argv[1:])
        output_dir = getattr(args, "output_dir", None)
        _RUNS[run_id] = (_report_key(prog, sub, args), Path(output_dir) if output_dir else None)
        return jsonify(events=url_for("commands.events", run_id=run_id),
                       stop=url_for("commands.stop", run_id=run_id),
                       report=url_for("commands.report", run_id=run_id))

    @bp.get("/runs/<run_id>/events")
    def events(run_id):
        try:
            process_stream.read_from(run_id, 0)
        except KeyError:
            abort(404)

        def stream():
            seen = 0
            while True:
                lines, seen, code = process_stream.read_from(run_id, seen)
                for line in lines:
                    yield f"data: {json.dumps(line)}\n\n"
                if code is not None:
                    end = {"code": code, "stopped": process_stream.was_stopped(run_id)}
                    yield f"event: end\ndata: {json.dumps(end)}\n\n"
                    return
                time.sleep(0.2)

        return Response(stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache"})

    @bp.post("/runs/<run_id>/stop")
    def stop(run_id):
        return jsonify(stopped=process_stream.stop(run_id))

    @bp.get("/runs/<run_id>/report")
    def report(run_id):
        key, output_dir = _RUNS.get(run_id, (None, None))
        if key is None or output_dir is None:
            return jsonify(message="This command writes tables, not a report.")
        found = find_report(REPORT_PATTERNS[key], output_dir)
        if found is None:
            # known to write a page and did not: a pattern left behind by a rename would
            # otherwise read as "nothing to show"
            return jsonify(message=f"The run finished but its report is missing: looked for "
                                   f"{', '.join(REPORT_PATTERNS[key])} under {output_dir}.")
        rate = []
        if key == "prep-raw":
            rate = [{"label": page.stem.removesuffix("_desc-raw_report"),
                     "href": url_for("commands.rate_raw", report=str(page), tree=str(output_dir))}
                    for page in sorted(output_dir.glob(REPORT_PATTERNS["prep-raw"][0]))]
        return jsonify(url=report_url(found), path=str(found), rate=rate)

    # (report, tree) -> its viewer's path, so a second click opens the same page
    rated: dict[tuple[Path, Path], str] = {}

    @bp.get("/rate/raw")
    def rate_raw():
        from nirspipe.qc.rating.app import RawRatingApp

        report_path, tree = Path(request.args["report"]), Path(request.args["tree"])
        if not report_path.is_file():
            abort(404)
        key = (report_path.resolve(), tree.resolve())
        if key not in rated:
            [rated[key]] = web.mount(RawRatingApp(report_path, tree).app, report_path.stem)
        return redirect("/" + rated[key])

    return bp
