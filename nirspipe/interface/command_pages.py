"""Pages that fill a command's flags from a form generated off its parser, run it, and stream its output."""

from __future__ import annotations

import contextlib
import io
import json
import os
import time

from flask import Blueprint, Response, abort, jsonify, render_template, request, url_for

from nirspipe.interface import process_stream
from nirspipe.interface.commands import (
    COMMANDS, VIEWERS, executable, parser_for, preview_text, subcommands,
)
from nirspipe.interface.forms import check, fields, to_argv

DEFAULT_SHELL = "powershell" if os.name == "nt" else "bash"


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
        parser, prefix, _top = _resolve(prog, sub)
        subs = list(subcommands(parser)) if sub is None else []
        return render_template(
            "command.html", prefix=" ".join(prefix), prog=prog, sub=sub, subs=subs,
            description=parser.description or "", groups=fields(parser),
            opens_viewer=(prog, sub) in VIEWERS, shell=DEFAULT_SHELL,
            preview_url=url_for("commands.preview", prog=prog, sub=sub),
            run_url=url_for("commands.run", prog=prog, sub=sub))

    @bp.post("/preview/<prog>/")
    @bp.post("/preview/<prog>/<sub>/")
    def preview(prog, sub=None):
        parser, prefix, top = _resolve(prog, sub)
        argv = to_argv(prefix, parser, request.form)
        return jsonify(preview=preview_text(argv, request.form.get("_shell", DEFAULT_SHELL)),
                       error=check(top, argv[1:]))

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
        return jsonify(events=url_for("commands.events", run_id=run_id),
                       stop=url_for("commands.stop", run_id=run_id))

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

    return bp
