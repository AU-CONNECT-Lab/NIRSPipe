"""A raw viewer in edit mode: a run's events edited in a table, read and saved through ``nirspipe-prep``."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Response, abort, jsonify, request, send_from_directory
from jinja2 import Environment, FileSystemLoader, select_autoescape
from werkzeug.datastructures import MultiDict

from nirspipe.interface import process_stream
from nirspipe.interface.commands import executable, parser_for, subcommands
from nirspipe.interface.forms import check, fields, required_first, to_argv
from nirspipe.io.derivatives import entity_of
from nirspipe.io.tables import read_table, write_tsv
from nirspipe.qc.rating.app import RawRatingApp
from nirspipe.utils import NOT_MARKERS, is_marker

_HERE = Path(__file__).resolve().parent
_JINJA = Environment(loader=FileSystemLoader(_HERE / "templates"), autoescape=select_autoescape())

PREP = "nirspipe-prep"
# the run's selection and the table come from the page; the one-step edits are the table's to make
_FILLED = {"bids_dir", "participant_label", "session_label", "task_label", "run_label",
           "tsv", "shift", "set_duration", "rename"}
_COLUMNS = ["onset", "duration", "trial_type"]
# what the page loads from prep/, by the name it asks for
_STATIC = {"edit.js": "prep_edit.js", "edit.css": "prep_edit.css", "forms.css": "forms.css"}


def _apply_parser():
    return subcommands(subcommands(parser_for(PREP))["edit-markers"])["apply"]


def selection(run_label: str) -> "dict[str, str]":
    """The selection flags that pick out one run, from its label (``sub-01_ses-1_task-x_run-2``)."""
    flags = {"sub": "--participant-label", "ses": "--session-label", "task": "--task-label",
             "run": "--run-label"}
    return {flag: value for key, flag in flags.items() if (value := entity_of(run_label, key))}


def _selection_argv(run_label: str) -> list[str]:
    return [token for flag, value in selection(run_label).items() for token in (flag, value)]


def form_html(values: "dict[str, list[str]]") -> str:
    """The apply form's fields, less the ones the page fills, as the command pages draw them."""
    groups = [(title, [f for f in group if f.action.dest not in _FILLED])
              for title, group in required_first(fields(_apply_parser()))]
    groups = [(title, group) for title, group in groups if group]
    return _JINJA.get_template("_fields.html").render(groups=groups, values=values)


def read_events(bids_dir: Path, run_label: str) -> dict:
    """The run's events table as ``edit-markers export`` writes it, or the command's own complaint."""
    with tempfile.TemporaryDirectory() as tmp:
        argv = [executable(PREP), "edit-markers", "export", str(bids_dir), tmp,
                *_selection_argv(run_label), "--skip-bids-validation"]
        proc = subprocess.run(argv, capture_output=True, text=True)
        written = sorted(Path(tmp).glob("*_events.tsv"))
        if proc.returncode or not written:
            tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-8:]
            return {"error": "\n".join(tail) or f"{PREP} edit-markers export wrote no table"}
        # as text, so a name such as 1 is not read back as 1.0
        table = read_table(written[0], dtype={"trial_type": str})
    missing = [c for c in _COLUMNS if c not in table.columns]
    if missing and len(table):
        return {"error": f"{written[0].name} has no {', '.join(missing)} column"}
    rows = [[float(r["onset"]), float(r["duration"]), r["trial_type"]]
            for r in table.fillna({"trial_type": ""}).to_dict("records")] if len(table) else []
    return {"rows": rows, "not_markers": list(NOT_MARKERS)}


def parse_rows(raw) -> "tuple[list[list], str | None]":
    """The page's table as [onset, duration, trial_type] rows, or what is wrong with it."""
    if not isinstance(raw, list):
        return [], "the events table did not arrive"
    rows = []
    for i, row in enumerate(raw, 1):
        try:
            onset, duration, name = float(row[0]), float(row[1]), str(row[2]).strip()
        except (TypeError, ValueError, IndexError):
            return [], f"row {i}: onset and duration must be numbers"
        if not (math.isfinite(onset) and math.isfinite(duration)) or onset < 0 or duration < 0:
            return [], f"row {i}: onset and duration must be 0 or more"
        if not name:
            return [], f"row {i}: the event has no name"
        rows.append([onset, duration, name])
    return rows, None


def drift_hint(rows: list[list]) -> "dict | None":
    """The drift high-pass a GLM on this run allows, set by the condition that repeats slowest."""
    onsets: dict[str, list[float]] = {}
    for onset, _duration, name in rows:
        if is_marker(name):
            onsets.setdefault(name, []).append(onset)
    gaps = {name: float(np.max(np.diff(sorted(on)))) for name, on in onsets.items() if len(on) >= 2}
    if not gaps:
        return None
    slowest = max(gaps, key=gaps.get)
    return {"cutoff": f"{1.0 / (2.0 * gaps[slowest]):.4g} Hz", "by": slowest,
            "per": " · ".join(f"{name} {gap:.0f} s"
                              for name, gap in sorted(gaps.items(), key=lambda kv: -kv[1]))}


def save_argv(form, bids_dir: Path, run_label: str, tsv: Path) -> list[str]:
    """``nirspipe-prep edit-markers apply`` for one run: the form's flags plus what the page fills."""
    parser = _apply_parser()
    names = {f.action.dest: f.name for _title, group in fields(parser) for f in group}
    merged = MultiDict(form)
    merged.setlist(names["bids_dir"], [str(bids_dir)])
    merged.setlist(names["tsv"], [str(tsv)])
    for flag, value in selection(run_label).items():
        merged.setlist(flag, [value])
    return to_argv([PREP, "edit-markers", "apply"], parser, merged)


class EditViewer(RawRatingApp):
    """A raw report served for editing: ``rate_config`` says so, and the page loads the editor from ``prep/``.

    Ratings and channel decisions stay read-only here; they are made in ``nirspipe-rate``.
    """

    rates = False

    def __init__(self, html_path: Path, output_dir: Path, bids_dir: Path):
        self.bids_dir = Path(bids_dir)
        super().__init__(html_path, output_dir)
        self._setup_prep_routes(self.app)

    def _config(self) -> dict:
        return {**super()._config(), "edit": True}

    def _setup_prep_routes(self, app) -> None:
        @app.get("/prep/<name>")
        def prep_static(name):
            if name not in _STATIC:
                abort(404)
            return send_from_directory(_HERE / "static", _STATIC[name])

        @app.get("/prep/form")
        def prep_form():
            # read just now from this tree, so validating it again only costs the save time
            return form_html({"--skip-bids-validation": ["yes"]})

        @app.get("/prep/events")
        def prep_events():
            return jsonify(read_events(self.bids_dir, request.args.get("run", "")))

        @app.post("/prep/drift")
        def prep_drift():
            rows, error = parse_rows((request.json or {}).get("rows"))
            return jsonify(hint=None if error else drift_hint(rows))

        @app.post("/prep/save")
        def prep_save():
            rows, error = parse_rows(_json_or_none(request.form.get("_rows")))
            if error:
                return jsonify(error=error), 400
            run_label = request.form.get("_run", "")
            if "--participant-label" not in selection(run_label):
                return jsonify(error=f"cannot tell the subject from the run {run_label!r}"), 400
            # kept until the command has read it; the run is in the background
            tsv = Path(tempfile.mkdtemp(prefix="nirspipe_events_")) / "events.tsv"
            write_tsv(pd.DataFrame(rows, columns=_COLUMNS), tsv)
            argv = save_argv(request.form, self.bids_dir, run_label, tsv)
            error = check(parser_for(PREP), argv[1:])
            if error:
                return jsonify(error=error), 400
            try:
                run_id = process_stream.start([executable(PREP), *argv[1:]])
            except OSError as exc:
                return jsonify(error=f"could not launch {PREP}: {exc}"), 500
            return jsonify(events=f"prep/runs/{run_id}/events", stop=f"prep/runs/{run_id}/stop",
                           command=" ".join(argv))

        @app.get("/prep/runs/<run_id>/events")
        def prep_run_events(run_id):
            try:
                stream = process_stream.sse(run_id)
            except KeyError:
                abort(404)
            return Response(stream, mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

        @app.post("/prep/runs/<run_id>/stop")
        def prep_run_stop(run_id):
            return jsonify(stopped=process_stream.stop(run_id))


def _json_or_none(text: "str | None"):
    try:
        return json.loads(text or "null")
    except ValueError:
        return None
