"""SNIRF to BIDS over a folder: one ``nirspipe-snirf2bids`` per SNIRF file, run one after another under one Stop."""

from __future__ import annotations

import re
from pathlib import Path

from flask import jsonify, request, url_for
from werkzeug.datastructures import MultiDict

from nirspipe.interface import process_stream
from nirspipe.interface.commands import executable, parser_for, preview_text
from nirspipe.interface.forms import check, fields, to_argv

# what changes from file to file: the table's column -> the snirf2bids flag it fills
PER_FILE = {"file": "input_file", "subject": "--participant-label", "task": "--task-label",
            "session": "--session-label", "run": "--run-label"}


def detect(folder: str) -> "tuple[list[dict], str]":
    """A row per SNIRF under ``folder``, its subject guessed from the file name, and what was found."""
    root = Path(folder)
    if not root.is_dir():
        return [], f"Folder not found: {folder}"
    rows = [{"file": str(f), "name": f.name, "subject": re.sub(r"[^0-9a-zA-Z]", "", f.stem)}
            for f in sorted(root.rglob("*.snirf"))]
    return rows, f"Found {len(rows)} SNIRF file(s)." if rows else "No .snirf files found."


def shared_groups() -> list:
    """The snirf2bids form's fields less the per-file ones, which the table holds."""
    per_file = set(PER_FILE.values())
    out = []
    for title, group in fields(parser_for("nirspipe-snirf2bids")):
        kept = [f for f in group if f.name not in per_file]
        if kept:
            out.append((title, kept))
    return out


def steps(form) -> "tuple[list[tuple[str, list[str]]], list[str], str | None]":
    """((header, argv) per file with a subject and a task, the files skipped, the first refusal)."""
    parser = parser_for("nirspipe-snirf2bids")
    shared = MultiDict([(k, v) for k, v in form.items(multi=True) if k not in PER_FILE])
    columns = {key: form.getlist(key) for key in PER_FILE}
    built, skipped = [], []
    for i, path in enumerate(columns["file"]):
        row = {key: (values[i] if i < len(values) else "").strip() for key, values in columns.items()}
        name = Path(path).name
        if not row["subject"] or not row["task"]:
            skipped.append(name)
            continue
        merged = MultiDict(shared)
        for key, flag in PER_FILE.items():
            merged.add(flag, row[key])
        argv = to_argv(["nirspipe-snirf2bids"], parser, merged)
        error = check(parser, argv[1:])
        if error:
            return [], skipped, f"{name}: {error}"
        built.append((f"{name} -> sub-{row['subject']}", argv))
    n = len(built)
    return [(f"[{i}/{n}] {header}", argv) for i, (header, argv) in enumerate(built, 1)], skipped, None


def register(bp, default_shell: str) -> None:
    @bp.post("/snirf2bids/batch/preview")
    def snirf2bids_batch_preview():
        built, skipped, error = steps(request.form)
        shell = request.form.get("_shell", default_shell)
        note = error or (f"Skipped, no subject or task: {', '.join(skipped)}" if skipped else None)
        if not built and not error:
            note = "Fill in subject and task for at least one file."
        return jsonify(preview="\n\n".join(preview_text(argv, shell) for _h, argv in built),
                       error=note)

    @bp.post("/snirf2bids/batch/run")
    def snirf2bids_batch_run():
        built, _skipped, error = steps(request.form)
        if error or not built:
            return jsonify(error=error or "Fill in subject and task for at least one file."), 400
        exe = executable("nirspipe-snirf2bids")
        try:
            run_id = process_stream.start_each([(header, [exe, *argv[1:]]) for header, argv in built])
        except OSError as exc:
            return jsonify(error=f"could not launch nirspipe-snirf2bids: {exc}"), 500
        return jsonify(events=url_for("commands.events", run_id=run_id),
                       stop=url_for("commands.stop", run_id=run_id))
