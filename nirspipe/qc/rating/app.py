import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory

from nirspipe.io.derivatives import channel_decisions_path, entity_of
from nirspipe.io.naming import bad_spans_path, rating_path, report_name
from nirspipe.qc.rating.export import HYPER_KEY
from nirspipe.utils.logging import get_logger

logger = get_logger("qc.rating")

# ---- Default ports ----
RATE_PORT = 8765
RAW_PORT = 5052
HYPER_PORT = 5053


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _read_json(path: Path, strict: bool = False) -> dict:
    """A JSON object from disk, or {} when it is missing, or unreadable unless ``strict``."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if strict:
            raise ValueError(_unreadable(path, exc)) from exc
        return {}
    if not isinstance(data, dict) and strict:
        raise ValueError(_unreadable(path, "not a JSON object"))
    return data if isinstance(data, dict) else {}


def _unreadable(path: Path, why) -> str:
    return (f"{path} could not be read ({why}), so saving would drop every other page's "
            "entries in it. Fix the file or move it aside, then save again")


def _write_json(path: Path, data: dict) -> None:
    # through a temporary file, so a save cut short leaves the old file whole
    tmp = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_rating_file(path: Path) -> dict:
    """A page's saved ratings and notes, or empty ones when it was never rated or is unreadable."""
    data = _read_json(path)
    return {"ratings": data.get("ratings", {}), "notes": data.get("notes", {})}


def _write_rating_file(path: Path, stem: str, ratings: dict, notes: dict) -> None:
    _write_json(path, {"stem": stem, "rated_at": _utc_now_iso(), "ratings": ratings, "notes": notes})


def _save_ratings(output_dir: Path, stem: str, data: dict, log_name: "str | None",
                  rater: "str | None" = None) -> None:
    """Write one page's rating file and, when the viewer keeps one, append the save to its log."""
    ratings, notes = data.get("ratings", {}), data.get("notes", {})
    _write_rating_file(rating_path(output_dir, stem, rater), stem, ratings, notes)
    if log_name is None:
        return
    # beside the run logs, which .bidsignore already waves through
    out = output_dir / "logs" / log_name
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"stem": stem, **({"rater": rater} if rater else {}), "rated_at": _utc_now_iso(),
              **ratings, "notes": notes}
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _respond(action, label: str):
    """Run a save and answer the page with its status."""
    try:
        action()
    except Exception as exc:
        logger.exception("%s failed", label)
        return jsonify({"status": "fail", "message": str(exc)}), 500
    return jsonify({"status": "success"})


def _empty_body():
    return jsonify({"status": "fail", "message": "empty body"}), 400


def _span_error(data) -> "str | None":
    """Why a page's ``{run label: [[start, stop], ...]}`` cannot be saved, or None when it can."""
    if not isinstance(data, dict):
        return "expected {run label: [[start, stop], ...]}"
    for key, spans in data.items():
        # a span applies to the whole run, so only the run page may mark one
        if entity_of(key, "cond"):
            return f"{key}: spans are marked on the run page, not a condition page"
        if not isinstance(spans, list):
            return f"{key}: expected a list of [start, stop]"
        for span in spans:
            if (not isinstance(span, list) or len(span) != 2
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in span)):
                return f"{key}: {span!r} is not [start, stop] in seconds"
            if not 0 <= span[0] < span[1]:
                return f"{key}: {span!r} needs 0 <= start < stop"
    return None


class FNIRSRatingApp:
    def __init__(self, output_dir: Path, subjects: list[str], rater: "str | None" = None):
        self.output_dir = output_dir
        self.subjects = subjects
        self.rater = rater
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self) -> None:
        app = self.app

        # a folder URL, so the index's relative links and rating calls resolve under sub-<pid>/
        @app.route("/sub-<pid>/")
        def participant(pid):
            html_file = (self.output_dir / f"sub-{pid}"
                         / report_name(f"sub-{pid}", desc="index"))
            if not html_file.exists():
                return f"<h2>Report not found: {html_file}</h2>", 404
            return html_file.read_text(encoding="utf-8")

        # the subject index links the per-run reports next to it, which need a route of
        # their own to be reachable here. Only report pages: this is a path from the URL
        @app.route("/sub-<pid>/<report>")
        def run_report(pid, report):
            if not report.endswith("_report.html"):
                return "", 404
            html_file = self.output_dir / f"sub-{pid}" / report
            if not html_file.exists():
                return f"<h2>Report not found: {html_file}</h2>", 404
            return html_file.read_text(encoding="utf-8")

        @app.route("/sub-<pid>/figures/<path:filename>")
        def figures(pid, filename):
            return send_from_directory(
                self.output_dir / f"sub-{pid}" / "figures", filename
            )

        @app.route("/sub-<pid>/nirs/<path:filename>")
        @app.route("/sub-<pid>/ses-<ses>/nirs/<path:filename>")
        def nirs_files(pid, filename, ses=None):
            folder = self.output_dir / f"sub-{pid}" / (f"ses-{ses}" if ses else "") / "nirs"
            return send_from_directory(folder, filename)

        # the report page owns its own module list: only it knows which panels were drawn
        @app.route("/sub-<pid>/load_ratings/<stem>", methods=["GET"])
        def load_ratings(pid, stem):
            return jsonify({**_read_rating_file(rating_path(self.output_dir, stem, self.rater)),
                            "rater": self.rater})

        @app.route("/sub-<pid>/save_ratings", methods=["POST"])
        def save_ratings(pid):
            data = request.json
            if not data or not data.get("id"):
                return jsonify({"status": "fail", "message": "Missing id"}), 400
            return _respond(
                lambda: _save_ratings(self.output_dir, data["id"], data, "group_ratings.jsonl",
                                      self.rater),
                f"save_ratings for {data['id']}")


class _ReportViewer:
    """One report page served with its section ratings; subclasses add the channel decisions."""

    route_tag: str
    default_port: int
    log_name: "str | None" = None
    # False serves the page read-only for ratings and decisions, which only nirspipe-rate makes
    rates: bool = True

    def __init__(self, html_path: Path, output_dir: Path, sci_threshold: "float | None" = None,
                 rater: "str | None" = None):
        self.html_path     = html_path
        self.stem          = html_path.stem
        self.output_dir    = output_dir
        self.sci_threshold = sci_threshold
        self.rater         = rater
        self.ratings_path  = rating_path(output_dir, self.stem, rater)
        self.app = Flask(__name__)
        self._setup_routes()

    def _config(self) -> dict:
        return {"sci_threshold": self.sci_threshold}

    def _setup_routes(self) -> None:
        app, tag = self.app, self.route_tag

        @app.route("/")
        def index():
            if not self.html_path.exists():
                return f"<h2>Report not found: {self.html_path}</h2>", 404
            return self.html_path.read_text(encoding="utf-8")

        # a second SCI line to outline cells against; None leaves the run's own colours alone
        @app.route("/rate_config", methods=["GET"])
        def rate_config():
            return jsonify(self._config())

        def load_ratings(stem=None):
            path = rating_path(self.output_dir, stem, self.rater) if stem else self.ratings_path
            return jsonify({**_read_rating_file(path), "rater": self.rater})

        def save_ratings():
            data = request.json
            if not data:
                return _empty_body()
            # the page says which report it is; a condition page must not write into the run's
            stem = data.get("id") or self.stem
            return _respond(
                lambda: _save_ratings(self.output_dir, stem, data, self.log_name, self.rater),
                f"save_{tag}_ratings")

        if self.rates:
            app.add_url_rule(f"/load_{tag}_ratings", "load_ratings", load_ratings)
            app.add_url_rule(f"/load_{tag}_ratings/<stem>", "load_ratings", load_ratings)
            app.add_url_rule(f"/save_{tag}_ratings", "save_ratings", save_ratings, methods=["POST"])
            self._setup_decision_routes(app)

        # the page's hrefs, its sibling pages and the subject index included, are relative
        # to the report's own folder
        @app.route("/<path:filename>")
        def static_files(filename):
            return send_from_directory(self.html_path.parent, filename)

    def _setup_decision_routes(self, app: Flask) -> None:
        raise NotImplementedError


def _merge_into(path: Path, key: str, data: dict) -> None:
    """Write one page's entry into a file the task's other pages and its dyad page share."""
    existing = _read_json(path, strict=True)
    if key in data:
        existing[key] = data[key]
    _write_json(path, existing)


class RawRatingApp(_ReportViewer):
    """Rate one recording's raw QC report and annotate its channel decisions."""

    route_tag = "raw"
    default_port = RAW_PORT
    log_name = "group_raw_ratings.jsonl"

    def __init__(self, html_path: Path, output_dir: Path, sci_threshold: "float | None" = None,
                 rater: "str | None" = None):
        run = {k: entity_of(html_path.stem, k) for k in ("sub", "task", "ses")}
        # the page's own key in the files below, which hold every run of its task
        self.run_label = html_path.stem.removesuffix("_desc-raw_report")
        self.decisions_path = channel_decisions_path(
            output_dir, run["sub"], run["task"], run["ses"], rater)
        self.spans_path = bad_spans_path(output_dir, run["sub"], run["task"], run["ses"], rater)
        super().__init__(html_path, output_dir, sci_threshold, rater)

    def _config(self) -> dict:
        # bad stretches are marked where decisions are made, never on a read-only page
        return {**super()._config(), "mark_spans": self.rates}

    def _setup_decision_routes(self, app: Flask) -> None:
        @app.route("/load_channel_decisions", methods=["GET"])
        def load_channel_decisions():
            return jsonify(_read_json(self.decisions_path))

        @app.route("/save_channel_decisions", methods=["POST"])
        def save_channel_decisions():
            data = request.json
            if data is None:
                return _empty_body()
            return _respond(lambda: _merge_into(self.decisions_path, self.run_label, data),
                            "save_channel_decisions")

        @app.route("/load_bad_spans", methods=["GET"])
        def load_bad_spans():
            return jsonify(_read_json(self.spans_path))

        @app.route("/save_bad_spans", methods=["POST"])
        def save_bad_spans():
            data = request.json
            if data is None:
                return _empty_body()
            error = _span_error(data)
            if error:
                return jsonify({"status": "fail", "message": error}), 400
            # millisecond seconds, in time order, so the file reads like the table it becomes
            data = {key: sorted([round(a, 3), round(b, 3)] for a, b in spans)
                    for key, spans in data.items()}
            return _respond(lambda: _merge_into(self.spans_path, self.run_label, data),
                            "save_bad_spans")


class HyperRatingApp(_ReportViewer):
    """Rate a dyad's raw QC report and annotate each member's channel decisions."""

    route_tag = "hyper"
    default_port = HYPER_PORT

    def __init__(
        self,
        html_path: Path,
        output_dir: Path,
        subject_ids: list[str],
        sci_threshold: "float | None" = None,
        rater: "str | None" = None,
    ):
        self.subject_ids   = list(subject_ids)
        stem = html_path.stem  # "group-A[_ses-01]_task-tapping_desc-raw_report"
        self.group_id  = entity_of(stem, "group") or "unknown"
        self.session   = entity_of(stem, "ses")
        self.task      = entity_of(stem, "task") or "unknown"
        super().__init__(html_path, output_dir, sci_threshold, rater)

    def _decisions_path(self, sid: str) -> Path:
        # the members' decisions live beside their raw pages, in the tree the dyad page is in
        return channel_decisions_path(self.output_dir, sid,
                                      task=self.task, session=self.session, rater=self.rater)

    # the dyad page's verdicts sit under their own key, apart from each run's own
    def _load_decisions(self) -> dict:
        return {sid: _read_json(self._decisions_path(sid)).get(HYPER_KEY, {})
                for sid in self.subject_ids}

    def _save_decisions(self, data: dict) -> None:
        # every member's file read before any is written, so one unreadable file saves none
        files = {sid: _read_json(self._decisions_path(sid), strict=True) for sid in data}
        for sid, ch_states in data.items():
            files[sid][HYPER_KEY] = dict(ch_states)
            _write_json(self._decisions_path(sid), files[sid])

    def _setup_decision_routes(self, app: Flask) -> None:
        @app.route("/load_decisions", methods=["GET"])
        def load_decisions():
            return jsonify(self._load_decisions())

        @app.route("/save_hyper_decisions", methods=["POST"])
        def save_hyper_decisions():
            data = request.json  # {sid: {ch: state}}
            if data is None:
                return _empty_body()
            # the member ids name the files written, so only this dyad's members are taken
            unknown = sorted(set(data) - set(self.subject_ids))
            if unknown:
                return jsonify({"status": "fail",
                                "message": f"not a member of this group: {unknown}"}), 400
            return _respond(lambda: self._save_decisions(data), "save_hyper_decisions")
