import json
import math
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
import logging as _logging

from flask import Flask, jsonify, request, send_from_directory

from nirspipe.io.derivatives import channel_decisions_path, entity_of
from nirspipe.io.naming import bad_spans_path, rating_path, report_name
from nirspipe.qc.rating.export import HYPER_KEY
from nirspipe.utils.logging import get_logger
from nirspipe.utils.net import resolve_port

logger = get_logger("qc.rating")

# ---- Default ports ----
RATE_PORT = 8765
RAW_PORT = 5052
HYPER_PORT = 5053


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _serve(app, port: "int | None", default_port: int, log_name: str, paths: list[str]) -> None:
    """Run a Flask app on a daemon thread, open each path in the browser, then block until Ctrl+C."""
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
    served = resolve_port(port or default_port, explicit=port is not None)
    failures: list[OSError] = []

    def _run():
        try:
            app.run(host="127.0.0.1", port=served, debug=False, use_reloader=False)
        except OSError as err:      # the thread would otherwise die silently, leaving the wait loop below
            failures.append(err)

    threading.Thread(target=_run, daemon=True).start()
    time.sleep(1.0)
    if failures:
        raise SystemExit(f"Could not serve on port {served}: {failures[0]}")
    for path in paths:
        url = f"http://localhost:{served}/{path}"
        logger.info("opening %s", url)
        webbrowser.open(url)

    logger.info("%s on http://localhost:%d - Ctrl+C to stop", log_name, served)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("shutting down")


def _read_json(path: Path) -> dict:
    """A JSON object from disk, or an empty one when the file is missing or unreadable."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_rating_file(path: Path) -> dict:
    """A page's saved ratings and notes, or empty ones when it was never rated or is unreadable."""
    data = _read_json(path)
    return {"ratings": data.get("ratings", {}), "notes": data.get("notes", {})}


def _write_rating_file(path: Path, stem: str, ratings: dict, notes: dict) -> None:
    _write_json(path, {"stem": stem, "rated_at": _utc_now_iso(), "ratings": ratings, "notes": notes})


def _save_ratings(output_dir: Path, stem: str, data: dict, log_name: "str | None") -> None:
    """Write one page's rating file and, when the viewer keeps one, append the save to its log."""
    ratings, notes = data.get("ratings", {}), data.get("notes", {})
    _write_rating_file(rating_path(output_dir, stem), stem, ratings, notes)
    if log_name is None:
        return
    # beside the run logs, which .bidsignore already waves through
    out = output_dir / "logs" / log_name
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"stem": stem, "rated_at": _utc_now_iso(), **ratings, "notes": notes}
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
    def __init__(self, output_dir: Path, subjects: list[str]):
        self.output_dir = output_dir
        self.subjects = subjects
        self.app = Flask(__name__)
        self._setup_routes()

    def _inject_base_href(self, html: str, subject: str) -> str:
        """Inject <base href> so relative figure URLs resolve under /sub-<pid> route."""
        base_tag = f'<base href="/sub-{subject}/">'
        return html.replace("<head>", f"<head>\n{base_tag}", 1)

    def _setup_routes(self) -> None:
        app = self.app

        @app.route("/sub-<pid>", strict_slashes=False)
        def participant(pid):
            html_file = (self.output_dir / f"sub-{pid}"
                         / report_name(f"sub-{pid}", desc="index"))
            if not html_file.exists():
                return f"<h2>Report not found: {html_file}</h2>", 404
            return self._inject_base_href(html_file.read_text(encoding="utf-8"), pid)

        # the subject index links the per-run reports next to it, which need a route of
        # their own to be reachable here. Only report pages: this is a path from the URL
        @app.route("/sub-<pid>/<report>")
        def run_report(pid, report):
            if not report.endswith("_report.html"):
                return "", 404
            html_file = self.output_dir / f"sub-{pid}" / report
            if not html_file.exists():
                return f"<h2>Report not found: {html_file}</h2>", 404
            return self._inject_base_href(html_file.read_text(encoding="utf-8"), pid)

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
        @app.route("/load_ratings/<stem>", methods=["GET"])
        def load_ratings(stem):
            return jsonify(_read_rating_file(rating_path(self.output_dir, stem)))

        @app.route("/save_ratings", methods=["POST"])
        def save_ratings():
            data = request.json
            if not data or not data.get("id"):
                return jsonify({"status": "fail", "message": "Missing id"}), 400
            return _respond(
                lambda: _save_ratings(self.output_dir, data["id"], data, "group_ratings.jsonl"),
                f"save_ratings for {data['id']}")

    def run(self, port: int | None = None) -> None:
        _serve(self.app, port, RATE_PORT, "nirspipe-rate",
               [f"sub-{subject}" for subject in self.subjects])


class _ReportViewer:
    """One report page served with its section ratings; subclasses add the channel decisions."""

    route_tag: str
    default_port: int
    log_name: "str | None" = None

    def __init__(self, html_path: Path, output_dir: Path, sci_threshold: "float | None" = None):
        self.html_path     = html_path
        self.stem          = html_path.stem
        self.output_dir    = output_dir
        self.sci_threshold = sci_threshold
        self.ratings_path  = rating_path(output_dir, self.stem)
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
            path = rating_path(self.output_dir, stem) if stem else self.ratings_path
            return jsonify(_read_rating_file(path))

        def save_ratings():
            data = request.json
            if not data:
                return _empty_body()
            # the page says which report it is; a condition page must not write into the run's
            stem = data.get("id") or self.stem
            return _respond(lambda: _save_ratings(self.output_dir, stem, data, self.log_name),
                            f"save_{tag}_ratings")

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

    def run(self, port: int | None = None) -> None:
        _serve(self.app, port, self.default_port, f"nirspipe-rate {self.route_tag}", [""])


class RawRatingApp(_ReportViewer):
    """Rate a single raw QC report and annotate its channel decisions."""

    route_tag = "raw"
    default_port = RAW_PORT
    log_name = "group_raw_ratings.jsonl"

    def __init__(self, html_path: Path, output_dir: Path, sci_threshold: "float | None" = None,
                 mark_spans: bool = False):
        run = {k: entity_of(html_path.stem, k) for k in ("sub", "task", "ses")}
        self.decisions_path = channel_decisions_path(
            output_dir, run["sub"], run["task"], run["ses"])
        self.mark_spans = mark_spans
        self.spans_path = bad_spans_path(output_dir, run["sub"], run["task"], run["ses"])
        super().__init__(html_path, output_dir, sci_threshold)

    def _config(self) -> dict:
        return {**super()._config(), "mark_spans": self.mark_spans}

    def _setup_decision_routes(self, app: Flask) -> None:
        @app.route("/load_channel_decisions", methods=["GET"])
        def load_channel_decisions():
            return jsonify(_read_json(self.decisions_path))

        @app.route("/save_channel_decisions", methods=["POST"])
        def save_channel_decisions():
            data = request.json
            if data is None:
                return _empty_body()
            return _respond(lambda: _write_json(self.decisions_path, data),
                            "save_channel_decisions")

        if not self.mark_spans:
            return

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
            return _respond(lambda: _write_json(self.spans_path, data), "save_bad_spans")


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
        decisions_dir: "Path | None" = None,
    ):
        # the members' decisions live beside their raw pages in the nirspipe tree
        self.decisions_dir = decisions_dir or output_dir
        self.subject_ids   = list(subject_ids)
        stem = html_path.stem  # "group-A[_ses-01]_task-tapping_desc-raw_report"
        self.group_id  = entity_of(stem, "group") or "unknown"
        self.session   = entity_of(stem, "ses")
        self.task      = entity_of(stem, "task") or "unknown"
        super().__init__(html_path, output_dir, sci_threshold)

    def _decisions_path(self, sid: str) -> Path:
        return channel_decisions_path(self.decisions_dir, sid,
                                      task=self.task, session=self.session)

    # the dyad page's verdicts sit under their own key, apart from each run's own
    def _load_decisions(self) -> dict:
        return {sid: _read_json(self._decisions_path(sid)).get(HYPER_KEY, {})
                for sid in self.subject_ids}

    def _save_decisions(self, data: dict) -> None:
        for sid, ch_states in data.items():
            path = self._decisions_path(sid)
            existing = _read_json(path)
            existing[HYPER_KEY] = dict(ch_states)
            _write_json(path, existing)

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
