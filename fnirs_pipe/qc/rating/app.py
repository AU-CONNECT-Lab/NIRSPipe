import json
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
import logging as _logging

from flask import Flask, jsonify, request, send_from_directory

from fnirs_pipe.io.derivatives import channel_decisions_path, entity_of
from fnirs_pipe.io.naming import rating_path
from fnirs_pipe.io.naming import report_name
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.utils.net import resolve_port
from fnirs_pipe.utils import load_toml

logger = get_logger("qc.rating")

# ---- Default ports ----
RATE_PORT = 8765
RAW_PORT = 5052
HYPER_PORT = 5053


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _serve_forever(app, port: int, log_name: str, ready_delay: float = 1.0, on_ready=None) -> None:
    """Run a Flask app on a daemon thread, fire on_ready, then block until Ctrl+C."""
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

    failures: list[OSError] = []

    def _serve():
        try:
            app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
        except OSError as err:      # the thread would otherwise die silently, leaving the wait loop below
            failures.append(err)

    threading.Thread(target=_serve, daemon=True).start()
    time.sleep(ready_delay)
    if failures:
        raise SystemExit(f"Could not serve on port {port}: {failures[0]}")
    if on_ready is not None:
        on_ready()

    logger.info("%s on http://localhost:%d - Ctrl+C to stop", log_name, port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("shutting down")


class FNIRSRatingApp:
    def __init__(self, output_dir: Path, subjects: list[str]):
        self.output_dir = output_dir
        self.subjects = subjects
        self.app = Flask(__name__)
        self._setup_routes()

    def _load_ratings(self, subject: str) -> dict:
        toml_path = (self.output_dir / f"sub-{subject}" / "figures"
                     / f"sub-{subject}_ratings.toml")
        if not toml_path.exists():
            return {"ratings": {}, "notes": {}}
        data = load_toml(toml_path)
        ratings = {k: v for k, v in data.items()
                   if isinstance(v, str) and k not in ("subject", "rated_at")}
        notes = data.get("notes", {})
        return {"ratings": ratings, "notes": (notes if isinstance(notes, dict) else {})}

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
        @app.route("/load_ratings/sub-<pid>", methods=["GET"])
        def load_ratings(pid):
            return jsonify(self._load_ratings(pid))

        @app.route("/save_ratings", methods=["POST"])
        def save_ratings():
            return self._handle_save()

    def _handle_save(self):
        data = request.json
        if not data or "id" not in data:
            return jsonify({"status": "fail", "message": "Missing id"}), 400
        # removeprefix, not lstrip: lstrip takes a character set, so "sub-bus01" would come
        # back as "01" and that subject's ratings would overwrite sub-01's
        subject = data["id"].removeprefix("sub-")
        ratings = data.get("ratings", {})
        notes   = data.get("notes", {})
        try:
            self._write_toml(subject, ratings, notes)
            self._append_jsonl(subject, ratings, notes)
            return jsonify({"status": "success"})
        except Exception as e:
            logger.exception("save_ratings failed for sub-%s", subject)
            return jsonify({"status": "fail", "message": str(e)}), 500

    def _write_toml(self, subject: str, ratings: dict, notes: dict) -> None:
        out_dir = self.output_dir / f"sub-{subject}" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"sub-{subject}_ratings.toml"
        ts = _utc_now_iso()
        lines = [f'subject = "{subject}"', f'rated_at = "{ts}"', ""]
        for k, v in ratings.items():
            lines.append(f'{k} = "{v}"')
        if notes:
            lines += ["", "[notes]"]
            for k, v in notes.items():
                escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{k} = "{escaped}"')
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _append_jsonl(self, subject: str, ratings: dict, notes: dict) -> None:
        # beside the run logs, which .bidsignore already waves through
        out = self.output_dir / "logs" / "group_ratings.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "subject": subject,
            "rated_at": _utc_now_iso(),
            **ratings,
            "notes": notes,
        }
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self, port: int | None = None, open_browser: bool = True) -> None:
        served = resolve_port(port or RATE_PORT, explicit=port is not None)

        def _on_ready():
            if not open_browser:
                return
            for subject in self.subjects:
                url = f"http://localhost:{served}/sub-{subject}"
                logger.info("opening %s", url)
                webbrowser.open(url)

        _serve_forever(self.app, served, "fnirs-rate", ready_delay=1.5, on_ready=_on_ready)


class RawRatingApp:
    """Flask server for rating a single raw QC HTML report and annotating channel decisions."""

    def __init__(self, html_path: Path, output_dir: Path, sci_threshold: float = SCI_PASS):
        self.html_path      = html_path
        self.stem           = html_path.stem        # "sub-01_task-rest_desc-raw_report"
        self.output_dir     = output_dir
        self.sci_threshold  = sci_threshold
        self.ratings_path   = self._ratings_path(self.stem)
        run = {k: entity_of(self.stem, k) for k in ("sub", "task", "ses")}
        self.decisions_path = channel_decisions_path(
            output_dir, run["sub"], run["task"], run["ses"])
        self.app = Flask(__name__)
        self._setup_routes()

    def _ratings_path(self, stem: str) -> Path:
        """One file per rated page, from the page's own stem. See :func:`rating_path`."""
        return rating_path(self.output_dir, stem)

    def _load_ratings(self, stem: "str | None" = None) -> dict:
        path = self._ratings_path(stem) if stem else self.ratings_path
        if not path.exists():
            return {"ratings": {}, "notes": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"ratings": data.get("ratings", {}), "notes": data.get("notes", {})}
        except Exception:
            return {"ratings": {}, "notes": {}}

    def _append_jsonl(self, stem: str, ratings: dict, notes: dict) -> None:
        """One line per save, so a tree's raw ratings can be read without walking it.

        The same arrangement ``FNIRSRatingApp`` writes for the subject reports, and for the
        same reason: the per-page files are what a page loads, this is what a group-level
        read needs. Append-only, so the history of a change survives.
        """
        # beside the run logs, which .bidsignore already waves through
        out = self.output_dir / "logs" / "group_raw_ratings.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {"stem": stem, "rated_at": _utc_now_iso(), **ratings, "notes": notes}
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_decisions(self) -> dict:
        if not self.decisions_path.exists():
            return {}
        try:
            return json.loads(self.decisions_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _setup_routes(self) -> None:
        app = self.app

        @app.route("/")
        def index():
            if not self.html_path.exists():
                return f"<h2>Report not found: {self.html_path}</h2>", 404
            return self.html_path.read_text(encoding="utf-8")

        # the per-condition pages sit beside the run's and link to it by name, so they need
        # a route of their own to be reachable here rather than being served as plain files
        @app.route("/<report>.html")
        def sibling_report(report):
            html_file = self.html_path.with_name(f"{report}.html")
            if not html_file.exists():
                return f"<h2>Report not found: {html_file}</h2>", 404
            return html_file.read_text(encoding="utf-8")

        @app.route("/load_raw_ratings", methods=["GET"])
        @app.route("/load_raw_ratings/<stem>", methods=["GET"])
        def load_raw_ratings(stem=None):
            return jsonify(self._load_ratings(stem))

        @app.route("/load_channel_decisions", methods=["GET"])
        def load_channel_decisions():
            return jsonify(self._load_decisions())

        @app.route("/save_raw_ratings", methods=["POST"])
        def save_raw_ratings():
            return self._handle_save_ratings()

        @app.route("/save_channel_decisions", methods=["POST"])
        def save_channel_decisions():
            return self._handle_save_decisions()

        @app.route("/<path:filename>")
        def static_files(filename):
            return send_from_directory(self.output_dir, filename)

    def _handle_save_ratings(self):
        data = request.json
        if not data:
            return jsonify({"status": "fail", "message": "empty body"}), 400
        # the page says which report it is; a per-condition page is not the run it sits
        # beside and must not write into the run's file
        stem = data.get("id") or self.stem
        try:
            ratings, notes = data.get("ratings", {}), data.get("notes", {})
            record = {
                "stem":     stem,
                "rated_at": _utc_now_iso(),
                "ratings":  ratings,
                "notes":    notes,
            }
            self._ratings_path(stem).write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._append_jsonl(stem, ratings, notes)
            return jsonify({"status": "success"})
        except Exception as exc:
            logger.exception("save_raw_ratings failed")
            return jsonify({"status": "fail", "message": str(exc)}), 500

    def _handle_save_decisions(self):
        data = request.json
        if data is None:
            return jsonify({"status": "fail", "message": "empty body"}), 400
        try:
            self.decisions_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return jsonify({"status": "success"})
        except Exception as exc:
            logger.exception("save_channel_decisions failed")
            return jsonify({"status": "fail", "message": str(exc)}), 500

    def run(self, port: int | None = None) -> None:
        served = resolve_port(port or RAW_PORT, explicit=port is not None)

        def _on_ready():
            url = f"http://localhost:{served}/"
            logger.info("raw viewer -> %s", url)
            webbrowser.open(url)

        _serve_forever(self.app, served, "fnirs-rate raw", on_ready=_on_ready)


class HyperRatingApp:
    """Flask server for the hyperscanning raw QC viewer with per-subject channel decisions."""

    def __init__(
        self,
        html_path: Path,
        output_dir: Path,
        subject_ids: list[str],
        sci_threshold: float = SCI_PASS,
    ):
        self.html_path     = html_path
        self.output_dir    = output_dir
        self.subject_ids   = list(subject_ids)
        self.sci_threshold = sci_threshold
        stem = html_path.stem  # "group-A[_ses-01]_task-tapping_desc-raw_report"
        self.group_id  = entity_of(stem, "group") or "unknown"
        self.session   = entity_of(stem, "ses")
        self.task      = entity_of(stem, "task") or "unknown"
        self.ratings_path = self._ratings_path(stem)
        self.app = Flask(__name__)
        self._setup_routes()

    def _ratings_path(self, stem: str) -> Path:
        """One file per rated page, from the page's own stem. See :func:`rating_path`.

        The raw dyad report, the post report and each of its per-condition and per-pairing
        pages are separate reports and are rated separately; the entities in the page's own
        name are what keeps their rating files apart.
        """
        return rating_path(self.output_dir, stem)

    def _load_ratings(self, stem: "str | None" = None) -> dict:
        path = self._ratings_path(stem) if stem else self.ratings_path
        if not path.exists():
            return {"ratings": {}, "notes": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"ratings": data.get("ratings", {}), "notes": data.get("notes", {})}
        except Exception:
            return {"ratings": {}, "notes": {}}

    def _decisions_path(self, sid: str) -> Path:
        return channel_decisions_path(self.output_dir, sid,
                                      task=self.task, session=self.session)

    def _load_decisions(self) -> dict:
        """Return {sid: {ch: state}} by flattening each subject's run-level JSON."""
        result: dict[str, dict] = {}
        for sid in self.subject_ids:
            path = self._decisions_path(sid)
            if not path.exists():
                result[sid] = {}
                continue
            try:
                run_data = json.loads(path.read_text(encoding="utf-8"))
                flat: dict[str, str] = {}
                for run_decisions in run_data.values():
                    if isinstance(run_decisions, dict):
                        flat.update(run_decisions)
                result[sid] = flat
            except Exception:
                result[sid] = {}
        return result

    def _setup_routes(self) -> None:
        app = self.app

        @app.route("/")
        def index():
            if not self.html_path.exists():
                return f"<h2>Report not found: {self.html_path}</h2>", 404
            return self.html_path.read_text(encoding="utf-8")

        @app.route("/load_hyper_ratings", methods=["GET"])
        @app.route("/load_hyper_ratings/<stem>", methods=["GET"])
        def load_hyper_ratings(stem=None):
            return jsonify(self._load_ratings(stem))

        @app.route("/load_decisions", methods=["GET"])
        def load_decisions():
            return jsonify(self._load_decisions())

        @app.route("/save_hyper_ratings", methods=["POST"])
        def save_hyper_ratings():
            return self._handle_save_ratings()

        @app.route("/save_hyper_decisions", methods=["POST"])
        def save_hyper_decisions():
            return self._handle_save_decisions()

        # the report's figure hrefs are relative to the report, which sits in the group's
        # own folder; serving from output_dir would resolve them against the wrong root
        @app.route("/<path:filename>")
        def static_files(filename):
            return send_from_directory(self.html_path.parent, filename)

    def _handle_save_ratings(self):
        data = request.json
        if not data:
            return jsonify({"status": "fail", "message": "empty body"}), 400
        # the page says which report it is; a condition page must not write into the run's
        stem = data.get("id") or self.html_path.stem
        try:
            record = {
                "stem":     stem,
                "rated_at": _utc_now_iso(),
                "ratings":  data.get("ratings", {}),
                "notes":    data.get("notes", {}),
            }
            self._ratings_path(stem).write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return jsonify({"status": "success"})
        except Exception as exc:
            logger.exception("save_hyper_ratings failed")
            return jsonify({"status": "fail", "message": str(exc)}), 500

    def _handle_save_decisions(self):
        """Merge chip states back into each subject's run-level JSON."""
        data = request.json  # {sid: {ch: state}}
        if data is None:
            return jsonify({"status": "fail", "message": "empty body"}), 400
        try:
            for sid, ch_states in data.items():
                path = self._decisions_path(sid)
                if path.exists():
                    try:
                        existing = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        existing = {}
                else:
                    existing = {}
                if existing:
                    for run_key in existing:
                        if isinstance(existing[run_key], dict):
                            existing[run_key].update(ch_states)
                else:
                    existing["_hyper"] = dict(ch_states)
                path.write_text(
                    json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            return jsonify({"status": "success"})
        except Exception as exc:
            logger.exception("save_hyper_decisions failed")
            return jsonify({"status": "fail", "message": str(exc)}), 500

    def run(self, port: int | None = None) -> None:
        served = resolve_port(port or HYPER_PORT, explicit=port is not None)

        def _on_ready():
            url = f"http://localhost:{served}/"
            logger.info("hyper viewer -> %s", url)
            webbrowser.open(url)

        _serve_forever(self.app, served, "fnirs-rate hyper", on_ready=_on_ready)
