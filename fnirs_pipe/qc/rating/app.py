import json
import re
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from fnirs_pipe.qc.rating.detect import detect_glm, detect_sessions
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.rating")

_SECTIONS        = ["Signal", "Motion", "HbO_HbR", "GLM", "Final"]
_SECTIONS_NO_GLM = ["Signal", "Motion", "HbO_HbR", "Final"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _serve_forever(app, port: int, log_name: str, ready_delay: float = 1.0, on_ready=None) -> None:
    """Run a Flask app on a daemon thread, fire on_ready, then block until Ctrl+C."""
    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

    def _serve():
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    threading.Thread(target=_serve, daemon=True).start()
    time.sleep(ready_delay)
    if on_ready is not None:
        on_ready()

    logger.info("%s on http://localhost:%d — Ctrl+C to stop", log_name, port)
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

    def _build_modules(self, subject: str) -> list[list[dict]]:
        sessions = detect_sessions(self.output_dir, subject)
        sections = _SECTIONS if detect_glm(self.output_dir, subject) else _SECTIONS_NO_GLM
        groups = []
        if sessions == [None]:
            groups.append([
                {"id": s, "name": s.replace("_", " ")}
                for s in sections
            ])
        else:
            for ses in sessions:
                groups.append([
                    {"id": f"{s}_ses-{ses}", "name": f"{s.replace('_', ' ')} s{ses}"}
                    for s in sections
                ])
        return groups

    def _load_ratings(self, subject: str) -> dict:
        toml_path = (self.output_dir / f"sub-{subject}" / "figures"
                     / f"sub-{subject}_ratings.toml")
        if not toml_path.exists():
            return {"ratings": {}, "notes": {}}
        from fnirs_pipe.utils import load_toml
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
            html_file = self.output_dir / f"sub-{pid}" / f"sub-{pid}_qc.html"
            if not html_file.exists():
                return f"<h2>Report not found: {html_file}</h2>", 404
            return self._inject_base_href(html_file.read_text(encoding="utf-8"), pid)

        @app.route("/sub-<pid>/figures/<path:filename>")
        def figures(pid, filename):
            return send_from_directory(
                self.output_dir / f"sub-{pid}" / "figures", filename
            )

        @app.route("/sub-<pid>/nirs/<path:filename>")
        def nirs_files(pid, filename):
            return send_from_directory(
                self.output_dir / f"sub-{pid}" / "nirs", filename
            )

        @app.route("/load_ratings/sub-<pid>", methods=["GET"])
        def load_ratings(pid):
            payload = self._load_ratings(pid)
            payload["modules"] = self._build_modules(pid)
            return jsonify(payload)

        @app.route("/save_ratings", methods=["POST"])
        def save_ratings():
            return self._handle_save()

    def _handle_save(self):
        data = request.json
        if not data or "id" not in data:
            return jsonify({"status": "fail", "message": "Missing id"}), 400
        subject = data["id"].lstrip("sub-")
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
        out = self.output_dir / "group_ratings.jsonl"
        record = {
            "subject": subject,
            "rated_at": _utc_now_iso(),
            **ratings,
            "notes": notes,
        }
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self, port: int = 8765, open_browser: bool = True) -> None:
        def _on_ready():
            if not open_browser:
                return
            for subject in self.subjects:
                url = f"http://localhost:{port}/sub-{subject}"
                logger.info("opening %s", url)
                webbrowser.open(url)

        _serve_forever(self.app, port, "fnirs-rate", ready_delay=1.5, on_ready=_on_ready)


class RawRatingApp:
    """Flask server for rating a single raw QC HTML report and annotating channel decisions."""

    def __init__(self, html_path: Path, output_dir: Path, sci_threshold: float = 0.8):
        self.html_path      = html_path
        self.stem           = html_path.stem          # e.g. "sub-01_task-rest_desc-raw_nirs"
        self.output_dir     = output_dir
        self.sci_threshold  = sci_threshold
        # Strip BIDS suffix so sidecar JSON keeps the legacy "_raw_*.json" naming
        # (also matches the hard-coded paths in hyper_align_callbacks / hyper rating app).
        bids_prefix = self.stem.removesuffix("_desc-raw_nirs")
        self.ratings_path   = output_dir / f"{bids_prefix}_raw_ratings.json"
        self.decisions_path = output_dir / f"{bids_prefix}_raw_channel_decisions.json"
        self.app = Flask(__name__)
        self._setup_routes()

    def _load_ratings(self) -> dict:
        if not self.ratings_path.exists():
            return {"ratings": {}, "notes": {}}
        try:
            data = json.loads(self.ratings_path.read_text(encoding="utf-8"))
            return {"ratings": data.get("ratings", {}), "notes": data.get("notes", {})}
        except Exception:
            return {"ratings": {}, "notes": {}}

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

        @app.route("/load_raw_ratings", methods=["GET"])
        def load_raw_ratings():
            return jsonify(self._load_ratings())

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
        try:
            record = {
                "stem":     self.stem,
                "rated_at": _utc_now_iso(),
                "ratings":  data.get("ratings", {}),
                "notes":    data.get("notes", {}),
            }
            self.ratings_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
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

    def run(self, port: int = 5052) -> None:
        def _on_ready():
            url = f"http://localhost:{port}/"
            logger.info("raw viewer → %s", url)
            webbrowser.open(url)

        _serve_forever(self.app, port, "fnirs-rate raw", on_ready=_on_ready)


class HyperRatingApp:
    """Flask server for the hyperscanning raw QC viewer with per-subject channel decisions."""

    def __init__(
        self,
        html_path: Path,
        output_dir: Path,
        subject_ids: list[str],
        sci_threshold: float = 0.8,
    ):
        self.html_path     = html_path
        self.output_dir    = output_dir
        self.subject_ids   = list(subject_ids)
        self.sci_threshold = sci_threshold
        stem = html_path.stem  # "group-A[_ses-01]_task-tapping_desc-hyperraw_nirs"
        m = re.match(r"group-([^_]+)(?:_ses-([^_]+))?_task-(.+?)_desc-hyperraw_nirs$", stem)
        self.group_id  = m.group(1) if m else "unknown"
        self.session   = m.group(2) if m else None
        self.task      = m.group(3) if m else "unknown"
        # Strip BIDS suffix so ratings JSON keeps the legacy "_hyper-raw_*.json" naming.
        bids_prefix = stem.removesuffix("_desc-hyperraw_nirs")
        self.ratings_path = output_dir / f"{bids_prefix}_hyper-raw_ratings.json"
        self.app = Flask(__name__)
        self._setup_routes()

    def _load_ratings(self) -> dict:
        if not self.ratings_path.exists():
            return {"ratings": {}, "notes": {}}
        try:
            data = json.loads(self.ratings_path.read_text(encoding="utf-8"))
            return {"ratings": data.get("ratings", {}), "notes": data.get("notes", {})}
        except Exception:
            return {"ratings": {}, "notes": {}}

    def _decisions_path(self, sid: str) -> Path:
        sub_prefix = sid if sid.startswith("sub-") else f"sub-{sid}"
        parts = [sub_prefix]
        if self.session:
            parts.append(f"ses-{self.session}")
        parts.append(f"task-{self.task}")
        return self.output_dir / ("_".join(parts) + "_raw_channel_decisions.json")

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
        def load_hyper_ratings():
            return jsonify(self._load_ratings())

        @app.route("/load_decisions", methods=["GET"])
        def load_decisions():
            return jsonify(self._load_decisions())

        @app.route("/save_hyper_ratings", methods=["POST"])
        def save_hyper_ratings():
            return self._handle_save_ratings()

        @app.route("/save_hyper_decisions", methods=["POST"])
        def save_hyper_decisions():
            return self._handle_save_decisions()

        @app.route("/<path:filename>")
        def static_files(filename):
            return send_from_directory(self.output_dir, filename)

    def _handle_save_ratings(self):
        data = request.json
        if not data:
            return jsonify({"status": "fail", "message": "empty body"}), 400
        try:
            record = {
                "stem":     self.html_path.stem,
                "rated_at": _utc_now_iso(),
                "ratings":  data.get("ratings", {}),
                "notes":    data.get("notes", {}),
            }
            self.ratings_path.write_text(
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

    def run(self, port: int = 5053) -> None:
        def _on_ready():
            url = f"http://localhost:{port}/"
            logger.info("hyper viewer → %s", url)
            webbrowser.open(url)

        _serve_forever(self.app, port, "fnirs-rate hyper", on_ready=_on_ready)
