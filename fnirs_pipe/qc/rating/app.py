import json
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

# Copied from fMRI_QCtoolkit/qc_rating.css, adapted for fNIRS
_QC_CSS = """
<style id="qc-rating-style">
:root {
  --qc-module-width: 110px;
  --qc-module-gap: 6px;
  --qc-run-gap: 16px;
}
#qc-container {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: white;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
  font-family: Arial, sans-serif;
  user-select: none;
  width: 100%;
}
#qc-rating-index, #qc-rating-bar {
  overflow-x: auto;
  width: 100%;
  padding: 6px 0;
  border-bottom: 1px solid #bbb;
  box-sizing: border-box;
}
#qc-rating-index {
  background-color: #e0e8f8;
  border-bottom: 2px solid #0078d7;
  scrollbar-width: none;
}
#qc-rating-index::-webkit-scrollbar { display: none; }
#qc-rating-bar {
  background-color: #f0f4ff;
  scrollbar-width: auto;
}
#qc-rating-bar::-webkit-scrollbar { height: 8px; display: block; }
#qc-rating-bar::-webkit-scrollbar-thumb { background: #bbb; border-radius: 4px; }
#qc-rating-index .qc-inner,
#qc-rating-bar .qc-inner {
  display: flex;
  gap: var(--qc-run-gap);
  width: max-content;
  min-width: 100%;
  box-sizing: content-box;
}
#qc-rating-index a {
  font-weight: 600;
  color: #004a9f;
  text-decoration: none;
  cursor: pointer;
  padding: 6px 8px;
  border-radius: 4px;
  font-size: 14px;
  white-space: nowrap;
  user-select: none;
  display: flex;
  align-items: center;
  justify-content: center;
  width: var(--qc-module-width);
  box-sizing: border-box;
}
.qc-run-group {
  display: flex;
  gap: var(--qc-module-gap);
}
.qc-module {
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 8px;
  border-radius: 6px;
  border: 1.5px solid transparent;
  min-width: var(--qc-module-width);
  box-sizing: border-box;
  user-select: none;
  white-space: nowrap;
  justify-content: center;
  font-size: 14px;
}
.qc-module .qc-icon {
  font-family: monospace;
  font-size: 18px;
  width: 18px;
  text-align: center;
}
.qc-module span.label { font-weight: 600; }
.qc-module.good  { background-color: #d4edda; border-color: #28a745; color: #155724; }
.qc-module.bad   { background-color: #f8d7da; border-color: #dc3545; color: #721c24; }
.qc-module.other { background-color: #fff3cd; border-color: #ffc107; color: #856404; }
</style>
"""

# Two-row sticky bar injected right after <body>, matching fMRI_QCtoolkit/base.html structure
_QC_BAR_HTML = """
<div id="qc-container" title="Ctrl+click to add comment, click to rate: + &#8594; &#8722; &#8594; ?">
  <div id="qc-rating-index"><div class="qc-inner"></div></div>
  <div id="qc-rating-bar"><div class="qc-inner"></div></div>
</div>
"""

# JS injected before </body>, adapted from fMRI_QCtoolkit/qc_rating.js
_QC_JS = """
<script id="qc-rating-script">
(function() {
  var modulesByRun  = __MODULES__;
  var ratings       = Object.assign({}, __RATINGS__);
  var notes         = Object.assign({}, __NOTES__);
  var participantId = "__SUBJECT__";

  var icons        = {NA: "◻", good: "+", bad: "−", other: "?"};
  var ratingStates = ["NA", "good", "bad", "other"];

  var indexInner = document.querySelector("#qc-rating-index .qc-inner");
  var barInner   = document.querySelector("#qc-rating-bar .qc-inner");

  // Build index row (module name links) and bar row (clickable rating buttons)
  modulesByRun.forEach(function(runModules) {
    var runGroupDiv = document.createElement("div");
    runGroupDiv.className = "qc-run-group";

    runModules.forEach(function(mod) {
      // Index link
      var link = document.createElement("a");
      link.textContent = mod.name;
      link.href = "#" + mod.id;
      indexInner.appendChild(link);

      if (!(mod.id in ratings)) ratings[mod.id] = "NA";
      if (!(mod.id in notes))   notes[mod.id]   = "";

      // Rating button
      var modEl = document.createElement("div");
      modEl.className = "qc-module" + (mod.name === "Final" || mod.name.startsWith("Final") ? " final-module" : "");
      modEl.innerHTML = '<span class="qc-icon">◻</span><span class="label">' + mod.name + '</span>';

      function updateDisplay(el, id) {
        var s = ratings[id] || "NA";
        el.classList.remove("good", "bad", "other");
        if (s !== "NA") el.classList.add(s);
        el.querySelector(".qc-icon").textContent = icons[s];
      }

      modEl.addEventListener("click", (function(m, el) {
        return function(e) {
          if (e.ctrlKey) {
            var v = prompt('Comment for "' + m.name + '"', notes[m.id] || "");
            if (v !== null) { notes[m.id] = v.trim(); save(); }
          } else {
            var idx = ratingStates.indexOf(ratings[m.id] || "NA");
            ratings[m.id] = ratingStates[(idx + 1) % ratingStates.length];
            updateDisplay(el, m.id);
            save();
          }
        };
      })(mod, modEl));

      updateDisplay(modEl, mod.id);
      runGroupDiv.appendChild(modEl);
    });

    barInner.appendChild(runGroupDiv);
  });

  function save() {
    fetch("/save_ratings", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ratings: ratings, notes: notes, id: participantId}),
    })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.status !== "success") alert("Failed to save: " + (d.message || ""));
    })
    .catch(function() { alert("Network error while saving rating"); });
  }

  // Synchronised horizontal scrolling between the two rows
  var indexContainer = document.querySelector("#qc-rating-index");
  var barContainer   = document.querySelector("#qc-rating-bar");
  var lockIndex = false, lockBar = false;

  indexContainer.addEventListener("scroll", function() {
    if (!lockIndex) { lockBar = true; barContainer.scrollLeft = indexContainer.scrollLeft; setTimeout(function(){lockBar=false;},10); }
  });
  barContainer.addEventListener("scroll", function() {
    if (!lockBar) { lockIndex = true; indexContainer.scrollLeft = barContainer.scrollLeft; setTimeout(function(){lockIndex=false;},10); }
  });
  indexContainer.addEventListener("wheel", function(e) {
    if (Math.abs(e.deltaX) <= Math.abs(e.deltaY)) {
      e.preventDefault();
      indexContainer.scrollLeft += e.deltaY * 0.5;
    }
  }, {passive: false});
})();
</script>
"""


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

    def _load_ratings(self, subject: str) -> tuple[dict, dict]:
        toml_path = (self.output_dir / f"sub-{subject}" / "figures"
                     / f"sub-{subject}_ratings.toml")
        if not toml_path.exists():
            return {}, {}
        from fnirs_pipe.utils import load_toml
        data = load_toml(toml_path)
        ratings = {k: v for k, v in data.items()
                   if isinstance(v, str) and k not in ("subject", "rated_at")}
        notes = data.get("notes", {})
        return ratings, (notes if isinstance(notes, dict) else {})

    def _inject(self, html: str, subject: str) -> str:
        modules = self._build_modules(subject)
        ratings, notes = self._load_ratings(subject)
        js = (
            _QC_JS
            .replace("__SUBJECT__", subject)
            .replace("__MODULES__", json.dumps(modules))
            .replace("__RATINGS__", json.dumps(ratings))
            .replace("__NOTES__",   json.dumps(notes))
        )
        base_tag = f'<base href="/sub-{subject}/">'
        html = html.replace("<head>",  f"<head>\n{base_tag}",   1)
        html = html.replace("</head>", f"{_QC_CSS}</head>",      1)
        html = html.replace("<body>",  f"<body>\n{_QC_BAR_HTML}", 1)
        html = html.replace("</body>", f"{js}</body>",            1)
        return html

    def _setup_routes(self) -> None:
        app = self.app

        @app.route("/sub-<pid>", strict_slashes=False)
        def participant(pid):
            html_file = self.output_dir / f"sub-{pid}" / f"sub-{pid}_qc.html"
            if not html_file.exists():
                return f"<h2>Report not found: {html_file}</h2>", 404
            return self._inject(html_file.read_text(encoding="utf-8"), pid)

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
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
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
            "rated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            **ratings,
            "notes": notes,
        }
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self, port: int = 8765) -> None:
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

        def _serve():
            self.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

        threading.Thread(target=_serve, daemon=True).start()
        time.sleep(1.5)

        for subject in self.subjects:
            url = f"http://localhost:{port}/sub-{subject}"
            logger.info("opening %s", url)
            webbrowser.open(url)

        logger.info("fnirs-rate on http://localhost:%d — Ctrl+C to stop", port)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("shutting down")
