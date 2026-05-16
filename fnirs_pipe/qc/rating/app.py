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

    def run(self, port: int = 8765, open_browser: bool = True) -> None:
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

        def _serve():
            self.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

        threading.Thread(target=_serve, daemon=True).start()
        time.sleep(1.5)

        if open_browser:
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


_RAW_SECTIONS = ["Raw_Signal", "SCI_PSP", "Final"]

_CD_CSS = """
<style id="cd-style">
.cd-chip{cursor:pointer;padding:.1rem .5rem;border-radius:3px;font-size:.78rem;
  font-weight:600;display:inline-block;min-width:4.2rem;text-align:center}
.cd-chip.cd-unrated{background:#e9ecef;color:#6c757d}
.cd-chip.cd-good{background:#d4edda;color:#155724}
.cd-chip.cd-bad{background:#f8d7da;color:#721c24}
</style>"""

_CD_SECTION_HTML = """
<div class="card" style="margin-bottom:.5rem" id="ch-decisions-card">
  <div class="panel-title" style="cursor:pointer;user-select:none;display:flex;align-items:center"
       onclick="toggleChDecisions()">
    <span>Channel Decisions &nbsp;<span style="font-weight:400;color:#aaa">good / bad / unrated per channel &bull; click to expand</span></span>
    <span id="ch-decisions-arrow" style="margin-left:auto">&#9658;</span>
  </div>
  <div id="ch-decisions-body" style="display:none;padding:.4rem .6rem .6rem">
    <p style="font-size:.75rem;color:#888;margin:0 0 .4rem">Rows in red have SCI below threshold. Click chip to cycle: &#8212; &#8594; good &#8594; bad. Saves automatically.</p>
    <table style="border-collapse:collapse;width:100%;font-size:.82rem;table-layout:fixed">
      <colgroup><col style="width:48%"/><col style="width:16%"/><col style="width:36%"/></colgroup>
      <thead><tr>
        <th style="text-align:left;padding:.22rem .5rem;border-bottom:2px solid #dde3ea;background:#f8f9fa;font-weight:600">Channel</th>
        <th style="text-align:left;padding:.22rem .5rem;border-bottom:2px solid #dde3ea;background:#f8f9fa;font-weight:600">SCI</th>
        <th style="text-align:left;padding:.22rem .5rem;border-bottom:2px solid #dde3ea;background:#f8f9fa;font-weight:600">Decision</th>
      </tr></thead>
      <tbody id="ch-decisions-tbody"></tbody>
    </table>
  </div>
</div>"""

_CD_JS = """
<script id="cd-script">
(function(){
  var _IS_FLASK     = (typeof _CHANNEL_DECISIONS !== "undefined");
  var _chDecisions  = _IS_FLASK ? (_CHANNEL_DECISIONS || {}) : {};
  var _sciThresh    = (typeof _SCI_THRESHOLD !== "undefined") ? _SCI_THRESHOLD : 0.8;
  var _cdOpen = false, _cdTimer = null;

  window.toggleChDecisions = function() {
    _cdOpen = !_cdOpen;
    document.getElementById("ch-decisions-body").style.display = _cdOpen ? "" : "none";
    document.getElementById("ch-decisions-arrow").innerHTML = _cdOpen ? "&#9660;" : "&#9658;";
    if (_cdOpen) buildTable();
  };

  function buildTable() {
    var tbody = document.getElementById("ch-decisions-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    var idx   = (typeof _run_idx !== "undefined") ? _run_idx : 0;
    var lbls  = (typeof _RUN_LABELS !== "undefined") ? _RUN_LABELS : [];
    var runLbl = lbls[idx] || "";
    var d      = ((typeof _STATIC_DATA !== "undefined") ? _STATIC_DATA : [])[idx] || {};
    var pairs  = d.channel_pairs || [];
    var sciPCh = ((d.iqm || {}).per_channel || {}).sci_per_channel || {};

    var channels = [];
    pairs.forEach(function(p){ channels.push(p+" hbo"); channels.push(p+" hbr"); });
    if (!channels.length) {
      tbody.innerHTML = "<tr><td colspan='3' style='color:#adb5bd;font-style:italic;text-align:center;padding:.5rem'>No channels</td></tr>";
      return;
    }
    var decided = _chDecisions[runLbl] || {};
    pairs.forEach(function(pair){
      var hbo  = pair+" hbo";
      var hbr  = pair+" hbr";
      var vals = Object.keys(sciPCh).filter(function(k){return k.startsWith(pair+" ");}).map(function(k){return sciPCh[k];});
      var sci  = vals.length ? vals[0] : null;
      var below= sci!==null && sci<_sciThresh;
      var tr   = document.createElement("tr");
      tr.style.background = below ? "#fff8f8" : "";

      // pair decision: one chip writes both hbo + hbr simultaneously
      var state= decided[hbo] || "unrated";
      var chip = document.createElement("span");
      chip.className  ="cd-chip cd-"+state;
      chip.textContent={unrated:"—",good:"good",bad:"bad"}[state];
      chip.addEventListener("click",function(){
        if(!_chDecisions[runLbl]) _chDecisions[runLbl]={};
        var next={unrated:"good",good:"bad",bad:"unrated"}[_chDecisions[runLbl][hbo]||"unrated"];
        _chDecisions[runLbl][hbo]=next;
        _chDecisions[runLbl][hbr]=next;
        chip.className  ="cd-chip cd-"+next;
        chip.textContent={unrated:"—",good:"good",bad:"bad"}[next];
        save();
      });

      var td1=document.createElement("td"); td1.style.cssText="padding:.18rem .5rem;border-bottom:1px solid #f0f0f0"; td1.textContent=pair;
      var td2=document.createElement("td"); td2.style.cssText="padding:.18rem .5rem;border-bottom:1px solid #f0f0f0;font-variant-numeric:tabular-nums;color:"+(below?"#c0392b":"#2c3e50"); td2.textContent=sci!==null?sci.toFixed(3):"—";
      var td3=document.createElement("td"); td3.style.cssText="padding:.18rem .5rem;border-bottom:1px solid #f0f0f0"; td3.appendChild(chip);
      tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3);
      tbody.appendChild(tr);
    });
  }

  window._cdBuildTable = buildTable;

  function save(){
    if(!_IS_FLASK) return;
    if(_cdTimer) clearTimeout(_cdTimer);
    _cdTimer=setTimeout(function(){
      fetch("/save_channel_decisions",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(_chDecisions)})
      .then(function(r){return r.json();})
      .then(function(d){if(d.status!=="success") console.warn("cd save failed:",d);})
      .catch(function(e){console.warn("cd save error:",e);});
    },500);
  }
})();
</script>"""


class RawRatingApp:
    """Flask server for rating a single raw QC HTML report and annotating channel decisions."""

    def __init__(self, html_path: Path, output_dir: Path, sci_threshold: float = 0.8):
        self.html_path      = html_path
        self.stem           = html_path.stem          # e.g. "sub-01_task-rest_raw"
        self.output_dir     = output_dir
        self.sci_threshold  = sci_threshold
        self.ratings_path   = output_dir / f"{self.stem}_ratings.json"
        self.decisions_path = output_dir / f"{self.stem}_channel_decisions.json"
        self.app = Flask(__name__)
        self._setup_routes()

    def _extract_run_labels(self, html: str) -> list[str]:
        m = re.search(r"var _RUN_LABELS\s*=\s*(\[.*?\]);", html)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return []

    def _build_modules(self, run_labels: list[str]) -> list[list[dict]]:
        return [
            [{"id": f"{lbl}_{s}", "name": s.replace("_", " ")}
             for s in _RAW_SECTIONS]
            for lbl in run_labels
        ]

    def _load_ratings(self) -> tuple[dict, dict]:
        if not self.ratings_path.exists():
            return {}, {}
        try:
            data = json.loads(self.ratings_path.read_text(encoding="utf-8"))
            return data.get("ratings", {}), data.get("notes", {})
        except Exception:
            return {}, {}

    def _load_decisions(self) -> dict:
        if not self.decisions_path.exists():
            return {}
        try:
            return json.loads(self.decisions_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _inject(self, html: str) -> str:
        run_labels = self._extract_run_labels(html)
        modules    = self._build_modules(run_labels)
        ratings, notes = self._load_ratings()
        decisions  = self._load_decisions()

        decisions_var = (
            f"<script>var _CHANNEL_DECISIONS={json.dumps(decisions, ensure_ascii=False)};"
            f"var _SCI_THRESHOLD={self.sci_threshold};</script>"
        )

        js = (
            _QC_JS
            .replace("__SUBJECT__",  self.stem)
            .replace("__MODULES__",  json.dumps(modules))
            .replace("__RATINGS__",  json.dumps(ratings))
            .replace("__NOTES__",    json.dumps(notes))
            .replace('fetch("/save_ratings"', 'fetch("/save_raw_ratings"')
        )

        html = html.replace("</head>", f"{_QC_CSS}{decisions_var}</head>", 1)
        html = html.replace("<body>",  f"<body>\n{_QC_BAR_HTML}",          1)

        # old HTML files (generated before template update) don't have the
        # channel decisions section — inject it alongside the rating JS
        if 'id="ch-decisions-card"' not in html:
            html = html.replace("</body>", f"{_CD_CSS}{_CD_SECTION_HTML}{_CD_JS}{js}</body>", 1)
        else:
            html = html.replace("</body>", f"{js}</body>", 1)

        return html

    def _setup_routes(self) -> None:
        app = self.app

        @app.route("/")
        def index():
            if not self.html_path.exists():
                return f"<h2>Report not found: {self.html_path}</h2>", 404
            return self._inject(self.html_path.read_text(encoding="utf-8"))

        @app.route("/save_raw_ratings", methods=["POST"])
        def save_raw_ratings():
            return self._handle_save_ratings()

        @app.route("/save_channel_decisions", methods=["POST"])
        def save_channel_decisions():
            return self._handle_save_decisions()

    def _handle_save_ratings(self):
        data = request.json
        if not data:
            return jsonify({"status": "fail", "message": "empty body"}), 400
        try:
            record = {
                "stem":     self.stem,
                "rated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
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
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

        def _serve():
            self.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

        threading.Thread(target=_serve, daemon=True).start()
        time.sleep(1.0)

        url = f"http://localhost:{port}/"
        logger.info("raw viewer → %s", url)
        webbrowser.open(url)

        logger.info("fnirs-rate raw on http://localhost:%d — Ctrl+C to stop", port)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("shutting down")
