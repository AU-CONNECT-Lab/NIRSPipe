// The raw viewer's events made editable for the GUI's Data Preparation; Save runs nirspipe-prep edit-markers apply.
// Works through the viewer's own globals: _markers, _selectedMk, _RUN_LABELS, _run_idx,
// renderMarkerTable, refreshTsShapes and switchRun.
(function () {
  "use strict";

  var rows = [];            // [onset, duration, name], the table as the command will get it
  var picked = {};          // row index -> true, for the shift buttons
  var colors = {};          // name -> the viewer's colour for it
  var notMarkers = [];      // lower-case prefixes that are not events, from the server
  var stopUrl = null;
  var storeKey = "nirspipe-prep-edit";
  var driftTimer = null;

  ["forms.css", "edit.css"].forEach(function (name) {
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "prep/" + name;
    document.head.appendChild(link);
  });

  var section = document.createElement("section");
  section.id = "EditEvents";
  section.className = "step";
  section.innerHTML =
    '<h3 class="step-title">Edit events</h3>' +
    '<div class="details-body">' +
    '<p class="step-desc">The plot and the Events list above follow this table. Save writes a new recording ' +
    'with <code>nirspipe-prep edit-markers apply</code>; the original is left as it is.</p>' +
    '<div class="pe-tools">' +
    '  <label>Step (s) <input type="number" id="pe-step" value="1" min="0" step="any"></label>' +
    '  <button type="button" data-shift="-1" data-which="all">&minus; All</button>' +
    '  <button type="button" data-shift="1" data-which="all">+ All</button>' +
    '  <button type="button" data-shift="-1" data-which="picked" disabled>&minus; Picked</button>' +
    '  <button type="button" data-shift="1" data-which="picked" disabled>+ Picked</button>' +
    '  <span class="pe-gap"></span>' +
    '  <label>Rename <input type="text" id="pe-from" placeholder="old"></label>' +
    '  <label>to <input type="text" id="pe-to" placeholder="new"></label>' +
    '  <button type="button" id="pe-rename">Rename all</button>' +
    '  <span class="pe-gap"></span>' +
    '  <button type="button" id="pe-add">+ Add</button>' +
    '  <button type="button" id="pe-download">Download TSV</button>' +
    '</div>' +
    '<div class="pe-table-wrap"><table class="pe-table">' +
    '  <thead><tr><th></th><th>Onset (s)</th><th>Duration (s)</th><th>Name</th><th></th></tr></thead>' +
    '  <tbody id="pe-tbody"></tbody>' +
    '</table></div>' +
    '<p class="qm-note" id="pe-drift"></p>' +
    '<form class="nform" id="pe-form" autocomplete="off">' +
    '  <div id="pe-fields"></div>' +
    '  <div class="pe-actions">' +
    '    <button type="submit" id="pe-save">Save</button>' +
    '    <button type="button" id="pe-stop" disabled>Stop</button>' +
    '    <span id="pe-status" class="pe-status" role="status"></span>' +
    '  </div>' +
    '</form>' +
    '<pre id="pe-command" class="pe-command" hidden></pre>' +
    '<pre id="pe-log" class="pe-log" hidden></pre>' +
    '</div>';
  var anchor = document.getElementById("RawSignal");
  anchor.parentNode.insertBefore(section, anchor.nextSibling);

  var tbody = document.getElementById("pe-tbody");
  var form = document.getElementById("pe-form");
  var status = document.getElementById("pe-status");
  var log = document.getElementById("pe-log");
  var command = document.getElementById("pe-command");
  var stop = document.getElementById("pe-stop");

  function json(r) { return r.json(); }

  function say(text, tone) {
    status.textContent = text;
    status.className = "pe-status" + (tone ? " " + tone : "");
  }

  function isMarker(name) {
    var low = String(name).toLowerCase();
    return !notMarkers.some(function (p) { return low.indexOf(p) === 0; });
  }

  // ---- the table ----

  function cell(i, k, value, type) {
    var input = document.createElement("input");
    input.type = type;
    if (type === "number") { input.step = "any"; input.min = "0"; }
    input.value = value;
    input.dataset.i = i;
    input.dataset.k = k;
    var td = document.createElement("td");
    td.appendChild(input);
    return td;
  }

  function render() {
    tbody.textContent = "";
    if (!rows.length) {
      tbody.innerHTML = "<tr><td colspan='5' class='pe-empty'>No events</td></tr>";
    }
    rows.forEach(function (r, i) {
      var tr = document.createElement("tr");
      var pick = document.createElement("td");
      var box = document.createElement("input");
      box.type = "checkbox";
      box.checked = !!picked[i];
      box.dataset.pick = i;
      box.setAttribute("aria-label", "pick row " + (i + 1));
      var dot = document.createElement("span");
      dot.className = "mk-dot";
      dot.style.background = colors[r[2]] || "#888";
      pick.appendChild(box);
      pick.appendChild(dot);
      tr.appendChild(pick);
      tr.appendChild(cell(i, 0, r[0], "number"));
      tr.appendChild(cell(i, 1, r[1], "number"));
      tr.appendChild(cell(i, 2, r[2], "text"));
      var del = document.createElement("td");
      var x = document.createElement("button");
      x.type = "button";
      x.className = "pe-del";
      x.dataset.del = i;
      x.textContent = "×";
      x.setAttribute("aria-label", "delete row " + (i + 1));
      del.appendChild(x);
      tr.appendChild(del);
      tbody.appendChild(tr);
    });
    var any = Object.keys(picked).length > 0;
    section.querySelectorAll("[data-which=picked]").forEach(function (b) { b.disabled = !any; });
  }

  // the viewer's plot and Events list, redrawn from the table
  function sync() {
    _markers = rows
      .filter(function (r) { return isMarker(r[2]) && isFinite(r[0]) && isFinite(r[1]); })
      .map(function (r) {
        return {onset: r[0], duration: r[1], description: r[2], color: colors[r[2]] || "#888"};
      });
    _selectedMk = -1;
    renderMarkerTable();
    if (document.getElementById("ts-plot").data) refreshTsShapes();
    clearTimeout(driftTimer);
    driftTimer = setTimeout(drift, 300);
  }

  function drift() {
    var p = document.getElementById("pe-drift");
    fetch("prep/drift", {method: "POST", headers: {"Content-Type": "application/json"},
                         body: JSON.stringify({rows: rows})})
      .then(json).then(function (d) {
        p.textContent = "";
        if (!d.hint) return;
        var b = document.createElement("b");
        b.textContent = d.hint.cutoff;
        p.appendChild(document.createTextNode("Drift high-pass for a GLM on this run: "));
        p.appendChild(b);
        p.appendChild(document.createTextNode(" or lower, set by " + d.hint.by +
                                              ". Slowest repeat per condition: " + d.hint.per + "."));
      });
  }

  tbody.addEventListener("input", function (e) {
    var t = e.target;
    if (t.dataset.i === undefined) return;
    var k = +t.dataset.k;
    rows[+t.dataset.i][k] = k < 2 ? parseFloat(t.value) : t.value;
    sync();
  });

  tbody.addEventListener("change", function (e) {
    var t = e.target;
    if (t.dataset.pick === undefined) return;
    if (t.checked) picked[t.dataset.pick] = true; else delete picked[t.dataset.pick];
    render();
  });

  tbody.addEventListener("click", function (e) {
    var t = e.target;
    if (t.dataset.del === undefined) return;
    var gone = +t.dataset.del;
    rows.splice(gone, 1);
    var kept = {};
    Object.keys(picked).forEach(function (i) {
      if (+i < gone) kept[i] = true; else if (+i > gone) kept[+i - 1] = true;
    });
    picked = kept;
    render();
    sync();
  });

  section.querySelectorAll("[data-shift]").forEach(function (b) {
    b.addEventListener("click", function () {
      var step = (parseFloat(document.getElementById("pe-step").value) || 0) * +b.dataset.shift;
      rows.forEach(function (r, i) {
        if (b.dataset.which === "all" || picked[i]) {
          r[0] = Math.max(0, Math.round((r[0] + step) * 1e4) / 1e4);
        }
      });
      render();
      sync();
    });
  });

  document.getElementById("pe-rename").addEventListener("click", function () {
    var from = document.getElementById("pe-from").value;
    var to = document.getElementById("pe-to").value;
    if (!from) return;
    if (colors[from] && !colors[to]) colors[to] = colors[from];
    rows.forEach(function (r) { if (r[2] === from) r[2] = to; });
    document.getElementById("pe-from").value = "";
    document.getElementById("pe-to").value = "";
    render();
    sync();
  });

  document.getElementById("pe-add").addEventListener("click", function () {
    rows.push([0, 0, ""]);
    render();
    sync();
  });

  document.getElementById("pe-download").addEventListener("click", function () {
    var text = "onset\tduration\ttrial_type\n" +
      rows.map(function (r) { return r.join("\t"); }).join("\n") + "\n";
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], {type: "text/tab-separated-values"}));
    a.download = _RUN_LABELS[_run_idx] + "_events.tsv";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  // ---- loading a run ----

  function load() {
    var label = _RUN_LABELS[_run_idx];
    colors = {};
    _markers.forEach(function (m) { colors[m.description] = m.color; });
    rows = [];
    picked = {};
    render();
    say("Reading the events of " + label + ".", "");
    fetch("prep/events?run=" + encodeURIComponent(label)).then(json).then(function (d) {
      if (d.error) { say(d.error, "bad"); return; }
      notMarkers = d.not_markers || [];
      rows = d.rows;
      render();
      sync();
      say("", "");
    });
  }

  var viewerSwitch = window.switchRun;
  window.switchRun = function (idx) {
    viewerSwitch(idx);
    load();
  };

  // ---- the command ----

  function remember() {
    var values = {};
    new FormData(form).forEach(function (v, k) { (values[k] = values[k] || []).push(v); });
    try { localStorage.setItem(storeKey, JSON.stringify(values)); } catch (e) { /* private window */ }
  }

  function recall() {
    var saved = null;
    try { saved = JSON.parse(localStorage.getItem(storeKey) || "null"); } catch (e) { saved = null; }
    if (!saved) return;
    Array.prototype.forEach.call(form.elements, function (el) {
      if (!el.name || !(el.name in saved)) return;
      if (el.type === "checkbox") el.checked = saved[el.name].indexOf(el.value) >= 0;
      else el.value = saved[el.name][0];
    });
  }

  fetch("prep/form").then(function (r) { return r.text(); }).then(function (html) {
    document.getElementById("pe-fields").innerHTML = html;
    recall();
  });
  form.addEventListener("input", remember);

  function follow(d) {
    stopUrl = d.stop;
    command.hidden = false;
    command.textContent = d.command;
    log.hidden = false;
    log.textContent = "";
    say("Saving.", "");
    stop.disabled = false;
    var source = new EventSource(d.events);
    source.onmessage = function (m) {
      log.textContent += JSON.parse(m.data) + "\n";
      log.scrollTop = log.scrollHeight;
    };
    source.addEventListener("end", function (m) {
      var end = JSON.parse(m.data);
      if (end.code === 0) say("Saved.", "ok");
      else if (end.stopped) say("Stopped.", "warn");
      else say("Failed (exit " + end.code + ").", "bad");
      stop.disabled = true;
      source.close();
    });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    remember();
    var body = new FormData(form);
    body.append("_run", _RUN_LABELS[_run_idx]);
    body.append("_rows", JSON.stringify(rows));
    fetch("prep/save", {method: "POST", body: body}).then(json).then(function (d) {
      if (d.error) { say(d.error, "bad"); return; }
      follow(d);
    });
  });

  stop.addEventListener("click", function () {
    if (stopUrl) fetch(stopUrl, {method: "POST"});
  });

  load();
})();
