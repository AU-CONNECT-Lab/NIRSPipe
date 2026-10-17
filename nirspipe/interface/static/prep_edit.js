// The raw viewer's edit mode for the GUI's Data Preparation: events edited in a table, the run cropped
// on the plot; each written by nirspipe-prep. Works through the viewer's own globals: _markers,
// _selectedMk, _bandShapes, _spanSync, _RUN_LABELS, _run_idx, _markerShapesTS, _spanShapes, _tsEnd,
// _attachSpanEditor, renderMarkerTable, refreshTsShapes and switchRun.
(function () {
  "use strict";

  ["forms.css", "edit.css"].forEach(function (name) {
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "prep/" + name;
    document.head.appendChild(link);
  });

  function json(r) { return r.json(); }

  function section(id, title, desc, body) {
    var el = document.createElement("section");
    el.id = id;
    el.className = "step pe-section";
    el.innerHTML = '<h3 class="step-title">' + title + '</h3><div class="details-body">' +
      '<p class="step-desc">' + desc + '</p>' + body + '<div class="pe-panel"></div></div>';
    return el;
  }

  function tsPlot() {
    var gd = document.getElementById("ts-plot");
    return gd && gd._fullLayout ? gd : null;
  }

  // ---- a form generated from the command's parser, run with Stop and its output streamed ----

  function commandPanel(holder, action, label, extra) {
    holder.innerHTML =
      '<form class="nform" autocomplete="off"><div class="pe-fields"></div>' +
      '<div class="pe-actions"><button type="submit" class="pe-go">' + label + '</button>' +
      '<button type="button" class="pe-stop" disabled>Stop</button>' +
      '<span class="pe-status" role="status"></span></div></form>' +
      '<p class="pe-report" hidden><a target="_blank">Make a raw report of the new recording</a></p>' +
      '<pre class="pe-command" hidden></pre><pre class="pe-log" hidden></pre>';
    var form = holder.querySelector("form");
    var status = holder.querySelector(".pe-status");
    var stop = holder.querySelector(".pe-stop");
    var command = holder.querySelector(".pe-command");
    var log = holder.querySelector(".pe-log");
    var report = holder.querySelector(".pe-report");
    var storeKey = "nirspipe-prep-" + action;
    var stopUrl = null;

    function say(text, tone) {
      status.textContent = text;
      status.className = "pe-status" + (tone ? " " + tone : "");
    }

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

    fetch("prep/form/" + action).then(function (r) { return r.text(); }).then(function (html) {
      holder.querySelector(".pe-fields").innerHTML = html;
      recall();
    });
    form.addEventListener("input", remember);

    function follow(d) {
      stopUrl = d.stop;
      command.hidden = false;
      command.textContent = d.command;
      log.hidden = false;
      log.textContent = "";
      say("Running.", "");
      report.hidden = true;
      stop.disabled = false;
      var source = new EventSource(d.events);
      source.onmessage = function (m) {
        log.textContent += JSON.parse(m.data) + "\n";
        log.scrollTop = log.scrollHeight;
      };
      source.addEventListener("end", function (m) {
        var end = JSON.parse(m.data);
        if (end.code === 0) say("Done.", "ok");
        // this page shows the recording it was opened on; the new one needs its own report
        if (end.code === 0 && d.report) {
          report.querySelector("a").href = d.report;
          report.hidden = false;
        }
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
      var more = extra();
      Object.keys(more).forEach(function (k) { body.append(k, more[k]); });
      fetch("prep/run/" + action, {method: "POST", body: body}).then(json).then(function (d) {
        if (d.error) { say(d.error, "bad"); return; }
        follow(d);
      });
    });

    stop.addEventListener("click", function () {
      if (stopUrl) fetch(stopUrl, {method: "POST"});
    });

    return {say: say};
  }

  // =========================== events ===========================

  var rows = [];            // [onset, duration, name], the table as the command will get it
  var picked = {};          // row index -> true, for the shift buttons
  var colors = {};          // name -> the viewer's colour for it
  var notMarkers = [];      // lower-case prefixes that are not events, from the server
  var driftTimer = null;

  var events = section("EditEvents", "Edit events",
    'The plot and the Events list above follow this table. Save writes a new recording with ' +
    '<code>nirspipe-prep edit-markers apply</code>; the original is left as it is.',
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
    '<p class="qm-note" id="pe-drift"></p>');
  var anchor = document.getElementById("RawSignal");
  anchor.parentNode.insertBefore(events, anchor.nextSibling);
  var tbody = document.getElementById("pe-tbody");
  var eventsPanel = commandPanel(events.querySelector(".pe-panel"), "markers", "Save", function () {
    return {_rows: JSON.stringify(rows)};
  });

  function isMarker(name) {
    var low = String(name).toLowerCase();
    return !notMarkers.some(function (p) { return low.indexOf(p) === 0; });
  }

  function cell(i, k, value, type) {
    var input = document.createElement("input");
    input.type = type;
    if (type === "number") { input.step = "any"; input.min = "0"; }
    // shown to 0.1 ms; the row keeps the stored value until the cell is edited
    input.value = type === "number" && isFinite(value) ? Math.round(value * 1e4) / 1e4 : value;
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
    events.querySelectorAll("[data-which=picked]").forEach(function (b) { b.disabled = !any; });
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
    if (tsPlot()) refreshTsShapes();
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

  events.querySelectorAll("[data-shift]").forEach(function (b) {
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

  function loadEvents() {
    var label = _RUN_LABELS[_run_idx];
    colors = {};
    _markers.forEach(function (m) { colors[m.description] = m.color; });
    rows = [];
    picked = {};
    render();
    eventsPanel.say("Reading the events of " + label + ".", "");
    fetch("prep/events?run=" + encodeURIComponent(label)).then(json).then(function (d) {
      if (d.error) { eventsPanel.say(d.error, "bad"); return; }
      notMarkers = d.not_markers || [];
      rows = d.rows;
      render();
      sync();
      eventsPanel.say("", "");
    });
  }

  // =========================== crop ===========================

  var single = {tmin: NaN, tmax: NaN};
  var segments = [];        // [onset, duration, task]
  var drawing = false;
  var CROP_FILL = "rgba(52, 152, 219, 0.15)";
  var CROP_LINE = {color: "rgba(41, 128, 185, 0.9)", width: 1};

  var crop = section("CropRecording", "Crop",
    'Draw the stretch to keep on the Raw Signal plot, or type it. Crop writes a new recording with ' +
    '<code>nirspipe-prep crop</code>; times are on the recording\'s own clock unless ' +
    '<code>--align trigger</code> is set below.',
    '<div class="pe-tools">' +
    '  <label><input type="radio" name="pc-mode" value="single" checked> One stretch</label>' +
    '  <label><input type="radio" name="pc-mode" value="multi"> Segments</label>' +
    '  <span class="pe-gap"></span>' +
    '  <button type="button" id="pc-draw">Draw on plot</button>' +
    '  <button type="button" id="pc-zoom">Use plot zoom</button>' +
    '  <button type="button" id="pc-clear">Clear</button>' +
    '</div>' +
    '<div class="pe-tools" id="pc-single">' +
    '  <label>Start (s) <input type="number" id="pc-tmin" min="0" step="any" placeholder="start"></label>' +
    '  <label>End (s) <input type="number" id="pc-tmax" min="0" step="any" placeholder="end"></label>' +
    '</div>' +
    '<div id="pc-multi" hidden>' +
    '  <div class="pe-table-wrap"><table class="pe-table">' +
    '    <thead><tr><th>Onset (s)</th><th>Duration (s)</th><th>Task</th><th></th></tr></thead>' +
    '    <tbody id="pc-tbody"></tbody>' +
    '  </table></div>' +
    '  <div class="pe-tools"><button type="button" id="pc-add">+ Segment</button></div>' +
    '</div>');
  events.parentNode.insertBefore(crop, events.nextSibling);
  var segBody = document.getElementById("pc-tbody");

  function mode() { return crop.querySelector("input[name=pc-mode]:checked").value; }

  function cropShapes() {
    var spans = [];
    if (mode() === "single") {
      if (isFinite(single.tmin) || isFinite(single.tmax)) {
        var gd = tsPlot();
        var end = gd ? _tsEnd(gd) : single.tmax;
        spans.push([isFinite(single.tmin) ? single.tmin : 0, isFinite(single.tmax) ? single.tmax : end]);
      }
    } else {
      segments.forEach(function (s) {
        if (isFinite(s[0]) && isFinite(s[1])) spans.push([s[0], s[0] + s[1]]);
      });
    }
    return spans.map(function (s) {
      return {type: "rect", xref: "x", yref: "paper", x0: s[0], x1: s[1], y0: 0, y1: 1,
              layer: "above", editable: true, fillcolor: CROP_FILL, line: CROP_LINE};
    });
  }

  // the crop boxes ride on the viewer's own shape list, so its redraws keep them
  var viewerSpanShapes = window._spanShapes;
  window._spanShapes = function () { return viewerSpanShapes().concat(cropShapes()); };

  function round3(x) { return Math.round(x * 1000) / 1000; }

  function renderSegments() {
    segBody.textContent = "";
    if (!segments.length) {
      segBody.innerHTML = "<tr><td colspan='4' class='pe-empty'>No segments</td></tr>";
    }
    segments.forEach(function (s, i) {
      var tr = document.createElement("tr");
      [[0, "number"], [1, "number"], [2, "text"]].forEach(function (c) {
        var input = document.createElement("input");
        input.type = c[1];
        if (c[1] === "number") { input.step = "any"; input.min = "0"; }
        input.value = isFinite(s[c[0]]) || c[1] === "text" ? s[c[0]] : "";
        input.dataset.i = i;
        input.dataset.k = c[0];
        var td = document.createElement("td");
        td.appendChild(input);
        tr.appendChild(td);
      });
      var td = document.createElement("td");
      var x = document.createElement("button");
      x.type = "button";
      x.className = "pe-del";
      x.dataset.del = i;
      x.textContent = "×";
      x.setAttribute("aria-label", "delete segment " + (i + 1));
      td.appendChild(x);
      tr.appendChild(td);
      segBody.appendChild(tr);
    });
  }

  function renderSingle() {
    document.getElementById("pc-tmin").value = isFinite(single.tmin) ? single.tmin : "";
    document.getElementById("pc-tmax").value = isFinite(single.tmax) ? single.tmax : "";
  }

  function redraw() {
    if (tsPlot()) refreshTsShapes();
  }

  // a box drawn or an edge dragged: read the crop boxes back off the figure
  function onRelayout(ev) {
    if (_spanSync || !ev) return;
    if (!Object.keys(ev).some(function (k) { return k.indexOf("shapes") === 0; })) return;
    var gd = tsPlot();
    if (!gd) return;
    var base = _bandShapes.length + _markerShapesTS().length + viewerSpanShapes().length;
    var boxes = (gd.layout.shapes || []).slice(base).map(function (s) {
      return [round3(Math.max(0, Math.min(s.x0, s.x1))), round3(Math.max(s.x0, s.x1))];
    });
    if (mode() === "single") {
      var last = boxes[boxes.length - 1];
      if (last) { single.tmin = last[0]; single.tmax = last[1]; }
      renderSingle();
    } else {
      segments = boxes.map(function (b, i) {
        return [b[0], round3(b[1] - b[0]), (segments[i] || [])[2] || ""];
      });
      renderSegments();
    }
    redraw();
  }

  function setDrawing(on) {
    drawing = on;
    document.getElementById("pc-draw").textContent = on ? "Stop drawing" : "Draw on plot";
    var gd = tsPlot();
    if (!gd) return;
    if (gd._cropDragmode0 === undefined) gd._cropDragmode0 = gd.layout.dragmode || "zoom";
    Plotly.relayout(gd, on
      ? {dragmode: "drawrect", newshape: {fillcolor: CROP_FILL, line: CROP_LINE, layer: "above"}}
      : {dragmode: gd._cropDragmode0});
  }

  // after each draw of the raw signal: a run switch draws a new figure without the listener
  function attachCrop() {
    var gd = tsPlot();
    if (!gd) return;
    gd.removeListener("plotly_relayout", onRelayout);
    gd.on("plotly_relayout", onRelayout);
    redraw();
    if (drawing) setDrawing(true);
  }
  var viewerAttach = window._attachSpanEditor;
  window._attachSpanEditor = function () { viewerAttach(); attachCrop(); };

  crop.querySelectorAll("input[name=pc-mode]").forEach(function (r) {
    r.addEventListener("change", function () {
      document.getElementById("pc-single").hidden = mode() !== "single";
      document.getElementById("pc-multi").hidden = mode() !== "multi";
      redraw();
    });
  });

  ["tmin", "tmax"].forEach(function (key) {
    document.getElementById("pc-" + key).addEventListener("input", function (e) {
      single[key] = parseFloat(e.target.value);
      redraw();
    });
  });

  segBody.addEventListener("input", function (e) {
    var t = e.target;
    if (t.dataset.i === undefined) return;
    var k = +t.dataset.k;
    segments[+t.dataset.i][k] = k < 2 ? parseFloat(t.value) : t.value;
    redraw();
  });

  segBody.addEventListener("click", function (e) {
    if (e.target.dataset.del === undefined) return;
    segments.splice(+e.target.dataset.del, 1);
    renderSegments();
    redraw();
  });

  document.getElementById("pc-add").addEventListener("click", function () {
    var gd = tsPlot();
    var start = gd ? round3(gd.layout.xaxis.range[0]) : 0;
    segments.push([Math.max(0, start), 30, ""]);
    renderSegments();
    redraw();
  });

  document.getElementById("pc-draw").addEventListener("click", function () { setDrawing(!drawing); });

  document.getElementById("pc-zoom").addEventListener("click", function () {
    var gd = tsPlot();
    if (!gd) return;
    var a = round3(Math.max(0, gd.layout.xaxis.range[0])), b = round3(gd.layout.xaxis.range[1]);
    if (mode() === "single") { single.tmin = a; single.tmax = b; renderSingle(); }
    else { segments.push([a, round3(b - a), ""]); renderSegments(); }
    redraw();
  });

  function clearCrop() {
    single = {tmin: NaN, tmax: NaN};
    segments = [];
    renderSingle();
    renderSegments();
    redraw();
  }
  document.getElementById("pc-clear").addEventListener("click", clearCrop);

  commandPanel(crop.querySelector(".pe-panel"), "crop", "Crop", function () {
    if (mode() === "single") {
      return {_mode: "single", _tmin: isFinite(single.tmin) ? single.tmin : "",
              _tmax: isFinite(single.tmax) ? single.tmax : ""};
    }
    return {_mode: "multi", _segments: JSON.stringify(segments)};
  });

  // ---- a run switch reloads both ----

  var viewerSwitch = window.switchRun;
  window.switchRun = function (idx) {
    viewerSwitch(idx);
    clearCrop();
    loadEvents();
  };

  renderSegments();
  attachCrop();
  loadEvents();
})();
