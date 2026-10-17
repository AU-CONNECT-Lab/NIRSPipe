// One command page: the preview follows the form, Run streams the output, Stop ends it,
// and a finished run shows the report it wrote. The form keeps its last values in this browser.
(function () {
  var form = document.getElementById("command-form");
  var preview = document.getElementById("preview");
  var status = document.getElementById("status");
  var opened = document.getElementById("opened");
  var log = document.getElementById("log");
  var report = document.getElementById("report");
  var stop = document.getElementById("stop");
  var stages = document.getElementById("stages");
  var storeKey = "nirspipe-form:" + form.dataset.run;
  var stopUrl = null;
  var timer = null;

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
      // a table of files is rebuilt from the folder each time, so its cells are not restored
      if (!el.name || !(el.name in saved) || el.dataset.recall === "no") return;
      if (el.type === "checkbox") el.checked = saved[el.name].indexOf(el.value) >= 0;
      else el.value = saved[el.name][0];
    });
  }

  // tone is ok, warn or bad, coloured as the reports colour a verdict
  function say(text, tone) {
    status.textContent = text;
    status.className = "status" + (tone ? " " + tone : "");
  }

  function post(url) {
    return fetch(url, {method: "POST", body: new FormData(form)})
      .then(function (r) { return r.json(); });
  }

  // levels left to right; a level of two runs both from the one before. Drawn as SVG with the
  // page's colours written into it, so the PNG download looks like the page without its CSS.
  var SVG = "http://www.w3.org/2000/svg";

  function svg(tag, attrs, parent) {
    var el = document.createElementNS(SVG, tag);
    Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(el);
    return el;
  }

  // every stage up to Beer-Lambert is preprocessing, the rest postprocessing, GLM its own
  var FAMILIES = [["Preprocessing", "--gui-stage-prep"], ["Postprocessing", "--gui-stage-post"],
                  ["GLM", "--gui-stage-glm"]];

  function families(levels) {
    var post = false;
    return levels.map(function (level) {
      return level.map(function (name) {
        var f = post ? (name === "GLM" ? 2 : 1) : 0;
        if (name === "Beer-Lambert Law") post = true;
        return f;
      });
    });
  }

  function drawStages(levels) {
    var css = getComputedStyle(document.documentElement);
    function token(name) { return css.getPropertyValue(name).trim(); }
    var font = token("--font") || "sans-serif";
    var measure = document.createElement("canvas").getContext("2d");
    measure.font = "13px " + font;
    var PAD = 10, BOX_H = 28, BOX_PAD = 12, GAP_Y = 8, ARROW = 34, KEY_H = 26, SWATCH = 12;
    var kinds = families(levels);
    var shown = FAMILIES.filter(function (f, k) {
      return kinds.some(function (level) { return level.indexOf(k) >= 0; });
    });
    var widths = levels.map(function (level) {
      return Math.max.apply(null, level.map(function (n) { return measure.measureText(n).width; })) + 2 * BOX_PAD;
    });
    var heights = levels.map(function (level) { return level.length * BOX_H + (level.length - 1) * GAP_Y; });
    var width = widths.reduce(function (a, b) { return a + b; }, 0) + ARROW * (levels.length - 1) + 2 * PAD;
    var height = Math.max.apply(null, heights) + 2 * PAD + KEY_H;
    var mid = KEY_H + (height - KEY_H) / 2;

    stages.textContent = "";
    var root = svg("svg", {xmlns: SVG, width: Math.ceil(width), height: height,
                           viewBox: "0 0 " + Math.ceil(width) + " " + height,
                           "font-family": font, "font-size": 13}, stages);
    var defs = svg("defs", {}, root);
    var head = svg("marker", {id: "stage-arrow", viewBox: "0 0 8 8", refX: 7, refY: 4,
                              markerWidth: 8, markerHeight: 8, orient: "auto"}, defs);
    svg("path", {d: "M0,0 L8,4 L0,8 z", fill: token("--faint")}, head);

    var kx = PAD;
    var key = svg("g", {"class": "key", "font-size": 12}, root);
    shown.forEach(function (f) {
      svg("rect", {x: kx + 1, y: PAD + 1, width: SWATCH, height: SWATCH, rx: 3, fill: token("--surface"),
                   stroke: token(f[1]), "stroke-width": 2}, key);
      svg("text", {x: kx + SWATCH + 7, y: PAD + 1 + SWATCH / 2, "dominant-baseline": "central",
                   fill: token("--muted")}, key).textContent = f[0];
      kx += SWATCH + 7 + measure.measureText(f[0]).width + 22;
    });

    var x = PAD;
    levels.forEach(function (level, i) {
      if (i) {
        svg("line", {x1: x + 3, y1: mid, x2: x + ARROW - 4, y2: mid, stroke: token("--faint"),
                     "stroke-width": 1.5, "marker-end": "url(#stage-arrow)"}, root);
        x += ARROW;
      }
      var g = svg("g", {"class": "level"}, root);
      var y = mid - heights[i] / 2;
      level.forEach(function (name, j) {
        var box = svg("g", {"class": "stage"}, g);
        svg("rect", {x: x + 1, y: y, width: widths[i] - 2, height: BOX_H, rx: 6, fill: token("--surface"),
                     stroke: token(FAMILIES[kinds[i][j]][1]), "stroke-width": 1.5}, box);
        svg("text", {x: x + widths[i] / 2, y: y + BOX_H / 2, "text-anchor": "middle",
                     "dominant-baseline": "central", fill: token("--ink")}, box).textContent = name;
        y += BOX_H + GAP_Y;
      });
      x += widths[i];
    });
    var download = document.getElementById("stages-png");
    if (download) download.hidden = false;
  }

  // the drawing as a PNG at twice its size, on its panel's background
  function stagesPng() {
    var root = stages.querySelector("svg");
    if (!root) return;
    var w = +root.getAttribute("width"), h = +root.getAttribute("height"), scale = 2;
    var image = new Image();
    image.onload = function () {
      var canvas = document.createElement("canvas");
      canvas.width = w * scale;
      canvas.height = h * scale;
      var ctx = canvas.getContext("2d");
      ctx.fillStyle = getComputedStyle(stages).backgroundColor || "#fff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(function (blob) {
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "pipeline_stages.png";
        a.click();
        URL.revokeObjectURL(a.href);
      }, "image/png");
    };
    image.src = "data:image/svg+xml;charset=utf-8," +
      encodeURIComponent(new XMLSerializer().serializeToString(root));
  }
  var stagesDownload = document.getElementById("stages-png");
  if (stagesDownload) stagesDownload.addEventListener("click", stagesPng);

  function refresh() {
    post(form.dataset.preview).then(function (d) {
      preview.textContent = d.preview;
      say(d.error || "", d.error ? "bad" : "");
      if (stages && d.stages) drawStages(d.stages);
      if (d.hints) applyHints(d.hints);
    });
  }

  // ---- groups fold; a folded one lists what is set inside it ----

  var foldKey = storeKey + ":folded";
  var groups = Array.prototype.slice.call(form.querySelectorAll("fieldset"));

  // a field as it would read on the command line, or "" when it sends nothing
  function setting(field) {
    var name = field.querySelector("label code").textContent;
    var kind = field.dataset.kind;
    var inputs = Array.prototype.slice.call(field.querySelectorAll("[name]"));
    if (kind === "switch") return inputs[0].checked ? name : "";
    if (kind === "boolean") return inputs[0].value ? inputs[0].selectedOptions[0].textContent : "";
    var values = kind === "choices"
      ? inputs.filter(function (el) { return el.checked; }).map(function (el) { return el.value; })
      : [inputs[0].value.trim().replace(/\s+/g, " ")].filter(Boolean);
    return values.length ? name + " " + values.join(" ") : "";
  }

  function summarise(group) {
    var parts = Array.prototype.map.call(group.querySelectorAll(".field:not(.inactive)"), setting)
      .filter(Boolean);
    var summary = group.querySelector(".group-summary");
    summary.textContent = summary.title = parts.join(" · ");
  }

  function saveFolds() {
    var folded = groups.filter(function (g) { return g.classList.contains("folded"); })
      .map(function (g) { return g.dataset.title; });
    try { localStorage.setItem(foldKey, JSON.stringify(folded)); } catch (e) { /* private window */ }
  }

  var savedFolds = null;
  try { savedFolds = JSON.parse(localStorage.getItem(foldKey) || "null"); } catch (e) { savedFolds = null; }
  groups.forEach(function (group) {
    var legend = group.querySelector("legend");
    var title = legend.textContent.trim();
    group.dataset.title = title;
    group.classList.add("folding");
    legend.textContent = "";
    var toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "fold";
    toggle.textContent = title;
    var summary = document.createElement("span");
    summary.className = "group-summary";
    legend.appendChild(toggle);
    legend.appendChild(summary);
    // first visit: what must be filled stays open, the rest folds
    var folded = savedFolds ? savedFolds.indexOf(title) >= 0 : title !== "Required";
    group.classList.toggle("folded", folded);
    toggle.setAttribute("aria-expanded", String(!folded));
    legend.addEventListener("click", function () {
      var now = group.classList.toggle("folded");
      toggle.setAttribute("aria-expanded", String(!now));
      saveFolds();
    });
  });

  // ---- a command's hints: the defaults an empty field takes, the flags that do nothing as the
  // form stands (the server leaves them out of the command too), suggestions and notes ----

  var hintNotes = document.getElementById("hint-notes");

  function applyHints(h) {
    Array.prototype.forEach.call(form.querySelectorAll(".field"), function (field) {
      var name = field.querySelector("label code").textContent;
      field.classList.toggle("inactive", h.inactive.indexOf(name) >= 0);
      var el = field.querySelector("[name]");
      if (!el) return;
      var text = h.placeholders[name];
      if (el.tagName === "SELECT") {
        var empty = el.querySelector("option[value='']");
        if (!empty) return;
        if (empty.dataset.text === undefined) empty.dataset.text = empty.textContent;
        empty.textContent = !text ? empty.dataset.text
          : text.indexOf("default") >= 0 ? text : "default (" + text + ")";
      } else {
        if (el.dataset.placeholder === undefined) el.dataset.placeholder = el.placeholder || "";
        el.placeholder = text || el.dataset.placeholder;
      }
    });
    Object.keys(h.suggestions).forEach(function (name) {
      var el = form.elements[name];
      if (!el || el.getAttribute("list")) return;
      var list = document.createElement("datalist");
      list.id = "suggest" + name;
      h.suggestions[name].forEach(function (value) {
        var option = document.createElement("option");
        option.value = value;
        list.appendChild(option);
      });
      form.appendChild(list);
      el.setAttribute("list", list.id);
    });
    if (hintNotes) {
      hintNotes.textContent = "";
      h.notes.forEach(function (note) {
        var p = document.createElement("p");
        p.className = "hint-note " + note[1];
        p.textContent = note[0];
        hintNotes.appendChild(p);
      });
    }
    groups.forEach(function (g) {
      g.classList.toggle("all-inactive", !g.querySelector(".field:not(.inactive)"));
      summarise(g);
    });
  }

  // ---- the BIDS and output folders follow from page to page; the last one typed wins ----

  var SHARED = ["bids_dir", "output_dir"];
  var sharedKey = "nirspipe-folders";

  function sharedFolders() {
    try { return JSON.parse(localStorage.getItem(sharedKey) || "{}"); } catch (e) { return {}; }
  }

  function shareFolder(el) {
    if (SHARED.indexOf(el.name) < 0 || !el.value.trim()) return;
    var saved = sharedFolders();
    saved[el.name] = el.value.trim();
    try { localStorage.setItem(sharedKey, JSON.stringify(saved)); } catch (e) { /* private window */ }
  }

  function takeFolders() {
    var saved = sharedFolders();
    SHARED.forEach(function (name) {
      if (form.elements[name] && saved[name]) form.elements[name].value = saved[name];
    });
  }

  // ---- Detect: the dataset's subjects, ticked into --participant-label ----

  var labels = form.elements["--participant-label"];
  if (labels && form.elements.bids_dir && form.dataset.subjects) {
    var field = labels.closest(".field");
    var detect = document.createElement("button");
    detect.type = "button";
    detect.className = "detect";
    detect.textContent = "Detect";
    field.querySelector(".field-head").appendChild(detect);
    var picks = document.createElement("div");
    picks.className = "subject-picks";
    field.appendChild(picks);

    function fill() {
      labels.value = Array.prototype.filter.call(picks.querySelectorAll("input"), function (b) {
        return b.checked;
      }).map(function (b) { return b.value; }).join(" ");
      labels.dispatchEvent(new Event("input", {bubbles: true}));
    }
    picks.addEventListener("change", fill);

    detect.addEventListener("click", function () {
      var url = form.dataset.subjects + "?bids_dir=" + encodeURIComponent(form.elements.bids_dir.value);
      picks.textContent = "Reading the dataset.";
      fetch(url).then(function (r) { return r.json(); }).then(function (d) {
        picks.textContent = "";
        if (d.error) { picks.textContent = d.error; return; }
        var typed = labels.value.split(/\s+/).filter(Boolean);
        // what is already typed stays ticked; an empty box takes every subject
        var ticked = typed.length ? typed : d.subjects;
        d.subjects.forEach(function (s) {
          var label = document.createElement("label");
          var box = document.createElement("input");
          box.type = "checkbox";
          box.value = s;
          box.checked = ticked.indexOf(s) >= 0;
          label.appendChild(box);
          label.appendChild(document.createTextNode(" " + s));
          picks.appendChild(label);
        });
        fill();
      });
    });

  }

  // one request after typing pauses, not one per key
  form.addEventListener("input", function (e) {
    shareFolder(e.target);
    var group = e.target.closest("fieldset");
    if (group) summarise(group);
    clearTimeout(timer);
    timer = setTimeout(function () { remember(); refresh(); }, 250);
  });
  // remembered values first, then the folders the other pages last used; a form filled from a
  // run record keeps its values, and a link that gives some folders keeps those
  if (form.dataset.given) {
    var given = {};
    form.dataset.given.split(" ").forEach(function (n) { given[n] = form.elements[n].value; });
    recall();
    takeFolders();
    Object.keys(given).forEach(function (n) { form.elements[n].value = given[n]; });
    remember();
  } else if (form.dataset.prefilled) {
    takeFolders();
    remember();
  } else {
    recall();
    takeFolders();
  }
  groups.forEach(summarise);

  // the Data Preparation finder reads the same two folders
  var finder = {tree: "output_dir", bids: "bids_dir"};
  Object.keys(finder).forEach(function (id) {
    var box = document.getElementById(id);
    var saved = sharedFolders()[finder[id]];
    if (box && !box.value && saved) box.value = saved;
  });
  refresh();

  function link(item) {
    var li = document.createElement("li");
    var a = document.createElement("a");
    a.href = item.href;
    a.target = "_blank";
    a.textContent = item.label;
    li.appendChild(a);
    opened.appendChild(li);
  }

  function showReport(url) {
    fetch(url).then(function (r) { return r.json(); }).then(function (d) {
      report.textContent = "";
      if (!d.url) { report.textContent = d.message; return; }
      var head = document.createElement("p");
      var a = document.createElement("a");
      a.href = d.url;
      a.target = "_blank";
      a.textContent = "Open in a new tab";
      head.textContent = d.path + " ";
      head.appendChild(a);
      var frame = document.createElement("iframe");
      frame.src = d.url;
      report.appendChild(head);
      // Raw QC links its reports for rating, Data Preparation for editing
      var kind = form.dataset.links || "rate";
      var items = d[kind] || [];
      if (items.length) {
        var links = document.createElement("p");
        links.className = "rate-links";
        links.textContent = kind === "edit" ? "Edit events: " : "Rate: ";
        items.forEach(function (item) {
          var r = document.createElement("a");
          r.href = item.href;
          r.target = "_blank";
          r.textContent = item.label;
          links.appendChild(r);
        });
        report.appendChild(links);
      }
      report.appendChild(frame);
    });
  }

  function follow(d) {
    stopUrl = d.stop;
    log.hidden = false;
    log.textContent = "";
    report.textContent = "";
    say("Running.", "");
    if (stop) stop.disabled = false;
    var source = new EventSource(d.events);
    source.onmessage = function (m) {
      log.textContent += JSON.parse(m.data) + "\n";
      log.scrollTop = log.scrollHeight;
    };
    source.addEventListener("end", function (m) {
      var end = JSON.parse(m.data);
      if (end.code === 0) say("Finished.", "ok");
      else if (end.stopped) say("Stopped.", "warn");
      else say("Failed (exit " + end.code + ").", "bad");
      if (stop) stop.disabled = true;
      source.close();
      if (end.code === 0 && d.report) showReport(d.report);
    });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    remember();
    post(form.dataset.run).then(function (d) {
      if (d.error) { say(d.error, "bad"); return; }
      if (d.opened) { say("Opened:", "ok"); d.opened.forEach(link); return; }
      follow(d);
    });
  });

  if (stop) stop.addEventListener("click", function () {
    if (stopUrl) fetch(stopUrl, {method: "POST"});
  });
})();
