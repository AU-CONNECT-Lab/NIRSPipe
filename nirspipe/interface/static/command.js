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

  // levels left to right; a level of two runs both from the one before
  function drawStages(levels) {
    stages.textContent = "";
    levels.forEach(function (level, i) {
      if (i) {
        var arrow = document.createElement("span");
        arrow.className = "arrow";
        arrow.textContent = "→";
        stages.appendChild(arrow);
      }
      var box = document.createElement("span");
      box.className = "level";
      level.forEach(function (name) {
        var chip = document.createElement("span");
        chip.className = "stage";
        chip.textContent = name;
        box.appendChild(chip);
      });
      stages.appendChild(box);
    });
  }

  function refresh() {
    post(form.dataset.preview).then(function (d) {
      preview.textContent = d.preview;
      say(d.error || "", d.error ? "bad" : "");
      if (stages && d.stages) drawStages(d.stages);
    });
  }

  // one request after typing pauses, not one per key
  form.addEventListener("input", function () {
    clearTimeout(timer);
    timer = setTimeout(function () { remember(); refresh(); }, 250);
  });
  // a form filled from a run record keeps those values over the remembered ones; a link that
  // gives some folders keeps those and takes the rest from memory
  if (form.dataset.given) {
    var given = {};
    form.dataset.given.split(" ").forEach(function (n) { given[n] = form.elements[n].value; });
    recall();
    Object.keys(given).forEach(function (n) { form.elements[n].value = given[n]; });
    remember();
  } else if (form.dataset.prefilled) remember(); else recall();
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
