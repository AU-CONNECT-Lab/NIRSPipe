// One command page: the preview follows the form, Run streams the output, Stop ends it.
(function () {
  var form = document.getElementById("command-form");
  var preview = document.getElementById("preview");
  var status = document.getElementById("status");
  var opened = document.getElementById("opened");
  var log = document.getElementById("log");
  var stop = document.getElementById("stop");
  var stopUrl = null;
  var timer = null;

  function post(url) {
    return fetch(url, {method: "POST", body: new FormData(form)})
      .then(function (r) { return r.json(); });
  }

  function refresh() {
    post(form.dataset.preview).then(function (d) {
      preview.textContent = d.preview;
      status.textContent = d.error || "";
    });
  }

  // one request after typing pauses, not one per key
  form.addEventListener("input", function () {
    clearTimeout(timer);
    timer = setTimeout(refresh, 250);
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

  function follow(d) {
    stopUrl = d.stop;
    log.hidden = false;
    log.textContent = "";
    status.textContent = "Running.";
    if (stop) stop.disabled = false;
    var source = new EventSource(d.events);
    source.onmessage = function (m) {
      log.textContent += JSON.parse(m.data) + "\n";
      log.scrollTop = log.scrollHeight;
    };
    source.addEventListener("end", function (m) {
      var end = JSON.parse(m.data);
      status.textContent = end.code === 0 ? "Finished."
        : end.stopped ? "Stopped." : "Failed (exit " + end.code + ").";
      if (stop) stop.disabled = true;
      source.close();
    });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    post(form.dataset.run).then(function (d) {
      if (d.error) { status.textContent = d.error; return; }
      if (d.opened) { status.textContent = "Opened:"; d.opened.forEach(link); return; }
      follow(d);
    });
  });

  if (stop) stop.addEventListener("click", function () {
    if (stopUrl) fetch(stopUrl, {method: "POST"});
  });
})();
