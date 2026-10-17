"""The one local web app: command pages, and each rating viewer mounted at run time under ``/view/<token>/``."""

from __future__ import annotations

import logging
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, Response, render_template, send_from_directory
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from nirspipe.interface import command_pages, report_serve
from nirspipe.qc.common.figure_io import PLOTLY_LOCAL_NAME
from nirspipe.utils.logging import get_logger
from nirspipe.utils.net import resolve_port

logger = get_logger("interface.server")

_REPORT_TEMPLATES = Path(__file__).resolve().parents[1] / "qc" / "templates"


def _plotly_js(_prefix: str = "") -> Response:
    from plotly.offline import get_plotlyjs
    return Response(get_plotlyjs(), mimetype="text/javascript")


class WebApp:
    """A Flask app with a home page; viewers are separate Flask apps mounted beside it."""

    def __init__(self) -> None:
        self.flask = Flask(__name__)
        # read on every request, so a viewer mounted after the server started is reachable at once
        self._dispatch = DispatcherMiddleware(self.flask.wsgi_app)
        self.flask.wsgi_app = self._dispatch
        self.links: list[tuple[str, str]] = []
        self.flask.add_url_rule("/", "home", self._home)
        # the reports' own colours and type, so the app looks like the pages it frames
        self.flask.add_url_rule("/tokens.css", "tokens",
                                lambda: send_from_directory(_REPORT_TEMPLATES, "_tokens.css"))
        # a served page's offline fallback climbs out of its viewer's mount, so it lands here
        for rule in (f"/{PLOTLY_LOCAL_NAME}", f"/<path:_prefix>/{PLOTLY_LOCAL_NAME}"):
            self.flask.add_url_rule(rule, rule, _plotly_js)
        self.flask.register_blueprint(command_pages.blueprint(self))
        report_serve.attach(self.flask)

    def mount(self, viewer: Flask, label: str, entries: "tuple[str, ...]" = ("",)) -> list[str]:
        """Serve viewer under ``/view/<token>/`` and return its entry pages' paths."""
        token = uuid.uuid4().hex[:12]
        self._dispatch.mounts[f"/view/{token}"] = viewer
        paths = [f"view/{token}/{entry}" for entry in entries]
        self.links += [(f"{label} {entry.rstrip('/')}".strip(), "/" + path)
                       for entry, path in zip(entries, paths)]
        return paths

    def _home(self):
        return render_template("home.html", links=self.links)

    def serve(self, port: "int | None", default_port: int, log_name: str, paths: list[str]) -> None:
        """Run on a daemon thread, open each path in the browser, then block until Ctrl+C."""
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        served = resolve_port(port or default_port, explicit=port is not None)
        failures: list[OSError] = []

        def _run():
            try:
                self.flask.run(host="127.0.0.1", port=served, debug=False, use_reloader=False)
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
