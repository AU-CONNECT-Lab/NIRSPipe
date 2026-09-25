"""Serving generated QC reports back into the GUI as an inline preview.

Reports are served as files, not through `srcDoc`: `desc-subjects_report.html` is an iframe
shell whose panels are sibling files, and a `srcdoc` document has no base URL to resolve
them against. The route is keyed by an opaque token rather than by the
path, which keeps Windows drive letters out of URLs and means the page can only ever ask for
a directory some run actually produced. Flask's `send_from_directory` refuses to escape the
root it is given, so the token cannot be used to walk the filesystem either.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("interface.report_serve")

# token -> directory a completed run wrote into. Populated by the run callbacks, read by the
# Flask route below, both in this one process.
_ROOTS: dict[str, Path] = {}


def register(directory: Path) -> str:
    """Make directory reachable over /report/<token>/ and return the token.

    The same directory always gets the same token, so re-running a report does not leak
    entries and the iframe's URL stays stable across runs.
    """
    resolved = Path(directory).resolve()
    for token, known in _ROOTS.items():
        if known == resolved:
            return token
    token = uuid.uuid4().hex[:12]
    _ROOTS[token] = resolved
    return token


def report_url(html_path: Path) -> str:
    """URL the iframe should point at for a report file already on disk."""
    html_path = Path(html_path).resolve()
    return f"/report/{register(html_path.parent)}/{html_path.name}"


def attach(server) -> None:
    """Add the /report/<token>/<path> route to a Flask server."""
    from flask import abort, send_from_directory

    @server.route("/report/<token>/<path:relative>")
    def _serve_report(token: str, relative: str):
        root = _ROOTS.get(token)
        if root is None:
            abort(404)
        # send_from_directory rejects anything resolving outside root, so a crafted
        # `relative` cannot climb out of the directory the token stands for
        return send_from_directory(root, relative)
