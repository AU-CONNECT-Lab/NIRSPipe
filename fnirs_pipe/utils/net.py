"""Choosing a local port for the interactive servers."""

from __future__ import annotations

import socket

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("utils.net")

SCAN_SPAN = 20


def _is_free(host: str, port: int) -> bool:
    # no SO_REUSEADDR: the probe has to fail exactly where the real server would
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def resolve_port(preferred: int, *, explicit: bool, host: str = "127.0.0.1") -> int:
    """Port to serve on, at or just above `preferred`.

    A server left running by an earlier session, another app sitting on the same default,
    or a Windows reserved port range all make `preferred` unbindable. Walking upwards is
    right when the port is just a default, wrong when the caller named it: a forwarding
    rule or a bookmarked URL points at that one number, so a silent move only hides the
    conflict. Hence `explicit`.

    resolve_port(8050, explicit=False) -> 8051, when 8050 is taken
    resolve_port(8050, explicit=True)  -> SystemExit, when 8050 is taken
    """
    if _is_free(host, preferred):
        return preferred

    if explicit:
        raise SystemExit(
            f"Port {preferred} is already in use. Stop whatever is holding it, "
            f"or pass a different --port."
        )

    for port in range(preferred + 1, preferred + 1 + SCAN_SPAN):
        if _is_free(host, port):
            logger.info("port %d in use, serving on %d instead", preferred, port)
            return port

    raise SystemExit(
        f"No free port between {preferred} and {preferred + SCAN_SPAN}. "
        f"Pass an explicit --port."
    )
