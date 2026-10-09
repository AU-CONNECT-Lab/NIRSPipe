"""What the interactive servers do when their port is already taken.

Every server here has a fixed default port, so the collisions are routine: a viewer left
running by an earlier session, another Dash app on 8050, a Windows reserved range. The rule
under test is that a default may move and a port the caller named may not, because a named
port is usually named for a reason (a forwarding rule, a bookmarked URL) that a silent move
would break without saying so.
"""

import socket

import pytest

from nirspipe.utils.net import SCAN_SPAN, resolve_port

HOST = "127.0.0.1"


@pytest.fixture
def taken():
    """Hold a listening socket on an OS-assigned port and yield its number."""
    held = []

    def _hold(port: int = 0) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((HOST, port))
        except OSError:
            sock.close()
            pytest.skip(f"port {port} is held by something outside the test")
        sock.listen(1)
        held.append(sock)
        return sock.getsockname()[1]

    yield _hold
    for sock in held:
        sock.close()


def test_free_port_is_returned_unchanged():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, 0))
    port = sock.getsockname()[1]
    sock.close()        # an ephemeral port nobody else has claimed yet

    assert resolve_port(port, explicit=False) == port
    assert resolve_port(port, explicit=True) == port


def test_taken_default_moves_up(taken):
    port = taken()
    assert resolve_port(port, explicit=False) == port + 1


def test_taken_default_skips_a_run_of_busy_ports(taken):
    port = taken()
    taken(port + 1)
    taken(port + 2)
    assert resolve_port(port, explicit=False) == port + 3


def test_named_port_fails_instead_of_moving(taken):
    port = taken()
    with pytest.raises(SystemExit, match=f"Port {port} is already in use"):
        resolve_port(port, explicit=True)


def test_exhausted_scan_fails(taken, monkeypatch):
    port = taken()
    monkeypatch.setattr("nirspipe.utils.net._is_free", lambda host, p: False)
    with pytest.raises(SystemExit, match=f"No free port between {port} and {port + SCAN_SPAN}"):
        resolve_port(port, explicit=False)
