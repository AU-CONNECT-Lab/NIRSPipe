"""nirspipe-gui CLI entry point (argparse)."""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nirspipe-gui",
        description="Dash-based GUI for interactive fNIRS data inspection and rating.",
    )
    p.add_argument("--port", type=int, default=None,
                   help="Local server port. Default: 8050, or the next free port above it.")
    # withheld until the Flask app replaces the Dash one
    p.add_argument("--new", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.new:
        from nirspipe.interface.server import WebApp
        WebApp().serve(args.port, 8050, "nirspipe-gui", [""])
        return
    from nirspipe.interface.app import launch as _launch
    _launch(port=args.port)
