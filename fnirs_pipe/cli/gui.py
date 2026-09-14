"""fnirs-gui CLI entry point (argparse)."""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-gui",
        description="Dash-based GUI for interactive fNIRS data inspection and rating.",
    )
    p.add_argument("--port", type=int, default=None,
                   help="Local server port. Default: 8050, or the next free port above it.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    from fnirs_pipe.interface.app import launch as _launch
    _launch(port=args.port)
