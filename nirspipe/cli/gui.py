"""nirspipe-gui CLI entry point (argparse)."""

from __future__ import annotations

import argparse

DEFAULT_PORT = 8050


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nirspipe-gui",
        description="Local web GUI: a page per command with its form, Data Preparation for "
                    "editing a recording's events and crop, and the rating viewers.",
    )
    p.add_argument("--port", type=int, default=None,
                   help=f"Local server port. Default: {DEFAULT_PORT}, or the next free port above it.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    from nirspipe.interface.server import WebApp
    # the Interactive page first, where the sidebar starts
    WebApp().serve(args.port, DEFAULT_PORT, "nirspipe-gui", ["data-prep/"])
