"""fnirs-recon CLI (argparse): convert raw snirf files to BIDS format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-recon",
        description="Convert a raw snirf file to BIDS format (one subject at a time). "
                    "For multiple subjects, loop over this command or use the GUI Recon page.",
    )
    p.add_argument("input_file", type=Path, help="Input snirf file.")
    p.add_argument("bids_dir",   type=Path, help="Output BIDS dataset directory.")
    p.add_argument("--subject", required=True,
                   help="Subject label, e.g. '01' or 'patient01'. BIDS has no group folders, "
                        "so encode patient/control in the label if IDs overlap.")
    p.add_argument("--task", required=True, help="Task label, e.g. tapping.")
    p.add_argument("--session", default=None, help="Session label. Omit if dataset has no session layer.")
    p.add_argument("--run", default=None, help="Run index, e.g. 01.")
    p.add_argument("--optode-frame", choices=["unknown", "head", "mri"], default="unknown",
                   help="Space the SNIRF's optode coordinates were measured in. SNIRF does "
                        "not record it, and without it no _optodes.tsv or _coordsystem.json "
                        "is written, both of which BIDS requires. Use 'head' for positions "
                        "digitised against the nasion and preauricular points.")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from fnirs_pipe.io.bids import write_bids_from_snirf

    if not args.input_file.exists():
        print(f"ERROR: input file not found: {args.input_file}", file=sys.stderr)
        raise SystemExit(1)

    write_bids_from_snirf(
        args.input_file, args.bids_dir, subject=args.subject, task=args.task,
        session=args.session, run=args.run, overwrite=args.overwrite,
        optode_frame=args.optode_frame,
    )
    print(f"BIDS output written to: {args.bids_dir}")
