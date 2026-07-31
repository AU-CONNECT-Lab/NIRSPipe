"""Pipeline-wide logging setup."""

import logging
import sys
from pathlib import Path


_CONSOLE_FMT = "[%(levelname)s] %(message)s"
_FILE_FMT    = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure root logger for the pipeline.

    Console (stderr): INFO normally, DEBUG when --verbose.
    File (log_file):  DEBUG always, full timestamp + logger name.
    MNE verbosity is suppressed to WARNING to avoid noise.
    """
    root = logging.getLogger("fnirs_pipe")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # mode="w": log name carries no timestamp, so a re-run overwrites instead of growing
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
        root.addHandler(fh)

    # suppress MNE's verbose output
    logging.getLogger("mne").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"fnirs_pipe.{name}")
