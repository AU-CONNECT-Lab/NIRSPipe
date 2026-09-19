"""Pipeline-wide logging setup."""

import logging
import threading
from contextlib import contextmanager
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


class _ThreadFileHandler(logging.FileHandler):
    """A log file that only takes records emitted by the thread that opened it."""

    def __init__(self, log_file: Path) -> None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(log_file, mode="w", encoding="utf-8")
        self._thread = threading.get_ident()
        self.setLevel(logging.DEBUG)
        self.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))

    def filter(self, record: logging.LogRecord) -> bool:
        return threading.get_ident() == self._thread


@contextmanager
def thread_log_file(log_file: Path):
    """One unit's log file, added beside whatever is already configured rather than replacing it.

    :func:`setup_logging` clears the root handlers, which is right for one run and wrong for
    several at once: the second unit would take the first one's file away and its records
    with it. This adds a handler and filters on the calling thread, so parallel units each
    get their own file and nothing lands in two of them.
    """
    root = logging.getLogger("fnirs_pipe")
    handler = _ThreadFileHandler(log_file)
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        handler.close()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"fnirs_pipe.{name}")
