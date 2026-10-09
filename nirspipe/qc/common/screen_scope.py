"""Which stretches of a recording the channel screening counts over.

A run holds time no analysis reads: the lead-in before the first block, the gaps between
blocks, the tail after the last one. ``"task"`` counts coupled windows only inside the
annotated blocks.

This is one channel set either way: only the *denominator* is restricted, so a contrast
between two conditions is never also a contrast between two montages.
"""

from __future__ import annotations

import mne

from nirspipe.utils.logging import get_logger

logger = get_logger("qc.screen_scope")

SCOPE_CHOICES = ("run", "task")


def resolve_screen_scope(
    raw: mne.io.Raw, scope: str = "run",
) -> "list[tuple[str, float, float]] | None":
    """The stretches to count over, or None for the whole recording.

    ::

      "run"                                    ->  None
      "task", blocks annotated                 ->  [("rest", 20.0, 320.0), ...]
      "task", only short triggers annotated    ->  None, with a warning

    ``"task"`` falls back to the whole recording rather than to nothing, since the case it
    falls back from is a recording carrying triggers instead of blocks. The fallback is logged
    with what it found, since a silent one leaves two runs screened differently with nothing
    on disk saying which.
    """
    if scope not in SCOPE_CHOICES:
        raise ValueError(f"screen scope must be one of {SCOPE_CHOICES}, got {scope!r}")
    if scope == "run":
        return None

    from nirspipe.qc.metrics.windowed import task_scope_windows

    windows = task_scope_windows(raw)
    if not windows:
        logger.warning(
            "--screen-scope task asked for, but no annotation is long enough to hold two "
            "screening windows (%d annotation(s) present); counting the whole recording "
            "instead", len(raw.annotations))
        return None
    covered = sum(t1 - t0 for _label, t0, t1 in windows)
    total = float(raw.times[-1]) or 1.0
    logger.info("--screen-scope task: %d block(s), %.0f of %.0f s (%.0f%% of the recording)",
                len(windows), covered, total, 100 * covered / total)
    return windows
