"""What happens when several workers process one dataset into one output tree at once.

A batch runs for hours, and a sharing violation near the end of one costs more than these
tests do.

Real processes, not threads. The Windows failure these guard against is a file being opened
for writing by one process while another holds it, which threads inside one interpreter do not
reproduce.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytest

from nirspipe.io.derivatives import write_dataset_description

WORKERS = 8


# ---- Workers. Module level and argument-only, so spawn can pickle them ----

def _write_description(output_dir: str) -> str:
    write_dataset_description(Path(output_dir))
    return "ok"


def _read_description_repeatedly(output_dir: str, times: int) -> list[str]:
    """Read the file while others rewrite it, and report anything that was not valid BIDS."""
    path = Path(output_dir) / "dataset_description.json"
    bad = []
    for _ in range(times):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            # the instant of the rename; a retry, not a corrupt tree
            continue
        try:
            if json.loads(text).get("Name") != "nirspipe output":
                bad.append(text[:80])
        except json.JSONDecodeError:
            bad.append(text[:80])
    return bad


# ---- dataset_description.json ----

def test_concurrent_writers_leave_a_valid_description(tmp_path):
    out = str(tmp_path / "derivatives")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_write_description, out) for _ in range(WORKERS * 3)]
        for f in as_completed(futures):
            assert f.result() == "ok"

    written = json.loads((Path(out) / "dataset_description.json").read_text(encoding="utf-8"))
    assert written["Name"] == "nirspipe output"
    assert written["BIDSVersion"] == "1.8.0"


def test_a_reader_never_sees_a_half_written_description(tmp_path):
    """The write is atomic: a reader never sees a truncated or empty file.

    A reader can still be told the path is busy at the instant of the rename, which Windows
    offers no way around; that is caught above and is a retry. What must never happen is
    reading a file that parses but says the wrong thing, or does not parse at all.
    """
    out = str(tmp_path / "derivatives")
    write_dataset_description(Path(out))

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        writers = [pool.submit(_write_description, out) for _ in range(WORKERS * 4)]
        readers = [pool.submit(_read_description_repeatedly, out, 200) for _ in range(3)]
        for f in writers:
            f.result()
        seen_bad = [b for f in readers for b in f.result()]

    assert not seen_bad, f"readers saw invalid content: {seen_bad[:3]}"


def test_no_temporary_files_are_left_behind(tmp_path):
    out = tmp_path / "derivatives"
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for f in [pool.submit(_write_description, str(out)) for _ in range(WORKERS * 2)]:
            f.result()
    leftovers = [p.name for p in out.iterdir() if p.name != "dataset_description.json"]
    assert not leftovers, f"temp files survived: {leftovers}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
