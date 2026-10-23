"""What happens when several workers process one dataset into one output tree at once.

A batch runs for hours, and a sharing violation near the end of one costs more than these
tests do.

Real processes, not threads. The Windows failure these guard against is a file being opened
for writing by one process while another holds it, which threads inside one interpreter do not
reproduce.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytest

from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import to_bids_uri, write_dataset_description

WORKERS = 8


# ---- Workers. Module level and argument-only, so spawn can pickle them ----

def _write_description(output_dir: str) -> str:
    write_dataset_description(Path(output_dir))
    return "ok"


def _link_at(output_dir: str, source: str, link: str, start: float) -> str:
    # every worker starts its read-merge-write at the same instant, where the race was
    while time.time() < start:
        pass
    write_dataset_description(Path(output_dir), source=Path(source), link=link)
    return link


def _add_links(output_dir: str, sources: list[str], start: float) -> int:
    # each link is new, so every write changes the file and replaces it
    while time.time() < start:
        pass
    for source in sources:
        write_dataset_description(Path(output_dir), source=Path(source), link=Path(source).name)
    return len(sources)


def _resolve_raw_until(output_dir: str, raw: str, start: float, stop_file: str) -> list[str]:
    """Name a raw file by its BIDS URI while others add links, and report every failure."""
    written = Path(output_dir) / "sub-01" / "sub-01_desc-x_nirs.json"
    target = Path(raw) / "sub-01" / "nirs" / "sub-01_nirs.snirf"
    while time.time() < start:
        pass
    failed = []
    while not Path(stop_file).exists():
        try:
            uri = to_bids_uri(target, written)
            if uri != "bids:raw:sub-01/nirs/sub-01_nirs.snirf":
                failed.append(uri)
        except StageError as exc:
            failed.append(str(exc)[:80])
    return failed


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
    kept = {"dataset_description.json", ".dataset_description.json.lock"}
    leftovers = [p.name for p in out.iterdir() if p.name not in kept]
    assert not leftovers, f"temp files survived: {leftovers}"


def test_links_resolve_while_other_runs_rewrite_the_description(tmp_path):
    """A run naming its sources reads the links while another run is replacing the file.

    The reader may meet the instant of the replace and the writer may meet an open reader;
    neither is allowed to fail or to read the links as absent.
    """
    out = tmp_path / "derivatives"
    raw = tmp_path / "raw"
    raw.mkdir()
    write_dataset_description(out, source=raw, link="raw")
    n_writers, n_links = 4, 30
    sources = [[str(tmp_path / f"src{w}_{i}") for i in range(n_links)] for w in range(n_writers)]
    for group in sources:
        for s in group:
            Path(s).mkdir()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for f in [pool.submit(_write_description, str(tmp_path / "warm")) for _ in range(WORKERS)]:
            f.result()
        start = time.time() + 1.0
        stop = tmp_path / "writers_done"
        writers = [pool.submit(_add_links, str(out), group, start) for group in sources]
        readers = [pool.submit(_resolve_raw_until, str(out), str(raw), start, str(stop))
                   for _ in range(WORKERS - n_writers)]
        try:
            assert sum(f.result() for f in writers) == n_writers * n_links
        finally:
            stop.touch()
        failed = [x for f in readers for x in f.result()]

    assert not failed, f"{len(failed)} resolutions failed, first: {failed[:2]}"
    written = json.loads((out / "dataset_description.json").read_text(encoding="utf-8"))
    assert len(written["DatasetLinks"]) == 1 + n_writers * n_links


def test_tools_writing_links_at_once_keep_every_link(tmp_path):
    """Each writer adds its own link to what is there; none may drop another's."""
    links = [f"source{i}" for i in range(WORKERS)]
    for name in links:
        (tmp_path / name).mkdir()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        # warm the workers, so the spawn and import time is not inside the race window
        for f in [pool.submit(_write_description, str(tmp_path / "warm")) for _ in range(WORKERS)]:
            f.result()
        for attempt in range(3):
            out = tmp_path / f"derivatives{attempt}"
            start = time.time() + 1.0
            futures = [pool.submit(_link_at, str(out), str(tmp_path / name), name, start)
                       for name in links]
            for f in futures:
                f.result()
            written = json.loads((out / "dataset_description.json").read_text(encoding="utf-8"))
            assert sorted(written["DatasetLinks"]) == sorted(links)
            assert len(written["SourceDatasets"]) == WORKERS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---- How long a sharing violation is waited out ----

def _held(times: int, real):
    """A stand-in that raises Windows' sharing violation `times` times, then does the real thing."""
    calls = {"n": 0}

    def call(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= times:
            raise PermissionError(13, "Permission denied")
        return real(*args, **kwargs)
    return call


def test_a_read_waits_out_a_holder_longer_than_five_retries(tmp_path, monkeypatch):
    """Eight refusals in a row, as a loaded machine gave: a count of five gave up at the fifth."""
    from nirspipe.io import derivatives

    write_dataset_description(tmp_path, source=tmp_path / "raw", link="raw")
    monkeypatch.setattr(derivatives.time, "sleep", lambda s: None)
    monkeypatch.setattr(Path, "read_text", _held(8, Path.read_text))
    assert "raw" in derivatives.read_dataset_description(tmp_path)["DatasetLinks"]


def test_a_replace_waits_out_a_holder_longer_than_five_retries(tmp_path, monkeypatch):
    from nirspipe.io import derivatives

    write_dataset_description(tmp_path)
    monkeypatch.setattr(derivatives.time, "sleep", lambda s: None)
    monkeypatch.setattr(derivatives.os, "replace", _held(8, derivatives.os.replace))
    write_dataset_description(tmp_path, source=tmp_path / "raw", link="raw")
    assert "raw" in json.loads((tmp_path / "dataset_description.json").read_text())["DatasetLinks"]
