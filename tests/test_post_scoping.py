"""Post-processing runs on this subject's own derivatives, and nothing else.

The BIDS layout is built over the whole output directory, so every nested derivative
tree under it gets indexed too. On a real run that meant sixteen unrelated trees under
output/pipeline_comparision/ were reprocessed as if they were this subject: post ran
seventeen times and the last unrelated file won. Nothing failed, and the outputs were
wrong.

The layout and run_post are faked here. What is under test is the file selection, and a
real layout would make the nested tree hard to construct while proving nothing extra.
"""

from pathlib import Path

import pytest

from fnirs_pipe.cli import workflows

ARGS = {"mode": "denoise", "task_label": None}


class _FakeLayout:
    def parse_file_entities(self, path):
        return {"subject": "01", "task": "tapping"}


@pytest.fixture
def post_calls(monkeypatch):
    """Record the file each run_post call received, and run nothing."""
    import fnirs_pipe.pipeline.post_pipeline as post_pipeline

    seen: list[Path] = []

    def _fake_run_post(raw, config, **kwargs):
        seen.append(kwargs["source_path"])
        return (None,) * 7

    monkeypatch.setattr(post_pipeline, "run_post", _fake_run_post)
    monkeypatch.setattr(workflows, "get_layout", lambda *a, **k: _FakeLayout())
    monkeypatch.setattr(workflows, "read_snirf", lambda path, **k: path)
    monkeypatch.setattr(workflows, "_build_post_config", lambda *a, **k: object())
    return seen


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _run(output_dir, files, monkeypatch):
    monkeypatch.setattr(workflows, "get_nirs_files", lambda *a, **k: list(files))
    workflows._run_post_for_subject("01", [None], ARGS, {}, output_dir)


def test_a_nested_output_tree_with_the_same_subject_is_ignored(tmp_path, post_calls, monkeypatch):
    out = tmp_path / "out"
    mine = _touch(out / "sub-01" / "nirs" / "sub-01_task-tapping_desc-preproc_nirs.snirf")
    nested = _touch(out / "pipeline_comparision" / "run2" / "sub-01" / "nirs"
                    / "sub-01_task-tapping_desc-preproc_nirs.snirf")

    _run(out, [mine, nested], monkeypatch)

    assert post_calls == [mine]


def test_every_file_of_this_subject_is_still_processed(tmp_path, post_calls, monkeypatch):
    # the filter narrows by directory, not by count: two tasks are two runs
    out = tmp_path / "out"
    nirs = out / "sub-01" / "nirs"
    tapping = _touch(nirs / "sub-01_task-tapping_desc-preproc_nirs.snirf")
    rest = _touch(nirs / "sub-01_task-rest_desc-preproc_nirs.snirf")

    _run(out, [tapping, rest], monkeypatch)

    assert post_calls == [tapping, rest]


def test_only_preproc_files_are_post_processed(tmp_path, post_calls, monkeypatch):
    out = tmp_path / "out"
    nirs = out / "sub-01" / "nirs"
    preproc = _touch(nirs / "sub-01_task-tapping_desc-preproc_nirs.snirf")
    intermediate = _touch(nirs / "sub-01_task-tapping_desc-od_nirs.snirf")

    _run(out, [intermediate, preproc], monkeypatch)

    assert post_calls == [preproc]


def test_no_matching_file_skips_the_subject_without_raising(tmp_path, post_calls, monkeypatch):
    out = tmp_path / "out"
    nested = _touch(out / "other" / "sub-01" / "nirs" / "sub-01_task-tapping_desc-preproc_nirs.snirf")

    _run(out, [nested], monkeypatch)

    assert post_calls == []
