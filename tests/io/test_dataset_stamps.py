"""What each output tree says about itself: who wrote it, from what, and what to skip.

The two stamps are the whole reason the dyad results live in a tree of their own:
`GeneratedBy` names the one tool that wrote the tree and `SourceDatasets` names the tree it
read, so a reader of a coherence table can recover which preprocessing produced its inputs.
"""

from __future__ import annotations

import json
from fnmatch import fnmatch

from fnirs_pipe.io.derivatives import (
    JSON_ONLY_DESCS, write_bidsignore, write_dataset_description,
)

# extensions a reader is expected to index. None of them may be waved through.
DATA_EXTENSIONS = (".tsv", ".snirf", ".json", ".npz", ".gz", ".csv")


def _description(path):
    return json.loads((path / "dataset_description.json").read_text(encoding="utf-8"))


def test_the_source_tree_is_named_with_the_version_that_wrote_it(tmp_path):
    source, out = tmp_path / "fnirs-pipe", tmp_path / "fnirs-hyper"
    write_dataset_description(source)
    write_dataset_description(out, name="fnirs-hyper output",
                              generated_by="fnirs-hyper", source=source)

    entry = _description(out)["SourceDatasets"][0]
    assert entry["Name"] == "fnirs-pipe"
    assert entry["Version"] == _description(source)["GeneratedBy"][0]["Version"]
    assert entry["URL"].startswith("file:")


def test_a_tree_with_no_source_claims_none(tmp_path):
    write_dataset_description(tmp_path)
    assert "SourceDatasets" not in _description(tmp_path)


def test_an_unreadable_source_still_leaves_a_usable_pointer(tmp_path):
    source, out = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    (source / "dataset_description.json").write_text("{not json", encoding="utf-8")
    write_dataset_description(out, source=source)

    entry = _description(out)["SourceDatasets"][0]
    assert entry["URL"].startswith("file:")
    assert "Version" not in entry


def _ignore_lines(tmp_path):
    write_bidsignore(tmp_path)
    return (tmp_path / ".bidsignore").read_text(encoding="utf-8").split()


def test_only_reports_logs_figures_and_json_only_records_are_waved_through(tmp_path):
    lines = _ignore_lines(tmp_path)

    # no trailing slash: bids-validator 3.0.2 matches nothing against `figures/`, so the
    # gitignore spelling for a directory leaves every figure on its books
    assert set(lines) == {"*.html", "logs", "figures",
                          *(f"*_desc-{desc}_qc.json" for desc in JSON_ONLY_DESCS)}
    assert not any(line.endswith("/") for line in lines)
    for line in lines:
        # a data extension only ever behind one exact desc, never as a bare wildcard
        assert not line.endswith(DATA_EXTENSIONS) or line.startswith("*_desc-"), (
            f"{line} exempts a data product; dyad tables and every other output stay on the "
            f"validator's books"
        )


def test_the_ignored_records_are_the_ones_the_writers_name(tmp_path):
    """Each JSON-only record the package writes is ignored, and nothing with a data file is.

    The desc names live in three modules, so the list is bound to the writers here rather
    than trusted to stay in step with them.
    """
    from fnirs_pipe.io.derivatives import channel_decisions_path
    from fnirs_pipe.io.naming import rating_path
    from fnirs_pipe.qc.subject.sqm_record import RECORD_SUFFIXES

    lines = _ignore_lines(tmp_path)
    written = [f"sub-01_task-rest{suffix}" for suffix in RECORD_SUFFIXES.values()] + [
        rating_path(tmp_path, "sub-01_task-rest_desc-raw_report").name,
        rating_path(tmp_path, "sub-01_task-rest_report").name,
        channel_decisions_path(tmp_path, "01", task="rest").name,
        "group-G1_task-rest_desc-sqm_qc.json",
    ]
    kept = ["sub-01_task-rest_desc-preproc_nirs.json", "sub-01_task-rest_desc-channel_qc.tsv",
            "group-G1_task-rest_desc-usable_qc.tsv", "group-G1_task-rest_stat-wtc_relmat.json"]

    assert all(any(fnmatch(name, line) for line in lines) for name in written)
    assert not any(fnmatch(name, line) for name in kept for line in lines)


def test_the_stamp_is_rewritten_when_the_source_changes(tmp_path):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    write_dataset_description(a)
    write_dataset_description(b)
    write_dataset_description(out, source=a)
    write_dataset_description(out, source=b)

    assert _description(out)["SourceDatasets"][0]["URL"].endswith("/b")


def test_a_prep_tree_restamps_a_description_an_older_version_left(tmp_path):
    """The crop, edit-markers and aligned trees used to keep whatever stamp first wrote them."""
    from fnirs_pipe import __version__
    from fnirs_pipe.pipeline.crop import _DERIV_NAME, _setup_deriv_dir

    root = tmp_path / _DERIV_NAME
    root.mkdir()
    (root / "dataset_description.json").write_text(json.dumps(
        {"Name": _DERIV_NAME, "GeneratedBy": [{"Name": "fnirs-prep crop", "Version": "0.0.1"}]}))
    _setup_deriv_dir(tmp_path, "01", None)

    assert _description(root)["GeneratedBy"][0]["Version"] == __version__
