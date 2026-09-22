"""What each output tree says about itself: who wrote it, from what, and what to skip.

The two stamps are the whole reason the dyad results live in a tree of their own. While one
tool wrote back into the tree it read, `GeneratedBy` could only name one of the two and
`SourceDatasets` had no correct value at all, so a reader of a coherence table had no way to
recover which preprocessing produced its inputs.
"""

from __future__ import annotations

import json

from fnirs_pipe.io.derivatives import write_bidsignore, write_dataset_description

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


def test_only_reports_logs_and_figures_are_waved_through(tmp_path):
    write_bidsignore(tmp_path)
    lines = (tmp_path / ".bidsignore").read_text(encoding="utf-8").split()

    assert set(lines) == {"*.html", "logs/", "figures/"}
    for line in lines:
        assert not line.endswith(DATA_EXTENSIONS), (
            f"{line} exempts a data product; dyad tables and every other output stay on the "
            f"validator's books"
        )


def test_the_stamp_is_rewritten_when_the_source_changes(tmp_path):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    write_dataset_description(a)
    write_dataset_description(b)
    write_dataset_description(out, source=a)
    write_dataset_description(out, source=b)

    assert _description(out)["SourceDatasets"][0]["URL"].endswith("/b")
