"""Sources as BIDS URIs: what every sidecar writes, how a reader gets the file back, and the
links in dataset_description.json both go through."""

import json
import shutil

import pytest

from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import (
    LINK_PREPROCESSED, LINK_RAW, read_json, resolve_bids_uri, to_bids_uri,
    write_dataset_description, write_sidecar_json,
)


def _dataset(root, kind="raw"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(json.dumps(
        {"Name": root.name, "BIDSVersion": "1.8.0", "DatasetType": kind}))
    return root


def _file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


@pytest.fixture
def trees(tmp_path):
    """A raw dataset and a derivative tree beside it that links to it."""
    raw = _dataset(tmp_path / "bids")
    out = tmp_path / "out"
    write_dataset_description(out, source=raw, link=LINK_RAW)
    return raw, out


# ---- writing ----

def test_a_file_of_the_tree_itself_is_named_from_its_root(trees):
    _, out = trees
    od = out / "sub-01" / "nirs" / "sub-01_task-rest_desc-od_nirs.snirf"
    assert to_bids_uri(od, od) == "bids::sub-01/nirs/sub-01_task-rest_desc-od_nirs.snirf"


def test_a_linked_dataset_is_named_by_its_link(trees):
    raw, out = trees
    src = raw / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf"
    od = out / "sub-01" / "nirs" / "sub-01_task-rest_desc-od_nirs.snirf"
    assert to_bids_uri(src, od) == "bids:raw:sub-01/nirs/sub-01_task-rest_nirs.snirf"


def test_a_tree_inside_its_raw_dataset_still_names_its_own_files_with_bids_colon_colon(tmp_path):
    raw = _dataset(tmp_path / "bids")
    out = raw / "derivatives" / "nirspipe"
    write_dataset_description(out, source=raw, link=LINK_RAW)
    od = out / "sub-01" / "nirs" / "sub-01_desc-od_nirs.snirf"
    assert to_bids_uri(od, od).startswith("bids::sub-01/")


def test_a_file_in_no_linked_dataset_has_no_uri_and_raises(trees, tmp_path):
    _, out = trees
    loose = tmp_path / "elsewhere" / "x.snirf"
    with pytest.raises(StageError, match="no dataset"):
        to_bids_uri(loose, out / "sub-01" / "nirs" / "y.json")


def test_a_uri_passes_through_as_the_bids_apps_leave_it(trees):
    _, out = trees
    assert to_bids_uri("bids:raw:sub-01/a.snirf", out / "sub-01" / "b.json") \
        == "bids:raw:sub-01/a.snirf"


def test_every_sidecar_writes_its_sources_as_uris(trees):
    raw, out = trees
    src = raw / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf"
    od = out / "sub-01" / "nirs" / "sub-01_task-rest_desc-od_nirs.snirf"
    od.parent.mkdir(parents=True)
    write_sidecar_json(od, {"step": "od_conversion", "Sources": [src.as_posix()]})
    assert read_json(od.with_suffix(".json"))["Sources"] \
        == ["bids:raw:sub-01/nirs/sub-01_task-rest_nirs.snirf"]


# ---- reading ----

def test_a_uri_resolves_to_the_file_and_still_does_after_both_trees_move(trees, tmp_path):
    raw, out = trees
    src = _file(raw / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf")
    side = out / "sub-01" / "nirs" / "sub-01_task-rest_desc-od_nirs.json"
    uri = to_bids_uri(src, side)
    assert resolve_bids_uri(uri, side) == src.resolve()

    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "bids", moved / "bids")
    shutil.copytree(out, moved / "out")
    found = resolve_bids_uri(uri, moved / "out" / "sub-01" / "nirs" / side.name)
    assert found == (moved / "bids" / "sub-01" / "nirs" / src.name).resolve()
    assert found.is_file()


def test_an_absolute_path_from_before_uris_is_refused(trees):
    _, out = trees
    with pytest.raises(StageError, match="not a BIDS URI"):
        resolve_bids_uri("C:/data/bids/sub-01/x.snirf", out / "sub-01" / "x.json")


def test_a_name_the_tree_does_not_link_is_refused(trees):
    _, out = trees
    with pytest.raises(StageError, match="does not link"):
        resolve_bids_uri("bids:preprocessed:sub-01/x.snirf", out / "sub-01" / "x.json")


# ---- the links ----

def test_the_link_is_relative_and_the_source_url_matches_it(trees):
    _, out = trees
    desc = read_json(out / "dataset_description.json")
    assert desc["DatasetLinks"] == {"raw": "../bids"}
    assert [s["URL"] for s in desc["SourceDatasets"]] == ["../bids"]


def test_two_commands_writing_one_tree_add_their_links_rather_than_take_turns(tmp_path):
    raw = _dataset(tmp_path / "bids")
    pipe = _dataset(tmp_path / "nirspipe", kind="derivative")
    hyper = tmp_path / "nirspipe-hyper"
    write_dataset_description(hyper, source=pipe, link=LINK_PREPROCESSED)
    write_dataset_description(hyper, source=raw, link=LINK_RAW)
    write_dataset_description(hyper, source=pipe, link=LINK_PREPROCESSED)

    desc = read_json(hyper / "dataset_description.json")
    assert desc["DatasetLinks"] == {"preprocessed": "../nirspipe", "raw": "../bids"}
    assert sorted(s["URL"] for s in desc["SourceDatasets"]) == ["../bids", "../nirspipe"]


def test_a_link_that_would_point_elsewhere_raises(trees, tmp_path):
    _, out = trees
    other = _dataset(tmp_path / "other_bids")
    with pytest.raises(StageError, match="new output directory"):
        write_dataset_description(out, source=other, link=LINK_RAW)


@pytest.mark.parametrize("kind, expected", [("raw", "raw"), ("derivative", "preprocessed")])
def test_an_unnamed_link_is_named_by_the_source_dataset_type(tmp_path, kind, expected):
    src = _dataset(tmp_path / "src", kind=kind)
    write_dataset_description(tmp_path / "out", source=src)
    assert list(read_json(tmp_path / "out" / "dataset_description.json")["DatasetLinks"]) \
        == [expected]
