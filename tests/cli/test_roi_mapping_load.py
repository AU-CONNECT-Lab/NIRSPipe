"""``--roi-mapping`` is read in one place, and a file that cannot be read stops the run."""

from __future__ import annotations

import json

import pytest

from fnirs_pipe.cli._shared import load_roi_mapping


def test_no_mapping_means_no_roi_map():
    assert load_roi_mapping(None) is None


def test_a_mapping_is_read_as_json(tmp_path):
    path = tmp_path / "roi.json"
    path.write_text(json.dumps({"L": ["S1_D1"]}))
    assert load_roi_mapping(path) == {"L": ["S1_D1"]}


@pytest.mark.parametrize("content", [None, "{not json"])
def test_a_mapping_that_cannot_be_read_exits_rather_than_dropping_the_roi_output(tmp_path, content):
    path = tmp_path / "roi.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(SystemExit) as exc:
        load_roi_mapping(path)
    assert exc.value.code == 1
