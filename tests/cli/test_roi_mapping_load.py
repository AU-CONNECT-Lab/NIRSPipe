"""``--roi-mapping`` is read in one place, and a file that cannot be read stops the run."""

from __future__ import annotations

import json

import pytest

from nirspipe.cli._shared import load_roi_mapping


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


def test_a_chromophore_suffix_on_an_entry_is_dropped(tmp_path):
    """Every path takes the chromophore from the pass, so the suffix only stopped hyper matching."""
    path = tmp_path / "roi.json"
    path.write_text(json.dumps({"L": ["S1_D1 hbo", "S1_D1", "S2_D2 hbr"]}))
    assert load_roi_mapping(path) == {"L": ["S1_D1", "S2_D2"]}


def test_a_channel_in_two_rois_is_logged_and_noted_in_the_report(tmp_path, caplog):
    from nirspipe.qc.common.channel_table import roi_overlap_note

    path = tmp_path / "roi.json"
    path.write_text(json.dumps({"L": ["S1_D1", "S1_D2 hbo"], "R": ["S1_D2"]}))
    with caplog.at_level("WARNING"):
        roi_map = load_roi_mapping(path)

    assert "S1_D2 (L, R)" in caplog.text
    note = roi_overlap_note(roi_map)
    assert "1 channel(s)" in note and "S1_D2 (L, R)" in note
    assert roi_overlap_note({"L": ["S1_D1"], "R": ["S1_D2"]}) is None
