"""Every segment written to its own file is named by a task label, or the crop is refused.

A segment of one recording is neither another run nor, without a label, anything with a
name: a numbered file carries no meaning a reader can select on, and two segments with one
label write the same file.
"""

from pathlib import Path

import pandas as pd
import pytest

from fnirs_pipe.io.snirf import write_snirf
from fnirs_pipe.pipeline.crop import crop_snirf_from_path
from tests._synth import synth_raw


@pytest.fixture
def source(tmp_path) -> Path:
    path = tmp_path / "sub-01_task-full_nirs.snirf"
    write_snirf(synth_raw("01", "full", duration=300.0, motion_onset=None, bad_pair=None), path)
    return path


def _crop(source: Path, tmp_path: Path, **columns) -> list[str]:
    segments = pd.DataFrame({"onset": [10.0, 150.0], "duration": [60.0, 60.0], **columns})
    outs = crop_snirf_from_path(source, tmp_path / "deriv", "01", segments_df=segments)
    return [p.name for p in outs]


def test_segments_without_a_label_are_refused(source, tmp_path):
    with pytest.raises(ValueError, match="no task column"):
        _crop(source, tmp_path)
    assert not list((tmp_path / "deriv").rglob("*_nirs.snirf"))


def test_an_all_blank_label_column_counts_as_no_labels(source, tmp_path):
    """The GUI table sends its Task column whether or not anybody typed in it."""
    with pytest.raises(ValueError, match="no task column"):
        _crop(source, tmp_path, task=[None, ""])


def test_a_label_used_twice_is_refused_rather_than_overwritten(source, tmp_path):
    with pytest.raises(ValueError, match="more than one segment"):
        _crop(source, tmp_path, task=["game", "game"])
    assert not list((tmp_path / "deriv").rglob("*_nirs.snirf"))


def test_a_segment_left_unlabelled_among_labelled_ones_is_refused(source, tmp_path):
    with pytest.raises(ValueError, match="segment 2"):
        _crop(source, tmp_path, task=["early", None])


def test_a_label_that_is_not_letters_and_digits_is_refused(source, tmp_path):
    """A hyphen or underscore would be read as the start of another entity."""
    with pytest.raises(ValueError, match="letters and digits"):
        _crop(source, tmp_path, task=["early", "late_1"])


def test_distinct_labels_name_the_segments(source, tmp_path):
    assert _crop(source, tmp_path, task=["early", "late"]) == [
        "sub-01_task-early_nirs.snirf", "sub-01_task-late_nirs.snirf"]


def test_one_unlabelled_segment_keeps_the_source_name(source, tmp_path):
    segments = pd.DataFrame({"onset": [10.0], "duration": [60.0]})
    outs = crop_snirf_from_path(source, tmp_path / "deriv", "01", segments_df=segments)
    assert [p.name for p in outs] == ["sub-01_task-full_nirs.snirf"]


def test_combining_needs_no_labels(source, tmp_path):
    segments = pd.DataFrame({"onset": [10.0, 150.0], "duration": [60.0, 60.0]})
    outs = crop_snirf_from_path(source, tmp_path / "deriv", "01", segments_df=segments,
                                combine=True)
    assert [p.name for p in outs] == ["sub-01_task-full_nirs.snirf"]
