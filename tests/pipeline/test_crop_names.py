"""Every segment written to its own file is named by a task label, or the crop is refused.

A segment of one recording is neither another run nor, without a label, anything with a
name: a numbered file carries no meaning a reader can select on, and two segments with one
label write the same file.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from fnirs_pipe.cli import prep as prep_cli
from fnirs_pipe.io.snirf import read_snirf, write_snirf
from fnirs_pipe.pipeline import crop
from fnirs_pipe.pipeline.crop import crop_snirf_from_path
from fnirs_pipe.qc.common.windows import crop_provenance
from tests._synth import _write_dataset_root, synth_raw


@pytest.fixture
def source(tmp_path) -> Path:
    _write_dataset_root(tmp_path, ["01"])
    path = tmp_path / "sub-01_task-main_nirs.snirf"
    write_snirf(synth_raw("01", "main", duration=300.0, motion_onset=None, bad_pair=None), path)
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
    assert [p.name for p in outs] == ["sub-01_task-main_nirs.snirf"]


def test_combining_needs_no_labels(source, tmp_path):
    segments = pd.DataFrame({"onset": [10.0, 150.0], "duration": [60.0, 60.0]})
    outs = crop_snirf_from_path(source, tmp_path / "deriv", "01", segments_df=segments,
                                combine=True)
    assert [p.name for p in outs] == ["sub-01_task-main_nirs.snirf"]


def test_the_command_checks_the_table_once_before_any_subject(tmp_path, monkeypatch, capsys):
    """One bad table is one error, not the same error once per subject."""
    table = tmp_path / "segments.tsv"
    pd.DataFrame({"onset": [10.0, 150.0], "duration": [60.0, 60.0]}).to_csv(
        table, sep="\t", index=False)
    called = []
    monkeypatch.setattr(crop, "crop_snirf", lambda *a, **k: called.append(a) or [])

    with pytest.raises(SystemExit) as exc:
        prep_cli.main(["crop", str(tmp_path), str(tmp_path / "out"),
                       "--participant-label", "01", "02", "03", "--segments-path", str(table)])
    assert exc.value.code == 1
    assert called == []
    assert capsys.readouterr().err.count("[error]") == 1


# ---- a segment cut from a recording says what it is ----

def _sidecar(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture
def source_with_sidecar(source) -> Path:
    source.with_suffix(".json").write_text(json.dumps(
        {"TaskName": "main", "RecordingDuration": 300.0, "SamplingFrequency": 10.0}))
    return source


def test_a_renamed_segment_names_its_own_task(source_with_sidecar, tmp_path):
    """BIDS derives the task label from TaskName, so a copied one would name the source's."""
    segments = pd.DataFrame({"onset": [10.0, 150.0], "duration": [60.0, 60.0],
                             "task": ["early", "late"]})
    outs = crop_snirf_from_path(source_with_sidecar, tmp_path / "deriv", "01",
                                segments_df=segments)
    assert [_sidecar(p)["TaskName"] for p in outs] == ["early", "late"]


def test_a_segment_records_its_own_length_and_where_it_came_from(source_with_sidecar, tmp_path):
    out = crop_snirf_from_path(source_with_sidecar, tmp_path / "deriv", "01",
                               tmin=10.0, tmax=70.0)[0]
    side = _sidecar(out)
    assert side["RecordingDuration"] == pytest.approx(60.0, abs=0.2)
    assert side["SamplingFrequency"] == 10.0                     # the rest is kept
    # named through the cropped tree's link to the dataset the source sits in
    assert side["Sources"] == [f"bids:raw:{source_with_sidecar.name}"]
    (lo, hi), = side["parameters"]["crop_windows_s"]
    assert (lo, hi) == (pytest.approx(10.0, abs=0.2), pytest.approx(70.0, abs=0.2))


def test_a_segment_of_a_recording_reads_back_as_a_segment(source_with_sidecar, tmp_path):
    """crop_provenance reads crop_windows_s, so a segment of a raw recording carries it too."""
    out = crop_snirf_from_path(source_with_sidecar, tmp_path / "deriv", "01",
                               tmin=10.0, tmax=70.0)[0]
    found = crop_provenance(read_snirf(out))
    assert found is not None
    assert found["window"][0] == pytest.approx(10.0, abs=0.2)
