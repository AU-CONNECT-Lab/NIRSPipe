"""Stage lineage and the write recorder.

These cover the mechanism itself: does the stamp survive the MNE operations the
pipeline actually performs, and does Recorder resolve an output's source to the
right file. Both have produced silent wrong answers before, so the assertions
here are about correctness of attribution, not about any numeric result.
"""

import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.utils.lineage import (
    Recorder,
    lineage_of,
    path_from,
    require_stage,
    stage_of,
    stamp,
)


# ---- stamp ----

def test_stamp_records_all_fields(fake_raw):
    stamp(fake_raw, stage="preproc", step="beer_lambert", dpf=[6.0, 6.0])
    lin = lineage_of(fake_raw)
    assert (lin.stage, lin.step, lin.source) == ("preproc", "beer_lambert", None)
    assert lin.params == {"dpf": [6.0, 6.0]}


def test_stamp_returns_raw_for_chaining(fake_raw):
    assert stamp(fake_raw, stage="od", step="od_conversion") is fake_raw


def test_unstamped_raw_has_no_lineage(fake_raw):
    assert lineage_of(fake_raw) is None
    assert stage_of(fake_raw) is None
    assert path_from(fake_raw) is None


@pytest.mark.parametrize("op", [
    pytest.param(lambda r: r.copy(), id="copy"),
    pytest.param(lambda r: r.copy().filter(0.5, 2.0, verbose="error"), id="filter"),
    pytest.param(lambda r: r.copy().resample(5.0, verbose="error"), id="resample"),
    pytest.param(lambda r: r.copy().crop(0, 10), id="crop"),
])
def test_stamp_survives_mne_operations(fake_raw, op):
    # info["temp"] is the only place MNE promises to leave alone; this is what makes
    # the whole approach viable, so it gets pinned rather than assumed.
    stamp(fake_raw, stage="preproc", step="beer_lambert")
    assert stage_of(op(fake_raw)) == "preproc"


def test_source_stage_is_read_from_the_input(fake_raw, make_raw):
    stamp(fake_raw, stage="od", step="od_conversion")
    out = stamp(make_raw(), stage="preproc", step="beer_lambert", source=fake_raw)
    assert lineage_of(out).source == "od"


def test_unstamped_source_is_recorded_as_raw(fake_raw, make_raw):
    out = stamp(make_raw(), stage="od", step="od_conversion", source=fake_raw)
    assert lineage_of(out).source == "raw"


def test_in_place_stamp_keeps_the_previous_stage_as_source(fake_raw):
    # sci_pruning stamps the object it was handed, so the old stage has to be read
    # before the new stamp overwrites it.
    stamp(fake_raw, stage="od", step="od_conversion")
    stamp(fake_raw, stage="sci", step="sci_pruning", source=fake_raw)
    lin = lineage_of(fake_raw)
    assert (lin.stage, lin.source) == ("sci", "od")


def test_stamp_preserves_unrelated_temp_keys(fake_raw):
    fake_raw.info["temp"] = {"other": 1}
    stamp(fake_raw, stage="od", step="od_conversion")
    assert fake_raw.info["temp"]["other"] == 1


# ---- require_stage ----

def test_require_stage_accepts_allowed(fake_raw):
    stamp(fake_raw, stage="preproc", step="beer_lambert")
    require_stage(fake_raw, "preproc")


def test_require_stage_rejects_other_stage(fake_raw):
    stamp(fake_raw, stage="filtered", step="bandpass")
    with pytest.raises(StageError):
        require_stage(fake_raw, "preproc")


def test_require_stage_rejects_unstamped(fake_raw):
    with pytest.raises(StageError):
        require_stage(fake_raw, "preproc")


# ---- Recorder ----

def test_sources_of_resolves_through_the_registered_input(fake_raw, make_raw, tmp_path):
    bids = tmp_path / "sub-01_nirs.snirf"
    stamp(fake_raw, stage="raw", step="load", path=bids.as_posix())
    rec = Recorder()
    rec.register_input(bids, fake_raw)

    od = stamp(make_raw(), stage="od", step="od_conversion", source=fake_raw)
    assert rec.sources_of(od) == [bids.as_posix()]


def test_register_input_records_an_input_that_carries_no_stamp(make_raw, tmp_path):
    # stamp() calls an unstamped source "raw"; the recorder has to agree, or the path is
    # dropped and every output of the run comes back with an empty Sources.
    bids = tmp_path / "sub-01_nirs.snirf"
    rec = Recorder()
    rec.register_input(bids, make_raw())                       # never stamped

    od = stamp(make_raw(), stage="od", step="od_conversion", source=make_raw())
    assert rec.sources_of(od) == [bids.as_posix()]


def test_sources_of_resolves_to_the_file_its_source_stage_was_written_to(make_raw, tmp_path):
    rec = Recorder()
    od = stamp(make_raw(), stage="od", step="od_conversion")
    od_path = tmp_path / "desc-od_nirs.snirf"
    rec.written(od_path, od)

    haemo = stamp(make_raw(), stage="preproc", step="beer_lambert", source=od)
    assert rec.sources_of(haemo) == [od_path.as_posix()]


def test_sources_of_falls_back_to_nearest_persisted_ancestor(make_raw, tmp_path):
    # Rest mode filters in memory without writing desc-filtered, so an output whose
    # source stage never reached disk still has to name something real.
    rec = Recorder()
    od = stamp(make_raw(), stage="od", step="od_conversion")
    od_path = tmp_path / "desc-od_nirs.snirf"
    rec.written(od_path, od)

    filtered = stamp(make_raw(), stage="filtered", step="bandpass", source=od)  # never written
    errts = stamp(make_raw(), stage="errts", step="glm_residuals", source=filtered)

    assert rec.sources_of(errts) == [od_path.as_posix()]


def test_sources_of_is_empty_when_nothing_was_written(make_raw):
    rec = Recorder()
    od = stamp(make_raw(), stage="od", step="od_conversion")
    assert rec.sources_of(od) == []


def test_path_of_returns_the_objects_own_file(make_raw, tmp_path):
    rec = Recorder()
    od = stamp(make_raw(), stage="od", step="od_conversion")
    od_path = tmp_path / "desc-od_nirs.snirf"
    rec.written(od_path, od)
    assert rec.path_of(od) == od_path.as_posix()


def test_written_records_the_entry_and_last(make_raw, tmp_path):
    rec = Recorder()
    od = stamp(make_raw(), stage="od", step="od_conversion", threshold=0.8)
    path = tmp_path / "desc-od_nirs.snirf"
    rec.written(path, od)

    entry = rec.entries[0]
    assert entry["stage"] == "od"
    assert entry["step"] == "od_conversion"
    assert entry["params"] == {"threshold": 0.8}
    assert rec.last == path


def test_writing_the_same_stage_twice_raises(make_raw, tmp_path):
    # Two files sharing one stage makes source resolution pick whichever was written
    # last, which once credited ALFF to the wrong residual. A parallel branch needs
    # its own stage name (that is why errtsbroad exists).
    rec = Recorder()
    first = stamp(make_raw(), stage="errts", step="glm_residuals")
    second = stamp(make_raw(), stage="errts", step="glm_residuals_broadband")
    rec.written(tmp_path / "a.snirf", first)

    with pytest.raises(StageError, match="already written"):
        rec.written(tmp_path / "b.snirf", second)
