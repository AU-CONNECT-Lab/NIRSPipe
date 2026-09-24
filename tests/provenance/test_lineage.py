"""Stage lineage and the write recorder.

These cover the mechanism itself: does the stamp survive the MNE operations the
pipeline actually performs, and does Recorder resolve an output's source to the
right file. Both fail silently, so the assertions here are about correctness
of attribution, not about any numeric result.
"""

import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.utils.lineage import (
    Recorder,
    carried_params,
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


# ---- carried_params ----

def test_carried_params_hands_the_previous_params_to_a_re_stamp(fake_raw):
    """A re-stamp replaces the whole entry, so a parallel branch has to hand them back in."""
    stamp(fake_raw, stage="errts", step="glm_residuals", noise_model="ar", drift_model="cosine")
    stamp(fake_raw, stage="errtsbroad", step="glm_residuals_broadband",
          **{**carried_params(fake_raw), "resample_sfreq": 5.0})

    lin = lineage_of(fake_raw)
    assert lin.stage == "errtsbroad"
    assert lin.params == {"noise_model": "ar", "drift_model": "cosine", "resample_sfreq": 5.0}


def test_carried_params_has_nothing_to_carry_from_an_unstamped_object(fake_raw):
    assert carried_params(fake_raw) == {}


@pytest.mark.parametrize("name", ["raw", "stage", "step", "source", "path"])
def test_carried_params_refuses_a_param_named_after_a_stamp_argument(fake_raw, name):
    """Expanded into stamp() it would bind twice, and the TypeError names no culprit."""
    stamp(fake_raw, stage="errts", step="glm_residuals")
    lineage_of(fake_raw).params[name] = "x"

    with pytest.raises(StageError, match=name):
        carried_params(fake_raw)


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
    # last. A parallel branch needs its own stage name (that is why errtsbroad exists).
    rec = Recorder()
    first = stamp(make_raw(), stage="errts", step="glm_residuals")
    second = stamp(make_raw(), stage="errts", step="glm_residuals_broadband")
    rec.written(tmp_path / "a.snirf", first)

    with pytest.raises(StageError, match="already written"):
        rec.written(tmp_path / "b.snirf", second)


# ---- what the alignment stamp carries forward ----

def test_alignment_carries_the_passband_forward(make_raw):
    """Every inter-brain consumer sees only the aligned stamp, so the filter has to survive.

    `stamp` replaces the whole entry, so the errts stage's `high_pass` has to be carried onto
    the aligned stamp. The ISC panel reads that key to decide whether a whole-record
    correlation is being run on drift, and without it would say so on every run, filtered or
    not.
    """
    from fnirs_pipe.pipeline.hyper.alignment import trim_to_shortest
    from fnirs_pipe.pipeline.hyper.group_io import unfiltered_stage_note

    raws = {}
    for sid in ("sub-01", "sub-02"):
        raw = make_raw()
        stamp(raw, stage="errts", step="load", path=f"{sid}.snirf",
              high_pass=0.01, low_pass=0.2, filter_method="iir", filter_order=4)
        raws[sid] = raw

    trimmed, _ = trim_to_shortest(raws)

    for sid, raw in trimmed.items():
        lin = lineage_of(raw)
        assert lin.stage == "aligned", sid
        assert lin.params["high_pass"] == 0.01, sid
        assert lin.params["low_pass"] == 0.2, sid
        # the step's own record is still there beside it
        assert lin.params["aligned"] is False, sid
    assert unfiltered_stage_note(trimmed) is None


def test_an_unfiltered_stage_is_still_reported_after_alignment(make_raw):
    """The carry-forward must not silence the warning it was blocking."""
    from fnirs_pipe.pipeline.hyper.alignment import trim_to_shortest
    from fnirs_pipe.pipeline.hyper.group_io import unfiltered_stage_note

    raws = {}
    for sid in ("sub-01", "sub-02"):
        raw = make_raw()
        stamp(raw, stage="preproc", step="load", path=f"{sid}.snirf")
        raws[sid] = raw

    trimmed, _ = trim_to_shortest(raws)
    note = unfiltered_stage_note(trimmed)
    assert note is not None and "record no bandpass" in note
