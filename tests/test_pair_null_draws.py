"""Which stand-ins are refused, and why each refusal is a refusal rather than a warning.

A draw only means something if it describes the same stretch of the same clock at the same
rate as the table it will be subtracted from. Coherence rises as a record shortens, so draws
of unequal length are a mixture rather than a null; a pair aligned on a different trigger
puts the fixed member on a different clock than the real table and invalidates the transforms
this reuses across draws; and two recordings at different rates cannot be compared at all.
Each is counted by name instead of raised, because one unusable stand-in is not a reason to
lose the other twenty.
"""

import numpy as np
import pytest

from fnirs_pipe.pipeline import group_io, group_quality, hyperscanning
from fnirs_pipe.pipeline.group_io import GroupEntry
from fnirs_pipe.pipeline.pair_null import _draw_pairs

FIXED = "sub-p1d01"
REAL_DURATION = 900.0
REAL_OFFSET = 22.4


class _Raw:
    def __init__(self, duration=1000.0, sfreq=10.0):
        self.info = {"sfreq": sfreq}
        self.times = np.array([0.0, duration])
        self.cropped_to = None

    def copy(self):
        out = _Raw(float(self.times[-1]), self.info["sfreq"])
        return out

    def crop(self, tmax=None):
        self.cropped_to = tmax
        self.times = np.array([0.0, float(tmax)])
        return self


@pytest.fixture
def wired(monkeypatch):
    """Stand in for the tree. `state` steers what each candidate looks like."""
    state = {"partners": {}, "offsets": {}, "aligned_duration": {}}

    def fake_haemo(output_dir, entries, desc="preproc"):
        entry = entries[0]
        return {entry.subject_id: state["partners"].get(entry.subject_id, _Raw())}

    def fake_align(raws, task):
        pid = next(k for k in raws if k != FIXED)
        duration = state["aligned_duration"].get(pid, 1000.0)
        for raw in raws.values():
            raw.times = np.array([0.0, duration])
        return raws, {FIXED: state["offsets"].get(pid, REAL_OFFSET), pid: 0.0}

    monkeypatch.setattr(group_io, "load_group_haemo", fake_haemo)
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    monkeypatch.setattr(hyperscanning, "align_recordings", fake_align)
    return state


def _run(candidates, refused=None, coverage=None, n_max=None, windows=None):
    refused = {} if refused is None else refused
    coverage = {} if coverage is None else coverage
    drawn = list(_draw_pairs(
        "/out", "full", FIXED, _Raw(), [GroupEntry("dXX", c, "full") for c in candidates],
        desc="preproc", bads_scope="run", scope_tasks=["full"],
        real_duration=REAL_DURATION, real_offset=REAL_OFFSET, n_max=n_max,
        refused=refused, coverage=coverage, windows=windows))
    return drawn, refused, coverage


def test_an_eligible_stand_in_is_drawn(wired):
    drawn, refused, _ = _run(["sub-p2d02"])
    assert [pid for pid, _ in drawn] == ["sub-p2d02"]
    assert refused == {}


def test_a_different_sampling_rate_is_refused(wired):
    wired["partners"]["sub-p2d02"] = _Raw(sfreq=25.0)
    drawn, refused, _ = _run(["sub-p2d02"])
    assert drawn == []
    assert refused["sampling_rate"] == ["sub-p2d02"]


def test_a_recording_too_short_to_reach_the_real_length_is_refused(wired):
    """Averaging draws of unequal length is a mixture, not a null."""
    wired["aligned_duration"]["sub-p2d02"] = REAL_DURATION - 10.0
    drawn, refused, _ = _run(["sub-p2d02"])
    assert drawn == []
    assert refused["too_short"] == ["sub-p2d02"]


def test_a_stand_in_that_moves_the_fixed_members_crop_is_refused(wired):
    """The earliest shared trigger differs, so the draw is on another clock."""
    wired["offsets"]["sub-p2d02"] = REAL_OFFSET + 5.0
    drawn, refused, _ = _run(["sub-p2d02"])
    assert drawn == []
    assert refused["moves_the_clock"] == ["sub-p2d02"]


def test_every_draw_is_cut_to_the_real_dyads_length(wired):
    wired["aligned_duration"]["sub-p2d02"] = 1500.0
    drawn, _, _ = _run(["sub-p2d02"])
    _, aligned = drawn[0]
    assert {float(r.times[-1]) for r in aligned.values()} == {REAL_DURATION}


def test_one_unusable_stand_in_does_not_lose_the_others(wired):
    wired["partners"]["sub-p2d03"] = _Raw(sfreq=25.0)
    drawn, refused, _ = _run(["sub-p2d02", "sub-p2d03", "sub-p2d04"])
    assert [pid for pid, _ in drawn] == ["sub-p2d02", "sub-p2d04"]
    assert refused["sampling_rate"] == ["sub-p2d03"]


def test_the_cap_stops_the_draws(wired):
    drawn, _, _ = _run(["sub-p2d02", "sub-p2d03", "sub-p2d04"], n_max=2)
    assert len(drawn) == 2


def test_an_unreadable_stand_in_is_counted_rather_than_raised(wired, monkeypatch):
    def boom(output_dir, entries, desc="preproc"):
        raise FileNotFoundError("no derivative for this one")

    monkeypatch.setattr(group_io, "load_group_haemo", boom)
    drawn, refused, _ = _run(["sub-p2d02"])
    assert drawn == []
    assert refused["unreadable"] == ["sub-p2d02"]


def test_the_condition_overlap_is_measured_for_every_draw(wired, monkeypatch):
    """Measured and reported, never used to drop a draw: the user asked for it that way."""
    monkeypatch.setattr("fnirs_pipe.pipeline.pair_null.condition_coverage",
                        lambda raw, windows: {"game1": 0.42})
    drawn, _, coverage = _run(["sub-p2d02"], windows=[("game1", 0.0, 300.0)])
    assert len(drawn) == 1                       # low overlap still counts
    assert coverage == {"sub-p2d02": {"game1": 0.42}}
