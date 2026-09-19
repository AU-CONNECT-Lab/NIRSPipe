"""Per-condition draws must ask for a crop the recording can actually give.

The alignment crop leaves the anchor marker a rounding error either side of zero, and the
duration tolerance lets a block run a millisecond past the end. Both reach `Raw.crop`, which
refuses outright rather than rounding, so a single stand-in took a whole dyad down.
"""

import numpy as np
import pytest

from fnirs_pipe.pipeline import group_io, group_quality, pair_null
from fnirs_pipe.pipeline.group_io import GroupEntry
from fnirs_pipe.pipeline.pair_null import _draw_condition_pairs

FIXED = "sub-p1d01"
WINDOWS = [("baseline", 0.0, 300.0), ("game1", 500.0, 1400.0)]


class _Raw:
    """Refuses out-of-range crops the way mne does, which is the point of these."""

    def __init__(self, duration=1400.0, sfreq=10.0):
        self.info = {"sfreq": sfreq}
        self.times = np.array([0.0, duration])

    def copy(self):
        return _Raw(float(self.times[-1]), self.info["sfreq"])

    def crop(self, tmin=0.0, tmax=None):
        if tmin < 0:
            raise ValueError(f"tmin ({tmin}) must be >= 0")
        if tmax is not None and tmax > float(self.times[-1]):
            raise ValueError(f"tmax ({tmax}) must be less than or equal to the end")
        self.times = np.array([0.0, float(tmax) - float(tmin)])
        return self


@pytest.fixture
def wired(monkeypatch):
    state = {"onsets": {}, "duration": {}}

    def fake_haemo(output_dir, entries, desc="preproc"):
        pid = entries[0].subject_id
        return {pid: _Raw(state["duration"].get(pid, 1400.0))}

    monkeypatch.setattr(group_io, "load_group_haemo", fake_haemo)
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    monkeypatch.setattr(pair_null, "_partner_condition_onsets",
                        lambda raw, labels: dict(state["onsets"]))
    return state


def _run(state, partners=("sub-p2d02",), fixed_duration=1400.0):
    refused: dict = {}
    drawn = list(_draw_condition_pairs(
        "/out", "full", FIXED, _Raw(fixed_duration),
        [GroupEntry("dXX", p, "full") for p in partners],
        desc="preproc", bads_scope="run", scope_tasks=["full"],
        windows=WINDOWS, n_max=None, refused=refused))
    return drawn, refused


def test_an_onset_a_hair_below_zero_is_still_drawn(wired):
    wired["onsets"] = {"baseline": -1.4210854715202004e-14, "game1": 480.0}
    drawn, refused = _run(wired)
    assert [label for _, label, _ in drawn] == ["baseline", "game1"]
    assert refused == {}


def test_a_block_a_hair_past_the_end_is_cut_to_the_end(wired):
    wired["onsets"] = {"baseline": 0.0, "game1": 500.0004}
    wired["duration"] = {"sub-p2d02": 1400.0}
    drawn, refused = _run(wired)
    assert [label for _, label, _ in drawn] == ["baseline", "game1"]
    assert refused == {}


def test_both_sides_of_a_draw_stay_the_same_length(wired):
    wired["onsets"] = {"baseline": 0.0, "game1": 500.0004}
    drawn, _ = _run(wired)
    for _, _, pair in drawn:
        spans = [float(raw.times[-1]) for raw in pair.values()]
        assert max(spans) - min(spans) < 0.1  # a sample period, what crop rounds to


def test_a_block_genuinely_past_the_end_is_still_refused(wired):
    wired["onsets"] = {"baseline": 0.0, "game1": 600.0}
    wired["duration"] = {"sub-p2d02": 1400.0}
    drawn, refused = _run(wired)
    assert [label for _, label, _ in drawn] == ["baseline"]
    assert refused["short_game1"] == ["sub-p2d02"]
