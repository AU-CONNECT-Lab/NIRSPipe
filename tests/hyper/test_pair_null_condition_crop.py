"""Per-condition draws must ask for a crop the recording can actually give.

The alignment crop leaves the anchor marker a rounding error either side of zero, and the
duration tolerance lets a block run a millisecond past the end. Both reach `Raw.crop`, which
refuses outright rather than rounding, so a single stand-in took a whole dyad down.
"""

import numpy as np
import pytest

from fnirs_pipe.pipeline.hyper import group_io, group_quality, pair_null
from fnirs_pipe.pipeline.hyper.group_io import GroupEntry
from fnirs_pipe.pipeline.hyper.pair_null import _draw_condition_pairs

FIXED = "sub-p1d01"
WINDOWS = [("baseline", 0.0, 300.0), ("game1", 500.0, 1400.0)]
BAND_FMIN = 0.06                 # cone_margin_s(0.06) is 47.1 s of pad either side


class _Raw:
    """Refuses out-of-range crops the way mne does, which is the point of these."""

    def __init__(self, duration=1600.0, sfreq=10.0):
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
        return {pid: _Raw(state["duration"].get(pid, 1600.0))}

    monkeypatch.setattr(group_io, "load_group_haemo", fake_haemo)
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    monkeypatch.setattr(pair_null, "_partner_condition_onsets",
                        lambda raw, labels: dict(state["onsets"]))
    return state


def _run(state, partners=("sub-p2d02",), fixed_duration=1600.0):
    refused: dict = {}
    drawn = list(_draw_condition_pairs(
        "/out", "full", FIXED, _Raw(fixed_duration),
        [GroupEntry("dXX", p, "full") for p in partners],
        desc="preproc", bads_scope="run", scope_tasks=["full"],
        windows=WINDOWS, band_fmin=BAND_FMIN, n_max=None, refused=refused))
    return drawn, refused


def test_an_onset_a_hair_below_zero_is_still_drawn(wired):
    wired["onsets"] = {"baseline": -1.4210854715202004e-14, "game1": 480.0}
    drawn, refused = _run(wired)
    assert [label for _, label, *_ in drawn] == ["baseline", "game1"]
    assert refused == {}


def test_a_block_a_hair_past_the_end_is_cut_to_the_end(wired):
    wired["onsets"] = {"baseline": 0.0, "game1": 500.0004}
    wired["duration"] = {"sub-p2d02": 1400.0}
    drawn, refused = _run(wired)
    assert [label for _, label, *_ in drawn] == ["baseline", "game1"]
    assert refused == {}


def test_both_sides_of_a_draw_stay_the_same_length(wired):
    wired["onsets"] = {"baseline": 0.0, "game1": 500.0004}
    drawn, _ = _run(wired)
    for _, _, pair, _ in drawn:
        spans = [float(raw.times[-1]) for raw in pair.values()]
        assert max(spans) - min(spans) < 0.1  # a sample period, what crop rounds to


def test_a_block_genuinely_past_the_end_is_still_refused(wired):
    wired["onsets"] = {"baseline": 0.0, "game1": 600.0}
    wired["duration"] = {"sub-p2d02": 1400.0}
    drawn, refused = _run(wired)
    assert [label for _, label, *_ in drawn] == ["baseline"]
    assert refused["short_game1"] == ["sub-p2d02"]


def test_the_condition_is_padded_and_its_place_reported(wired):
    """A draw carries context either side, and says where the condition sits inside it."""
    wired["onsets"] = {"baseline": 0.0, "game1": 500.0}
    drawn, _ = _run(wired)
    by_label = {label: (pair, inner) for _, label, pair, inner in drawn}

    # game1 has room for the full 47.1 s either side
    pair, (lo, hi) = by_label["game1"]
    assert hi - lo == pytest.approx(900.0)
    assert lo == pytest.approx(47.14, abs=0.01)
    for raw in pair.values():
        assert float(raw.times[-1]) == pytest.approx(900.0 + 2 * 47.14, abs=0.02)

    # baseline starts at 0 on both sides, so there is no lead to take and it says so
    pair, (lo, hi) = by_label["baseline"]
    assert lo == 0.0 and hi == pytest.approx(300.0)
    for raw in pair.values():
        assert float(raw.times[-1]) == pytest.approx(300.0 + 47.14, abs=0.02)


# ---- equal-length windows: the stand-in is cut at its own marker plus the offset ----

def _run_windows(state, windows, sources, partners=("sub-p2d02",), fixed_duration=1600.0):
    refused: dict = {}
    drawn = list(_draw_condition_pairs(
        "/out", "full", FIXED, _Raw(fixed_duration),
        [GroupEntry("dXX", p, "full") for p in partners],
        desc="preproc", bads_scope="run", scope_tasks=["full"],
        windows=windows, band_fmin=BAND_FMIN, n_max=None, refused=refused,
        window_sources=sources))
    return drawn, refused


def test_a_window_is_cut_at_the_partners_own_marker_plus_its_offset(wired):
    """The whole point: the offset rides on the stand-in's clock, not the real dyad's."""
    from fnirs_pipe.qc.common.windows import split_windows

    windows, sources = split_windows([("game1", 500.0, 1400.0)], 300.0)
    assert [w[0] for w in windows] == ["game1-w1", "game1-w2", "game1-w3"]
    # the stand-in entered game1 100 s earlier than the real dyad did
    wired["onsets"] = {"game1": 400.0}
    drawn, refused = _run_windows(wired, windows, sources)
    assert refused == {}
    assert [label for _, label, *_ in drawn] == ["game1-w1", "game1-w2", "game1-w3"]


def test_every_window_of_a_draw_is_the_same_length(wired):
    from fnirs_pipe.qc.common.windows import split_windows

    windows, sources = split_windows([("baseline", 0.0, 300.0), ("game1", 500.0, 1400.0)], 300.0)
    wired["onsets"] = {"baseline": 0.0, "game1": 450.0}
    drawn, _ = _run_windows(wired, windows, sources)
    spans = set()
    for _, _, pair, inner in drawn:
        for raw in pair.values():
            spans.add(round(float(raw.times[-1]), 3))
        assert round(inner[1] - inner[0], 3) == 300.0
    # one span per distinct pad, and every analysed stretch is 300 s whatever the pad
    assert len(spans) <= 2


def test_a_partner_too_short_for_a_late_window_is_refused_by_condition_name(wired):
    """Refused under the condition, not the window: the marker is what it lacks."""
    from fnirs_pipe.qc.common.windows import split_windows

    windows, sources = split_windows([("game1", 500.0, 1400.0)], 300.0)
    wired["onsets"] = {}                    # never entered game1
    _, refused = _run_windows(wired, windows, sources)
    assert list(refused) == ["no_game1"]


def test_without_a_mapping_the_labels_are_the_conditions_themselves(wired):
    """No window grid means the old behaviour, exactly."""
    wired["onsets"] = {"baseline": 0.0, "game1": 480.0}
    drawn, refused = _run_windows(wired, WINDOWS, {})
    assert [label for _, label, *_ in drawn] == ["baseline", "game1"]
    assert refused == {}
