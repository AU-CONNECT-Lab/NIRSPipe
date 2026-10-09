"""A per-condition cut must come out croppable and one length on both sides, whatever the floats do.

Two rounding faults reach `Raw.crop` or the transform: a pad that runs to a recording's end
sums to a hair past its last sample, which crop refuses, and two cuts of one duration can round
to sample counts one apart, which the wavelet grid refuses. Real `RawArray`s here, because the
faults are in where the samples fall.
"""

import mne
import numpy as np
import pytest

from nirspipe.pipeline.hyper import group_io, group_quality, pair_null
from nirspipe.pipeline.hyper.group_io import GroupEntry
from nirspipe.pipeline.hyper.pair_null import _draw_condition_pairs

FIXED, PARTNER = "sub-01G01", "sub-02G02"


def _raw(n_samples: int, sfreq: float) -> mne.io.RawArray:
    info = mne.create_info(["S1_D1 hbo"], sfreq, "hbo")
    return mne.io.RawArray(np.zeros((1, n_samples)), info, verbose="error")


@pytest.fixture
def wired(monkeypatch):
    state = {"partner": None, "onsets": {}}

    def partner(output_dir, entries, desc="preproc"):
        return {entries[0].subject_id: state["partner"].copy()}

    monkeypatch.setattr(group_io, "load_group_haemo", partner)
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    monkeypatch.setattr(pair_null, "_partner_condition_onsets",
                        lambda raw, labels: dict(state["onsets"]))
    return state


def _draw(fixed, windows, band_fmin):
    refused: dict = {}
    drawn = list(_draw_condition_pairs(
        "/out", "main", FIXED, fixed, [GroupEntry("G02", PARTNER, "main")], desc="preproc",
        bads_scope="run", scope_tasks=["main"], windows=windows, band_fmin=band_fmin, n_max=None,
        refused=refused))
    return drawn, refused


def test_a_pad_reaching_the_recordings_end_is_cut_at_its_last_sample(wired):
    # the stand-in's game1 ends 129.9 s before its recording does, inside the 141 s pad
    start, t0, t1 = 358.9, 406.2, 1208.8
    wired["partner"] = _raw(12799, 10.0)
    wired["onsets"] = {"game1": start}
    end = float(wired["partner"].times[-1])
    span = t1 - t0
    assert start + span + (end - start - span) > end   # the overshoot this guards against
    drawn, refused = _draw(_raw(30001, 10.0), [("game1", t0, t1)], band_fmin=0.02)
    assert refused == {}
    (_, label, pair, (lo, hi), _, _), = drawn
    assert label == "game1" and hi - lo == pytest.approx(span)
    assert pair[PARTNER].n_times == pair[FIXED].n_times


def test_cuts_rounding_a_sample_apart_come_out_one_length(wired):
    # at 5 Hz, a cut from 100.0 s and one from 200.05 s of the same duration round to 1973 and 1972
    wired["partner"] = _raw(5001, 5.0)
    wired["onsets"] = {"game1": 200.05}
    drawn, refused = _draw(_raw(5001, 5.0), [("game1", 100.0, 400.0)], band_fmin=0.06)
    assert refused == {}
    (_, _, pair, (lo, hi), white, _), = drawn
    assert pair[FIXED].n_times == pair[PARTNER].n_times == 1972
    assert white is pair
    assert hi <= float(pair[FIXED].times[-1])


def test_equal_cuts_are_left_as_they_were(wired):
    wired["partner"] = _raw(5001, 5.0)
    wired["onsets"] = {"game1": 200.0}
    drawn, _ = _draw(_raw(5001, 5.0), [("game1", 100.0, 400.0)], band_fmin=0.06)
    (_, _, pair, _, _, _), = drawn
    assert pair[FIXED].n_times == pair[PARTNER].n_times == 1973
