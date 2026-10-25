"""Frames at either end that measured no light in most channels are held at the nearest lit
frame and marked BAD_edge before OD conversion, where one NaN would take a whole channel."""

import json

import mne
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nirspipe.pipeline.prep_pipeline import (PrepConfig, dark_edge_frames, repair_dark_edges,
                                             run_prep)
from nirspipe.qc.boilerplate.vocabulary import span_keys
from nirspipe.utils.spans import EDGE, bad_spans, excluded_spans, frames_under

from tests._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)


def _with(raw, edit):
    data = raw.get_data()
    edit(data)
    out = mne.io.RawArray(data, raw.info, verbose="error")
    return out.set_annotations(raw.annotations)


def _first_frame_dark(data):
    """As NinjaNIRS writes it: frame 0 NaN in most channels and near zero in the rest."""
    n_nan = int(0.9 * len(data))
    data[:n_nan, 0] = np.nan
    data[n_nan:, 0] = 1e-6 * data[n_nan:, 1]


@pytest.fixture(scope="module")
def raw():
    return synth_raw("01", "rest", duration=60.0, bad_pair=None, motion_onset=None)


# ---- which frames ----

def test_a_first_frame_dark_in_most_channels_and_a_dead_tail_are_found(raw):
    def edit(d):
        _first_frame_dark(d)
        d[:, -3:] = 0.0
    assert dark_edge_frames(_with(raw, edit)) == (1, 3)


@pytest.mark.parametrize("edit", [
    lambda d: None,
    lambda d: d.__setitem__((slice(0, 2), 0), np.nan),                # a minority of channels
    lambda d: d.__setitem__((slice(None), 300), np.nan),             # a dark frame mid-record
    lambda d: d.__setitem__((slice(None), slice(0, 5)), 0.5 * d[:, :5]),  # dim, still lit
], ids=["clean", "minority", "middle", "dim"])
def test_nothing_else_counts(raw, edit):
    assert dark_edge_frames(_with(raw, edit)) == (0, 0)


def test_a_dead_channel_has_no_vote(raw):
    def edit(d):
        d[: len(d) // 2 + 1] = 0.0
    assert dark_edge_frames(_with(raw, edit)) == (0, 0)


def test_optical_density_is_dark_only_where_it_is_not_finite(raw):
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    od._data[:, :2] = -1.0
    assert dark_edge_frames(od) == (0, 0)
    od._data[:, 0] = np.nan
    assert dark_edge_frames(od) == (1, 0)


# ---- the repair ----

def test_the_dark_frames_take_the_nearest_lit_frame_and_nothing_else_moves(raw):
    def edit(d):
        _first_frame_dark(d)
        d[:, -2:] = 0.0
    dark = _with(raw, edit)
    fixed, frames = repair_dark_edges(dark)
    data, before = fixed.get_data(), dark.get_data()
    assert frames == (1, 2)
    assert_array_equal(data[:, 0], before[:, 1])
    assert_array_equal(data[:, -2:], np.repeat(before[:, [-3]], 2, axis=1))
    assert_array_equal(data[:, 1:-2], before[:, 1:-2])
    assert np.isnan(before[0, 0])
    under = frames_under(fixed, bad_spans(fixed))
    assert np.flatnonzero(under).tolist() == [0, fixed.n_times - 2, fixed.n_times - 1]
    assert {(s["description"], s["kind"], s["source"]) for s in excluded_spans(fixed)} == {
        (EDGE, "corrupted", "prep (no light)")}


def test_a_lit_recording_comes_back_as_it_was(raw):
    fixed, frames = repair_dark_edges(raw)
    assert frames == (0, 0) and fixed is raw


# ---- through prep ----

def test_prep_keeps_every_channel_and_records_the_repair(raw, tmp_path):
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.8, motion_correction="tddr",
                        **_BANDS)
    result = run_prep(_with(raw, _first_frame_dark), config, tmp_path,
                      source_entities={"task": "rest"})
    # before the repair every channel NaN at frame 0 was NaN throughout in OD, and bad
    assert result.bad_channels == []
    assert np.isfinite(result.raw_haemo.get_data()).all()
    side = json.loads(next(tmp_path.rglob("*desc-od_nirs.json")).read_text(encoding="utf-8"))
    assert side["dark_edge_frames"] == {"start": 1, "end": 0}
    assert "non_finite_samples" not in side
    assert [s["description"] for s in side["excluded_spans"]] == [EDGE]


def test_the_methods_say_so_on_the_screening_step():
    spans = [{"description": EDGE, "kind": "corrupted", "source": "prep (no light)"}]
    assert span_keys("sci_pruning", {}, spans) == ["excluded_edge"]
    assert span_keys("sci_pruning", {}, []) == []
