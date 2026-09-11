"""The trial half of a per-condition page: which trials it holds and when it holds none.

A condition window is built from one annotation, so a single-level design gives every window
exactly one trial and these panels stay empty. They fill in on a two-level design, which is
what ``task-mixed`` in ``_synth`` is for.
"""

from __future__ import annotations

import mne
import pytest

from fnirs_pipe.qc.figures.raw_figures import (
    _trial_image_by_span, build_trial_image_by_condition,
)
from fnirs_pipe.qc.hyper_report import condition_windows
from fnirs_pipe.qc.metrics.windowed import SCREEN_WINDOW_S
from fnirs_pipe.qc.report import _condition_trial_qc
from tests._synth import synth_raw

EPOCH = (-5.0, 25.0)


def _haemo(task: str) -> mne.io.Raw:
    raw = synth_raw("01", task)
    od = mne.preprocessing.nirs.optical_density(raw)
    return mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)


def _spans(raw: mne.io.Raw) -> list:
    return condition_windows(raw, min_duration=2 * SCREEN_WINDOW_S)


@pytest.fixture(scope="module")
def mixed():
    raw = synth_raw("01", "mixed")
    return _haemo("mixed"), _spans(raw)


def test_two_level_design_gives_one_window_per_block(mixed):
    _, spans = mixed
    assert [label for label, _, _ in spans] == ["talk", "listen"]


def test_trials_are_assigned_by_window_not_by_event_name(mixed):
    haemo, spans = mixed
    picks = [haemo.ch_names.index(next(c for c in haemo.ch_names if c.endswith(" hbo")))]
    res = _trial_image_by_span(haemo, picks, spans, *EPOCH)
    # every trial carries the same description, so a split by event name would pool them
    assert {label: data.shape[0] for label, (data, _, _) in res.items()} == {
        "talk": 6, "listen": 6}
    assert all(set(rows) == {"trial"} for _, _, rows in res.values())


def test_the_annotation_that_defines_a_window_is_not_one_of_its_trials(mixed):
    haemo, spans = mixed
    picks = [haemo.ch_names.index(next(c for c in haemo.ch_names if c.endswith(" hbo")))]
    res = _trial_image_by_span(haemo, picks, spans, *EPOCH)
    # 14 annotations, 12 of them trials; the two block markers are dropped
    assert sum(data.shape[0] for data, _, _ in res.values()) == 12


def test_condition_panels_share_one_colour_scale(mixed):
    haemo, spans = mixed
    ch = next(c for c in haemo.ch_names if c.endswith(" hbo"))
    figs = build_trial_image_by_condition(haemo, ch, spans, *EPOCH)
    scales = {(f[0].data[0].zmin, f[0].data[0].zmax) for f in figs.values()}
    assert len(figs) == 2 and len(scales) == 1


def test_a_block_design_gets_no_condition_trial_images():
    raw = synth_raw("01", "tapping")
    haemo = _haemo("tapping")
    ch = next(c for c in haemo.ch_names if c.endswith(" hbo"))
    # its blocks are shorter than two screening windows, so it has no condition windows at all
    assert _spans(raw) == []
    assert build_trial_image_by_condition(haemo, ch, _spans(raw), *EPOCH) is None


def test_condition_trial_qc_slices_the_runs_rows(tmp_path):
    rows = [(30.0, "trial-002", {"sci_mean": 0.9}), (55.0, "trial-003", {"sci_mean": 0.8}),
            (200.0, "trial-009", {"sci_mean": 0.7})]
    out = _condition_trial_qc(rows, (20.0, 170.0), "_talk", "01", [], tmp_path, min_trials=2)
    assert out["trial_qc_path"] and out["condition_trial_reason"] == ""
    assert (tmp_path / "trial_qc_talk.html").exists()


def test_a_window_holding_only_its_own_annotation_says_so(tmp_path):
    rows = [(20.0, "trial-001_20s_talk", {"sci_mean": 0.9})]
    out = _condition_trial_qc(rows, (20.0, 170.0), "_talk", "01", [], tmp_path, min_trials=2)
    assert out["trial_qc_path"] is None
    assert "annotation that defines it" in out["condition_trial_reason"]


def test_too_few_trials_names_the_count(tmp_path):
    rows = [(30.0, "trial-002", {"sci_mean": 0.9})]
    out = _condition_trial_qc(rows, (20.0, 170.0), "_talk", "01", [], tmp_path, min_trials=2)
    assert out["trial_qc_path"] is None
    assert "1 trial inside its window" in out["condition_trial_reason"]
