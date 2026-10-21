"""Epoch figures on a planted design: condition order, a one-trial condition, the raw low-pass."""

import mne
import numpy as np
import pytest

from nirspipe.qc.figures.common.topomap import evoked_channel_map_figure
from nirspipe.qc.figures.subject.raw_figures import (
    build_epoch_preview_figure, build_evoked_topo_figure, build_trial_image_figure,
)
from nirspipe.qc.common.windows import markers_on_data_axis
from nirspipe.qc.metrics._helpers import epochable_events
from nirspipe.qc.subject.prep_raw_report import _below_cardiac
from nirspipe.qc.subject.report import _no_epoch_reason, _single_trial_conditions
from tests._synth import synth_raw


def _haemo(onsets, labels):
    raw = synth_raw("01", "rest", bad_pair=None, motion_onset=None)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    haemo = mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)
    haemo.set_annotations(mne.Annotations(onsets, [5.0] * len(onsets), labels))
    return haemo


@pytest.fixture(scope="module")
def mixed():
    # "once" first in time, then "zeta" before "alpha": neither a string sort nor mne's order
    onsets = [20.0] + [60.0 + 40 * k for k in range(8)]
    labels = ["once"] + ["zeta", "alpha"] * 4
    return _haemo(onsets, labels)


def _titles(fig):
    return [a.text for a in fig.layout.annotations]


def test_conditions_come_in_order_of_first_appearance(mixed):
    _, event_id = epochable_events(mixed, -5.0, 25.0)
    assert list(event_id) == ["once", "zeta", "alpha"]


def test_the_grand_mean_rows_follow_first_appearance(mixed):
    fig = build_epoch_preview_figure(mixed, epoch_tmin=-5.0, epoch_tmax=25.0)
    assert _titles(fig) == ["once (n=1), own scale", "zeta (n=4)", "alpha (n=4)"]


def test_a_one_trial_row_scales_itself_and_the_others_share_one_axis(mixed):
    fig = build_epoch_preview_figure(mixed, epoch_tmin=-5.0, epoch_tmax=25.0)
    lay = fig.layout
    assert lay.yaxis.matches is None and lay.yaxis.autorange is True
    assert lay.yaxis2.range is not None and lay.yaxis2.matches is None
    assert lay.yaxis3.matches == "y2"


def test_the_channel_map_columns_follow_first_appearance_and_name_the_unscaled_one(mixed):
    fig = evoked_channel_map_figure(mixed)
    if fig is None:
        pytest.skip("synthetic montage has no projectable head")
    heads = [t for t in _titles(fig) if t][:3]
    assert heads[0].startswith("once<br>") and "too few trials" in heads[0]
    assert heads[1:] == ["zeta", "alpha"]


def test_a_design_where_no_condition_repeats_is_still_drawn():
    haemo = _haemo([40.0, 150.0, 260.0], ["a", "b", "c"])
    assert _no_epoch_reason(haemo, -5.0, 25.0) is None
    assert _single_trial_conditions(haemo, -5.0, 25.0) == ["a", "b", "c"]
    fig = build_epoch_preview_figure(haemo, epoch_tmin=-5.0, epoch_tmax=25.0)
    # nothing clears the bar, so all three share one scale and none is called out
    assert _titles(fig) == ["a (n=1)", "b (n=1)", "c (n=1)"]


def test_the_single_trial_note_names_only_the_conditions_that_occur_once(mixed):
    assert _single_trial_conditions(mixed, -5.0, 25.0) == ["once"]


def test_the_raw_display_low_pass_drops_the_heartbeat_and_keeps_the_slow_signal():
    sfreq, n = 10.0, 6000
    t = np.arange(n) / sfreq
    slow, heart = np.sin(2 * np.pi * 0.1 * t), np.sin(2 * np.pi * 1.2 * t)
    info = mne.create_info(["S1_D1 hbo"], sfreq, ["hbo"])
    raw = mne.io.RawArray((slow + heart)[None] * 1e-6, info, verbose="error")
    out = _below_cardiac(raw, 0.7).get_data()[0] * 1e6
    mid = slice(1000, 5000)
    np.testing.assert_allclose(out[mid], slow[mid], atol=0.02)


def test_signal_by_position_shows_one_condition_at_a_time_in_first_appearance_order(mixed):
    fig = build_evoked_topo_figure(mixed, markers_on_data_axis(mixed), 2000, -5.0, 25.0)
    menu = fig.layout.updatemenus[0]
    assert [b.label for b in menu.buttons] == ["once (n=1)", "zeta (n=4)", "alpha (n=4)"]
    shown = {t.legendgroup.split(" ", 1)[1] for t in fig.data if t.visible}
    assert shown == {"once"}
    picked = menu.buttons[2].args[0]["visible"]
    assert {t.legendgroup.split(" ", 1)[1] for t, v in zip(fig.data, picked) if v} == {"alpha"}


def test_a_trial_image_draws_the_hbr_average_of_the_same_trials(mixed):
    figs = build_trial_image_figure(mixed, "S1_D1 hbo", -5.0, 25.0)
    events, event_id = epochable_events(mixed, -5.0, 25.0)
    epochs = mne.Epochs(mixed, events, event_id, tmin=-5.0, tmax=25.0, baseline=(-5.0, 0),
                        picks=["S1_D1 hbr"], preload=True, verbose="error")
    alpha = next(f for f in figs if "/ alpha" in f.layout.annotations[0].text)
    hbr = next(t for t in alpha.data if t.name == "HbR")
    np.testing.assert_allclose(hbr.y, epochs["alpha"].average().data[0] * 1e6, rtol=1e-6)
