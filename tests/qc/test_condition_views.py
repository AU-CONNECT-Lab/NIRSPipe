"""Per-condition QC views: what they are named, and what they refuse to carry over.

A view that cannot honestly fill a column drops it rather than printing a whole-run number
beside per-condition ones, and a figure reaches a condition page only if its own name says
it is that condition's.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from fnirs_pipe.qc.common.figure_io import figure_namer
from fnirs_pipe.qc.subject.condition_views import (
    UNSLICEABLE, figure_leaks, rescale_y_to_window, slice_record,
    carpet_window_spec, window_view_spec, zoom_to_condition,
)


# ---- which figures a raw condition page keeps ----

def _src(name: str) -> dict:
    return {"src": f"figures/{name}"}


def test_a_figure_named_for_the_condition_is_kept():
    """The name `prep-raw` gives a condition's own figure is the one this check accepts.

    It used to look for a `_<slug>_nirs` ending, which no figure has carried since the
    condition moved into its own entity, so every rewritten panel was dropped as a leak.
    """
    name = figure_namer("sub-01_task-rest", "game1", prefix="raw")("psd")
    assert figure_leaks({"psd": _src(name)}, "game1") == []


def test_the_run_wide_figure_and_another_conditions_are_both_leaks():
    run_wide = figure_namer("sub-01_task-rest", prefix="raw")("psd")
    other = figure_namer("sub-01_task-rest", "game10", prefix="raw")("psd")
    assert figure_leaks({"a": _src(run_wide), "b": _src(other)}, "game1") == ["a", "b"]


def test_a_window_fragment_and_the_whole_run_panels_pass():
    carpet = figure_namer("sub-01_task-rest", prefix="raw")("carpet")
    layout = figure_namer("sub-01_task-rest", prefix="raw")("layout")
    assert figure_leaks({"carpet": {"src": f"figures/{carpet}#game1"},
                         "layout": _src(layout)}, "game1") == []


# ---- what a view carries ----

def _record() -> dict:
    return {
        "raw": {"sci_mean": 0.9},
        "per_channel": {
            "raw_long": {"sci_per_channel": {"A": 0.9, "B": 0.8},
                         "psp_per_channel": {"A": 0.5, "B": 0.4},
                         "snr_per_channel": {"A": 100.0, "B": 90.0},
                         "cv_per_channel": {"A": 0.1, "B": 0.2}},
            "raw_short": {"sci_per_channel": {"S": 0.99}},
        },
    }


def test_the_sliced_values_replace_the_whole_run_ones():
    out = slice_record(_record(), {"sci_per_channel": {"A": 0.2, "B": 0.3, "S": 0.5}})
    assert out["per_channel"]["raw_long"]["sci_per_channel"] == {"A": 0.2, "B": 0.3}


def test_a_section_only_gets_the_channels_it_already_described():
    # channel_rows reads the sections to decide which block a channel is in, so leaking a
    # short channel into the long section would give it a long channel's verdict
    out = slice_record(_record(), {"sci_per_channel": {"A": 0.2, "B": 0.3, "S": 0.5}})
    assert set(out["per_channel"]["raw_long"]["sci_per_channel"]) == {"A", "B"}
    assert set(out["per_channel"]["raw_short"]["sci_per_channel"]) == {"S"}


def test_the_metrics_with_no_windowed_series_are_dropped_not_carried_over():
    # a whole-run SNR beside a per-condition SCI is two time scopes in one table
    out = slice_record(_record(), {"sci_per_channel": {"A": 0.2, "B": 0.3}})
    for key in UNSLICEABLE:
        assert key not in out["per_channel"]["raw_long"]


def test_a_metric_the_condition_has_no_value_for_comes_back_empty_not_stale():
    # nothing was sliced for psp, so it must not keep the whole-run numbers
    out = slice_record(_record(), {"sci_per_channel": {"A": 0.2, "B": 0.3}})
    assert out["per_channel"]["raw_long"]["psp_per_channel"] == {}


def test_scalars_and_other_sections_pass_through():
    out = slice_record(_record(), {"sci_per_channel": {"A": 0.2}})
    assert out["raw"] == {"sci_mean": 0.9}


def test_the_original_record_is_not_modified():
    rec = _record()
    slice_record(rec, {"sci_per_channel": {"A": 0.2, "B": 0.3}})
    assert rec["per_channel"]["raw_long"]["sci_per_channel"] == {"A": 0.9, "B": 0.8}
    assert "snr_per_channel" in rec["per_channel"]["raw_long"]


def test_a_record_with_no_per_channel_half_is_not_an_error():
    assert slice_record({"raw": {}}, {})["per_channel"] == {}


# ---- narrowing a time-axis figure instead of recomputing it ----

def test_every_shared_x_axis_is_narrowed():
    # the carpet is stacked subplots on one time axis; leaving one unset would show a panel
    # spanning a different stretch from the one above it
    fig = {"data": [{"x": [1, 2]}],
           "layout": {"xaxis": {"title": "s"}, "xaxis2": {}, "yaxis": {}}}
    out = zoom_to_condition(fig, 100.0, 200.0)
    assert out["layout"]["xaxis"]["range"] == [100.0, 200.0]
    assert out["layout"]["xaxis2"]["range"] == [100.0, 200.0]
    assert out["layout"]["yaxis"] == {}


def test_narrowing_pins_the_range_against_autoscaling():
    out = zoom_to_condition({"layout": {"xaxis": {}}}, 100.0, 200.0)
    assert out["layout"]["xaxis"]["autorange"] is False


def test_narrowing_keeps_the_rest_of_the_axis_and_the_data():
    fig = {"data": [{"x": [1, 2], "y": [3, 4]}], "layout": {"xaxis": {"title": "time (s)"}}}
    out = zoom_to_condition(fig, 10.0, 20.0)
    assert out["layout"]["xaxis"]["title"] == "time (s)"
    assert out["data"] == fig["data"]


def test_narrowing_does_not_modify_the_run_wide_figure():
    # the whole-run figure is written to its own file as well, so a mutation would narrow it
    fig = {"layout": {"xaxis": {}}}
    zoom_to_condition(fig, 10.0, 20.0)
    assert "range" not in fig["layout"]["xaxis"]


def test_a_figure_with_no_x_axis_is_returned_unchanged():
    fig = {"layout": {"yaxis": {}}}
    assert zoom_to_condition(fig, 10.0, 20.0) is fig


# ---- one file per channel, every condition's window in a table ----

def _two_row_figure():
    """A stand-in for the motion panel: a zero-pinned row carrying a shaded span, and a row
    that autoranges. The first row is quiet up to t=60 and twenty times louder after, which
    is the case the y rescale exists for.
    """
    t = list(range(120))
    quiet = [0.001 * (1 + i % 5) for i in range(60)]
    loud = [0.020 * (1 + i % 5) for i in range(60)]
    quiet_then_loud = quiet + loud
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[10, 10, 20, 20], y=[0, 1, 1, 0], fill="toself"))
    fig.add_trace(go.Scatter(x=t, y=quiet_then_loud))
    fig.add_trace(go.Scatter(x=t, y=[-0.5 + 0.01 * (i % 7) for i in t],
                             xaxis="x2", yaxis="y2"))
    fig.update_layout(yaxis=dict(range=[0, 0.11]), yaxis2=dict(), xaxis2=dict())
    return fig


def test_the_spec_says_what_rescaling_would_have_done():
    # the saved-per-condition path and the picked-at-load path have to be one measurement,
    # or a condition page and a fragment of the run's file drift apart silently
    spec = window_view_spec(_two_row_figure(), 0.0, 59.0)
    applied = rescale_y_to_window(_two_row_figure(), 0.0, 59.0)
    for key, (floor, top) in spec["y"].items():
        assert list(applied.layout[key].range) == [floor, top]
    notes = [a.text for a in applied.layout.annotations]
    assert notes == [n["text"] for n in spec["notes"]]


def test_a_zero_pinned_row_keeps_its_floor_in_the_spec():
    spec = window_view_spec(_two_row_figure(), 0.0, 59.0)
    assert spec["y"]["yaxis"][0] == 0.0
    assert spec["y"]["yaxis"][1] < 0.01        # the quiet stretch, not the run's 0.1


def test_a_shaded_span_is_named_by_trace_index_so_the_page_can_reach_it():
    # the span is trace data rather than layout, so narrowing the view has to restyle it
    spec = window_view_spec(_two_row_figure(), 0.0, 59.0)
    assert [b["i"] for b in spec["bands"]] == [0]
    assert spec["bands"][0]["hi"] == spec["y"]["yaxis"][1]


def test_measuring_a_window_leaves_the_run_wide_figure_alone():
    fig = _two_row_figure()
    window_view_spec(fig, 0.0, 59.0)
    assert list(fig.layout.yaxis.range) == [0, 0.11]
    assert fig.layout.annotations == ()


# ---- the carpet: its lines follow the condition, its colours do not ----

@pytest.fixture(scope="module")
def carpet_fig():
    """A real carpet through the shipped builder, quiet for the first half of the run."""
    import mne
    from tests._synth import synth_raw
    from fnirs_pipe.qc.figures.common.motion_panel import carpet_gvtd_figure
    from fnirs_pipe.qc.metrics import gvtd_channel_blocks

    raw = synth_raw("01", "tapping", duration=600.0)
    rng = np.random.default_rng(0)
    scale = 1 + np.where(raw.times < 300, 1.0, 20.0) * 1e-3 * rng.standard_normal(
        raw.get_data().shape)
    raw = mne.io.RawArray(raw.get_data() * scale, raw.info, verbose="error")
    blocks = gvtd_channel_blocks(raw)
    return carpet_gvtd_figure(
        raw, raw.ch_names, raw_after=raw, channel_set=blocks[0][0], blocks=blocks,
        corrected_segments=[(400.0, 20.0)], spike_segments={blocks[0][0]: [(100.0, 5.0)]})


def test_both_gvtd_rows_are_given_one_top(carpet_fig):
    # long and short are the same unit at comparable magnitudes, and scaling each to itself
    # would hide the difference the second row was added to show
    spec = carpet_window_spec(carpet_fig, 0.0, 300.0)
    tops = {tuple(v) for v in spec["y"].values()}
    assert len(spec["y"]) == 2 and len(tops) == 1


def test_the_top_is_measured_over_the_window(carpet_fig):
    quiet = carpet_window_spec(carpet_fig, 0.0, 300.0)["y"]["yaxis2"]
    loud = carpet_window_spec(carpet_fig, 300.0, 600.0)["y"]["yaxis2"]
    assert quiet[0] == loud[0] == 0.0          # a magnitude row keeps its floor
    assert quiet[1] < loud[1]


def test_the_heatmaps_keep_the_runs_colour_scale(carpet_fig):
    # the z-score is against a per-channel mean and SD taken over the whole run from the
    # uncorrected side; re-deriving it per condition would make one colour mean a different
    # deviation on each page, with one colour bar for the whole image and nowhere to say so
    spec = carpet_window_spec(carpet_fig, 0.0, 300.0)
    assert set(spec["y"]) == {"yaxis2", "yaxis3"}
    assert "coloraxis" not in spec


def test_each_row_restates_its_numbers_over_the_window(carpet_fig):
    # the run's maximum printed beside an axis that no longer reaches it is worse than no
    # label, so the view rewrites the line rather than only moving the axis
    quiet = {n["name"]: n["text"] for n in carpet_window_spec(carpet_fig, 0.0, 300.0)["notes"]}
    loud = {n["name"]: n["text"] for n in carpet_window_spec(carpet_fig, 300.0, 600.0)["notes"]}
    assert set(quiet) == set(loud)
    assert quiet["gvtd-stat-long"] != loud["gvtd-stat-long"]


def test_the_threshold_stays_the_runs(carpet_fig):
    # a condition is counted against the run's line rather than given one of its own
    import re
    texts = [n["text"] for n in carpet_window_spec(carpet_fig, 0.0, 300.0)["notes"]]
    run = [n["text"] for n in carpet_window_spec(carpet_fig, 300.0, 600.0)["notes"]]
    def thresholds(lines):
        return {re.search(r"thresh (\S+)", t).group(1) for t in lines if "thresh" in t}

    assert thresholds(texts) == thresholds(run)


def test_the_shaded_spans_follow_the_new_top(carpet_fig):
    spec = carpet_window_spec(carpet_fig, 0.0, 300.0)
    assert spec["bands"]
    for band in spec["bands"]:
        assert band["lo"] == 0.0
        assert band["hi"] == spec["y"]["yaxis2"][1]


def test_measuring_the_carpet_leaves_it_alone(carpet_fig):
    before = [list(carpet_fig.layout.yaxis2.range), list(carpet_fig.layout.yaxis3.range)]
    carpet_window_spec(carpet_fig, 0.0, 300.0)
    assert [list(carpet_fig.layout.yaxis2.range),
            list(carpet_fig.layout.yaxis3.range)] == before
