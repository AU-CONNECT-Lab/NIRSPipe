"""Per-condition QC views: what they are named, and what they refuse to carry over.

The naming follows `fnirs-prep crop`, which already made this call: a labelled segment takes
the label as its task- entity, an unlabelled one gets seg-NN. Two things differ from crop and
both are pinned here. The labels come from annotations rather than from a table someone
wrote, so they can hold characters a filename cannot and can collide once reduced; and a
view that cannot honestly fill a column drops it rather than printing a whole-run number
beside per-condition ones.
"""

import plotly.graph_objects as go

from fnirs_pipe.qc.condition_views import (
    UNSLICEABLE, condition_stem, condition_stems, rescale_y_to_window, slice_record,
    window_view_spec, zoom_to_condition,
)

STEM = "sub-01_task-full_desc-raw_nirs"


# ---- naming, crop's rule ----

def test_a_labelled_condition_replaces_the_task_entity():
    assert condition_stem(STEM, "game1", 1) == "sub-01_task-game1_desc-raw_nirs"


def test_the_desc_entity_survives_the_way_it_does_through_crop():
    assert condition_stem("sub-01_task-full_desc-errts_nirs", "baseline", 1) == \
        "sub-01_task-baseline_desc-errts_nirs"


def test_an_unlabelled_condition_appends_a_segment_number():
    assert condition_stem(STEM, None, 3) == f"{STEM}_seg-03"
    assert condition_stem(STEM, "", 3) == f"{STEM}_seg-03"


def test_a_label_is_reduced_to_what_a_bids_entity_allows():
    # condition_windows numbers a repeated description "desc#1", and a # cannot go in a
    # filename; a label that reduces to nothing falls back to seg-NN rather than to task-
    assert condition_stem(STEM, "game 1", 1) == "sub-01_task-game1_desc-raw_nirs"
    assert condition_stem(STEM, "talk#2", 1) == "sub-01_task-talk2_desc-raw_nirs"
    assert condition_stem(STEM, "###", 4) == f"{STEM}_seg-04"


def test_labels_that_collide_once_reduced_do_not_share_a_file():
    # two views writing one path would leave the second silently winning
    stems = condition_stems(STEM, ["game-1", "game 1"])
    assert len(set(stems)) == 2
    assert stems[0] == "sub-01_task-game1_desc-raw_nirs"


def test_distinct_labels_are_left_alone():
    stems = condition_stems(STEM, ["baseline", "game1", "video"])
    assert stems == ["sub-01_task-baseline_desc-raw_nirs",
                     "sub-01_task-game1_desc-raw_nirs",
                     "sub-01_task-video_desc-raw_nirs"]


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
