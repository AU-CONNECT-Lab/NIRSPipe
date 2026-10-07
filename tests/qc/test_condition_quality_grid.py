"""The run page's per-condition channel grid: regrouped by metric, cell for cell the condition pages'."""

from fnirs_pipe.qc.common.channel_table import heatmap_args
from fnirs_pipe.qc.metrics.coupling import CV_WINDOW_S, PSP_WINDOW_S, SCI_WINDOW_S
from fnirs_pipe.qc.common.figure_io import figure_namer
from fnirs_pipe.qc.figures import channel_quality_heatmap, condition_quality_heatmap
from fnirs_pipe.qc.subject.report import _condition_channel_rows, _section_condition_summary


def _args(sci_b: float, bad: bool) -> dict:
    return {
        "ch_names": ["A", "B"], "is_bad": [False, bad],
        "good_frac_per_ch": {"A": 0.9, "B": 0.9},
        "sci_per_ch": {"A": 0.95, "B": sci_b},
        "cv_per_ch": {"A": 0.01, "B": 0.01},
        "snr_per_ch": {"A": 100.0, "B": 100.0},
        "psp_per_ch": {"A": 0.3, "B": 0.3},
        "split_at": None,
    }


def _cells(fig) -> dict:
    """{hover prefix up to the metric: colour}, e.g. {"rest · B · SCI (10 s)": "#F8786E"}."""
    trace = fig.data[0]
    return {text.rsplit(":", 1)[0]: colour
            for text, colour in zip(trace.text, trace.marker.color)}


def test_a_channel_failing_in_one_condition_is_red_only_in_that_row():
    fig = condition_quality_heatmap([("rest", _args(0.95, False)), ("task", _args(0.3, True))])
    cells = _cells(fig)
    sci = f"SCI ({SCI_WINDOW_S:g} s)"
    assert cells[f"rest · B · {sci}"] != cells[f"task · B · {sci}"]
    assert cells["task · B · Status"] == cells[f"task · B · {sci}"]
    assert cells["rest · B · Status"] == cells["rest · A · Status"]


def test_rows_are_grouped_by_metric_then_condition():
    fig = condition_quality_heatmap([("rest", _args(0.95, False)), ("task", _args(0.3, True))])
    assert list(fig.layout.yaxis.ticktext) == ["rest", "task"] * 6
    headings = [a.text for a in fig.layout.annotations]
    assert headings == [f"<b>{m}</b>" for m in
                        ("Status", "Coupled", f"SCI ({SCI_WINDOW_S:g} s)",
                         f"CV ({CV_WINDOW_S:g} s)", f"PSP ({PSP_WINDOW_S:g} s)",
                         f"SNR ({CV_WINDOW_S:g} s)")]


def test_every_cell_matches_the_condition_pages_own_grid():
    conditions = [("rest", _args(0.95, False)), ("task", _args(0.3, True))]
    regrouped = _cells(condition_quality_heatmap(conditions, sci_thresh=0.8))
    for label, args in conditions:
        own = _cells(channel_quality_heatmap(sci_thresh=0.8, **args))
        assert {f"{label} · {k}": v for k, v in own.items()} == {
            k: v for k, v in regrouped.items() if k.startswith(f"{label} · ")}


def test_a_single_condition_draws_nothing():
    assert condition_quality_heatmap([("rest", _args(0.95, False))]) is None


# ---- the run-page section ----

def _record(labels) -> dict:
    return {
        "raw_long": {},
        "per_channel": {"raw_long": {"sci_win_per_channel": {"A": 0.9, "B": 0.9},
                                     "psp_per_channel": {"A": 0.3, "B": 0.3}}},
        "by_condition": {
            lab: {"per_channel": {"sci_win_per_channel": {"A": 0.9, "B": 0.2 if i else 0.9},
                                  "psp_per_channel": {"A": 0.3, "B": 0.3}},
                  "bad_channels": ["B"] if i else []}
            for i, lab in enumerate(labels)},
    }


class _Config:
    sci_threshold = 0.8


def test_the_section_reads_each_conditions_slice_and_verdict():
    record = _record(["rest", "task"])
    rows = _condition_channel_rows(record, record["by_condition"]["task"], {"A": 0.9, "B": 0.9})
    args = heatmap_args(rows)
    assert args["sci_per_ch"]["B"] == 0.2
    assert dict(zip(args["ch_names"], args["is_bad"])) == {"A": False, "B": True}


def test_the_section_writes_one_figure_for_two_conditions(tmp_path):
    errors: list = []
    out = _section_condition_summary(_record(["rest", "task"]), {"A": 0.9, "B": 0.9},
                                     _Config(), "01", errors, tmp_path,
                                     figure_namer("sub-01_task-rest"))
    assert errors == []
    assert out["condition_summary_path"] and out["condition_summary_h"] > 0
    assert (tmp_path / figure_namer("sub-01_task-rest")("condsummary", suffix="qc")).exists()


def test_the_section_is_empty_for_one_condition(tmp_path):
    out = _section_condition_summary(_record(["rest"]), {"A": 0.9, "B": 0.9}, _Config(),
                                     "01", [], tmp_path, figure_namer("sub-01_task-rest"))
    assert out == {"condition_summary_path": None, "condition_summary_h": 0}
