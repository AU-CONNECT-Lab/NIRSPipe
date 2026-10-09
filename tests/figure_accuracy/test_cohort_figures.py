"""The cohort page (fnirs-qc cohort): each figure against the cohort table and each run's record."""

import numpy as np
import pandas as pd
import pytest

from tests._dyad_fingerprint import (BLOCKS, COHORT_ONLY, EXTRA_SPIKES, MEMBERS, NO_SHORT,
                                     OWN_SPIKE, SHARED_SPIKE, TASK)
from tests.figure_accuracy._dyad import html_tables
from tests.figure_accuracy._payload import one_figure, plotly_figures

CONDITIONS = [name for name, _, _ in BLOCKS]
SUBJECTS = [m[1] for m in MEMBERS + COHORT_ONLY]


@pytest.fixture(scope="module")
def cohort(groups):
    return pd.read_csv(groups.deriv / "desc-subjects_qc.tsv", sep="\t").set_index("bids_name")


def _figure(groups, desc):
    return groups.deriv / "figures" / f"desc-{desc}_nirs.html"


def _column(row_label: str, channel_set: str) -> str:
    stage, metric = row_label.rsplit(" ", 1)
    return f"{stage}_{metric}" if channel_set == "all" else f"{stage}_{channel_set}_{metric}"


def _robust_z(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    return (values - median) / (1.4826 * mad if mad > 0 else np.std(finite))


def test_every_strip_point_is_its_run_s_robust_distance_on_its_own_row(groups, cohort):
    fig = one_figure(_figure(groups, "strip"))
    ticks = dict(zip(fig["layout"]["yaxis"]["tickvals"], fig["layout"]["yaxis"]["ticktext"]))
    for trace in fig["data"]:
        if trace.get("customdata") is None:
            continue
        column = _column(ticks[int(round(float(np.mean(trace["y"]))))], trace["name"])
        runs = list(trace["customdata"])
        want = dict(zip(cohort.index, _robust_z(cohort[column].to_numpy())))
        if all(np.isnan(v) for v in want.values()):
            continue                 # no spread at the table's precision: nothing to place
        drawn = dict(zip(runs, np.asarray(trace["x"], float)))
        assert drawn == pytest.approx({r: want[r] for r in runs}, rel=1e-9, abs=1e-9,
                                      nan_ok=True), column


def test_every_box_point_is_its_run_s_value(groups, cohort):
    boxes = sorted(groups.deriv.joinpath("figures").glob("desc-box*_nirs.html"))
    assert boxes
    seen = 0
    for path in boxes:
        for trace in one_figure(path)["data"]:
            if trace.get("customdata") is None:
                continue
            for (run, column), y in zip(trace["customdata"], trace["y"]):
                want = cohort.loc[run, column]
                assert (y is None and np.isnan(want)) or y == pytest.approx(want, rel=1e-9), \
                    (path.name, run, column)
                seen += 1
    assert seen > 100


@pytest.mark.parametrize("column, rising", [("raw_long_gvtd_filt_mean", True),
                                            ("raw_long_sci_win_mean", False)])
def test_the_boxes_rank_the_runs_by_their_planted_noise(groups, column, rising):
    for path in sorted(groups.deriv.joinpath("figures").glob("desc-box*_nirs.html")):
        for trace in one_figure(path)["data"]:
            points = {run: y for (run, col), y in zip(trace.get("customdata") or [], trace["y"])
                      if col == column}
            if points:
                by_noise = [points[f"sub-{m[1]}_task-{TASK}"]
                            for m in sorted(MEMBERS, key=lambda m: m[5])]
                assert by_noise == sorted(by_noise, reverse=not rising), by_noise
                return
    raise AssertionError(f"{column} is on no box")


def _condition_values(groups, metric):
    """{subject: {condition: value}} off each run's own record."""
    out = {}
    for subject in SUBJECTS:
        rec = groups.member_record(f"sub-{subject}")
        out[f"sub-{subject}"] = {c: rec["by_condition"][c]["scalars"].get(metric, np.nan)
                                 for c in rec["by_condition"]}
    return out


_PANELS = {"SCI": "sci_win_mean", "PSP": "psp_mean", "CV": "cv_mean", "SNR": "snr_mean",
           "GVTD 0.02-0.5 Hz": "gvtd_filt_mean", "GVTD above threshold": "gvtd_pct_above_thresh"}


def _metric_of(title):
    return next((v for k, v in _PANELS.items() if title.startswith(k)), None)


def test_every_condition_matrix_cell_is_its_run_s_own_condition_value(groups):
    fig = one_figure(_figure(groups, "conditionmatrix"))
    titles = [a["text"] for a in fig["layout"]["annotations"]]
    heat = [t for t in fig["data"] if t["type"] == "heatmap"]
    assert len(titles) == len(heat)
    for title, panel in zip(titles, heat):
        if _metric_of(title) is None:
            continue
        values = _condition_values(groups, _metric_of(title))
        for i, label in enumerate(panel["y"]):
            for j, condition in enumerate(panel["x"]):
                assert panel["text"][i][j] == f"{values[label][condition]:.4g}", \
                    (title, label, condition)


def test_the_condition_matrix_shows_each_run_s_movement_in_its_own_block(groups):
    fig = one_figure(_figure(groups, "conditionmatrix"))
    titles = [a["text"] for a in fig["layout"]["annotations"]]
    panel = [t for t in fig["data"] if t["type"] == "heatmap"][titles.index("GVTD above threshold")]
    for i, label in enumerate(panel["y"]):
        subject = label.removeprefix("sub-")
        for j, condition in enumerate(panel["x"]):
            if condition not in CONDITIONS:
                continue
            onset, span = next((on, dur) for n, on, dur in BLOCKS if n == condition)
            spikes = (OWN_SPIKE[subject], SHARED_SPIKE, *EXTRA_SPIKES.get(subject, ()))
            moved = any(onset <= t < onset + span for t in spikes)
            assert (float(panel["text"][i][j]) > 0) == moved, (label, condition)


def test_the_condition_lines_put_the_cohort_median_through_each_condition(groups):
    fig = plotly_figures(_figure(groups, "conditions"))[0]
    titles = [a["text"] for a in fig["layout"]["annotations"]]
    medians = [t for t in fig["data"] if t.get("name") == "cohort median"]
    for title, trace in zip(titles, medians):
        if _metric_of(title) is None:
            continue
        values = _condition_values(groups, _metric_of(title))
        for condition, y in zip(trace["x"], np.asarray(trace["y"], float)):
            want = np.nanmedian([v.get(condition, np.nan) for v in values.values()])
            assert y == pytest.approx(want, rel=1e-9), (title, condition)


@pytest.mark.xfail(strict=True, reason="D7: a condition's windowed values come from the run's "
                   "--window-length grid, and the panel titles print the 10 s constants")
@pytest.mark.parametrize("desc", ["conditions", "conditionmatrix"])
def test_the_condition_panels_name_the_window_the_runs_were_measured_in(groups, desc):
    window = groups.member_record("sub-01")["windowed"]["qc_window_s"]
    titles = [a["text"] for a in one_figure(_figure(groups, desc))["layout"]["annotations"]]
    for title in titles:
        if title.split(" (")[0] in ("SCI", "PSP", "CV", "SNR"):
            assert f"({window:g} s)" in title, title


def test_the_metrics_table_is_the_cohort_table(groups, cohort):
    rows = next(rows for _, rows in html_tables(groups.deriv / "desc-subjects_report.html")
                if rows and rows[0][0].startswith("bids_name"))
    head, body = [h.rstrip("⇅") for h in rows[0]], rows[1:]
    assert [r[0] for r in body] == list(cohort.index)
    for r in body:
        for column, text in zip(head[1:], r[1:]):
            want = cohort.loc[r[0], column]
            if isinstance(want, (float, np.floating)) and np.isfinite(want):
                assert float(text) == pytest.approx(want, rel=1e-3, abs=1e-9), (r[0], column)


# ---- the windowed grid: every run pale, the outliers in colour ----

_GRID_ROWS = {"SCI (windowed)": "sci", "CV (windowed)": "cv"}


def _own_series(groups, sid, key, channel_set):
    """One run's windowed series for one channel set, off its own record and the channel
    names the record stores beside each matrix, smoothed and sampled as the grid does."""
    from fnirs_pipe.qc.figures.subject.group_figures import SMOOTH_S, _smooth

    rec = groups.member_record(sid)
    windowed = rec["windowed"]
    times = np.asarray(windowed[f"{key}_window_times_s"], float)
    names = windowed[f"{key}_channels"]
    split = {s: set(rec["per_channel"].get(f"raw_{s}", {}).get("sci_per_channel", {}))
             for s in ("long", "short")}
    # a montage with no short channel records no split: every channel is long
    members = {"all": set(names), "long": split["long"] or set(names), "short": split["short"]}
    rows = [i for i, name in enumerate(names) if name in members[channel_set]]
    if not rows:
        return None
    with np.errstate(invalid="ignore"):
        values = np.nanmean(np.asarray(windowed[f"{key}_matrix"], float)[rows], axis=0)
    step = max(1, int(round(SMOOTH_S / windowed["qc_window_s"])))
    return times[::step], _smooth(values, step)[::step]


def _grid_lines(groups):
    """(run, metric key, channel set, x, y) for every highlighted line on the grid."""
    fig = one_figure(_figure(groups, "windows"))
    sets = [a["text"] for a in fig["layout"]["annotations"] if a["text"] in ("all", "long", "short")]
    metrics = [fig["layout"][k]["title"]["text"] for k in sorted(
        (k for k in fig["layout"] if k.startswith("yaxis") and (fig["layout"][k].get("title") or {}).get("text")),
        key=lambda k: int(k[5:] or 1))]
    out = []
    for trace in fig["data"]:
        if not trace.get("name") or trace["name"] == "cohort median":
            continue
        n = int(trace["xaxis"][1:] or 1) - 1
        metric, channel_set = metrics[n // len(sets)], sets[n % len(sets)]
        if metric in _GRID_ROWS:
            out.append((trace["name"], _GRID_ROWS[metric], channel_set,
                        np.asarray(trace["x"], float), np.asarray(trace["y"], float)))
    return out


def test_the_grid_highlights_the_planted_outliers(groups):
    assert {line[0] for line in _grid_lines(groups)} == {f"sub-{s}" for s in ("00", "03b")}


@pytest.mark.xfail(strict=True, reason="D8: a run missing a channel set shifts the stack, so "
                   "a highlighted line carries the next run's values under its own name")
def test_no_run_is_drawn_in_a_channel_set_it_does_not_have(groups):
    for run, _, channel_set, _, _ in _grid_lines(groups):
        if channel_set == "short":
            assert run.removeprefix("sub-") not in NO_SHORT, run


@pytest.mark.xfail(strict=True, reason="D8: a highlighted line is the stack row at the run's "
                   "index, which is another run's once an earlier run lacks the set")
def test_every_highlighted_line_is_its_own_run_s_series(groups):
    lines = _grid_lines(groups)
    assert lines
    for run, key, channel_set, x, y in lines:
        own = _own_series(groups, run, key, channel_set)
        assert own is not None, (run, key, channel_set)
        times, values = own
        keep = np.isfinite(y)
        np.testing.assert_allclose(y[keep], np.interp(x[keep], times, values), rtol=1e-9,
                                   err_msg=f"{run} {key} {channel_set}")
