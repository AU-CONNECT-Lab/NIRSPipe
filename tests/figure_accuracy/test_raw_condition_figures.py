"""The raw viewer's condition pages draw their own condition, and agree with the pipeline's files."""

import json
import re

import mne
import numpy as np
import pytest

from fnirs_pipe.qc.subject.record_io import read_record
from tests._fingerprint import EVENT_ONSETS
from tests.figure_accuracy._payload import _decode, one_figure, plotly_figures
from fnirs_pipe.qc.boilerplate.notes import section_note
from tests.figure_accuracy._read import (
    _blocks, _check_condition_grid, _failing_in, _grid, _trial_panel, xy,
)

TMIN, TMAX = -5.0, 25.0
BLOCKS = ["ca", "cb"]


def _figure(run, desc, block, suffix="nirs", chan=None):
    middle = f"chan-{chan}_cond-{block}" if chan else f"cond-{block}"
    return run.figures / f"sub-{run.subject}_task-{run.task}_{middle}_desc-{desc}_{suffix}.html"


def _page(run, block):
    page = run.out / "sub-01" / f"sub-01_task-main_cond-{block}_desc-raw_report.html"
    html = page.read_text(encoding="utf-8")
    return _decode(json.loads(re.search(r"var _STATIC_DATA = (\[.*?\]);\n", html, re.S).group(1))[0])


def _views(path):
    html = path.read_text(encoding="utf-8")
    return json.loads(re.search(r"window\.__COND_VIEWS__=(\{.*?\});</script>", html, re.S).group(1))


@pytest.fixture(scope="module")
def epochs_by_block(condition_run):
    """Each block's own trials on the pipeline's uncorrected haemoglobin, which the raw viewer draws."""
    raw = condition_run.read("uncorrected")
    events, event_id = mne.events_from_annotations(raw, event_id={"trial": 1}, verbose="error")
    onsets = events[:, 0] / raw.info["sfreq"]
    return {block: mne.Epochs(raw, events[(onsets >= t0) & (onsets < t1)], event_id, tmin=TMIN,
                              tmax=TMAX, baseline=(TMIN, 0), preload=True,
                              reject_by_annotation=False, verbose="error")
            for block, (t0, t1, _) in _blocks(condition_run).items()}


def test_there_is_one_page_per_block(raw_condition_run):
    pages = sorted(p.name for p in (raw_condition_run.out / "sub-01").glob("*_cond-*_desc-raw_report.html"))
    assert pages == [f"sub-01_task-main_cond-{b}_desc-raw_report.html" for b in _blocks(raw_condition_run)]


# ---- quality, sliced from the record ----

@pytest.mark.parametrize("block", BLOCKS)
def test_each_page_rejects_what_the_run_rejected_and_colours_its_own_coupling(raw_condition_run, block):
    cells = _grid(_figure(raw_condition_run, "rawchsummary", block, "qc"))
    _check_condition_grid(cells, raw_condition_run, block)


@pytest.mark.parametrize("block", BLOCKS)
def test_a_condition_page_header_counts_what_fails_on_its_own_stretch(raw_condition_run, block):
    page = raw_condition_run.out / "sub-01" / f"sub-01_task-main_cond-{block}_desc-raw_report.html"
    assert f"<th>{section_note('summary.condition_failing')}</th>" in page.read_text(encoding="utf-8")
    summary = _page(raw_condition_run, block)["summary"]
    assert (summary["n_bad"], summary["n_total"]) == (
        2 * len(_failing_in(raw_condition_run, block)), 2 * len(raw_condition_run.truth.pairs))


@pytest.mark.parametrize("block", BLOCKS)
def test_the_sci_panel_holds_the_condition_s_windows_and_only_those(raw_condition_run, block):
    t0, t1, _ = _blocks(raw_condition_run)[block]
    windowed = read_record(raw_condition_run.nirs / "sub-01_task-main_desc-sqmraw_qc.json")["windowed"]
    heat = next(t for t in one_figure(_figure(raw_condition_run, "rawscipsp", block))["data"]
                if t["type"] == "heatmap" and t["name"] == "SCI")
    x = np.asarray(heat["x"], float)
    assert x.min() >= t0 and x.max() <= t1
    times = np.asarray(windowed["sci_times"], float)
    centres = times.mean(axis=1) if times.ndim == 2 else times
    columns = [int(np.argmin(np.abs(centres - c))) for c in x]
    rows = [list(windowed["sci_channels"]).index(name) for name in heat["y"]]
    np.testing.assert_allclose(np.asarray(heat["z"], float),
                               np.asarray(windowed["sci_matrix"], float)[np.ix_(rows, columns)],
                               rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("block", BLOCKS)
def test_the_trial_panel_holds_the_condition_s_own_trials(raw_condition_run, block):
    t0, t1, _ = _blocks(raw_condition_run)[block]
    fig = one_figure(_figure(raw_condition_run, "rawtrialqc", block, "qc"))
    onsets = [float(re.search(r"_(\d+(?:\.\d+)?)s_", label).group(1))
              for label in fig["layout"]["xaxis"]["ticktext"]]
    assert onsets == [o for o in EVENT_ONSETS if t0 <= o < t1]


# ---- haemoglobin, rebuilt on the condition's cut ----

@pytest.mark.parametrize("block", BLOCKS)
def test_the_grand_mean_averages_the_condition_s_trials_over_the_kept_long_channels(
        raw_condition_run, epochs_by_block, block):
    _, y = xy(_trial_panel(one_figure(_figure(raw_condition_run, "rawepochmean", block))))
    epochs = epochs_by_block[block]
    assert len(epochs) == 3
    kept = [f"{p.name} hbo" for p in raw_condition_run.truth.long_pairs if not p.bad]
    np.testing.assert_allclose(y, epochs.average(picks=kept).data.mean(axis=0) * 1e6,
                               rtol=1e-4, atol=1e-3)


def test_the_two_pages_differ_by_the_planted_response_gain(raw_condition_run):
    # a window mean, not the peak: unfiltered, the decoupled trial's pulse tops cb's curve
    response = {}
    for block in BLOCKS:
        x, y = xy(_trial_panel(one_figure(_figure(raw_condition_run, "rawepochmean", block))))
        response[block] = y[(x >= 5) & (x <= 15)].mean()
    gains = {block: gain for block, (_, _, gain) in _blocks(raw_condition_run).items()}
    assert response["cb"] / response["ca"] == pytest.approx(gains["cb"] / gains["ca"], abs=0.1)


@pytest.mark.parametrize("block", BLOCKS)
def test_the_correlation_dots_are_measured_on_the_condition_s_stretch(raw_condition_run, condition_run, block):
    t0, t1, _ = _blocks(raw_condition_run)[block]
    fig = one_figure(_figure(raw_condition_run, "rawhbohbrcorr", block))
    rejected = {p.name for p in raw_condition_run.truth.pairs if p.bad}
    for name, desc in (("before motion correction", "uncorrected"), ("after tddr", "preproc")):
        dots = next(t for t in fig["data"] if t.get("name") == name)
        raw = condition_run.read(desc).copy().crop(t0, t1)
        for pair, r in zip(dots["customdata"], dots["y"]):
            hbo, hbr = raw.get_data(picks=[f"{pair} hbo", f"{pair} hbr"])
            assert r == pytest.approx(np.corrcoef(hbo, hbr)[0, 1], abs=1e-6), (block, name, pair)
        assert set(list(dots["customdata"])[-len(rejected):]) == rejected


@pytest.mark.parametrize("block", BLOCKS)
def test_the_table_prints_the_condition_s_own_correlation(raw_condition_run, condition_run, block):
    t0, t1, _ = _blocks(raw_condition_run)[block]
    raw = condition_run.read("uncorrected").copy().crop(t0, t1)
    rows = [r for _, rows in _page(raw_condition_run, block)["channels"]["blocks"] for r in rows]
    assert len(rows) == len(raw_condition_run.truth.pairs)
    for row in rows:
        pair = raw_condition_run.truth.pair(row["name"])
        if pair.bad:
            assert row["corr"] in (None, "", "—"), row["name"]
            continue
        hbo, hbr = raw.get_data(picks=[f"{pair.name} hbo", f"{pair.name} hbr"])
        assert float(row["corr"]) == pytest.approx(np.corrcoef(hbo, hbr)[0, 1], abs=6e-4), row["name"]


@pytest.mark.parametrize("block", BLOCKS)
def test_each_trial_image_is_its_channel_s_trials_in_the_condition(raw_condition_run, epochs_by_block, block):
    epochs = epochs_by_block[block]
    for pair in raw_condition_run.truth.long_pairs:
        figs = plotly_figures(_figure(raw_condition_run, "rawtrialimg", block, chan=pair.name.replace("_", "")))
        heats = [t for fig in figs for t in fig["data"]
                 if t["type"] == "heatmap" and np.shape(t["z"])[0] == len(epochs)]
        assert len(heats) == 1, pair.name
        expected = epochs.get_data(picks=[f"{pair.name} hbo"])[:, 0, :] * 1e6
        np.testing.assert_allclose(np.asarray(heats[0]["z"], float), expected, rtol=1e-4, atol=1e-3)


# ---- run-wide figures, opened on the condition's window ----

@pytest.mark.parametrize("block", BLOCKS)
def test_the_run_figures_are_addressed_at_the_condition(raw_condition_run, block):
    paths = _page(raw_condition_run, block)["figure_paths"]
    for key in ("carpet", "ch_detail_template", "motion_detail_template"):
        src = paths[key]["src"] if isinstance(paths[key], dict) else paths[key]
        assert src.endswith(f"#{block}"), (key, src)


@pytest.mark.parametrize("block", BLOCKS)
def test_the_channel_detail_view_fits_the_condition_s_stretch(raw_condition_run, block):
    t0, t1, _ = _blocks(raw_condition_run)[block]
    for pair in raw_condition_run.truth.pairs:
        path = raw_condition_run.figure("rawdetail", chan=pair.name.replace("_", ""))
        view = _views(path)[block]
        assert view["x"] == pytest.approx([t0, t1])
        series = plotly_figures(path)[0]
        for key, (floor, top) in view["y"].items():
            axis = "y" + key[len("yaxis"):]
            inside = np.concatenate([y[(x >= t0) & (x <= t1)] for x, y in
                                     (xy(t) for t in series["data"]
                                      if (t.get("yaxis") or "y") == axis and t.get("fill") != "toself")])
            lo, hi = np.nanmin(inside), np.nanmax(inside)
            assert floor <= lo and hi <= top, (pair.name, key)
            assert top - floor <= 1.2 * (hi - lo), (pair.name, key)


@pytest.mark.parametrize("block", BLOCKS)
def test_the_motion_and_carpet_views_open_on_the_condition(raw_condition_run, block):
    t0, t1, _ = _blocks(raw_condition_run)[block]
    pair = raw_condition_run.truth.long_pairs[0]
    for path in (raw_condition_run.figure("rawmotion", chan=f"{pair.name.replace('_', '')}760"),
                 raw_condition_run.figure("rawcarpet")):
        assert _views(path)[block]["x"] == pytest.approx([t0, t1]), path.name
