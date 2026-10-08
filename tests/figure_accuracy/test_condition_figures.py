"""Each condition page draws its own condition: its window, its trials, its verdict."""

import re

import mne
import numpy as np
import pytest

from tests._fingerprint import EVENT_ONSETS, RESPONSE_AMP
from tests.figure_accuracy._payload import one_figure, plotly_figures
from tests.figure_accuracy._read import _blocks, _grid, _trial_panel, traces, xy

TMIN, TMAX = -5.0, 25.0


def _figure(run, desc, block, suffix="nirs", chan=None):
    entities = f"cond-{block}"
    if chan:
        return run.figures / f"sub-{run.subject}_task-{run.task}_chan-{chan}_{entities}_desc-{desc}_{suffix}.html"
    return run.figure(desc, suffix=suffix, entities=entities)


def test_there_is_one_page_per_block(condition_run):
    pages = sorted(p.name for p in condition_run.report.parent.glob("*_cond-*_report.html"))
    assert pages == [f"sub-01_task-main_cond-{b}_report.html" for b in _blocks(condition_run)]


@pytest.mark.parametrize("block", ["ca", "cb"])
def test_each_page_rejects_what_its_own_windows_rejected(condition_run, block):
    truth = condition_run.truth
    pair, onset, span = truth.block_bad
    t0, t1, _ = _blocks(condition_run)[block]
    inside = t0 <= onset < t1
    for (channel, metric), (value, _) in _grid(_figure(condition_run, "chsummary", block, "qc")).items():
        if metric != "Status":
            continue
        name = channel.rsplit(" ", 1)[0]
        bad = truth.pair(name).bad or (name == pair and inside)
        assert (value != "OK") if bad else (value == "OK"), (block, channel)


def test_the_run_page_grid_carries_each_condition_s_own_verdict(condition_run):
    pair = condition_run.truth.block_bad[0]
    cells = _grid(condition_run.figure("condsummary", suffix="qc"))
    status = {where[0]: value for where, (value, _) in cells.items()
              if where[-1] == "Status" and where[1].startswith(pair)}
    assert status["ca"] == "OK" and status["cb"] != "OK"


@pytest.mark.parametrize("block", ["ca", "cb"])
def test_a_condition_page_has_no_whole_run_sci_column(condition_run, block):
    html = (condition_run.report.parent / f"sub-01_task-main_cond-{block}_report.html").read_text(
        encoding="utf-8")
    assert "<th>SCI (whole run)</th>" not in html and "<th>SCI (10 s)</th>" in html


@pytest.mark.parametrize("block", ["ca", "cb"])
def test_the_sci_panel_holds_the_condition_s_windows_and_only_those(condition_run, block):
    t0, t1, _ = _blocks(condition_run)[block]
    windowed = condition_run.record()["windowed"]
    heat = next(t for t in one_figure(_figure(condition_run, "scipsp", block))["data"]
                if t["type"] == "heatmap" and t["name"] == "SCI")
    x = np.asarray(heat["x"], float)
    assert x.min() >= t0 and x.max() <= t1
    centres = np.asarray(windowed["sci_times"], float).mean(axis=1) \
        if np.ndim(windowed["sci_times"]) == 2 else np.asarray(windowed["sci_times"], float)
    columns = [int(np.argmin(np.abs(centres - c))) for c in x]
    np.testing.assert_allclose(np.asarray(heat["z"], float),
                               np.asarray(windowed["sci_matrix"], float)[:, columns],
                               rtol=1e-5, equal_nan=True)


@pytest.fixture(scope="module")
def evoked_by_block(condition_run):
    raw = condition_run.read("filtered")
    events, event_id = mne.events_from_annotations(raw, event_id={"trial": 1}, verbose="error")
    out = {}
    for block, (t0, t1, _) in _blocks(condition_run).items():
        inside = events[(events[:, 0] / raw.info["sfreq"] >= t0)
                        & (events[:, 0] / raw.info["sfreq"] < t1)]
        out[block] = mne.Epochs(raw, inside, event_id, tmin=TMIN, tmax=TMAX, baseline=(TMIN, 0),
                                preload=True, reject_by_annotation=False, verbose="error")
    return out


@pytest.mark.parametrize("block", ["ca", "cb"])
def test_the_grand_mean_averages_the_condition_s_own_trials(condition_run, evoked_by_block, block):
    _, y = xy(_trial_panel(one_figure(_figure(condition_run, "epochmean", block))))
    epochs = evoked_by_block[block]
    assert len(epochs) == 3
    kept = [f"{p.name} hbo" for p in condition_run.truth.long_pairs if not p.bad]
    expected = epochs.average(picks=kept).data.mean(axis=0) * 1e6
    np.testing.assert_allclose(y, expected, rtol=1e-3, atol=2e-3)


def test_the_two_pages_differ_by_the_planted_response_gain(condition_run):
    peaks = {}
    for block in _blocks(condition_run):
        _, y = xy(_trial_panel(one_figure(_figure(condition_run, "epochmean", block))))
        peaks[block] = y.max()
    gains = {block: gain for block, (_, _, gain) in _blocks(condition_run).items()}
    assert peaks["cb"] / peaks["ca"] == pytest.approx(gains["cb"] / gains["ca"], abs=0.15)
    assert peaks["ca"] > 0.25 * RESPONSE_AMP * 1e6


@pytest.mark.parametrize("block", ["ca", "cb"])
def test_the_trial_panel_holds_the_condition_s_own_trials(condition_run, block):
    t0, t1, _ = _blocks(condition_run)[block]
    fig = one_figure(_figure(condition_run, "trialqc", block, "qc"))
    onsets = [float(re.search(r"_(\d+(?:\.\d+)?)s_", label).group(1))
              for label in fig["layout"]["xaxis"]["ticktext"]]
    assert onsets == [o for o in EVENT_ONSETS if t0 <= o < t1]


@pytest.mark.parametrize("block", ["ca", "cb"])
def test_a_channel_page_is_the_condition_s_stretch_of_the_channel(condition_run, block):
    t0, t1, _ = _blocks(condition_run)[block]
    pair = next(p for p in condition_run.truth.long_pairs if not p.bad)
    series = plotly_figures(_figure(condition_run, "detail", block, chan=pair.name.replace("_", "")))[0]
    x, y = xy(traces(series, "HbO")[0])
    assert x.min() >= t0 - 0.1 and x.max() <= t1 + 0.1
    times, full = condition_run.channel("uncorrected", f"{pair.name} hbo")
    np.testing.assert_allclose(y, np.interp(x, times, full) * 1e6, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("block", ["ca", "cb"])
def test_the_correlation_dots_are_measured_on_the_condition_s_stretch(condition_run, block):
    t0, t1, _ = _blocks(condition_run)[block]
    dots = next(t for t in one_figure(_figure(condition_run, "hbohbrcorr", block))["data"]
                if t.get("name") == "before denoising")
    raw = condition_run.read("preproc").copy().crop(t0, t1)
    for pair, r in zip(dots["customdata"], dots["y"]):
        hbo, hbr = raw.get_data(picks=[f"{pair} hbo", f"{pair} hbr"])
        assert r == pytest.approx(np.corrcoef(hbo, hbr)[0, 1], abs=0.02), (block, pair)
