"""The raw viewer draws the same recording the pipeline does, so its figures must agree with the pipeline's files."""

import csv
import json
import re

import mne
import numpy as np
import pytest
from mne.time_frequency import psd_array_welch

from nirspipe.qc.figures.common._utils import PSD_NFFT
from nirspipe.qc.metrics.coupling import SCI_WINDOW_S
from nirspipe.qc.subject.record_io import read_record
from tests._fingerprint import CARDIAC_FREQ, CLI_ARGS, EVENT_DURATION, EVENT_ONSETS, HBR_FREQ
from tests.figure_accuracy._payload import _decode, one_figure, plotly_figures
from tests.figure_accuracy._read import share_at, traces, which_pair, xy


def _arg(flag):
    return float(CLI_ARGS[CLI_ARGS.index(flag) + 1])


@pytest.fixture(scope="module")
def page(raw_viewer_run):
    html = (raw_viewer_run.out / "sub-01" / "sub-01_task-tapping_desc-raw_report.html").read_text(
        encoding="utf-8")
    data = json.loads(re.search(r"var _STATIC_DATA = (\[.*?\]);\n", html, re.S).group(1))[0]
    return html, _decode(data)


def _fname(pair):
    return pair.replace("_", "")


# ---- inline figures ----

def test_each_raw_signal_row_is_the_channel_it_is_labelled(raw_viewer_run, page):
    truth = raw_viewer_run.truth
    for trace in page[1]["ts"]["figure"]["data"]:
        pair = truth.pair(trace["name"].rsplit(" ", 1)[0])
        if not pair.short and not pair.bad:
            assert which_pair(*xy(trace), truth) == pair.name, trace["name"]


def test_the_layout_marks_the_rejected_pair_and_no_other(raw_viewer_run, page):
    channels = next(t for t in page[1]["layout"]["layout_2d_figure"]["data"]
                    if t.get("customdata") is not None and np.size(t["customdata"]) == 16)
    good = channels["marker"]["color"][0]
    for name, colour in zip(channels["customdata"], channels["marker"]["color"]):
        bad = raw_viewer_run.truth.pair(name.rsplit(" ", 1)[0]).bad
        assert (colour != good) if bad else (colour == good), name


def test_the_channel_table_heads_both_sci_columns(page):
    html = page[0]
    assert f"<th>SCI ({SCI_WINDOW_S:g} s)</th>" in html and "<th>SCI (whole run)</th>" in html


# ---- the same recording as the pipeline's files ----

def test_the_channel_detail_is_the_pipeline_s_uncorrected_haemoglobin(raw_viewer_run, denoise_run):
    for pair in raw_viewer_run.truth.pairs:
        series = plotly_figures(raw_viewer_run.figure("rawdetail", chan=_fname(pair.name)))[0]
        for chroma, label in (("hbo", "HbO"), ("hbr", "HbR")):
            x, y = xy(traces(series, label)[0])
            times, expected = denoise_run.channel("uncorrected", f"{pair.name} {chroma}")
            np.testing.assert_allclose(y, np.interp(x, times, expected) * 1e6, rtol=1e-5, atol=1e-6)
        if not pair.bad and not pair.short:
            hbo, hbr = traces(series, "HbO")[0], traces(series, "HbR")[0]
            assert share_at(*xy(hbr), HBR_FREQ) > 10 * share_at(*xy(hbo), HBR_FREQ), pair.name


def test_the_picker_names_each_pair_as_the_pipeline_report_does(raw_viewer_run, denoise_run, page):
    html = denoise_run.report.read_text(encoding="utf-8")
    labels = page[1]["channel_labels"]
    assert list(page[1]["channel_pairs"]) == [p.name for p in raw_viewer_run.truth.pairs]
    for pair in raw_viewer_run.truth.pairs:
        path = denoise_run.figure("detail", chan=_fname(pair.name))
        option = re.search(rf'<option value="figures/{re.escape(path.name)}">([^<]*)</option>', html)
        assert labels[pair.name] == option.group(1)
        assert ("rejected" in labels[pair.name]) == pair.bad, labels[pair.name]


def test_the_motion_rows_are_the_pipeline_s_od_either_side_of_the_correction(raw_viewer_run, denoise_run):
    for pair in raw_viewer_run.truth.pairs:
        fig = plotly_figures(raw_viewer_run.figure("rawmotion", chan=f"{_fname(pair.name)}760"))[0]
        for label, desc in (("Before", "sci"), ("After", "motcorrected")):
            _, y = xy(traces(fig, label)[0])
            _, expected = denoise_run.channel(desc, f"{pair.name} 760")
            np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-7)


def test_the_correlation_panel_is_the_pipeline_s_two_stages(raw_viewer_run, denoise_run):
    fig = one_figure(raw_viewer_run.figure("rawhbohbrcorr"))
    rejected = {p.name for p in raw_viewer_run.truth.pairs if p.bad}
    for name, desc in (("before motion correction", "uncorrected"), ("after tddr", "preproc")):
        dots = next(t for t in fig["data"] if t.get("name") == name)
        raw = denoise_run.read(desc)
        for pair, r in zip(dots["customdata"], dots["y"]):
            hbo, hbr = raw.get_data(picks=[f"{pair} hbo", f"{pair} hbr"])
            assert r == pytest.approx(np.corrcoef(hbo, hbr)[0, 1], abs=1e-6), (name, pair)
        assert set(list(dots["customdata"])[-len(rejected):]) == rejected


def _corr(raw, pair):
    return np.corrcoef(*raw.get_data(picks=[f"{pair} hbo", f"{pair} hbr"]))[0, 1]


def test_the_correlation_record_leaves_the_rejected_pair_out(raw_viewer_run, denoise_run):
    record = read_record(raw_viewer_run.nirs / "sub-01_task-tapping_desc-sqmraw_qc.json")
    kept = [p.name for p in raw_viewer_run.truth.long_pairs if not p.bad]
    for section, desc in (("rawhaemo_long", "uncorrected"), ("rawhaemo_post_long", "preproc")):
        expected = np.mean([_corr(denoise_run.read(desc), p) for p in kept])
        assert record[section]["hbo_hbr_corr_mean"] == pytest.approx(expected, abs=1e-6), section


def test_the_table_prints_no_correlation_for_the_rejected_pair(raw_viewer_run, denoise_run):
    with open(raw_viewer_run.table("_desc-rawchannel_qc.tsv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    raw = denoise_run.read("uncorrected")
    for row in rows:
        pair = raw_viewer_run.truth.pair(row["name"].rsplit(" ", 1)[0])
        if pair.bad:
            assert row["corr"] == "n/a", row["name"]
        else:
            assert float(row["corr"]) == pytest.approx(_corr(raw, pair.name), abs=1e-6), row["name"]


# ---- quality panels ----

def test_the_sci_panel_draws_the_record_and_the_run_s_lines(raw_viewer_run):
    record = read_record(raw_viewer_run.nirs / "sub-01_task-tapping_desc-sqmraw_qc.json")
    fig = one_figure(raw_viewer_run.figure("rawscipsp"))
    heat = next(t for t in fig["data"] if t["type"] == "heatmap" and t["name"] == "SCI")
    assert list(heat["y"]) == list(record["windowed"]["sci_channels"])
    np.testing.assert_allclose(np.asarray(heat["z"], float),
                               np.asarray(record["windowed"]["sci_matrix"], float),
                               rtol=1e-5, equal_nan=True)
    lines = {s["xref"]: s["x0"] for s in fig["layout"]["shapes"] if s["type"] == "line"}
    assert lines["x2"] == pytest.approx(_arg("--sci-threshold"))
    assert lines["x4"] == pytest.approx(_arg("--psp-threshold"))


def test_the_grid_prints_the_raw_channel_table(raw_viewer_run):
    with open(raw_viewer_run.table("_desc-rawchannel_qc.tsv"), encoding="utf-8") as f:
        table = {row["name"]: row for row in csv.DictReader(f, delimiter="\t")}
    dots = one_figure(raw_viewer_run.figure("rawchsummary", suffix="qc"))["data"][0]
    for text in dots["text"]:
        channel, rest = text.split(" · ", 1)
        metric, value = rest.split(": ", 1)
        if metric.startswith("SCI"):
            assert float(value) == pytest.approx(float(table[channel]["sci_win"]), abs=1e-3), channel
        if metric == "Status":
            assert (value == "OK") == (table[channel]["is_bad"] == "False"), channel


# ---- spectrum and carpet ----

def test_the_mean_psd_is_the_welch_mean_of_each_separation_group(raw_viewer_run, denoise_run):
    fig = one_figure(raw_viewer_run.figure("rawpsd"))
    od = denoise_run.read("sci")
    for label, short in (("Long channels", False), ("Short channels", True)):
        names = [c for c in od.ch_names if raw_viewer_run.truth.pair(c.rsplit(" ", 1)[0]).short == short]
        x, y = xy(traces(fig, f"{label} (n={len(names)})")[0])
        data = od.get_data(picks=names)
        psd, freqs = psd_array_welch(data, od.info["sfreq"], n_fft=min(PSD_NFFT, data.shape[1]),
                                     verbose=False)
        shown = freqs <= x.max() + 1e-9
        np.testing.assert_allclose(x, freqs[shown], atol=1e-9)
        np.testing.assert_allclose(y, psd[:, shown].mean(axis=0), rtol=1e-6)
        above = x > 0.5
        assert x[above][np.argmax(y[above])] == pytest.approx(CARDIAC_FREQ, abs=od.info["sfreq"] / PSD_NFFT)


def test_the_mean_psd_shades_the_run_s_cardiac_band(raw_viewer_run):
    fig = one_figure(raw_viewer_run.figure("rawpsd"))
    spans = [(s["x0"], s["x1"]) for s in fig["layout"]["shapes"] if s["type"] == "rect"]
    named = dict(zip([a["text"] for a in fig["layout"]["annotations"]], spans))
    # prep-raw takes no respiration band, so it draws none
    assert named == {"Mayer": (0.07, 0.13),
                     "Cardiac": (_arg("--cardiac-l-freq"), _arg("--cardiac-h-freq"))}


def test_the_carpet_is_the_pipeline_s_carpet(raw_viewer_run, denoise_run):
    raw_fig = one_figure(raw_viewer_run.figure("rawcarpet"))
    pipe_fig = one_figure(denoise_run.figure("carpet"))
    assert [t.get("name") for t in raw_fig["data"]] == [t.get("name") for t in pipe_fig["data"]]
    for ours, theirs in zip(raw_fig["data"], pipe_fig["data"]):
        if ours["type"] == "heatmap":
            assert list(ours["y"]) == list(theirs["y"])
            pairs = ((ours["x"], theirs["x"]), (ours["z"], theirs["z"]))
        else:
            pairs = zip(xy(ours), xy(theirs))
        for a, b in pairs:
            np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-9,
                                       equal_nan=True)
    lines = [[(s["yref"], s["y0"]) for s in fig["layout"]["shapes"] if s["type"] == "line"]
             for fig in (raw_fig, pipe_fig)]
    assert lines[0] == lines[1]


# ---- events and trials ----

def test_the_timeline_bars_are_the_events(raw_viewer_run):
    bar = one_figure(raw_viewer_run.figure("rawtrigger"))["data"][0]
    starts = np.asarray(bar.get("base") if bar.get("base") is not None else bar["x"], float)
    np.testing.assert_allclose(starts, EVENT_ONSETS, atol=1e-6)
    if bar.get("base") is not None:
        np.testing.assert_allclose(np.asarray(bar["x"], float), EVENT_DURATION, atol=1e-6)


def test_the_grand_mean_averages_the_kept_long_channels_only(raw_viewer_run, denoise_run):
    _, y = xy(traces(one_figure(raw_viewer_run.figure("rawepochmean")), "HBO long")[0])
    raw = denoise_run.read("uncorrected")
    events, event_id = mne.events_from_annotations(raw, verbose="error")
    epochs = mne.Epochs(raw, events, event_id, tmin=-5.0, tmax=25.0, baseline=(-5.0, 0),
                        preload=True, reject_by_annotation=False, verbose="error")
    kept = [f"{p.name} hbo" for p in raw_viewer_run.truth.long_pairs if not p.bad]
    np.testing.assert_allclose(y, epochs.average(picks=kept).data.mean(axis=0) * 1e6,
                               rtol=1e-4, atol=1e-3)


def test_the_decoupled_trial_has_the_lowest_sci(raw_viewer_run):
    fig = one_figure(raw_viewer_run.figure("rawtrialqc", suffix="qc"))
    sci = {}
    for text in fig["data"][0]["text"]:
        trial, rest = text.split(" · ", 1)
        metric, value = rest.split(": ", 1)
        if metric.startswith("SCI"):
            sci[trial] = float(value)
    dead = next(t for t in sci if f"_{raw_viewer_run.truth.dead_trial:g}s_" in t)
    assert min(sci, key=sci.get) == dead
