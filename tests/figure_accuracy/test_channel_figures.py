"""The per-channel pages draw the named channel, the named chromophore and the named stage."""

import re

import numpy as np
import pytest
from mne.time_frequency import psd_array_welch

from nirspipe.qc.figures.common._utils import PSD_NFFT
from nirspipe.qc.metrics.gvtd import GVTD_MOTION_BAND, gvtd_timetrace
from nirspipe.qc.subject.record_io import read_record
from tests._fingerprint import CLI_ARGS, HBR_FREQ, SFREQ
from tests.figure_accuracy._payload import plotly_figures
from tests.figure_accuracy._read import share_at, traces, which_pair, xy


def _arg(flag):
    return float(CLI_ARGS[CLI_ARGS.index(flag) + 1])


def _fname(pair):
    return pair.replace("_", "")


# ---- channel detail: HbO and HbR over time, then their PSD ----

@pytest.fixture(scope="module")
def details(denoise_run):
    return {p.name: plotly_figures(denoise_run.figure("detail", chan=_fname(p.name)))
            for p in denoise_run.truth.pairs}


@pytest.mark.parametrize("desc", ["detail", "psddetail"])
def test_every_pair_has_a_page_and_a_rejected_one_says_so(denoise_run, desc):
    html = denoise_run.report.read_text(encoding="utf-8")
    for pair in denoise_run.truth.pairs:
        path = denoise_run.figure(desc, chan=_fname(pair.name))
        assert path.exists(), path.name
        option = re.search(rf'<option value="figures/{re.escape(path.name)}">([^<]*)</option>', html)
        assert option, path.name
        assert ("rejected" in option.group(1)) == pair.bad, option.group(1)


def test_the_detail_series_is_haemoglobin_from_the_od_before_motion_correction(denoise_run, details):
    for pair, (series, _) in details.items():
        for chroma, label in (("hbo", "HbO"), ("hbr", "HbR")):
            x, y = xy(traces(series, label)[0])
            times, expected = denoise_run.channel("uncorrected", f"{pair} {chroma}")
            np.testing.assert_allclose(x, times, atol=1e-6)
            np.testing.assert_allclose(y, expected * 1e6, rtol=1e-5, atol=1e-6)


def test_the_detail_series_is_its_own_pair_and_not_swapped(denoise_run, details):
    for pair in denoise_run.truth.long_pairs:
        if pair.bad:
            continue
        series = details[pair.name][0]
        hbo, hbr = traces(series, "HbO")[0], traces(series, "HbR")[0]
        assert which_pair(*xy(hbo), denoise_run.truth) == pair.name
        assert share_at(*xy(hbr), HBR_FREQ) > 10 * share_at(*xy(hbo), HBR_FREQ)


def test_the_detail_psd_is_the_welch_spectrum_of_the_same_series(denoise_run, details):
    for pair, (_, psd) in details.items():
        raw = denoise_run.read("uncorrected")
        data = raw.get_data(picks=[f"{pair} hbo", f"{pair} hbr"])
        expected, freqs = psd_array_welch(data, SFREQ, n_fft=PSD_NFFT, verbose=False)
        for row, label in enumerate(("HbO", "HbR")):
            x, y = xy(traces(psd, label)[0])
            keep = freqs <= x[-1] + 1e-9
            np.testing.assert_allclose(x, freqs[keep], atol=1e-6)
            np.testing.assert_allclose(y, expected[row][keep], rtol=1e-4)


def test_the_detail_psd_shades_the_bands_the_run_was_given(details):
    _, psd = next(iter(details.values()))
    spans = {(s["x0"], s["x1"]) for s in psd["layout"]["shapes"] if s["type"] == "rect"}
    assert (_arg("--cardiac-l-freq"), _arg("--cardiac-h-freq")) in spans
    assert (_arg("--resp-l-freq"), _arg("--resp-h-freq")) in spans


# ---- motion detail: one OD channel before and after correction ----

def _motion(run, pair, wavelength="760"):
    return plotly_figures(run.figure("motion", chan=f"{_fname(pair)}{wavelength}"))[0]


def test_the_motion_rows_are_the_od_on_either_side_of_the_correction(denoise_run):
    for pair in denoise_run.truth.pairs:
        for wavelength in ("760", "850"):
            fig = _motion(denoise_run, pair.name, wavelength)
            for label, desc in (("Before", "sci"), ("After", "motcorrected")):
                x, y = xy(traces(fig, label)[0])
                times, expected = denoise_run.channel(desc, f"{pair.name} {wavelength}")
                np.testing.assert_allclose(x, times, atol=1e-6)
                np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-7)


def test_the_derivative_row_finds_each_artefact_on_its_own_channel_at_its_own_time(denoise_run):
    truth = denoise_run.truth
    clean = next(p.name for p in truth.long_pairs if not p.bad and p.name not in truth.moved)
    clean_peak = xy(traces(_motion(denoise_run, clean), "|dOD/dt|")[0])[1].max()
    for pair, when in (truth.spike, truth.step):
        x, y = xy(traces(_motion(denoise_run, pair), "|dOD/dt|")[0])
        # the bump spans 2 s and the row is band-limited, so the peak lands within a few seconds
        assert abs(x[np.argmax(y)] - when) < 3.0
        assert y.max() > 2 * clean_peak


def test_the_gvtd_row_rises_at_each_artefact(denoise_run):
    truth = denoise_run.truth
    gvtd = next(t for t in _motion(denoise_run, truth.spike[0])["data"]
                if (t.get("name") or "").startswith("GVTD"))
    x, y = xy(gvtd)
    for _, when in (truth.spike, truth.step):
        near = np.abs(x - when) < 3.0
        assert y[near].max() > 3 * np.median(y), when


# ---- the corrected spans shaded on each channel's motion figure ----

# the pipeline's figures and the raw viewer's, each beside the record its run wrote
MOTION_VIEWS = [("denoise_run", "motion", "sqm"), ("raw_viewer_run", "rawmotion", "sqmraw")]


def _shaded(fig, name="corrected"):
    """A shading trace's spans as (onset, duration); each span is one four-corner outline."""
    found = traces(fig, name)
    if not found:
        return np.empty((0, 2))
    x = np.asarray(found[0]["x"], float)
    corners = x[np.isfinite(x)].reshape(-1, 4)
    return np.column_stack([corners[:, 0], corners[:, 2] - corners[:, 0]])


def _spans_for(request, run_name, desc, record, short):
    run = request.getfixturevalue(run_name)
    windowed = read_record(run.nirs / f"sub-01_task-tapping_desc-{record}_qc.json")["windowed"]
    key = "motion_corrected_spans_short_s" if short else "motion_corrected_spans_s"
    expected = np.asarray(windowed[key], float).reshape(-1, 2)
    pairs = [p for p in run.truth.pairs if p.short == short]
    return expected, {p.name: _shaded(plotly_figures(run.figure(desc, chan=f"{_fname(p.name)}760"))[0])
                      for p in pairs}


@pytest.mark.parametrize("run_name, desc, record", MOTION_VIEWS)
def test_a_long_channel_shades_the_long_set_s_corrected_spans(request, run_name, desc, record):
    expected, shaded = _spans_for(request, run_name, desc, record, short=False)
    assert len(expected)
    for pair, spans in shaded.items():
        np.testing.assert_allclose(spans, expected, atol=0.05, err_msg=pair)


@pytest.mark.parametrize("run_name, desc, record", MOTION_VIEWS)
def test_a_short_channel_shades_the_short_set_s_corrected_spans(request, run_name, desc, record):
    expected, shaded = _spans_for(request, run_name, desc, record, short=True)
    for pair, spans in shaded.items():
        assert spans.shape == expected.shape, pair
        np.testing.assert_allclose(spans, expected, atol=0.05, err_msg=pair)


@pytest.mark.parametrize("run_name, desc, record", MOTION_VIEWS)
def test_only_the_long_channels_shade_the_long_pairs_artefacts(request, run_name, desc, record):
    run = request.getfixturevalue(run_name)
    for pair in run.truth.pairs:
        spans = _shaded(plotly_figures(run.figure(desc, chan=f"{_fname(pair.name)}760"))[0])
        for _, when in (run.truth.spike, run.truth.step):
            shaded = any(onset - 1 <= when <= onset + duration + 1 for onset, duration in spans)
            assert shaded == (not pair.short), (pair.name, when)


# ---- the GVTD rows and their thresholds ----

def _threshold_lines(fig):
    return [s["y0"] for s in fig["layout"].get("shapes", [])
            if s["type"] == "line" and not str(s.get("yref", "")).endswith("domain")]


@pytest.mark.parametrize("run_name, desc, record", MOTION_VIEWS)
def test_each_gvtd_line_is_the_threshold_the_record_holds(request, run_name, desc, record):
    run = request.getfixturevalue(run_name)
    rec = read_record(run.nirs / f"sub-01_task-tapping_desc-{record}_qc.json")
    carpet = plotly_figures(run.figure(desc.replace("motion", "carpet")))[0]
    assert _threshold_lines(carpet) == pytest.approx(
        [rec["raw_long"]["gvtd_thresh"], rec["raw_short"]["gvtd_thresh"]], rel=1e-9)
    for pair in run.truth.pairs:
        fig = plotly_figures(run.figure(desc, chan=f"{_fname(pair.name)}760"))[0]
        own = rec["raw_short" if pair.short else "raw_long"]["gvtd_thresh"]
        assert _threshold_lines(fig) == pytest.approx([own], rel=1e-9), pair.name


def test_the_gvtd_row_is_its_set_s_gvtd_in_the_motion_band(denoise_run):
    od = denoise_run.read("sci")
    for short in (False, True):
        pair = next(p for p in denoise_run.truth.pairs if p.short == short)
        names = [c for c in od.ch_names if denoise_run.truth.pair(c.rsplit(" ", 1)[0]).short == short]
        expected = gvtd_timetrace(od.get_data(picks=names), od.info["sfreq"], *GVTD_MOTION_BAND)
        gvtd = next(t for t in _motion(denoise_run, pair.name)["data"]
                    if (t.get("name") or "").startswith("GVTD"))
        x, y = xy(gvtd)
        np.testing.assert_allclose(x, od.times[1:], atol=1e-6)
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-12)


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


@pytest.mark.parametrize("run_name, desc", [("denoise_run", "carpet"), ("raw_viewer_run", "rawcarpet")])
def test_every_band_the_carpet_names_is_the_one_its_gvtd_was_filtered_in(request, run_name, desc):
    fig = plotly_figures(request.getfixturevalue(run_name).figure(desc))[0]
    named = {band for text in _strings(fig) for band in re.findall(r"[\d.]+–[\d.]+ Hz", text)}
    assert named == {f"{GVTD_MOTION_BAND[0]:g}–{GVTD_MOTION_BAND[1]:g} Hz"}


def test_the_motion_figure_names_the_band_it_filtered_in(denoise_run):
    band = f"{GVTD_MOTION_BAND[0]:g}–{GVTD_MOTION_BAND[1]:g} Hz"
    pair = denoise_run.truth.long_pairs[0].name
    fig = _motion(denoise_run, pair)
    names = [t.get("name") or "" for t in fig["data"]]
    notes = [a.get("text") or "" for a in fig["layout"].get("annotations", [])]
    assert f"GVTD {band}" in names, names
    assert f"{pair} 760, {band}" in notes, notes
