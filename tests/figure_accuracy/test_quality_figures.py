"""The quality panels draw the record's numbers on the record's channels, against the run's own lines."""

import csv

import numpy as np
import pytest

from fnirs_pipe.qc.subject.record_io import read_record
from tests._fingerprint import CLI_ARGS
from tests.figure_accuracy._payload import one_figure

PASS_COLOUR, FAIL_COLOUR = "#C5E0B3", "#F8786E"


def _arg(flag):
    return float(CLI_ARGS[CLI_ARGS.index(flag) + 1])


@pytest.fixture(scope="module")
def record(denoise_run):
    return read_record(denoise_run.nirs / f"sub-{denoise_run.subject}_task-{denoise_run.task}_desc-sqm_qc.json")


@pytest.fixture(scope="module")
def scipsp(denoise_run):
    return one_figure(denoise_run.figure("scipsp"))


def _heatmap(fig, name):
    return next(t for t in fig["data"] if t["type"] == "heatmap" and t["name"] == name)


# ---- windowed SCI / PSP / CV ----

@pytest.mark.parametrize("name, key", [("SCI", "sci"), ("PSP", "psp"), ("CV", "cv")])
def test_each_heatmap_is_the_record_matrix_under_the_record_channel_names(record, scipsp, name, key):
    windowed = record["windowed"]
    heat = _heatmap(scipsp, name)
    assert list(heat["y"]) == list(windowed[f"{key}_channels"])
    np.testing.assert_allclose(np.asarray(heat["z"], float), np.asarray(windowed[f"{key}_matrix"], float),
                               rtol=1e-5, equal_nan=True)


def test_the_rejected_pair_is_the_dark_row(denoise_run, scipsp):
    heat = _heatmap(scipsp, "SCI")
    means = dict(zip(heat["y"], np.nanmean(np.asarray(heat["z"], float), axis=1)))
    for channel, mean in means.items():
        bad = denoise_run.truth.pair(channel.rsplit(" ", 1)[0]).bad
        assert (mean < 0.3) if bad else (mean > 0.9), channel


def _threshold_line(fig, axis):
    return next(s["x0"] for s in fig["layout"]["shapes"] if s["type"] == "line" and s["xref"] == axis)


def test_the_sci_line_is_the_run_s_threshold(scipsp):
    assert _threshold_line(scipsp, "x2") == pytest.approx(_arg("--sci-threshold"))


@pytest.mark.xfail(strict=True, reason="report.py never passes psp_threshold to build_sci_psp_figure")
def test_the_psp_line_is_the_run_s_threshold(scipsp):
    assert _threshold_line(scipsp, "x4") == pytest.approx(_arg("--psp-threshold"))


# ---- channel quality grid ----

@pytest.fixture(scope="module")
def grid(denoise_run):
    fig = one_figure(denoise_run.figure("chsummary", suffix="qc"))
    dots = fig["data"][0]
    cells = {}
    for text, colour in zip(dots["text"], dots["marker"]["color"]):
        channel, rest = text.split(" · ", 1)
        metric, value = rest.split(": ", 1)
        cells[(channel, metric)] = (value, colour)
    return cells


@pytest.fixture(scope="module")
def table(denoise_run):
    path = denoise_run.nirs / f"sub-{denoise_run.subject}_task-{denoise_run.task}_desc-channel_qc.tsv"
    with open(path, encoding="utf-8") as f:
        return {row["name"]: row for row in csv.DictReader(f, delimiter="\t")}


@pytest.mark.parametrize("metric, column", [("PSP", "psp"), ("CV", "cv"),
                                            ("SNR", "snr"), ("Coupled", "good_frac")])
def test_every_grid_cell_prints_the_channel_table_value(grid, table, metric, column):
    for channel, row in table.items():
        printed = float(grid[(channel, metric)][0])
        assert printed == pytest.approx(float(row[column]), rel=0.01, abs=0.001), (channel, metric)


def test_the_grid_sci_is_the_windowed_estimate_the_screening_counted(grid, record):
    windowed = {**record["per_channel"]["raw_long"]["sci_win_per_channel"],
                **record["per_channel"]["raw_short"]["sci_win_per_channel"]}
    for channel, value in windowed.items():
        assert float(grid[(channel, "SCI")][0]) == pytest.approx(value, abs=0.001), channel


def test_only_the_rejected_pair_is_not_ok(denoise_run, grid):
    for (channel, metric), (value, _) in grid.items():
        if metric == "Status":
            bad = denoise_run.truth.pair(channel.rsplit(" ", 1)[0]).bad
            assert (value != "OK") if bad else (value == "OK"), channel


def test_the_grid_judges_psp_against_the_run_s_threshold(denoise_run, grid, table):
    line = _arg("--psp-threshold")
    for channel, row in table.items():
        expected = PASS_COLOUR if float(row["psp"]) >= line else FAIL_COLOUR
        assert grid[(channel, "PSP")][1] == expected, channel
