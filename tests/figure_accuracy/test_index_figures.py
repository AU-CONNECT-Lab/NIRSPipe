"""The subject index: each condition's numbers from the record, under that condition's name."""

import re

import numpy as np
import pytest

from tests.figure_accuracy._payload import one_figure
from tests.figure_accuracy._read import _failing_in
from tests.figure_accuracy.conftest import HAND_MARKED

# a condition's windowed values are columns of the run's --window-length grid, and say so
PANELS = {"SCI": "sci_win_mean", "PSP": "psp_mean", "CV": "cv_mean", "SNR": "snr_mean"}


def _grid(run) -> str:
    return f"{run.record()['windowed']['qc_window_s']:g} s"


def _index_figure(run, name):
    return one_figure(run.figures / name)


def test_the_condition_profile_draws_each_condition_s_recorded_scalars(condition_run):
    by_condition = condition_run.record()["by_condition"]
    fig = _index_figure(condition_run, "sub-01_desc-condprofile_nirs.html")
    titles = [a["text"] for a in fig["layout"]["annotations"]]
    for k, (name, key) in enumerate(PANELS.items()):
        title = f"{name} ({_grid(condition_run)})"
        assert titles[k] == title
        axis = "x" if k == 0 else f"x{k + 1}"
        median = next(t for t in fig["data"] if t.get("name") == "cohort median"
                      and (t.get("xaxis") or "x") == axis)
        for condition, value in zip(median["x"], median["y"]):
            assert value == pytest.approx(by_condition[condition]["scalars"][key], rel=1e-6), title


def test_the_condition_with_the_decoupled_stretch_has_the_lower_sci(condition_run):
    scalars = {c: e["scalars"]["sci_win_mean"] for c, e in condition_run.record()["by_condition"].items()}
    assert scalars["cb"] < scalars["ca"] - 0.1


def test_the_channel_matrix_prints_each_condition_s_recorded_channel_sci(condition_run):
    by_condition = condition_run.record()["by_condition"]
    fig = _index_figure(condition_run, "sub-01_task-main_desc-condchannels_nirs.html")
    assert fig["layout"]["annotations"][0]["text"] == f"SCI ({_grid(condition_run)})"
    heat = fig["data"][0]
    for channel, row in zip(heat["y"], np.asarray(heat["text"])):
        for condition, printed in zip(heat["x"], row):
            stored = by_condition[condition]["per_channel"]["sci_win_per_channel"][channel]
            assert float(printed) == pytest.approx(stored, abs=1e-4), (channel, condition)
    # rows run worst first: the pair decoupled in one condition only moves the most
    assert heat["y"][0].startswith(condition_run.truth.block_bad[0])


def test_the_timeline_shades_each_condition_over_its_own_block(condition_run):
    fig = _index_figure(condition_run, "sub-01_task-main_desc-condtimeline_nirs.html")
    spans = {(s["x0"], s["x1"]) for s in fig["layout"]["shapes"] if s["type"] == "rect"}
    labels = {a["text"]: a["x"] for a in fig["layout"]["annotations"]}
    for name, onset, duration, _ in condition_run.truth.blocks:
        assert (onset, onset + duration) in spans
        assert labels[name] == pytest.approx(onset + duration / 2)


def _conditions_table(run):
    html = (run.out / "sub-01" / "sub-01_desc-index_report.html").read_text(encoding="utf-8")
    table = html[html.index('<h2 id="Conditions">'):]
    table = table[:table.index("</table>")]
    rows = [[re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", cell)).strip()
             for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)]
    return rows[0], {row[0]: row for row in rows[1:]}


@pytest.mark.parametrize("run_name, hand", [("condition_run", ()), ("prep_condition_run", (HAND_MARKED,))])
def test_the_conditions_table_counts_the_channels_passing_on_each_row(request, run_name, hand):
    run = request.getfixturevalue(run_name)
    header, rows = _conditions_table(run)
    column = header.index("Channels passing")
    n = 2 * len(run.truth.pairs)
    rejected = {p.name for p in run.truth.pairs if p.bad} | set(hand)
    assert rows["whole run"][column] == f"{n - 2 * len(rejected)}/{n}"
    for block in ("ca", "cb"):
        assert rows[block][column] == f"{n - 2 * len(_failing_in(run, block, hand))}/{n}", block

