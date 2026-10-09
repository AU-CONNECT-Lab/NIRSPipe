"""What alignment leaves between members, per block, and its row on the alignment timeline."""

import mne
import pytest

from nirspipe.pipeline.hyper.alignment import onset_residuals
from nirspipe.qc.figures.hyper.hyper_figures import build_alignment_timeline
from tests._synth import synth_raw


def _member(sid, blocks):
    raw = synth_raw(sid, "main", duration=200.0)
    raw.set_annotations(mne.Annotations(*zip(*blocks)) if blocks else mne.Annotations([], [], []))
    return raw


def _pair():
    a = _member("01", [(0.0, 0.0, "sync"), (30.0, 20.0, "ca"), (90.0, 20.0, "ca"),
                       (60.0, 10.0, "cb"), (120.0, 5.0, "BAD_motion")])
    b = _member("02", [(0.0, 0.0, "sync"), (30.098, 20.0, "ca"), (90.0, 20.0, "ca"),
                       (130.0, 3.0, "BAD_motion")])
    return {"sub-01": a, "sub-02": b}


def test_each_block_is_measured_against_the_first_member_s_copy():
    rows = {r["condition"]: r["residual_s"]
            for r in onset_residuals(_pair(), ["sub-01", "sub-02"])}
    # a repeat is matched by order; a block the member lacks and BAD spans have no row
    assert rows == {"sync": pytest.approx(0.0), "ca (1)": pytest.approx(0.098, abs=1e-9),
                    "ca (2)": pytest.approx(0.0)}


def test_a_lone_member_has_nothing_to_differ_from():
    assert onset_residuals(_pair(), ["sub-01"]) == []


def test_the_timeline_draws_the_residuals_with_the_ones_past_tolerance_red():
    raws = _pair()
    rows = onset_residuals(raws, ["sub-01", "sub-02"])
    fig = build_alignment_timeline(raws, raws, ["sub-01", "sub-02"], rows, 0.05)
    points = [t for t in fig.data if t.type == "scatter"]
    assert len(points) == 1 and points[0].name == "sub-02"
    drawn = dict(zip(points[0].x, points[0].y))
    assert drawn == {r["condition"]: r["residual_s"] for r in rows}
    colours = dict(zip(points[0].x, points[0].marker.color))
    assert colours["ca (1)"] == "#c0392b" and colours["sync"] != "#c0392b"
    assert "minus sub-01" in fig.layout.annotations[-1].text


def test_without_residuals_the_timeline_keeps_its_two_rows():
    raws = _pair()
    fig = build_alignment_timeline(raws, raws, ["sub-01", "sub-02"])
    assert len(fig.layout.annotations) == 2
    assert all(t.type == "bar" for t in fig.data)
