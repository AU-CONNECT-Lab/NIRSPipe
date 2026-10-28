"""With --events-path the run report's epoch figures read the design's events, not the recording's markers."""

from datetime import datetime, timezone

import mne
import numpy as np

from nirspipe.qc.subject.report import _with_design_events
from tests.figure_accuracy.conftest import run_capturing
from tests._fingerprint import EVENT_DURATION, EVENT_ONSETS


def _raw(first_samp=0):
    info = mne.create_info(["a"], 10.0, "misc")
    raw = mne.io.RawArray(np.zeros((1, 1000)), info, first_samp=first_samp, verbose=False)
    raw.set_meas_date(datetime(2024, 1, 1, tzinfo=timezone.utc))
    # onsets are seconds since the measurement date, so data time t sits at t + first_time
    shift = raw.first_time
    raw.set_annotations(mne.Annotations([5 + shift, 10 + shift], [2, 1], ["BAD_motion", "1.0"],
                                        orig_time=raw.annotations.orig_time))
    return raw


def test_the_markers_are_replaced_and_the_bad_spans_kept(tmp_path):
    table = tmp_path / "events.tsv"
    table.write_text("onset\tduration\ttrial_type\n12\t3\tTap\n30\t3\tRest\n40\t1\tBAD_x\n",
                     encoding="utf-8")
    for first_samp in (0, 50):
        out = _with_design_events(_raw(first_samp), str(table))
        rows = sorted(zip(out.annotations.onset - out.first_time, out.annotations.duration,
                          out.annotations.description))
        assert [(round(o, 6), d, n) for o, d, n in rows] == [
            (5.0, 2.0, "BAD_motion"), (12.0, 3.0, "Tap"), (30.0, 3.0, "Rest")]


def test_the_epoch_preview_names_the_design_s_conditions(tmp_path):
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n" + "".join(
        f"{o:g}\t{EVENT_DURATION:g}\tFinger\n" for o in EVENT_ONSETS), encoding="utf-8")
    done = run_capturing(tmp_path, ["--mode", "glm", "--drift-high-pass", "0.008",
                                    "--short-channel", "mean", "--events-path", str(events)],
                         ("build_epoch_preview_figure", "design_matrix_static_figure"))
    drawn = done["captured"]["build_epoch_preview_figure"][0][0]
    markers = {d for d in drawn.annotations.description if not d.startswith("BAD_")}
    assert markers == {"Finger"}
    onsets = sorted(o - drawn.first_time for o, d in zip(drawn.annotations.onset,
                                                         drawn.annotations.description)
                    if d == "Finger")
    np.testing.assert_allclose(onsets, EVENT_ONSETS)
    assert "Finger" in done["captured"]["design_matrix_static_figure"][0][0].columns
