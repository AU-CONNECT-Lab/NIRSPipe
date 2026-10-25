"""The glm block average: trial means, standard errors and counts, against values worked by hand."""
import json

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.pipeline.block_average import block_average_table, write_block_average
from nirspipe.qc.figures.common.topomap import _condition_evokeds

SF = 10.0
ONSETS = (20.0, 60.0, 100.0)
# MNE's baseline (tmin, 0) takes the onset sample too: 1 of the 51 baseline samples has stepped
KEPT = 50 / 51
SOURCE = "sub-01/nirs/sub-01_task-tap_desc-filtered_nirs.snirf"


def _raw(levels=(1.0, 2.0, 3.0), offset=5.0, bad_spans=()):
    """HbO steps to ``levels[k]`` above ``offset`` for 10 s after trial k; HbR the negative."""
    n = int(140 * SF)
    t = np.arange(n) / SF
    hbo = np.full(n, offset)
    for onset, level in zip(ONSETS, levels):
        hbo[(t >= onset) & (t < onset + 10)] += level
    raw = mne.io.RawArray(np.vstack([hbo, -hbo]),
                          mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], SF, ["hbo", "hbr"]),
                          verbose=False)
    onsets = [*ONSETS, *(a for a, _ in bad_spans)]
    durations = [10.0] * len(ONSETS) + [d for _, d in bad_spans]
    names = ["tap"] * len(ONSETS) + ["BAD_motion"] * len(bad_spans)
    raw.set_annotations(mne.Annotations(onsets, durations, names))
    return raw


def _at(table, chroma, time):
    row = table[(table.chroma == chroma) & np.isclose(table.time, time)]
    assert len(row) == 1
    return row.iloc[0]


def test_each_trial_loses_its_baseline_and_the_trials_average():
    table = block_average_table(_raw(), -5.0, 15.0)
    hbo = _at(table, "hbo", 5.0)
    # levels 1, 2, 3 over a baseline of 5: mean 2, sd 1, se 1 / sqrt(3), each scaled by KEPT
    assert hbo["mean"] == pytest.approx(2.0 * KEPT)
    assert hbo["se"] == pytest.approx(KEPT / np.sqrt(3))
    assert hbo["n_trials"] == 3
    assert _at(table, "hbr", 5.0)["mean"] == pytest.approx(-2.0 * KEPT)
    # after the step ends the signal is back at its level, less the onset sample's share
    assert _at(table, "hbo", 12.0)["mean"] == pytest.approx(-2.0 / 51)


def test_a_trial_touching_a_bad_span_is_left_out():
    table = block_average_table(_raw(bad_spans=[(65.0, 2.0)]), -5.0, 15.0)
    hbo = _at(table, "hbo", 5.0)
    # trial 2 (level 2) dropped: levels 1 and 3, sd sqrt(2), se 1
    assert hbo["n_trials"] == 2
    assert hbo["mean"] == pytest.approx(2.0 * KEPT)
    assert hbo["se"] == pytest.approx(KEPT)


def test_without_a_pre_onset_window_nothing_is_subtracted():
    table = block_average_table(_raw(), 0.0, 15.0)
    assert _at(table, "hbo", 5.0)["mean"] == pytest.approx(7.0)


def test_one_trial_has_no_standard_error():
    raw = _raw()
    raw.set_annotations(mne.Annotations([20.0], [10.0], ["tap"]))
    row = _at(block_average_table(raw, -5.0, 15.0), "hbo", 5.0)
    assert row["n_trials"] == 1 and np.isnan(row["se"])


def test_the_evoked_map_draws_the_table_values():
    raw = _raw(levels=(0.5, 2.0, 4.0))
    table = block_average_table(raw, -5.0, 15.0)
    evoked = _condition_evokeds(raw, -5.0, 15.0)["tap"]
    hbo = table[table.chroma == "hbo"].sort_values("time")["mean"].to_numpy()
    np.testing.assert_allclose(evoked.get_data(picks="hbo")[0], hbo)


def test_an_events_table_replaces_the_annotations(tmp_path):
    raw = _raw()
    # the table names only the first two trials, under its own condition name
    events = pd.DataFrame({"onset": [20.0, 60.0], "duration": [10.0, 10.0],
                           "trial_type": ["press", "press"]})
    path = write_block_average(raw, tmp_path / "sub-01" / "nirs", str(tmp_path / SOURCE),
                               -5.0, 15.0, events=events)
    table = pd.read_csv(path, sep="\t")
    assert set(table.condition) == {"press"}
    assert _at(table, "hbo", 5.0)["mean"] == pytest.approx(1.5 * KEPT)


def test_the_file_and_its_sidecar(tmp_path):
    raw = _raw()
    raw.info["bads"] = ["S1_D1 hbr"]
    path = write_block_average(raw, tmp_path / "sub-01" / "nirs", str(tmp_path / SOURCE))
    assert path.name == "sub-01_task-tap_desc-blockavg_timeseries.tsv"
    table = pd.read_csv(path, sep="\t")
    assert table.groupby("chroma")["bad"].all().to_dict() == {"hbo": False, "hbr": True}
    assert table.time.min() == pytest.approx(-5.0) and table.time.max() == pytest.approx(25.0)
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["step"] == "block_average"
    assert side["parameters"]["trials"] == {"tap": 3}
    assert side["parameters"]["baseline"] == [-5.0, 0.0]


def test_no_trial_fitting_the_window_writes_nothing(tmp_path):
    # the first trial at 20 s would need the recording to run to 145 s; it stops at 140
    out = tmp_path / "sub-01" / "nirs"
    assert write_block_average(_raw(), out, str(tmp_path / SOURCE), -5.0, 125.0) is None
    assert not out.exists()
