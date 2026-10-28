"""The wavelength check: HbO over HbR power of the pulse with the labels as given and swapped,
and the cohort page's per-task warning.

The recordings are built forward from concentrations, so the ratio a pulse should read is set
by construction: an HbO pulse of amplitude A over an HbR pulse of A / 10 reads 100.
"""

import mne
import numpy as np
import pytest
from mne.preprocessing.nirs._beer_lambert_law import _load_absorption

from nirspipe.qc.metrics.coupling import _cardiac_ratio_metrics, labels_look_swapped
from nirspipe.qc.subject.group_writer import build_group_raw_report, swapped_label_tasks
from nirspipe.qc.subject.record_io import write_record
from nirspipe.qc.subject.sqm_record import raw_sections
from tests._synth import WAVELENGTHS, _channel_layout, synth_raw

SFREQ, PULSE_HZ = 10.0, 1.2


def _recording(pairs, swap_labels=False, seed=0):
    """Intensity of one pair per ``(hbo_amp, hbr_amp)``: a 1.2 Hz pulse of that HbO and HbR
    over white noise; ``swap_labels`` writes each wavelength's light under the other's label."""
    names, locs, _ = _channel_layout(len(pairs), short=False)
    rng = np.random.default_rng(seed)
    t = np.arange(int(SFREQ * 180)) / SFREQ
    pulse = np.sin(2 * np.pi * PULSE_HZ * t)
    extinction = _load_absorption(list(WAVELENGTHS))
    data = []
    for hbo, hbr in pairs:
        od = extinction @ np.vstack([hbo * pulse, hbr * pulse])
        od = od + 1e-4 * rng.normal(size=od.shape)
        data += list(0.1 * np.exp(-od))
    info = mne.create_info(names, SFREQ, ["fnirs_cw_amplitude"] * len(names))
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc.copy()
        if swap_labels:
            ch["loc"][9] = WAVELENGTHS[1] if loc[9] == WAVELENGTHS[0] else WAVELENGTHS[0]
    raw = mne.io.RawArray(np.array(data), info, verbose="error")
    # MNE wants each pair's wavelengths in ascending order, which the swap undid
    return raw.reorder_channels(sorted(raw.ch_names, key=lambda c: (c.split()[0], float(
        next(ch["loc"][9] for ch in raw.info["chs"] if ch["ch_name"] == c))))) if swap_labels else raw


def test_an_arterial_pulse_reads_its_hbo_over_hbr_power_as_labelled():
    out = _cardiac_ratio_metrics(_recording([(1e-3, 1e-4)] * 3), [], 0.7, 1.8)
    assert out["cardiac_ratio_n"] == 3
    assert out["cardiac_ratio"] == pytest.approx(100, rel=0.05)
    assert out["cardiac_ratio_swapped"] < 1
    assert labels_look_swapped(out["cardiac_ratio"], out["cardiac_ratio_swapped"]) is False


def test_swapped_labels_read_the_other_way_round():
    given = _cardiac_ratio_metrics(_recording([(1e-3, 1e-4)] * 3), [], 0.7, 1.8)
    swapped = _cardiac_ratio_metrics(_recording([(1e-3, 1e-4)] * 3, swap_labels=True), [], 0.7, 1.8)
    assert swapped["cardiac_ratio"] == pytest.approx(given["cardiac_ratio_swapped"], rel=1e-6)
    assert swapped["cardiac_ratio_swapped"] == pytest.approx(given["cardiac_ratio"], rel=1e-6)
    assert labels_look_swapped(swapped["cardiac_ratio"], swapped["cardiac_ratio_swapped"]) is True


def test_a_pair_without_a_clear_pulse_or_rejected_is_not_counted():
    raw = _recording([(1e-3, 1e-4), (0.0, 0.0), (1e-3, 1e-4)])
    out = _cardiac_ratio_metrics(raw, ["S3_D3 850"], 0.7, 1.8)
    assert sorted(out["cardiac_ratio_per_channel"]) == ["S1_D1"]
    assert out["cardiac_ratio_n"] == 1


def test_a_flat_pair_is_not_counted():
    raw = _recording([(1e-3, 1e-4), (1e-3, 1e-4)])
    raw._data[2:4] = 0.1
    assert sorted(_cardiac_ratio_metrics(raw, [], 0.7, 1.8)["cardiac_ratio_per_channel"]) == ["S1_D1"]


def test_no_clear_pulse_leaves_no_verdict():
    out = _cardiac_ratio_metrics(_recording([(0.0, 0.0)] * 2), [], 0.7, 1.8)
    assert out["cardiac_ratio"] is None and out["cardiac_ratio_n"] == 0
    assert labels_look_swapped(out["cardiac_ratio"], out["cardiac_ratio_swapped"]) is None


def test_the_record_carries_it_on_the_long_set():
    raw = synth_raw("01", "rest", duration=120.0, motion_onset=None)
    sections, per_channel = raw_sections(raw, {ch: 0.9 for ch in raw.ch_names}, [], 0.7, 1.8,
                                         resp_h_freq=0.5)
    assert sections["raw_long"]["cardiac_ratio_n"] == len(
        per_channel["raw_long"]["cardiac_ratio_per_channel"]) > 0
    assert "cardiac_ratio_swapped" in sections["raw_long"]


def _row(name, ratio, swapped):
    return {"bids_name": name, "raw_long": {"cardiac_ratio": ratio, "cardiac_ratio_swapped": swapped}}


def test_a_task_is_named_when_most_of_its_runs_read_swapped():
    rows = [_row("sub-01_task-rest", 1.0, 9.0), _row("sub-02_task-rest", 1.2, 8.0),
            _row("sub-03_task-rest", 30.0, 0.5), _row("sub-04_task-rest", None, None),
            _row("sub-01_task-tap_acq-a", 1.0, 9.0), _row("sub-02_task-tap_acq-b", 30.0, 0.5)]
    assert swapped_label_tasks(rows) == [
        {"task": "task-rest", "runs": ["sub-01_task-rest", "sub-02_task-rest"], "n": 3}]


def test_the_cohort_page_warns_for_such_a_task(tmp_path):
    for i, (ratio, swapped) in enumerate([(1.0, 9.0), (1.1, 8.0), (25.0, 0.4)], 1):
        nirs = tmp_path / f"sub-0{i}" / "nirs"
        nirs.mkdir(parents=True)
        write_record(nirs / f"sub-0{i}_task-rest_desc-sqm_qc.json",
                     {"step": "sqm", "raw": {"psp_mean": 0.3},
                      "raw_long": {"cardiac_ratio": ratio, "cardiac_ratio_swapped": swapped}})
    page = build_group_raw_report(tmp_path).read_text(encoding="utf-8")
    assert "2 of the 3 runs of task-rest with a clear pulse read as if" in page
