"""The raw viewer measures HbO-HbR correlation on its own Beer-Lambert, and after the correction."""

import mne
import pytest

from fnirs_pipe.io.derivatives import write_dataset_description
from fnirs_pipe.qc.figures import hbo_hbr_correlation_figure
from fnirs_pipe.qc.subject.prep_raw_report import _process_run
from fnirs_pipe.qc.subject.record_io import read_record
from fnirs_pipe.qc.subject.sqm_record import OPTIONAL_SECTIONS
from tests._synth import _write_dataset_root, _write_subject, synth_raw


def _process(tmp_path, motion_correction, conditions=False):
    bids, out = tmp_path / "bids", tmp_path / "out"
    _write_dataset_root(bids, ["01"])
    write_dataset_description(out, source=bids)
    raw = synth_raw("01", "tap", duration=300.0)
    if conditions:
        raw.set_annotations(mne.Annotations([20.0, 150.0], [100.0, 100.0], ["rest", "talk"]))
    path = _write_subject(bids, "01", "tap", raw)
    run = {"label": "sub-01_task-tap", "snirf_path": path, "session": None}
    return _process_run(run, 0.8, out / "sub-01", 0.7, 1.5, [6.0],
                        motion_correction=motion_correction)


def _run(tmp_path, motion_correction):
    payload, ctx = _process(tmp_path, motion_correction)
    return payload, read_record(ctx["sqm_path"])


@pytest.fixture(scope="module")
def uncorrected(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("none"), None)


@pytest.fixture(scope="module")
def corrected(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("tddr"), "tddr")


def test_the_record_carries_the_correlation_per_channel_set(uncorrected):
    _, record = uncorrected
    for name in ("rawhaemo", "rawhaemo_long", "rawhaemo_short"):
        assert -1.0 <= record[name]["hbo_hbr_corr_mean"] <= 1.0
        assert name in record["data"]["sections"]
    assert "rawhaemo_post" not in record


def test_rejected_channels_are_measured_too(uncorrected):
    payload, record = uncorrected
    per_pair = record["per_channel"]["rawhaemo"]["hbo_hbr_corr_per_channel"]
    assert set(per_pair) == set(payload["channel_pairs"])


def test_the_channel_table_prints_the_column(uncorrected):
    payload, record = uncorrected
    per_pair = record["per_channel"]["rawhaemo"]["hbo_hbr_corr_per_channel"]
    rows = [r for _, block in payload["channels"]["blocks"] for r in block]
    assert rows and all(r["corr"] not in (None, "", "—") for r in rows)
    assert {r["name"] for r in rows} == set(per_pair)


def test_the_panel_and_the_table_are_drawn(uncorrected):
    payload, _ = uncorrected
    assert payload["figure_paths"]["hbo_hbr"]["src"].endswith("hbohbrcorr_nirs.html")
    split = payload["sqm"]["haemo_split"]
    assert [r["name"] for r in split["rows"]] == ["All", "Long", "Short"]
    assert all("post" not in c for r in split["rows"] for c in r["cells"])


def test_a_correction_adds_the_after_side(corrected):
    payload, record = corrected
    for name in ("rawhaemo_post", "rawhaemo_post_long", "rawhaemo_post_short"):
        assert -1.0 <= record[name]["hbo_hbr_corr_mean"] <= 1.0
    cells = [c for r in payload["sqm"]["haemo_split"]["rows"] for c in r["cells"]]
    assert cells and all("post" in c for c in cells)


def test_no_condition_entry_carries_the_run_correlation(tmp_path):
    _, ctx = _process(tmp_path, "tddr", conditions=True)
    by_condition = read_record(ctx["sqm_path"])["by_condition"]
    assert set(by_condition) == {"rest", "talk"}
    assert "hbo_hbr_corr" not in repr(by_condition)


def test_the_new_sections_are_registered():
    assert {"rawhaemo", "rawhaemo_long", "rawhaemo_short", "rawhaemo_post",
            "rawhaemo_post_long", "rawhaemo_post_short"} <= set(OPTIONAL_SECTIONS)


def test_the_panel_takes_the_stage_names_it_is_given():
    haemo = mne.preprocessing.nirs.beer_lambert_law(
        mne.preprocessing.nirs.optical_density(synth_raw("01", "tap", duration=120.0)), ppf=6.0)
    fig = hbo_hbr_correlation_figure(haemo, raw_after=haemo.copy(),
                                     stage_labels=("before motion correction", "after tddr"))
    titles = {a.text for a in fig.layout.annotations}
    assert {"before motion correction", "after tddr"} <= titles
    assert "before denoising" not in titles
