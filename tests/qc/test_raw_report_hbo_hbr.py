"""The raw viewer measures HbO-HbR correlation on its own Beer-Lambert, and after the correction."""

import json
import re

import mne
import pytest

from fnirs_pipe.io.derivatives import write_dataset_description
from fnirs_pipe.qc.figures import hbo_hbr_correlation_figure
from fnirs_pipe.qc.metrics import haemo_quality_metrics
from fnirs_pipe.qc.subject.prep_raw_report import _process_run, build_prep_raw_report
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
    assert payload["figure_paths"]["hbo_hbr_corr"]["src"].endswith("hbohbrcorr_nirs.html")
    split = payload["sqm"]["haemo_split"]
    assert [r["name"] for r in split["rows"]] == ["All", "Long", "Short"]
    assert all("post" not in c for r in split["rows"] for c in r["cells"])


def test_a_correction_adds_the_after_side(corrected):
    payload, record = corrected
    for name in ("rawhaemo_post", "rawhaemo_post_long", "rawhaemo_post_short"):
        assert -1.0 <= record[name]["hbo_hbr_corr_mean"] <= 1.0
    cells = [c for r in payload["sqm"]["haemo_split"]["rows"] for c in r["cells"]]
    assert cells and all("post" in c for c in cells)


# ---- per condition ----

@pytest.fixture(scope="module")
def conditioned(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cond")
    bids, out = tmp / "bids", tmp / "out"
    _write_dataset_root(bids, ["01"])
    write_dataset_description(out, source=bids)
    raw = synth_raw("01", "tap", duration=300.0)
    raw.set_annotations(mne.Annotations([20.0, 150.0], [100.0, 100.0], ["rest", "talk"]))
    path = _write_subject(bids, "01", "tap", raw)
    run = {"label": "sub-01_task-tap", "snirf_path": path, "session": None,
           "subject_id": "01", "task": "tap"}
    page = out / "sub-01" / "sub-01_task-tap_desc-raw_report.html"
    build_prep_raw_report([run], page, 0.7, 1.5, [6.0], by_condition=True,
                          motion_correction="tddr")
    record = read_record(next(out.rglob("*_desc-sqmraw_qc.json")))
    return path, record, page.parent


def _page_data(html_path):
    html = html_path.read_text(encoding="utf-8")
    return json.loads(re.search(r"var _STATIC_DATA = (\[.*?\]);\n", html, re.S).group(1))[0]


def test_each_condition_is_measured_on_its_own_cut(conditioned):
    path, record, _ = conditioned
    od = mne.preprocessing.nirs.optical_density(mne.io.read_raw_snirf(path, preload=True))
    haemo = mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)
    for label, entry in record["by_condition"].items():
        t0, t1 = entry["window_s"]
        expected = haemo_quality_metrics(haemo.copy().crop(t0, t1))
        assert entry["haemo_by_set"]["all"]["hbo_hbr_corr_mean"] == pytest.approx(
            expected["hbo_hbr_corr_mean"], abs=1e-12)
        assert entry["per_channel"]["hbo_hbr_corr_per_channel"] == pytest.approx(
            expected["hbo_hbr_corr_per_channel"], abs=1e-12)
        for set_name in ("all", "long", "short"):
            assert "hbo_hbr_corr_mean_post" in entry["haemo_by_set"][set_name]
        assert entry["haemo_by_set"]["all"]["hbo_hbr_corr_mean"] != pytest.approx(
            record["rawhaemo"]["hbo_hbr_corr_mean"])


def test_a_condition_page_prints_its_own_correlation(conditioned):
    _, record, folder = conditioned
    page = next(folder.glob("*cond-rest_desc-raw_report.html"))
    data = _page_data(page)
    rows = data["sqm"]["haemo_split"]["rows"]
    assert [r["name"] for r in rows] == ["All", "Long", "Short"]
    assert all("post" in c for r in rows for c in r["cells"])
    assert "cond-rest" in data["figure_paths"]["hbo_hbr_corr"]["src"]
    assert (folder / data["figure_paths"]["hbo_hbr_corr"]["src"]).exists()
    own = record["by_condition"]["rest"]["per_channel"]["hbo_hbr_corr_per_channel"]
    cells = {r["name"]: r["corr"] for _, block in data["channels"]["blocks"] for r in block}
    assert cells == {pair: f"{value:.3f}" for pair, value in own.items()}


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
