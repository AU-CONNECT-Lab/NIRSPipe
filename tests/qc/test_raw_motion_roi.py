"""Under an ROI map the raw viewer's carpet groups its long rows by ROI, and every page says which set."""

import json
import re

import mne
import pytest

from nirspipe.io.derivatives import write_dataset_description
from nirspipe.qc.subject.prep_raw_report import _process_run, build_prep_raw_report
from tests._synth import _write_dataset_root, _write_subject, synth_raw

ROI_MAP = {"B": ["S1_D1 hbo"], "A": ["S3_D3 hbo"]}


def _run(tmp):
    bids, out = tmp / "bids", tmp / "out"
    _write_dataset_root(bids, ["01"])
    write_dataset_description(out, source=bids)
    raw = synth_raw("01", "tap", duration=300.0)
    raw.set_annotations(mne.Annotations([20.0, 150.0], [100.0, 100.0], ["rest", "talk"]))
    path = _write_subject(bids, "01", "tap", raw)
    return out, {"label": "sub-01_task-tap", "snirf_path": path, "session": None,
                 "subject_id": "01", "task": "tap"}


def _page_data(html_path):
    html = html_path.read_text(encoding="utf-8")
    return json.loads(re.search(r"var _STATIC_DATA = (\[.*?\]);\n", html, re.S).group(1))[0]


def _legend_squares(fig) -> list:
    return [t["name"] for t in fig["data"]
            if t.get("type") == "scatter" and (t.get("marker") or {}).get("symbol") == "square"]


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    out, run = _run(tmp_path_factory.mktemp("roi"))
    page = out / "sub-01" / "sub-01_task-tap_desc-raw_report.html"
    build_prep_raw_report([run], page, 0.7, 1.5, 0.1, 0.5, [6.0], 0.8, by_condition=True,
                          roi_map=ROI_MAP)
    return page, sorted(page.parent.glob("*_cond-*_desc-raw_report.html"))


def test_the_run_page_s_carpet_draws_the_strip_and_names_the_rois(pages):
    page, _ = pages
    data = _page_data(page)
    fig = data["carpet_gvtd"]["figure"]
    assert _legend_squares(fig)[:2] == ["A", "B"]
    assert [sh for sh in fig["layout"]["shapes"] if sh["type"] == "rect" and sh["x1"] < -0.015]
    assert data["figure_paths"]["carpet"]["roi_set"] == "long"


def test_every_condition_page_names_the_set_the_strip_groups(pages):
    _, conditions = pages
    assert len(conditions) == 2
    for cond in conditions:
        carpet = _page_data(cond)["figure_paths"]["carpet"]
        assert carpet["roi_set"] == "long"
        assert "#" in carpet["src"]


def test_without_a_map_the_carpet_has_no_strip_and_no_note(tmp_path):
    out, run = _run(tmp_path)
    payload, _ = _process_run(run, 0.8, out / "sub-01", 0.7, 1.5, 0.1, 0.5, [6.0])
    fig = payload["carpet_gvtd"]["figure"]
    assert not _legend_squares(fig)
    assert not [sh for sh in fig["layout"]["shapes"] if sh["type"] == "rect" and sh["x1"] < -0.015]
    assert "roi_set" not in payload["figure_paths"]["carpet"]
