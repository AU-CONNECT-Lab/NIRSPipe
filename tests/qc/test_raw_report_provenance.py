"""The dyad raw report draws the provenance graph its footer table describes; the raw viewer has none."""

import json

from nirspipe.io.derivatives import group_data_dir, group_report_dir
from nirspipe.io.naming import report_name
from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.hyper import hyper_report
from nirspipe.qc.subject.prep_raw_report import _shell_vars


def _sidecar(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps({"Sources": [], "step": "od_conversion"}))
    (directory / f"{name}.snirf").write_bytes(b"")


_RUNS = [{"subject_id": "01", "session": None, "task": "hold"}]


# ---- the subject raw viewer ----

def test_the_viewer_carries_no_provenance_section(tmp_path):
    # its lineage is one input and the record, which the record's sidecar already lists
    _sidecar(tmp_path / "nirs", "sub-01_task-hold_desc-od_nirs")
    shell = _shell_vars(_RUNS, tmp_path / "x_report.html", 0.8)
    assert "provenance_rows" not in shell and "provenance_path" not in shell


# ---- the dyad raw report ----

def test_the_dyad_raw_report_shows_the_graph_not_the_placeholder(tmp_path, monkeypatch):
    data_dir = group_data_dir(tmp_path, "G1")
    _sidecar(data_dir, "group-G1_task-hold_desc-od_nirs")
    meta = {"subject_ids": ["01", "02"], "label": "group-G1_task-hold", "sqm_dir": data_dir,
            "alignment": [], "ch_pairs": [], "sqm": {}, "figure_paths": {}}
    monkeypatch.setattr(hyper_report, "_process_hyper_raw_group", lambda **_: meta)
    monkeypatch.setattr(hyper_report, "group_methods", lambda *a, **k: None)

    path = hyper_report.build_hyper_report("G1", "hold", [], {}, {}, {}, tmp_path, sci_threshold=0.8)

    assert path == group_report_dir(tmp_path, "G1") / report_name("group-G1_task-hold",
                                                                    desc="raw")
    html = path.read_text(encoding="utf-8")
    assert 'class="prov-graph"' in html
    assert section_note("footer.provenance_not_rendered") not in html
    pngs = list((path.parent / "figures").glob("*provenance*.png"))
    assert len(pngs) == 1 and f"figures/{pngs[0].name}" in html
