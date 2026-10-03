"""Both raw reports draw the provenance graph their footer table describes, and link it."""

import json

from fnirs_pipe.io.derivatives import group_data_dir, group_report_dir
from fnirs_pipe.io.naming import report_name
from fnirs_pipe.qc.boilerplate.notes import section_note
from fnirs_pipe.qc.hyper import hyper_report
from fnirs_pipe.qc.subject.prep_raw_report import _page_provenance, _shell_vars


def _sidecar(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps({"Sources": [], "step": "od_conversion"}))
    (directory / f"{name}.snirf").write_bytes(b"")


_RUNS = [{"subject_id": "01", "session": None, "task": "hold"}]


# ---- the subject raw viewer ----

def test_the_viewer_draws_the_graph_beside_its_figures(tmp_path):
    _sidecar(tmp_path / "nirs", "sub-01_task-hold_desc-od_nirs")
    href = _page_provenance(_RUNS, tmp_path / "x_report.html", tmp_path)
    assert href and href.startswith("figures/") and href.endswith(".png")
    assert (tmp_path / href).exists()


def test_the_viewer_footer_links_the_graph_it_drew(tmp_path):
    _sidecar(tmp_path / "nirs", "sub-01_task-hold_desc-od_nirs")
    out = tmp_path / "x_report.html"
    shell = _shell_vars(_RUNS, out, tmp_path, 0.8, _page_provenance(_RUNS, out, tmp_path))
    assert shell["provenance_rows"]
    assert shell["provenance_path"]


def test_a_tree_without_sidecars_draws_nothing(tmp_path):
    assert _page_provenance(_RUNS, tmp_path / "x_report.html", tmp_path) is None


# ---- the dyad raw report ----

def test_the_dyad_raw_report_shows_the_graph_not_the_placeholder(tmp_path, monkeypatch):
    data_dir = group_data_dir(tmp_path, "G1")
    _sidecar(data_dir, "group-G1_task-hold_desc-od_nirs")
    meta = {"subject_ids": ["01", "02"], "label": "group-G1_task-hold", "sqm_dir": data_dir,
            "alignment": [], "ch_pairs": [], "sqm": {}, "figure_paths": {}}
    monkeypatch.setattr(hyper_report, "_process_hyper_raw_group", lambda **_: meta)
    monkeypatch.setattr(hyper_report, "group_methods", lambda *a, **k: None)

    path = hyper_report.build_hyper_report("G1", "hold", [], {}, {}, {}, tmp_path)

    assert path == group_report_dir(tmp_path, "G1") / report_name("group-G1_task-hold",
                                                                    desc="raw")
    html = path.read_text(encoding="utf-8")
    assert 'class="prov-graph"' in html
    assert section_note("footer.provenance_not_rendered") not in html
    pngs = list((path.parent / "figures").glob("*provenance*.png"))
    assert len(pngs) == 1 and f"figures/{pngs[0].name}" in html
