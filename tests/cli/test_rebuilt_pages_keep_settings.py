"""A page a second command rebuilds says what the run that wrote it said: its mode, its ROI map, its members."""

import json

import pandas as pd

from nirspipe.cli import qc as qc_cli
from nirspipe.cli.run_record import recorded_post
from nirspipe.qc.figures.common import provenance_figure
from nirspipe.qc.hyper.hyper_index import write_hyper_index
from nirspipe.qc.subject import subject_index

from tests._raw_qc import SCREENING
from tests._synth import make_bids_dataset
from tests.hyper._names import name


def _record(sub_dir, stem, mode, stamp, **execution):
    logs = sub_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lines = ["[execution]", f'run_timestamp = "{stamp}"']
    lines += [f"{key} = [ " + ", ".join(f'"{v}"' for v in values) + " ]"
              for key, values in execution.items()]
    lines += ["[post]", f'mode = "{mode}"']
    (logs / f"{stem}.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_newest_record_covering_the_run_gives_its_settings(tmp_path):
    _record(tmp_path, "sub-01", "denoise", "2026-10-01T10:00:00")
    _record(tmp_path, "sub-01_task-rest", "rest", "2026-10-02T10:00:00", task_label=["rest"])
    assert recorded_post(tmp_path, "sub-01_task-rest")["mode"] == "rest"
    assert recorded_post(tmp_path, "sub-01_task-tap")["mode"] == "denoise"
    assert recorded_post(tmp_path)["mode"] == "rest"


def test_raw_qc_records_are_not_the_pipeline_s(tmp_path):
    _record(tmp_path, "sub-01_desc-raw", "denoise", "2026-10-01T10:00:00")
    assert recorded_post(tmp_path, "sub-01_task-rest") == {}


def test_a_rerendered_graph_keeps_the_mode_in_its_title(tmp_path, monkeypatch):
    sub = tmp_path / "sub-01"
    (sub / "nirs").mkdir(parents=True)
    (sub / "nirs" / "sub-01_task-rest_desc-preproc_nirs.snirf").write_bytes(b"")
    _record(sub, "sub-01", "rest", "2026-10-01T10:00:00")
    titles = []
    monkeypatch.setattr(provenance_figure, "write_provenance",
                        lambda *a, title=None, **k: titles.append(title) or [])
    qc_cli.main_provenance([str(tmp_path)])
    # the root's own graph is drawn too, under the tree's name
    assert provenance_figure.provenance_title("sub-01_task-rest", "rest") in titles
    assert "sub-01_task-rest" not in titles


def test_raw_qc_rebuilding_an_analysis_tree_s_index_keeps_its_mode(tmp_path, monkeypatch):
    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("rest",), duration=60.0)
    out = tmp_path / "out"
    _record(out / "sub-01", "sub-01", "rest", "2026-10-01T10:00:00")
    seen = []
    real = subject_index.write_subject_index
    monkeypatch.setattr(subject_index, "write_subject_index",
                        lambda *a, **k: seen.append(k.get("mode")) or real(*a, **k))
    qc_cli.main([str(bids), str(out), "participant", *SCREENING, "--skip-bids-validation"])
    assert seen == ["rest"]


def test_a_rebuilt_dyad_index_names_the_members_its_tables_hold(tmp_path):
    (tmp_path / "nirs").mkdir()
    table = tmp_path / "nirs" / name("G01", "main")
    pd.DataFrame({"sub1": "sub-11", "sub2": "sub-12", "chromophore": "hbo",
                  "condition": ["whole_run"], "label": "S1_D1", "coherence": [0.4]}).to_csv(
        table, sep="\t", index=False, na_rep="n/a")
    table.with_suffix(".json").write_text(json.dumps({"parameters": {
        "conditions": [], "condition_windows_s": {}}}))
    page = write_hyper_index(tmp_path, "G01")
    assert "sub-11 &times; sub-12" in page.read_text(encoding="utf-8")
