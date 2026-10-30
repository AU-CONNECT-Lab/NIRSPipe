"""A raw viewer in edit mode: events read by edit-markers export, saved by edit-markers apply."""

import json
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd

from nirspipe.interface import process_stream
from nirspipe.interface.prep_edit import (
    EditViewer, command_argv, drift_hint, form_html, page_flags, parse_crop, parse_rows, selection,
)
from nirspipe.io.naming import report_name
from nirspipe.qc.rating.app import RawRatingApp
from tests._synth import make_bids_dataset

TEMPLATE = Path(__file__).resolve().parents[2] / "nirspipe" / "qc" / "templates" / "raw_viewer.html"


def _report(tree: Path, subject: str = "01", task: str = "tapping") -> Path:
    html = tree / f"sub-{subject}" / report_name(f"sub-{subject}_task-{task}", desc="raw")
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text("<html></html>", encoding="utf-8")
    return html


def _wait(run_id: str) -> "tuple[list[str], int]":
    for _ in range(600):
        lines, code, _dropped = process_stream.poll(run_id)
        if code is not None:
            return lines, code
        time.sleep(0.1)
    raise AssertionError("the command did not finish")


def test_only_an_edit_viewer_asks_the_page_for_the_editor(tmp_path):
    report = _report(tmp_path)
    edit = EditViewer(report, tmp_path, tmp_path / "bids").app.test_client()
    rate = RawRatingApp(report, tmp_path).app.test_client()

    assert edit.get("rate_config").get_json()["edit"] is True
    assert "edit" not in rate.get("rate_config").get_json()
    assert rate.get("prep/edit.js").status_code == 404


def test_the_edit_viewer_marks_no_bad_stretches(tmp_path):
    report = _report(tmp_path)
    edit = EditViewer(report, tmp_path, tmp_path / "bids").app.test_client()
    rate = RawRatingApp(report, tmp_path).app.test_client()

    assert edit.get("rate_config").get_json()["mark_spans"] is False
    assert edit.post("save_bad_spans", json={}).status_code in (404, 405)
    assert rate.get("rate_config").get_json()["mark_spans"] is True


def test_the_raw_viewer_loads_the_editor_only_when_told_to():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert 'editor.src = "prep/edit.js"' in text
    assert "cfg.edit && _DECISIONS" in text


def test_the_editor_files_are_served_beside_the_page_and_nothing_else(tmp_path):
    client = EditViewer(_report(tmp_path), tmp_path, tmp_path / "bids").app.test_client()

    for name in ("edit.js", "edit.css", "forms.css"):
        assert client.get(f"prep/{name}").status_code == 200
    assert client.get("prep/app.css").status_code == 404


def test_the_form_leaves_out_what_the_page_fills():
    html = form_html("markers", {"--skip-bids-validation": ["yes"]})

    assert 'name="output_dir"' in html
    for filled in ("bids_dir", "--participant-label", "--run-label", "--tsv", "--shift",
                   "--set-duration", "--rename"):
        assert f'name="{filled}"' not in html
    assert '<option value="yes" selected>--skip-bids-validation</option>' in html


def test_a_run_label_selects_its_own_run():
    assert selection("sub-01_ses-2_task-tap_run-3") == {
        "--participant-label": "01", "--session-label": "2", "--task-label": "tap",
        "--run-label": "3"}
    assert selection("sub-01_task-rest") == {"--participant-label": "01", "--task-label": "rest"}


def test_the_table_is_refused_with_the_row_that_is_wrong():
    assert parse_rows([[1, 2, "a"], ["x", 2, "b"]])[1] == "row 2: onset and duration must be numbers"
    assert parse_rows([[-1, 2, "a"]])[1] == "row 1: onset and duration must be 0 or more"
    assert parse_rows([[1, None, "a"]])[1] == "row 1: onset and duration must be numbers"
    assert parse_rows([[1, 2, " "]])[1] == "row 1: the event has no name"
    assert parse_rows([[1, 2, "a"]]) == ([[1.0, 2.0, "a"]], None)


def test_the_drift_hint_follows_the_slowest_repeat_and_skips_bad_spans():
    rows = [[0, 5, "a"], [40, 5, "a"], [100, 5, "a"], [10, 5, "b"], [30, 5, "b"],
            [0, 300, "BAD_motion"], [400, 1, "BAD_motion"]]
    assert drift_hint(rows) == {"cutoff": "0.008333 Hz", "by": "a", "per": "a 60 s · b 20 s"}
    assert drift_hint([[1, 1, "a"]]) is None


def test_a_doubled_marker_is_no_repeat_for_the_drift_hint():
    assert drift_hint([[5, 1, "a"], [5, 1, "a"], [20, 1, "b"]]) is None
    assert drift_hint([[5, 1, "a"], [5, 1, "a"], [0, 1, "b"], [50, 1, "b"]])["by"] == "b"


def test_the_drift_hint_is_the_cutoff_auto_runs_with():
    # every 30 s gives 1/(2 x 30) = 0.0167 Hz, above the 0.01 Hz auto never exceeds
    assert drift_hint([[0, 1, "a"], [30, 1, "a"], [60, 1, "a"]])["cutoff"] == "0.01 Hz"


def test_the_save_command_is_the_form_plus_the_run_and_the_table(tmp_path):
    from werkzeug.datastructures import MultiDict

    argv = command_argv("markers",
                        MultiDict({"output_dir": str(tmp_path / "out"), "--skip-bids-validation": "yes"}),
                        tmp_path / "bids", "sub-01_task-tap_run-2", {"tsv": str(tmp_path / "events.tsv")})

    assert argv[:5] == ["nirspipe-prep", "edit-markers", "apply", str(tmp_path / "bids"),
                        str(tmp_path / "out")]
    tail = argv[5:]
    for flag, value in (("--participant-label", "01"), ("--task-label", "tap"), ("--run-label", "2"),
                        ("--tsv", str(tmp_path / "events.tsv"))):
        assert tail[tail.index(flag) + 1] == value
    assert "--skip-bids-validation" in tail


def test_events_are_read_and_saved_through_nirspipe_prep(tmp_path):
    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("tapping",))
    out = tmp_path / "edited"
    client = EditViewer(_report(tmp_path / "tree"), tmp_path / "tree", bids).app.test_client()

    events = client.get("prep/events?run=sub-01_task-tapping").get_json()
    assert events["not_markers"] == ["bad", "edge"]
    rows = events["rows"]
    assert len(rows) > 1 and all(len(r) == 3 for r in rows)

    # argparse's own refusal, before anything runs
    refused = client.post("prep/run/markers", data={"_run": "sub-01_task-tapping", "_rows": "[]"})
    assert refused.status_code == 400 and "output_dir" in refused.get_json()["error"]

    rows[0][0] += 1.5
    rows[1][2] = "renamed"
    saved = client.post("prep/run/markers", data={
        "_run": "sub-01_task-tapping", "_rows": json.dumps(rows),
        "output_dir": str(out), "--skip-bids-validation": "yes"}).get_json()
    assert saved["command"].startswith("nirspipe-prep edit-markers apply")
    run_id = saved["events"].split("/")[2]
    lines, code = _wait(run_id)
    assert code == 0, "\n".join(lines)

    written = pd.read_csv(out / "marker_edited" / "sub-01" / "nirs" / "sub-01_task-tapping_events.tsv",
                          sep="\t")
    assert written["onset"].iloc[0] == rows[0][0]
    assert written["trial_type"].iloc[1] == "renamed"
    assert len(written) == len(rows)


def test_a_run_the_label_cannot_place_is_refused(tmp_path):
    client = EditViewer(_report(tmp_path), tmp_path, tmp_path / "bids").app.test_client()
    refused = client.post("prep/run/markers", data={"_run": "nolabel", "_rows": "[]"})
    assert refused.status_code == 400
    assert "cannot tell the subject" in refused.get_json()["error"]


def test_event_names_are_read_as_written(tmp_path):
    from nirspipe.interface.prep_edit import read_events

    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("tapping",))
    events = bids / "sub-01" / "nirs" / "sub-01_task-tapping_events.tsv"
    events.write_text("onset\tduration\ttrial_type\n10\t5\t1\n50\t5\tn/a\n90\t5\t2\n", encoding="utf-8")

    assert [r[2] for r in read_events(bids, "sub-01_task-tapping")["rows"]] == ["1", "", "2"]


def test_an_edit_viewer_rates_nothing(tmp_path):
    client = EditViewer(_report(tmp_path), tmp_path, tmp_path / "bids").app.test_client()

    assert client.post("save_raw_ratings", json={"id": "x"}).status_code != 200
    assert client.post("save_channel_decisions", json={}).status_code != 200
    assert client.get("load_channel_decisions").status_code != 200
    assert client.get("load_raw_ratings").status_code != 200


def _data_prep_tree(tree: Path) -> None:
    for stem in ("sub-01_task-rest", "sub-02_task-rest"):
        _report(tree, stem[4:6], "rest")
    # a condition page is a slice of its run's and is not offered for editing
    (tree / "sub-01" / report_name("sub-01_task-rest_cond-a", desc="raw")).write_text("<html></html>")


def test_data_preparation_finds_a_trees_reports_and_opens_one_for_editing(tmp_path):
    from nirspipe.interface.server import WebApp

    tree, bids = tmp_path / "tree", tmp_path / "bids"
    _data_prep_tree(tree)
    client = WebApp().flask.test_client()

    page = client.get("/data-prep/raw-qc/", query_string={"tree": str(tree), "bids": str(bids)})
    html = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "sub-01_task-rest</a>" in html and "sub-02_task-rest</a>" in html
    assert "cond-a" not in html

    href = re.search(r'<ul class="edit-links">\s*<li><a href="([^"]+)"', html).group(1).replace("&amp;", "&")
    opened = client.get(href)
    assert opened.status_code == 302
    viewer = opened.headers["Location"]
    assert client.get(f"{viewer}rate_config").get_json()["edit"] is True
    assert client.get(href).headers["Location"] == viewer


def test_data_preparation_says_what_it_could_not_find(tmp_path):
    from nirspipe.interface.server import WebApp

    client = WebApp().flask.test_client()
    one = client.get("/data-prep/raw-qc/", query_string={"tree": str(tmp_path)}).get_data(as_text=True)
    assert "Give both folders" in one
    empty = client.get("/data-prep/raw-qc/",
                       query_string={"tree": str(tmp_path), "bids": "x"}).get_data(as_text=True)
    assert f"No raw reports under {tmp_path}" in empty


def test_a_prep_raw_run_links_its_reports_for_editing_against_the_dataset_it_read(tmp_path):
    from nirspipe.interface import command_pages
    from nirspipe.interface.server import WebApp

    tree = tmp_path / "tree"
    _data_prep_tree(tree)
    command_pages._RUNS["prep"] = ("raw-qc", tree)
    command_pages._SOURCES["prep"] = tmp_path / "bids"

    edit = WebApp().flask.test_client().get("/runs/prep/report").get_json()["edit"]
    assert [item["label"] for item in edit] == ["sub-01_task-rest", "sub-02_task-rest"]
    assert parse_qs(urlparse(edit[0]["href"]).query)["bids"] == [str(tmp_path / "bids")]


def test_data_preparation_heads_the_sidebar_on_its_own():
    from nirspipe.interface.sidebar import FAMILIES, PAGES

    assert FAMILIES == ("Interactive", "Batch")
    assert [p.slug for p in PAGES if p.family == "Interactive"] == ["data-prep"]


def test_the_crop_form_leaves_out_the_span_and_the_stage():
    html = form_html("crop", {})

    for shown in ("output_dir", "--align", "--trigger-name", "--combine", "--margin", "--band-fmin"):
        assert f'name="{shown}"' in html
    for filled in ("bids_dir", "--participant-label", "--tmin", "--tmax", "--segments-path",
                   "--input-desc"):
        assert f'name="{filled}"' not in html


def test_a_crop_is_one_stretch_or_segments_and_refused_with_what_is_wrong():
    from werkzeug.datastructures import MultiDict

    assert parse_crop(MultiDict({"_tmin": "10", "_tmax": ""})) == ({"tmin": 10.0}, None)
    assert parse_crop(MultiDict({"_tmin": "", "_tmax": ""}))[1] == "give a start, an end or both"
    assert parse_crop(MultiDict({"_tmin": "20", "_tmax": "10"}))[1] == "the start must come before the end"
    assert parse_crop(MultiDict({"_tmin": "-1"}))[1] == "tmin must be 0 or more"
    multi = {"_mode": "multi"}
    assert parse_crop(MultiDict({**multi, "_segments": "[]"}))[1] == "add at least one segment"
    assert parse_crop(MultiDict({**multi, "_segments": '[[5, 0, ""]]'}))[1].startswith("segment 1:")
    assert parse_crop(MultiDict({**multi, "_segments": '[[5, 20, " a "]]'})) == ([[5.0, 20.0, "a"]], None)


def test_segments_carry_a_task_column_only_when_one_is_named():
    from werkzeug.datastructures import MultiDict

    def table(segments):
        flags, error = page_flags("crop", MultiDict({"_mode": "multi", "_segments": json.dumps(segments)}))
        assert error is None
        return pd.read_csv(flags["segments_path"], sep="\t")

    assert list(table([[0, 10, ""], [20, 10, ""]]).columns) == ["onset", "duration"]
    named = table([[0, 10, "a"], [20, 10, "b"]])
    assert list(named.columns) == ["onset", "duration", "task"]
    assert list(named["task"]) == ["a", "b"]


def test_a_stretch_drawn_on_the_plot_is_cropped_and_links_a_report_of_the_result(tmp_path):
    import mne
    from nirspipe.interface.server import WebApp

    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("tapping",))
    out = tmp_path / "out"
    report = _report(tmp_path / "tree")
    client = WebApp().flask.test_client()
    viewer = client.get("/prep/edit", query_string={
        "report": str(report), "tree": str(tmp_path / "tree"), "bids": str(bids)}).headers["Location"]

    started = client.post(f"{viewer}prep/run/crop", data={
        "_run": "sub-01_task-tapping", "_mode": "single", "_tmin": "10", "_tmax": "100",
        "output_dir": str(out), "--skip-bids-validation": "yes"}).get_json()
    assert started["command"].startswith("nirspipe-prep crop")
    assert "--tmin 10.0 --tmax 100.0" in started["command"]
    lines, code = _wait(started["events"].split("/")[2])
    assert code == 0, "\n".join(lines)
    link = urlparse(started["report"])
    assert link.path == "/data-prep/raw-qc/"
    assert parse_qs(link.query, keep_blank_values=True) == {"bids_dir": [str(out / "cropped")],
                                                            "output_dir": [""]}

    [written] = (out / "cropped").rglob("*_nirs.snirf")
    raw = mne.io.read_raw_snirf(written, verbose="error")
    assert abs(raw.times[-1] - 90) < 1


def test_a_crop_without_a_span_is_refused_before_anything_runs(tmp_path):
    client = EditViewer(_report(tmp_path), tmp_path, tmp_path / "bids").app.test_client()
    refused = client.post("prep/run/crop", data={"_run": "sub-01_task-tapping", "_mode": "single"})
    assert refused.status_code == 400
    assert refused.get_json()["error"] == "give a start, an end or both"
    assert client.post("prep/run/nothing", data={}).status_code == 404


def test_the_report_link_fills_the_new_tree_and_clears_the_output_folder(tmp_path):
    from nirspipe.interface.server import WebApp

    html = WebApp().flask.test_client().get("/data-prep/raw-qc/", query_string={
        "bids_dir": str(tmp_path / "out" / "marker_edited"), "output_dir": ""}).get_data(as_text=True)

    assert f'name="bids_dir" value="{tmp_path / "out" / "marker_edited"}"' in html
    assert 'data-given="bids_dir output_dir"' in html and "data-prefilled" not in html
    assert "Pick a new output folder" in html


def test_an_edit_viewer_opened_on_its_own_links_no_report_page(tmp_path):
    from nirspipe.interface.prep_edit import report_link

    argv = ["nirspipe-prep", "edit-markers", "apply", "bids", str(tmp_path / "o"), "--participant-label", "01",
            "--tsv", "x.tsv"]
    assert report_link("/p/", "markers", argv) == (
        "/p/?" + urlencode({"bids_dir": str(tmp_path / "o" / "marker_edited"), "output_dir": ""}))
    assert EditViewer(_report(tmp_path), tmp_path, tmp_path / "bids")._report_link("markers", argv) is None
