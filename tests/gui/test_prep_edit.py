"""A raw viewer in edit mode: events read by edit-markers export, saved by edit-markers apply."""

import json
import time
from pathlib import Path

import pandas as pd

from nirspipe.interface import process_stream
from nirspipe.interface.prep_edit import (
    EditViewer, drift_hint, form_html, parse_rows, save_argv, selection,
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
    html = form_html({"--skip-bids-validation": ["yes"]})

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


def test_the_save_command_is_the_form_plus_the_run_and_the_table(tmp_path):
    from werkzeug.datastructures import MultiDict

    argv = save_argv(MultiDict({"output_dir": str(tmp_path / "out"), "--skip-bids-validation": "yes"}),
                     tmp_path / "bids", "sub-01_task-tap_run-2", tmp_path / "events.tsv")

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
    refused = client.post("prep/save", data={"_run": "sub-01_task-tapping", "_rows": "[]"})
    assert refused.status_code == 400 and "output_dir" in refused.get_json()["error"]

    rows[0][0] += 1.5
    rows[1][2] = "renamed"
    saved = client.post("prep/save", data={
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
    refused = client.post("prep/save", data={"_run": "nolabel", "_rows": "[]"})
    assert refused.status_code == 400
    assert "cannot tell the subject" in refused.get_json()["error"]


def test_event_names_are_read_as_written(tmp_path):
    from nirspipe.interface.prep_edit import read_events

    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("tapping",))
    events = bids / "sub-01" / "nirs" / "sub-01_task-tapping_events.tsv"
    events.write_text("onset\tduration\ttrial_type\n10\t5\t1\n50\t5\tn/a\n90\t5\t2\n", encoding="utf-8")

    assert [r[2] for r in read_events(bids, "sub-01_task-tapping")["rows"]] == ["1", "", "2"]
