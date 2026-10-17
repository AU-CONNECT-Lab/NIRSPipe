"""Recon over a folder: the files found, one checked command per file, and the run one after another."""

import json

from werkzeug.datastructures import MultiDict

from nirspipe.interface import recon_batch
from nirspipe.interface.server import WebApp


def _snirfs(tmp_path):
    for rel in ("a/sub_01-rest.snirf", "b/S02.snirf"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return tmp_path


def test_every_snirf_under_the_folder_is_found_with_a_subject_guess(tmp_path):
    rows, found = recon_batch.detect(str(_snirfs(tmp_path)))
    assert [(r["name"], r["subject"]) for r in rows] == [("sub_01-rest.snirf", "sub01rest"),
                                                         ("S02.snirf", "S02")]
    assert found == "Found 2 SNIRF file(s)."
    assert recon_batch.detect(str(tmp_path / "nope"))[1].startswith("Folder not found")


def test_the_shared_fields_leave_the_per_file_flags_to_the_table():
    shown = {f.name for _title, group in recon_batch.shared_groups() for f in group}
    assert not shown & set(recon_batch.PER_FILE.values())
    assert {"bids_dir", "--optode-frame", "--overwrite"} <= shown


def _form(**shared):
    form = MultiDict({"bids_dir": "/bids", **shared})
    for file, subject, task in [("/raw/a.snirf", "01", "rest"), ("/raw/b.snirf", "02", ""),
                                ("/raw/c.snirf", "03", "tap")]:
        for key, value in (("file", file), ("subject", subject), ("task", task),
                           ("session", ""), ("run", "")):
            form.add(key, value)
    return form


def test_each_file_with_a_subject_and_a_task_gets_its_own_command():
    built, skipped, error = recon_batch.steps(_form(**{"--optode-frame": "head"}))
    assert error is None and skipped == ["b.snirf"]
    assert [header for header, _argv in built] == ["[1/2] a.snirf -> sub-01", "[2/2] c.snirf -> sub-03"]
    assert built[1][1] == ["nirspipe-recon", "/raw/c.snirf", "/bids", "--participant-label", "03",
                           "--task-label", "tap", "--optode-frame", "head"]


def test_a_refused_command_names_its_file():
    _built, _skipped, error = recon_batch.steps(_form(**{"--optode-frame": "nose"}))
    assert error.startswith("a.snirf: ") and "--optode-frame" in error


def test_the_files_run_one_after_another_under_one_stream(tmp_path):
    client = WebApp().flask.test_client()
    form = _form()
    form.setlist("bids_dir", [str(tmp_path / "bids")])
    r = client.post("/recon/batch/run", data=form)
    events = client.get(r.get_json()["events"]).get_data(as_text=True)

    lines = [json.loads(raw[len("data: "):]) for raw in events.splitlines() if raw.startswith("data: ")]
    assert "[1/2] a.snirf -> sub-01" in lines and "[2/2] c.snirf -> sub-03" in lines
    assert lines[-1] == {"code": 1, "stopped": False}
    assert "0 of 2 succeeded" in lines


def test_a_table_with_nothing_to_run_is_refused():
    form = MultiDict({"bids_dir": "/bids", "file": "/raw/a.snirf", "subject": "", "task": ""})
    r = WebApp().flask.test_client().post("/recon/batch/run", data=form)
    assert r.status_code == 400 and "at least one file" in r.get_json()["error"]
