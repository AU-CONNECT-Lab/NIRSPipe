"""The command pages: what a run streams, how Stop ends it, and a viewer opened in this app."""

import json
import sys

from nirspipe.interface import process_stream
from nirspipe.interface.commands import COMMANDS
from nirspipe.interface.server import WebApp
from nirspipe.io.naming import report_name


def _events(client, url: str) -> tuple[list[str], dict]:
    """The lines a stream sent, and its end event."""
    lines, end, kind = [], None, None
    for raw in client.get(url).get_data(as_text=True).split("\n"):
        if raw.startswith("event: "):
            kind = raw[len("event: "):]
        elif raw.startswith("data: "):
            data = json.loads(raw[len("data: "):])
            if kind == "end":
                end = data
            else:
                lines.append(data)
            kind = None
    return lines, end


def test_a_run_streams_every_line_then_its_exit_code():
    client = WebApp().flask.test_client()
    run_id = process_stream.start([sys.executable, "-c", "print('one'); print('two')"])

    lines, end = _events(client, f"/runs/{run_id}/events")
    assert lines == ["one", "two"]
    assert end == {"code": 0, "stopped": False}


def test_stop_ends_a_run_and_the_stream_says_so():
    client = WebApp().flask.test_client()
    run_id = process_stream.start(
        [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(60)"])

    assert client.post(f"/runs/{run_id}/stop").get_json() == {"stopped": True}
    _lines, end = _events(client, f"/runs/{run_id}/events")
    assert end["stopped"] is True


def test_an_unknown_run_is_not_found():
    assert WebApp().flask.test_client().get("/runs/nope/events").status_code == 404


def test_a_form_that_does_not_parse_is_not_run():
    client = WebApp().flask.test_client()
    r = client.post("/run/nirspipe-qc/", data={"bids_dir": "/bids", "output_dir": "/out"})
    assert r.status_code == 400 and "required" in r.get_json()["error"]


def test_a_form_runs_the_installed_command(tmp_path):
    client = WebApp().flask.test_client()
    r = client.post("/run/nirspipe-rate/export/", data={"trees": str(tmp_path)})
    lines, end = _events(client, r.get_json()["events"])
    assert end["code"] == 0, lines
    assert any("--bad-channels" in line for line in lines)


def test_a_rating_subcommand_opens_its_viewer_in_this_app(tmp_path):
    web = WebApp()
    client = web.flask.test_client()
    form = {"output_dir": str(tmp_path), "--participant-label": "01"}

    missing = client.post("/run/nirspipe-rate/raw/", data=form)
    assert missing.status_code == 400 and "raw report not found" in missing.get_json()["error"]

    html = tmp_path / "sub-01" / report_name("sub-01", desc="raw")
    html.parent.mkdir(parents=True)
    html.write_text("<html></html>", encoding="utf-8")
    [opened] = client.post("/run/nirspipe-rate/raw/", data=form).get_json()["opened"]
    assert client.get(opened["href"]).status_code == 200
    config = client.get(f"{opened['href']}rate_config").get_json()
    assert config["sci_threshold"] is None and config["mark_spans"] is True


def test_the_raw_qc_rating_link_opens_a_viewer_that_marks_bad_stretches(tmp_path):
    client = WebApp().flask.test_client()
    html = tmp_path / "sub-01" / report_name("sub-01", desc="raw")
    html.parent.mkdir(parents=True)
    html.write_text("<html></html>", encoding="utf-8")

    r = client.get("/rate/raw", query_string={"report": str(html), "tree": str(tmp_path)})
    assert r.status_code == 302
    assert client.get(r.location.rstrip("/") + "/rate_config").get_json()["mark_spans"] is True


def test_the_command_list_links_every_command():
    page = WebApp().flask.test_client().get("/commands/").get_data(as_text=True)
    for prog in COMMANDS:
        assert f">{prog}<" in page
