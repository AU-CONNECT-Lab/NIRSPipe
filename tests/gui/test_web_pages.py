"""The sidebar's pages: every tab is a real command, and a finished run finds the report it wrote."""

from types import SimpleNamespace

import pytest

from nirspipe.interface import command_pages
from nirspipe.interface.commands import COMMANDS, parser_for, subcommands
from nirspipe.interface.server import WebApp
from nirspipe.interface.sidebar import PAGES
from nirspipe.io.naming import report_name

TABS = [(page, tab) for page in PAGES for tab in page.tabs]


@pytest.mark.parametrize("page, tab", TABS, ids=[f"{p.slug}/{t.key}" for p, t in TABS])
def test_every_tab_is_a_command_and_renders(page, tab):
    assert tab.prog in COMMANDS
    assert tab.sub is None or tab.sub in subcommands(parser_for(tab.prog))
    assert WebApp().flask.test_client().get(f"/{page.slug}/{tab.key}/").status_code == 200


def test_a_page_opens_on_its_first_tab():
    page = PAGES[0]
    r = WebApp().flask.test_client().get(f"/{page.slug}/")
    assert r.status_code == 302 and r.headers["Location"].endswith(f"/{page.slug}/{page.tabs[0].key}/")


def test_a_finished_run_links_the_report_it_wrote(tmp_path):
    written = tmp_path / "sub-01" / report_name("sub-01_task-rest", desc="raw")
    written.parent.mkdir(parents=True)
    written.write_text("<html>raw</html>", encoding="utf-8")
    command_pages._RUNS["done"] = ("prep-raw", tmp_path)
    client = WebApp().flask.test_client()

    found = client.get("/runs/done/report").get_json()
    assert found["path"] == str(written)
    assert client.get(found["url"]).get_data(as_text=True) == "<html>raw</html>"


def test_a_raw_qc_run_links_each_recording_for_rating(tmp_path):
    for subject in ("01", "02"):
        page = tmp_path / f"sub-{subject}" / report_name(f"sub-{subject}_task-rest", desc="raw")
        page.parent.mkdir(parents=True)
        page.write_text("<html></html>", encoding="utf-8")
    command_pages._RUNS["raw"] = ("prep-raw", tmp_path)
    client = WebApp().flask.test_client()

    rate = client.get("/runs/raw/report").get_json()["rate"]
    assert [item["label"] for item in rate] == ["sub-01_task-rest", "sub-02_task-rest"]
    first = client.get(rate[0]["href"])
    assert first.status_code == 302
    viewer = first.headers["Location"]
    assert client.get(f"{viewer}rate_config").status_code == 200
    assert client.get(rate[0]["href"]).headers["Location"] == viewer


def test_a_missing_report_is_said_to_be_missing(tmp_path):
    command_pages._RUNS["empty"] = ("prep-raw", tmp_path)
    message = WebApp().flask.test_client().get("/runs/empty/report").get_json()["message"]
    assert "report is missing" in message and str(tmp_path) in message


def test_a_command_that_writes_tables_says_so():
    command_pages._RUNS["tables"] = (None, None)
    message = WebApp().flask.test_client().get("/runs/tables/report").get_json()["message"]
    assert "tables, not a report" in message


@pytest.mark.parametrize("level, key", [("participant", "participant"), ("group", "cohort")])
def test_nirspipe_lands_on_the_page_its_level_writes(level, key):
    assert command_pages._report_key("nirspipe", None, SimpleNamespace(analysis_level=level)) == key


def test_nirspipe_gui_opens_this_app_on_data_preparation(monkeypatch):
    from nirspipe.cli import gui

    opened = {}
    monkeypatch.setattr(WebApp, "serve", lambda self, port, default, name, paths: opened.update(
        port=port, default=default, paths=paths))
    gui.main(["--port", "9100"])
    assert opened == {"port": 9100, "default": gui.DEFAULT_PORT, "paths": ["data-prep/"]}
    assert WebApp().flask.test_client().get("/data-prep/").status_code == 302
