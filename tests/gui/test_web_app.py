"""One web app serves every viewer at once, each under its own prefix."""

import re
from pathlib import Path

from nirspipe.interface.server import WebApp
from nirspipe.io.naming import rating_path, report_name
from nirspipe.qc.rating.app import FNIRSRatingApp, RawRatingApp

TEMPLATES = Path(__file__).resolve().parents[2] / "nirspipe" / "qc" / "templates"


def _report(tree: Path, subject: str, desc: str) -> Path:
    stem = f"sub-{subject}" if desc == "index" else f"sub-{subject}_task-rest"
    html = tree / f"sub-{subject}" / report_name(stem, desc=desc)
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text("<html></html>", encoding="utf-8")
    return html


def test_two_viewers_answer_each_under_their_own_prefix(tmp_path):
    web = WebApp()
    [first] = web.mount(RawRatingApp(_report(tmp_path, "01", "raw"), tmp_path, 0.6).app, "a")
    [second] = web.mount(RawRatingApp(_report(tmp_path, "02", "raw"), tmp_path).app, "b")
    client = web.flask.test_client()

    assert client.get(f"/{first}rate_config").get_json()["sci_threshold"] == 0.6
    assert client.get(f"/{second}rate_config").get_json()["sci_threshold"] is None
    stem = "sub-02_task-rest_desc-raw_report"
    assert client.post(f"/{second}save_raw_ratings",
                       json={"id": stem, "ratings": {"Final": "good"}}).status_code == 200
    assert rating_path(tmp_path, stem).exists()


def test_a_viewer_mounted_after_the_server_started_is_reachable(tmp_path):
    web = WebApp()
    client = web.flask.test_client()
    assert client.get("/").status_code == 200

    [path] = web.mount(RawRatingApp(_report(tmp_path, "01", "raw"), tmp_path).app, "late")
    assert client.get(f"/{path}").status_code == 200


def test_the_subject_pages_rate_relative_to_their_folder(tmp_path):
    _report(tmp_path, "01", "index")
    web = WebApp()
    [page] = web.mount(FNIRSRatingApp(tmp_path, ["01"]).app, "Run reports", ("sub-01/",))
    client = web.flask.test_client()
    stem = "sub-01_task-rest_report"

    assert client.get(f"/{page}").status_code == 200
    client.post(f"/{page}save_ratings", json={"id": stem, "ratings": {"Final": "bad"}})
    assert client.get(f"/{page}load_ratings/{stem}").get_json()["ratings"] == {"Final": "bad"}


def test_the_home_page_links_every_page_opened(tmp_path):
    web = WebApp()
    [raw] = web.mount(RawRatingApp(_report(tmp_path, "01", "raw"), tmp_path).app, "Raw sub-01")
    [s1, s2] = web.mount(FNIRSRatingApp(tmp_path, ["01", "02"]).app, "Run reports",
                         ("sub-01/", "sub-02/"))
    home = web.flask.test_client().get("/").get_data(as_text=True)

    for text, path in [("Raw sub-01", raw), ("Run reports sub-01", s1), ("Run reports sub-02", s2)]:
        assert f'<a href="/{path}">{text}</a>' in home


def test_no_report_page_calls_an_endpoint_from_the_server_root():
    absolute = re.compile(r'fetch\("/|_endpoint = "/')
    calls = {page.name: absolute.findall(page.read_text(encoding="utf-8"))
             for page in TEMPLATES.iterdir() if page.suffix in (".html", ".j2")}
    assert not {name: found for name, found in calls.items() if found}


def test_the_app_wears_the_reports_tokens():
    css = WebApp().flask.test_client().get("/tokens.css").get_data(as_text=True)
    assert "--accent:" in css
