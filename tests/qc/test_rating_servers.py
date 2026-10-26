"""What the three rating servers write and serve, exercised through their routes."""

import json

from nirspipe.io.derivatives import channel_decisions_path
from nirspipe.io.naming import rating_path, report_name
from nirspipe.qc.rating.app import FNIRSRatingApp, HyperRatingApp, RawRatingApp


def test_each_subject_page_keeps_its_own_rating_file(tmp_path):
    client = FNIRSRatingApp(tmp_path, ["01"]).app.test_client()
    run, cond = "sub-01_task-rest_report", "sub-01_task-rest_cond-a_report"
    client.post("/sub-01/save_ratings", json={"id": run, "ratings": {"Final": "good"}})
    client.post("/sub-01/save_ratings", json={"id": cond, "ratings": {"Final": "bad"}})

    assert client.get(f"/sub-01/load_ratings/{run}").get_json()["ratings"] == {"Final": "good"}
    assert client.get(f"/sub-01/load_ratings/{cond}").get_json()["ratings"] == {"Final": "bad"}
    assert rating_path(tmp_path, run).exists() and rating_path(tmp_path, cond).exists()


def test_a_note_with_a_line_break_reads_back(tmp_path):
    client = FNIRSRatingApp(tmp_path, ["01"]).app.test_client()
    note = 'line one\nline "two" \\ end'
    client.post("/sub-01/save_ratings", json={"id": "sub-01_task-rest_report",
                                              "ratings": {"Final": "good"}, "notes": {"Final": note}})

    loaded = client.get("/sub-01/load_ratings/sub-01_task-rest_report")
    assert loaded.status_code == 200
    assert loaded.get_json()["notes"] == {"Final": note}


def test_a_save_without_a_page_id_is_refused(tmp_path):
    client = FNIRSRatingApp(tmp_path, ["01"]).app.test_client()
    assert client.post("/sub-01/save_ratings", json={"ratings": {"Final": "good"}}).status_code == 400


def test_the_raw_viewer_serves_the_files_beside_its_report(tmp_path):
    sub = tmp_path / "sub-01"
    html = sub / report_name("sub-01_task-rest", desc="raw")
    (sub / "figures").mkdir(parents=True)
    html.write_text("<html><head></head></html>", encoding="utf-8")
    (sub / report_name("sub-01", desc="index")).write_text("<html></html>", encoding="utf-8")
    (sub / "figures" / "panel.html").write_text("<html></html>", encoding="utf-8")
    client = RawRatingApp(html, tmp_path).app.test_client()

    assert client.get("/").status_code == 200
    assert client.get("/" + report_name("sub-01", desc="index")).status_code == 200
    assert client.get("/figures/panel.html").status_code == 200


def test_the_hyper_viewer_writes_only_this_dyad_s_members(tmp_path):
    html = tmp_path / "group-G1" / report_name("group-G1_task-rest", desc="raw")
    client = HyperRatingApp(html, tmp_path, ["sub-01", "sub-02"]).app.test_client()

    r = client.post("/save_hyper_decisions", json={"sub-09": {"S1_D1 hbo": "bad"}})
    assert r.status_code == 400
    assert not channel_decisions_path(tmp_path, "sub-09", task="rest").exists()


def test_hyper_decisions_live_under_their_own_key_beside_the_runs(tmp_path):
    html = tmp_path / "group-G1" / report_name("group-G1_task-rest", desc="raw")
    app = HyperRatingApp(html, tmp_path, ["sub-01", "sub-02"])
    path = app._decisions_path("sub-01")
    run_key = "sub-01_task-rest_run-01"
    path.write_text(json.dumps({run_key: {"S1_D1 hbo": "good"},
                                "_hyper": {"S3_D3 hbo": "bad"}}), encoding="utf-8")
    client = app.app.test_client()

    assert client.get("/load_decisions").get_json() == {"sub-01": {"S3_D3 hbo": "bad"},
                                                         "sub-02": {}}
    r = client.post("/save_hyper_decisions", json={"sub-01": {"S2_D2 hbo": "bad"}})
    assert r.get_json() == {"status": "success"}
    assert json.loads(path.read_text(encoding="utf-8")) == {
        run_key: {"S1_D1 hbo": "good"}, "_hyper": {"S2_D2 hbo": "bad"}}


def _raw_page(tmp_path, label):
    html = tmp_path / "sub-01" / report_name(label, desc="raw")
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text("<html><head></head></html>", encoding="utf-8")
    return html


def test_two_runs_of_a_task_keep_each_other_s_decisions_and_the_dyad_s(tmp_path):
    # each page sends everything it loaded; the other page's later save must not undo this one's
    run1, run2 = "sub-01_task-rest_run-01", "sub-01_task-rest_run-02"
    page1 = RawRatingApp(_raw_page(tmp_path, run1), tmp_path).app.test_client()
    page2 = RawRatingApp(_raw_page(tmp_path, run2), tmp_path).app.test_client()
    path = channel_decisions_path(tmp_path, "01", task="rest")
    path.write_text(json.dumps({"_hyper": {"S3_D3 hbo": "bad"}}), encoding="utf-8")

    stale = page2.get("/load_channel_decisions").get_json()
    page1.post("/save_channel_decisions", json={run1: {"S1_D1 hbo": "bad"}, "_hyper": {}})
    page2.post("/save_channel_decisions", json={**stale, run2: {"S2_D2 hbo": "good"}})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "_hyper": {"S3_D3 hbo": "bad"}, run1: {"S1_D1 hbo": "bad"}, run2: {"S2_D2 hbo": "good"}}


def test_rate_raw_opens_every_report_the_selection_matches(tmp_path):
    from nirspipe.cli.rate import open_raw

    for label in ("sub-01_task-rest_run-01", "sub-01_task-rest_run-02", "sub-01_task-tap"):
        _raw_page(tmp_path, label)
    _raw_page(tmp_path, "sub-01_task-rest_run-01_cond-a")

    stems = lambda opened: sorted(label for _viewer, label, _entries in opened)
    assert stems(open_raw(tmp_path, "01", None, "rest", None)) == [
        "sub-01_task-rest_run-01_desc-raw_report", "sub-01_task-rest_run-02_desc-raw_report"]
    assert stems(open_raw(tmp_path, "01", None, "rest", None, run_label="02")) == [
        "sub-01_task-rest_run-02_desc-raw_report"]
    assert len(open_raw(tmp_path, "01", None, None, None)) == 3
