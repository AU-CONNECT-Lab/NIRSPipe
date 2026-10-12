"""What the three rating servers write and serve, exercised through their routes."""

import json

from nirspipe.io.derivatives import channel_decisions_path
from nirspipe.io.naming import rating_path, report_name
from nirspipe.qc.rating.app import FNIRSRatingApp, HyperRatingApp, RawRatingApp


def test_each_subject_page_keeps_its_own_rating_file(tmp_path):
    client = FNIRSRatingApp(tmp_path, ["01"]).app.test_client()
    run, cond = "sub-01_task-rest_report", "sub-01_task-rest_cond-a_report"
    client.post("/save_ratings", json={"id": run, "ratings": {"Final": "good"}})
    client.post("/save_ratings", json={"id": cond, "ratings": {"Final": "bad"}})

    assert client.get(f"/load_ratings/{run}").get_json()["ratings"] == {"Final": "good"}
    assert client.get(f"/load_ratings/{cond}").get_json()["ratings"] == {"Final": "bad"}
    assert rating_path(tmp_path, run).exists() and rating_path(tmp_path, cond).exists()


def test_a_note_with_a_line_break_reads_back(tmp_path):
    client = FNIRSRatingApp(tmp_path, ["01"]).app.test_client()
    note = 'line one\nline "two" \\ end'
    client.post("/save_ratings", json={"id": "sub-01_task-rest_report",
                                       "ratings": {"Final": "good"}, "notes": {"Final": note}})

    loaded = client.get("/load_ratings/sub-01_task-rest_report")
    assert loaded.status_code == 200
    assert loaded.get_json()["notes"] == {"Final": note}


def test_a_save_without_a_page_id_is_refused(tmp_path):
    client = FNIRSRatingApp(tmp_path, ["01"]).app.test_client()
    assert client.post("/save_ratings", json={"ratings": {"Final": "good"}}).status_code == 400


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


def test_hyper_decisions_merge_into_every_run_or_one_hyper_entry(tmp_path):
    html = tmp_path / "group-G1" / report_name("group-G1_task-rest", desc="raw")
    app = HyperRatingApp(html, tmp_path, ["sub-01", "sub-02"])
    with_runs = app._decisions_path("sub-01")
    with_runs.write_text(json.dumps({"sub-01_task-rest_run-01": {"S1_D1 hbo": "good"},
                                     "sub-01_task-rest_run-02": {}}), encoding="utf-8")

    r = app.app.test_client().post("/save_hyper_decisions", json={
        "sub-01": {"S2_D2 hbo": "bad"}, "sub-02": {"S2_D2 hbo": "bad"}})
    assert r.get_json() == {"status": "success"}

    runs = json.loads(with_runs.read_text(encoding="utf-8"))
    assert runs == {"sub-01_task-rest_run-01": {"S1_D1 hbo": "good", "S2_D2 hbo": "bad"},
                    "sub-01_task-rest_run-02": {"S2_D2 hbo": "bad"}}
    fresh = json.loads(app._decisions_path("sub-02").read_text(encoding="utf-8"))
    assert fresh == {"_hyper": {"S2_D2 hbo": "bad"}}
