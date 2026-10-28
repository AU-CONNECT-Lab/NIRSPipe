"""--rater: each rater's ratings, channel decisions and marked stretches kept in files of their own."""

import json

import pytest

from nirspipe.cli import rate
from nirspipe.io.naming import report_name
from nirspipe.qc.rating.app import FNIRSRatingApp, HyperRatingApp, RawRatingApp


def _raw_page(tree, label="sub-01_task-rest"):
    html = tree / "sub-01" / report_name(label, desc="raw")
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text("<html><head></head></html>", encoding="utf-8")
    return html


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rate_raw(tree, rater, verdict, state, span):
    client = RawRatingApp(_raw_page(tree), tree, rater=rater).app.test_client()
    for url, body in (("/save_raw_ratings", {"ratings": {"Final": verdict}}),
                      ("/save_channel_decisions", {"sub-01_task-rest": {"S1_D1 hbo": state}}),
                      ("/save_bad_spans", {"sub-01_task-rest": [span]})):
        assert client.post(url, json=body).get_json() == {"status": "success"}
    return client


def test_two_raters_of_one_raw_page_keep_their_own_files(tmp_path):
    first = _rate_raw(tmp_path, "QY", "good", "bad", [1.0, 2.0])
    second = _rate_raw(tmp_path, "rater2", "bad", "good", [5.0, 6.0])
    nirs = tmp_path / "sub-01" / "nirs"

    assert _json(nirs / "sub-01_task-rest_rater-QY_desc-rawrating_qc.json")["ratings"] == {
        "Final": "good"}
    assert _json(nirs / "sub-01_task-rest_rater-rater2_desc-rawrating_qc.json")["ratings"] == {
        "Final": "bad"}
    assert _json(nirs / "sub-01_task-rest_rater-QY_desc-rawdecision_qc.json") == {
        "sub-01_task-rest": {"S1_D1 hbo": "bad"}}
    assert _json(nirs / "sub-01_task-rest_rater-rater2_desc-rawdecision_qc.json") == {
        "sub-01_task-rest": {"S1_D1 hbo": "good"}}
    assert _json(nirs / "sub-01_task-rest_rater-QY_desc-rawspans_qc.json") == {
        "sub-01_task-rest": [[1.0, 2.0]]}
    # each viewer reads back its own rater's work and says whose it is
    assert first.get("/load_raw_ratings").get_json() == {
        "ratings": {"Final": "good"}, "notes": {}, "rater": "QY"}
    assert second.get("/load_channel_decisions").get_json() == {
        "sub-01_task-rest": {"S1_D1 hbo": "good"}}
    assert not (nirs / "sub-01_task-rest_desc-rawrating_qc.json").exists()


def test_without_a_rater_the_shared_files_are_written_as_before(tmp_path):
    client = _rate_raw(tmp_path, None, "good", "bad", [1.0, 2.0])
    nirs = tmp_path / "sub-01" / "nirs"

    assert sorted(p.name for p in nirs.iterdir()) == [
        "sub-01_task-rest_desc-rawdecision_qc.json", "sub-01_task-rest_desc-rawrating_qc.json",
        "sub-01_task-rest_desc-rawspans_qc.json"]
    assert client.get("/load_raw_ratings").get_json()["rater"] is None


def test_the_run_report_viewer_keeps_each_rater_apart_and_logs_who_saved(tmp_path):
    stem = "sub-01_task-rest_cond-a_report"
    for rater, verdict in (("QY", "good"), ("AB", "bad")):
        client = FNIRSRatingApp(tmp_path, ["01"], rater).app.test_client()
        client.post("/sub-01/save_ratings", json={"id": stem, "ratings": {"Final": verdict}})
    nirs = tmp_path / "sub-01" / "nirs"

    assert _json(nirs / "sub-01_task-rest_cond-a_rater-QY_desc-rating_qc.json")["ratings"] == {
        "Final": "good"}
    assert _json(nirs / "sub-01_task-rest_cond-a_rater-AB_desc-rating_qc.json")["ratings"] == {
        "Final": "bad"}
    log = (tmp_path / "logs" / "group_ratings.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["rater"] for line in log] == ["QY", "AB"]
    assert client.get(f"/sub-01/load_ratings/{stem}").get_json()["ratings"] == {"Final": "bad"}


def test_the_dyad_viewer_writes_each_member_s_decisions_under_the_rater(tmp_path):
    html = tmp_path / "qc" / "group-G1" / report_name("group-G1_task-rest", desc="raw")
    tree = tmp_path / "nirspipe"
    client = HyperRatingApp(html, tmp_path / "qc", ["sub-01", "sub-02"],
                            decisions_dir=tree, rater="QY").app.test_client()

    client.post("/save_hyper_decisions", json={"sub-01": {"S2_D2 hbo": "bad"}})
    client.post("/save_hyper_ratings", json={"ratings": {"Final": "good"}})

    assert _json(tree / "sub-01" / "nirs" / "sub-01_task-rest_rater-QY_desc-rawdecision_qc.json") == {
        "_hyper": {"S2_D2 hbo": "bad"}}
    assert _json(tmp_path / "qc" / "group-G1" / "nirs"
                 / "group-G1_task-rest_rater-QY_desc-rawrating_qc.json")["ratings"] == {
        "Final": "good"}
    assert client.get("/load_decisions").get_json() == {"sub-01": {"S2_D2 hbo": "bad"},
                                                         "sub-02": {}}


@pytest.mark.parametrize("name", ["Q Y", "Q-Y", "Q_Y", "Qiu.Yu", "", "秋雨"])
def test_a_name_that_cannot_go_into_a_file_name_is_refused(name, capsys):
    with pytest.raises(SystemExit) as stop:
        rate._build_parser().parse_args(["raw", "out", "--participant-label", "01",
                                         "--rater", name])
    assert stop.value.code == 2
    assert "letters and digits" in capsys.readouterr().err


def test_each_viewer_subcommand_takes_a_rater_and_hands_it_to_its_viewer(tmp_path):
    _raw_page(tmp_path)
    args = rate._build_parser().parse_args(
        ["raw", str(tmp_path), "--participant-label", "01", "--rater", "QY"])
    [(viewer, _label, _entries)] = rate.open_raw(
        args.output_dir, args.participant_label, None, None, None, rater=args.rater)
    assert viewer.rater == "QY"
    for sub in (["rate", "out"], ["hyper", "out", "--group-id", "G1", "--task-label", "rest",
                                  "--pairs-csv", "p.csv"]):
        assert rate._build_parser().parse_args([*sub, "--rater", "QY"]).rater == "QY"
        assert rate._build_parser().parse_args(sub).rater is None


def test_the_raw_qc_page_offers_an_optional_rater_field_that_reaches_the_viewer(tmp_path):
    from nirspipe.interface.server import WebApp

    client = WebApp().flask.test_client()
    page = client.get("/raw-qc/rate-raw/").get_data(as_text=True)
    assert 'name="--rater"' in page
    _raw_page(tmp_path)
    form = {"output_dir": str(tmp_path), "--participant-label": "01", "--rater": "QY"}
    [opened] = client.post("/run/nirspipe-rate/raw/", data=form).get_json()["opened"]

    client.post(f"{opened['href']}save_raw_ratings", json={"ratings": {"Final": "good"}})
    assert (tmp_path / "sub-01" / "nirs" / "sub-01_task-rest_rater-QY_desc-rawrating_qc.json").exists()
