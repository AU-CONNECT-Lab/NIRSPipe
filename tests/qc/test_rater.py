"""--rater: each rater's ratings, channel decisions and marked stretches kept in files of their own."""

import json

import pandas as pd
import pytest

from nirspipe.cli import rate
from nirspipe.cli.workflows import _bad_channels_for, _keep_channels_for
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
    # the members' raw pages sit in the tree the dyad page was written into
    tree = tmp_path / "qc"
    client = HyperRatingApp(html, tree, ["sub-01", "sub-02"], rater="QY").app.test_client()

    client.post("/save_hyper_decisions", json={"sub-01": {"S2_D2 hbo": "bad"}})
    client.post("/save_hyper_ratings", json={"ratings": {"Final": "good"}})

    assert _json(tree / "sub-01" / "nirs" / "sub-01_task-rest_rater-QY_desc-rawdecision_qc.json") == {
        "_hyper": {"S2_D2 hbo": "bad"}}
    assert _json(tmp_path / "qc" / "group-G1" / "nirs"
                 / "group-G1_task-rest_rater-QY_desc-rawrating_qc.json")["ratings"] == {
        "Final": "good"}
    assert client.get("/load_decisions").get_json() == {"sub-01": {"S2_D2 hbo": "bad"},
                                                         "sub-02": {}}


def test_the_dyad_viewer_takes_no_second_tree_for_the_decisions(capsys):
    with pytest.raises(SystemExit) as stop:
        rate._build_parser().parse_args(["hyper", "out", "--group-id", "G1", "--task-label",
                                         "rest", "--pairs-csv", "p.csv",
                                         "--derivatives-dir", "nirspipe"])
    assert stop.value.code == 2
    assert "--derivatives-dir" in capsys.readouterr().err


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


# ---- the exports and what nirspipe reads ----

def _two_raters(tree):
    _rate_raw(tree, "QY", "good", "bad", [1.0, 2.0])
    _rate_raw(tree, "rater2", "bad", "good", [5.0, 6.0])


def test_the_export_names_each_row_s_rater(tmp_path):
    from nirspipe.qc.rating.export import export_ratings, export_spans

    _two_raters(tmp_path)
    channels, verdicts, conflicts = export_ratings([tmp_path], tmp_path / "out")
    spans = export_spans([tmp_path], tmp_path / "out")
    read = lambda path: pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    assert conflicts == []
    assert read(channels)[["rater", "bad_channels", "keep_channels"]].values.tolist() == [
        ["QY", "S1_D1", ""], ["rater2", "", "S1_D1"]]
    assert read(verdicts)[["rater", "verdict"]].values.tolist() == [["QY", "good"],
                                                                    ["rater2", "bad"]]
    assert read(spans)[["rater", "onset", "duration"]].values.tolist() == [
        ["QY", "1.0", "1.0"], ["rater2", "5.0", "1.0"]]


def test_export_with_a_rater_keeps_only_that_rater_s_files(tmp_path, capsys):
    _two_raters(tmp_path)
    rate.main(["export", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--rater", "rater2"])
    table = pd.read_csv(tmp_path / "out" / "desc-ratedchannels_qc.tsv", sep="\t", dtype=str,
                        keep_default_na=False)

    assert table[["rater", "keep_channels"]].values.tolist() == [["rater2", "S1_D1"]]
    assert _keep_channels_for(str(tmp_path / "out" / "desc-ratedchannels_qc.tsv"), "01",
                              {"task": "rest"}) == ["S1_D1"]
    assert "several raters" not in capsys.readouterr().err


def test_export_of_every_rater_says_nirspipe_takes_one(tmp_path, capsys):
    _two_raters(tmp_path)
    _rate_raw(tmp_path, None, "good", "bad", [8.0, 9.0])
    rate.main(["export", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    err = capsys.readouterr().err

    assert "several raters' decisions ((saved without --rater), QY, rater2)" in err
    assert "--rater NAME" in err


def test_export_of_a_rater_with_no_files_warns(tmp_path, capsys):
    _two_raters(tmp_path)
    rate.main(["export", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--rater", "XY"])
    assert "no files of rater XY" in capsys.readouterr().err


def _table(tmp_path, text, name="table.tsv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("raters", [("QY", "rater2"), ("QY", "")])
def test_a_channel_table_holding_several_raters_is_refused(tmp_path, raters):
    path = _table(tmp_path, "participant_id\trater\tbad_channels\n"
                            f"sub-01\t{raters[0]}\tS1_D1\nsub-01\t{raters[1]}\tS2_D2\n")
    with pytest.raises(ValueError, match=r"several raters' decisions .*--rater NAME"):
        _bad_channels_for(str(path), "01")


def test_a_channel_table_of_one_rater_or_settled_reads_as_before(tmp_path):
    one = _table(tmp_path, "participant_id\trater\tbad_channels\n"
                           "sub-01\tQY\tS1_D1\nsub-01\tQY\tS2_D2\n", "one.tsv")
    settled = _table(tmp_path, "participant_id\trater\tbad_channels\n"
                               "sub-01\t\tS1_D1\nsub-02\t\tS2_D2\n", "settled.tsv")
    assert _bad_channels_for(str(one), "01") == ["S1_D1", "S2_D2"]
    assert _bad_channels_for(str(settled), "01") == ["S1_D1"]


def test_a_span_table_holding_several_raters_stops_the_run(mini_bids, tmp_path):
    from nirspipe.cli.workflows import _refuse_bad_span_rows
    from nirspipe.io.bids import get_layout

    path = _table(tmp_path, "task\trater\tonset\tduration\nrest\tQY\t0\t10\nrest\tAB\t20\t10\n")
    with pytest.raises(SystemExit, match=r"several raters' decisions \(AB, QY\)"):
        _refuse_bad_span_rows(str(path), get_layout(mini_bids), flag="--bad-spans")


def test_a_channel_table_holding_several_raters_stops_the_run(mini_bids, tmp_path):
    from nirspipe.cli.workflows import _refuse_unmatched_bad_channel_rows
    from nirspipe.io.bids import get_layout

    path = _table(tmp_path, "participant_id\trater\tbad_channels\n"
                            "sub-01\tQY\tS1_D1\nsub-01\tAB\tS1_D1\n")
    with pytest.raises(SystemExit, match="several raters"):
        _refuse_unmatched_bad_channel_rows(str(path), get_layout(mini_bids))
