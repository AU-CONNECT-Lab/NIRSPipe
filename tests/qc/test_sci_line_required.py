"""The SCI line has one source: the command that screens is given it, everything after reads it back."""

import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.cli import rate as rate_cli
from nirspipe.cli.hyper import _parsers as hyper_parsers
from nirspipe.interface.commands import parser_for
from nirspipe.interface.forms import fields, required_first
from nirspipe.qc.common.channel_table import format_rows
from nirspipe.qc.rating.app import HyperRatingApp, RawRatingApp

SCREEN = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
          "--resp-l-freq", "0.1", "--resp-h-freq", "0.5"]


def test_the_participant_level_refuses_to_run_without_a_line(capsys):
    positionals = ["/bids", "/out", "participant", "--participant-label", "01"]
    with pytest.raises(SystemExit):
        qc_cli.main([*positionals, *SCREEN])
    assert "Missing option '--sci-threshold'" in capsys.readouterr().err
    args = qc_cli._build_parser().parse_args([*positionals, *SCREEN, "--sci-threshold", "0.7"])
    assert args.sci_threshold == 0.7


def test_the_dyad_analysis_takes_no_line_of_its_own():
    parser = hyper_parsers()["nirspipe-hyper"]
    assert "--sci-threshold" not in parser._option_string_actions


@pytest.mark.parametrize("command, positionals", [
    ("raw", ["/out", "--participant-label", "01"]),
    ("hyper", ["/out", "--group-id", "A", "--task-label", "tap", "--pairs-csv", "p.csv"]),
])
def test_the_rater_s_line_is_off_unless_given(command, positionals):
    parser = rate_cli._build_parser()
    assert parser.parse_args([command, *positionals]).sci_threshold is None
    assert parser.parse_args([command, *positionals, "--sci-threshold", "0.6"]).sci_threshold == 0.6


def test_with_no_line_the_sci_cell_is_uncoloured_and_keeps_its_number():
    row = {"name": "S1_D1 760", "sci_win": 0.412, "is_bad": False}
    assert format_rows([row], 0.8)[0]["sci_win_cls"] == "bad"
    unlined = format_rows([row], None)[0]
    assert unlined["sci_win_cls"] == ""
    assert unlined["sci_win_value"] == pytest.approx(0.412)


def test_the_rating_servers_hand_the_page_the_rater_s_line(tmp_path):
    html = tmp_path / "sub-01_task-rest_desc-raw_report.html"
    html.write_text("<html></html>", encoding="utf-8")

    def line(server):
        return server.app.test_client().get("/rate_config").get_json()["sci_threshold"]

    assert line(RawRatingApp(html, tmp_path)) is None
    assert line(RawRatingApp(html, tmp_path, 0.6)) == 0.6
    assert line(HyperRatingApp(html, tmp_path, ["01", "02"], 0.6)) == 0.6


def test_the_gui_shows_the_line_among_the_participant_level_settings():
    groups = dict(required_first(fields(parser_for("nirspipe-qc"))))
    assert "--sci-threshold" in [f.name for f in groups["participant level"]]

