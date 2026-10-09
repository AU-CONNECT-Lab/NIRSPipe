"""The SCI line has one source: the command that screens is given it, everything after reads it back."""

import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.cli import rate as rate_cli
from nirspipe.cli.hyper import _parsers as hyper_parsers
from nirspipe.interface.cli_args import missing_raw_qc
from nirspipe.qc.common.channel_table import format_rows
from nirspipe.qc.rating.app import HyperRatingApp, RawRatingApp

SCREEN = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5"]


@pytest.mark.parametrize("command, positionals", [
    ("prep-raw", ["/bids", "/out", "--participant-label", "01"]),
    ("hyper-raw", ["/bids", "/out", "group", "--pairs-csv", "pairs.csv"]),
])
def test_a_screening_command_refuses_to_run_without_a_line(command, positionals):
    parser = qc_cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([command, *positionals, *SCREEN])
    args = parser.parse_args([command, *positionals, *SCREEN, "--sci-threshold", "0.7"])
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
    assert RawRatingApp(html, tmp_path).app.test_client().get("/rate_config").get_json() \
        == {"sci_threshold": None}
    assert RawRatingApp(html, tmp_path, 0.6).app.test_client().get("/rate_config").get_json() \
        == {"sci_threshold": 0.6}
    hyper = HyperRatingApp(html, tmp_path, ["01", "02"], 0.6)
    assert hyper.app.test_client().get("/rate_config").get_json() == {"sci_threshold": 0.6}


@pytest.mark.parametrize("command", ["prep-raw", "hyper-raw"])
def test_the_gui_asks_for_the_line_before_running_a_raw_report(command):
    opts = {"bids_dir": "/bids", "output_dir": "/out", "dpf": 6.0, "cardiac_l": 0.7,
            "cardiac_h": 1.5, "subject": "01", "pairs_csv": "p.csv"}
    assert missing_raw_qc(command, opts) == "SCI threshold is required."
    assert missing_raw_qc(command, dict(opts, sci_threshold=0.8)) is None


def test_the_alignment_table_colours_against_the_page_s_line_or_not_at_all():
    from nirspipe.interface.callbacks.hyper_align_callbacks import _build_ha_decisions_table

    def red(line):
        table = _build_ha_decisions_table(["A"], ["S1_D1"], {"A": {"S1_D1": 0.7}}, {}, line)
        return "#dc3545" in str(table)

    assert red(0.8) and not red(0.6) and not red(None)
