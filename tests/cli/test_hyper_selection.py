"""The dyad commands take the BIDS Apps selection flags: --participant-label and --task-label."""

import pytest

from fnirs_pipe.cli import _shared
from fnirs_pipe.cli.hyper import _parsers, _select_groups


@pytest.fixture
def pairs_csv(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("group_id,subject_id,task\n"
                    "G01,sub-01,main\nG01,sub-02,main\n"
                    "G02,sub-03,main\nG02,sub-04,main\n"
                    "G03,01,rest\nG03,05,rest\n")
    return path


def test_participant_label_keeps_every_group_the_subject_is_in(pairs_csv):
    groups = _select_groups(pairs_csv, None, None, ["01"])
    assert sorted(groups) == [("G01", "main"), ("G03", "rest")]


def test_the_partner_comes_along(pairs_csv):
    groups = _select_groups(pairs_csv, None, None, ["02"])
    assert [e.subject_id for e in groups[("G01", "main")]] == ["sub-01", "sub-02"]


def test_participant_label_combines_with_task_label(pairs_csv):
    assert sorted(_select_groups(pairs_csv, None, ["main"], ["01"])) == [("G01", "main")]


def test_unknown_participant_exits(pairs_csv, capsys):
    with pytest.raises(SystemExit) as exc:
        _select_groups(pairs_csv, None, None, ["99"])
    assert exc.value.code == 1
    assert "participant(s) ['99']" in capsys.readouterr().err


def test_participant_label_parses_both_spellings_and_strips_sub():
    p = _shared.pairs_selection()
    a = p.parse_args(["--pairs-csv", "x.csv", "--participant-label", "sub-01", "02"])
    b = p.parse_args(["--pairs-csv", "x.csv", "--participant_label", "01", "--participant_label", "02"])
    assert a.participant_label == b.participant_label == ["01", "02"]


@pytest.mark.parametrize("flag", ["--task-label", "--task_label", "--task"])
def test_groupnull_takes_task_label(flag):
    args = _parsers()["fnirs-hyper-groupnull"].parse_args(["out", "group", flag, "task-main"])
    assert args.task == "main"
