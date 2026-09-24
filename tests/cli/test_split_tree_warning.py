"""`fnirs-pipe group` says so when quality records sit in a subtree it does not aggregate."""

from fnirs_pipe.cli.workflows import _warn_on_split_tree
from fnirs_pipe.qc.subject.sqm_record import record_path

_LINE = "not part of this cohort page"


def test_an_empty_subfolder_says_nothing(tmp_path, caplog):
    (tmp_path / "qc").mkdir()
    with caplog.at_level("WARNING"):
        _warn_on_split_tree(tmp_path)
    assert _LINE not in caplog.text


def test_a_record_in_a_subtree_is_reported(tmp_path, caplog):
    nirs = tmp_path / "qc" / "sub-01" / "nirs"
    nirs.mkdir(parents=True)
    record_path(nirs, "sub-01_task-rest").write_text("{}")
    with caplog.at_level("WARNING"):
        _warn_on_split_tree(tmp_path)
    assert _LINE in caplog.text
