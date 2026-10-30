"""Each participant level goes on in a tree that holds the other command's records, keeps both, and says once that they belong apart."""

import logging

from nirspipe.cli import qc as qc_cli
from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _warn_on_split_tree
from nirspipe.qc.subject.sqm_record import RECORD_SUFFIXES, record_path

from tests._raw_qc import SCREENING
from tests._synth import make_bids_dataset

_MIXED = "meant for trees of their own"
_FLAGS = (*SCREENING, "--skip-bids-validation")


def _plant(out, desc):
    nirs = out / "sub-01" / "nirs"
    nirs.mkdir(parents=True)
    path = nirs / ("sub-01_task-other" + RECORD_SUFFIXES[desc])
    path.write_text("{}", encoding="utf-8")
    return path


def _bids(tmp_path):
    return make_bids_dataset(tmp_path, subjects=("01",), tasks=("rest",), duration=60.0)


def test_raw_qc_in_an_analysis_tree_says_so_once_and_keeps_both(tmp_path, caplog):
    bids, out = _bids(tmp_path), tmp_path / "out"
    planted = _plant(out, "sqm")
    with caplog.at_level(logging.WARNING, logger="nirspipe.cli"):
        qc_cli.main([str(bids), str(out), "participant", *_FLAGS])
    assert caplog.text.count(_MIXED) == 1
    assert planted.read_text(encoding="utf-8") == "{}"
    assert list(out.rglob("*_desc-sqmraw_qc.json"))


def test_the_analysis_in_a_raw_qc_tree_says_so_once_and_keeps_both(tmp_path, caplog):
    bids, out = _bids(tmp_path), tmp_path / "out"
    planted = _plant(out, "sqmraw")
    with caplog.at_level(logging.WARNING, logger="nirspipe.cli"):
        run_cli.main([str(bids), str(out), "participant", *_FLAGS, "--no-report"])
    assert caplog.text.count(_MIXED) == 1
    assert planted.read_text(encoding="utf-8") == "{}"
    assert list(out.rglob("*_desc-sqm_qc.json"))


def test_a_tree_of_one_command_says_nothing(tmp_path, caplog):
    bids, out = _bids(tmp_path), tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger="nirspipe.cli"):
        qc_cli.main([str(bids), str(out), "participant", *_FLAGS])
        qc_cli.main([str(bids), str(out), "participant", *_FLAGS])
    assert _MIXED not in caplog.text


def test_a_record_in_a_subtree_is_sent_to_its_own_group_level(tmp_path, caplog):
    nirs = tmp_path / "qc" / "sub-01" / "nirs"
    nirs.mkdir(parents=True)
    record_path(nirs, "sub-01_task-rest").write_text("{}")
    with caplog.at_level(logging.WARNING):
        _warn_on_split_tree(tmp_path)
    assert "one output directory" not in caplog.text
    assert "its own `group` level" in caplog.text
