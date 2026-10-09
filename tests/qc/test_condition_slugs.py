"""Two conditions whose file names would be the same are not drawn over each other.

A condition reaches a file name through a slug that keeps letters and digits only, so
"game 1" and "game1" name one file, and so do a second "game1" block ("game1#2") and a
condition called "game12", and the later file would replace the earlier one without a word.

What is measured is keyed by the label itself and keeps both; what is named by the slug is
refused. The dyad report fails the group, since its per-condition tables are named by the
slug too; a subject report draws no per-condition views or pages and says why, the run
itself going ahead. The dyad tables also reserve ``cond-all`` for the file of every
condition.
"""

import json

import mne
import numpy as np
import pytest

from nirspipe.exceptions import StageError
from nirspipe.qc.common.windows import condition_windows, refuse_colliding_labels
from tests._synth import _write_dataset_root, _write_subject, synth_raw


def _raw(descriptions, block=100.0):
    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], 5.0, ["hbo", "hbr"])
    n = int(5.0 * block * (len(descriptions) + 1))
    raw = mne.io.RawArray(np.zeros((2, n)), info, verbose="ERROR")
    onsets = [block * i for i in range(len(descriptions))]
    raw.set_annotations(mne.Annotations(onsets, [block] * len(descriptions), descriptions))
    return raw


def test_distinct_labels_pass():
    refuse_colliding_labels(["rest", "talk"])


def test_two_spellings_of_one_name_are_refused():
    with pytest.raises(StageError, match="game 1.*game1|game1.*game 1"):
        refuse_colliding_labels(["game 1", "game1"])


def test_a_numbered_repeat_colliding_with_another_condition_is_refused():
    labels = [w[0] for w in condition_windows(_raw(["game1", "game1", "game12"]),
                                              min_duration=10.0)]
    with pytest.raises(StageError, match="game12"):
        refuse_colliding_labels(labels)


def test_a_reserved_name_is_refused_only_where_it_is_reserved():
    refuse_colliding_labels(["all"])
    with pytest.raises(StageError, match="cond-all"):
        refuse_colliding_labels(["all"], reserved=("all",))


def test_the_windows_themselves_keep_every_label():
    """The measurement is keyed by the label, so it has nothing to refuse."""
    labels = [w[0] for w in condition_windows(_raw(["game 1", "game1"]), min_duration=10.0)]
    assert labels == ["game 1", "game1"]


def _dyad(tmp_path, descriptions):
    from nirspipe.pipeline.hyper import GroupEntry
    from nirspipe.qc.hyper.hyper_report import build_hyper_post_report

    raws = {sid: _raw(descriptions) for sid in ("sub-01", "sub-02")}
    build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", s, "tap") for s in raws],
        aligned_raws=raws, offsets={s: 0.0 for s in raws}, output_dir=tmp_path,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_chroma=("hbo",), wtc_by_condition=True)


def test_a_condition_named_all_is_refused_by_the_dyad_report(tmp_path):
    with pytest.raises(StageError, match="all"):
        _dyad(tmp_path, ["rest", "all"])


def test_colliding_conditions_are_refused_by_the_dyad_report(tmp_path):
    with pytest.raises(StageError, match="game1"):
        _dyad(tmp_path, ["game 1", "game1"])


def test_a_subject_report_keeps_the_record_and_draws_no_condition_pages(tmp_path):
    from nirspipe.cli import qc as qc_cli

    bids = tmp_path / "bids"
    _write_dataset_root(bids, ["01"])
    raw = synth_raw("01", "tap", duration=300.0, motion_onset=None, bad_pair=None)
    raw.set_annotations(mne.Annotations([20.0, 150.0], [100.0, 100.0], ["game 1", "game1"]))
    _write_subject(bids, "01", "tap", raw)
    out = tmp_path / "out"
    qc_cli.main(["prep-raw", str(bids), str(out), "--participant-label", "01",
                 "--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
                 "--sci-threshold", "0.8", "--by-condition", "--skip-bids-validation"])

    records = list(out.rglob("*_desc-sqmraw_qc.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert set(record["by_condition"]) == {"game 1", "game1"}

    pages = list(out.rglob("*_report.html"))
    assert not [p for p in pages if "cond-" in p.name]
    assert any("game1" in p.read_text(encoding="utf-8")
               and "would be written to one file" in p.read_text(encoding="utf-8")
               for p in pages)


def test_a_pipeline_report_keeps_the_record_and_draws_no_condition_pages(tmp_path):
    from nirspipe.cli import run as run_cli

    bids = tmp_path / "bids"
    _write_dataset_root(bids, ["01"])
    raw = synth_raw("01", "tap", duration=300.0, motion_onset=None, bad_pair=None)
    raw.set_annotations(mne.Annotations([20.0, 150.0], [100.0, 100.0], ["game 1", "game1"]))
    _write_subject(bids, "01", "tap", raw)
    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", "--participant-label", "01",
                  "--dpf", "6", "--sci-threshold", "0.5",
                  "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
                  "--resp-l-freq", "0.1", "--resp-h-freq", "0.5",
                  "--by-condition", "--skip-bids-validation", "--work-dir", str(tmp_path / "w")])

    records = list(out.rglob("*_desc-sqm_qc.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert set(record["by_condition"]) == {"game 1", "game1"}

    pages = list(out.rglob("*_report.html"))
    assert pages and not [p for p in pages if "cond-" in p.name]
    assert any("would be written to one file" in p.read_text(encoding="utf-8") for p in pages)
