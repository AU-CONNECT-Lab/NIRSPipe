"""A raw record with conditions reads back, so its per-condition pages are written."""

import mne

from fnirs_pipe.cli import qc as qc_cli
from fnirs_pipe.io.derivatives import write_dataset_description
from fnirs_pipe.qc.subject.prep_raw_report import _process_run
from fnirs_pipe.qc.subject.record_io import read_record
from tests._synth import _write_dataset_root, _write_subject, synth_raw


def _bids(tmp_path):
    bids = tmp_path / "bids"
    _write_dataset_root(bids, ["01"])
    raw = synth_raw("01", "tap", duration=300.0)
    raw.set_annotations(mne.Annotations([20.0, 150.0], [100.0, 100.0], ["rest", "talk"]))
    return bids, _write_subject(bids, "01", "tap", raw)


def test_the_record_reads_back_with_each_condition_share(tmp_path):
    bids, path = _bids(tmp_path)
    out = tmp_path / "out"
    write_dataset_description(out, source=bids)
    _, ctx = _process_run({"label": "sub-01_task-tap", "snirf_path": path, "session": None},
                          0.8, out / "sub-01", 0.7, 1.5, [6.0])
    record = read_record(ctx["sqm_path"])
    assert set(record["by_condition"]) == {"rest", "talk"}
    for entry in record["by_condition"].values():
        assert entry["per_channel"]["good_frac_per_channel"]
    assert "good_frac_by_condition" not in record["raw"]


def test_by_condition_writes_a_page_per_condition(tmp_path):
    bids, _ = _bids(tmp_path)
    out = tmp_path / "out"
    qc_cli.main(["prep-raw", str(bids), str(out), "--participant-label", "01",
                 "--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
                 "--by-condition", "--skip-bids-validation"])
    pages = sorted(p.name for p in out.rglob("*cond-*_report.html"))
    assert len(pages) == 2
    assert any("cond-rest" in p for p in pages) and any("cond-talk" in p for p in pages)
