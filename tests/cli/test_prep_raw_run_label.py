"""Raw QC writes one report per recording, which the subject index and the cohort page link."""

import json
import re

import pytest

from nirspipe.cli import qc
from nirspipe.io.naming import report_name
from tests._synth import _write_dataset_root, _write_subject, synth_raw

PHYS = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5", "--sci-threshold", "0.8",
        "--resp-l-freq", "0.1", "--resp-h-freq", "0.5", "--skip-bids-validation"]


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    """sub-01, task-tapping, runs 1 and 2."""
    bids = tmp_path_factory.mktemp("runs") / "bids"
    _write_dataset_root(bids, ["01"])
    for run in ("1", "2"):
        snirf = _write_subject(bids, "01", "tapping", synth_raw("01", "tapping", duration=120.0,
                                                                seed=int(run)))
        for path in list(snirf.parent.glob("sub-01_task-tapping_*")):
            if "_run-" not in path.name:
                path.rename(path.with_name(path.name.replace("_task-tapping_", f"_task-tapping_run-{run}_")))
    return bids


def _run_labels(html):
    return json.loads(re.search(r"var _RUN_LABELS\s*=\s*(\[.*?\]);", html.read_text(encoding="utf-8")).group(1))


RUNS = ("sub-01_task-tapping_run-1", "sub-01_task-tapping_run-2")


def test_every_recording_gets_a_report_of_its_own(two_runs, tmp_path):
    out = tmp_path / "qc"
    qc.main([str(two_runs), str(out), "participant", "--participant-label", "01", *PHYS])
    for label in RUNS:
        assert _run_labels(out / "sub-01" / report_name(label, desc="raw")) == [label]
    assert not (out / "sub-01" / report_name("sub-01_task-tapping", desc="raw")).exists()


def test_the_subject_index_and_the_cohort_link_every_raw_report(two_runs, tmp_path):
    out = tmp_path / "qc"
    qc.main([str(two_runs), str(out), "participant", "--participant-label", "01", *PHYS])
    qc.main([str(two_runs), str(out), "group", "--skip-bids-validation"])
    index = (out / "sub-01" / "sub-01_desc-index_report.html").read_text(encoding="utf-8")
    for label in RUNS:
        assert f'href="{report_name(label, desc="raw")}' in index
    # the cohort page opens a run's raw report by the row's bids_name
    rows = (out / "desc-subjects_qc.tsv").read_text(encoding="utf-8").splitlines()[1:]
    names = sorted(row.split("\t")[0] for row in rows)
    assert names == list(RUNS)
    assert all((out / "sub-01" / report_name(name, desc="raw")).exists() for name in names)


def test_a_chosen_run_rewrites_its_own_report_only(two_runs, tmp_path):
    out = tmp_path / "qc"
    qc.main([str(two_runs), str(out), "participant", "--participant-label", "01", *PHYS])
    other = out / "sub-01" / report_name(RUNS[0], desc="raw")
    before = other.read_text(encoding="utf-8")

    qc.main([str(two_runs), str(out), "participant", "--participant-label", "01", "--run-label", "2", *PHYS])

    assert other.read_text(encoding="utf-8") == before
    assert _run_labels(out / "sub-01" / report_name(RUNS[1], desc="raw")) == [RUNS[1]]


def test_a_run_label_no_recording_has_writes_no_report(two_runs, tmp_path, capsys):
    out = tmp_path / "qc"
    try:
        qc.main([str(two_runs), str(out), "participant", "--participant-label", "01", "--run-label", "9", *PHYS])
    except SystemExit:
        pass
    assert not list(out.glob("sub-01/*_desc-raw_report.html"))
    assert "no SNIRF files found for sub-01" in capsys.readouterr().err
