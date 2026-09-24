"""`fnirs-pipe participant` on a subject recorded in two sessions, without `--session-label`.

Each config takes its session from the file being processed, not from the `--session-label`
loop (None when the flag is not given), so every output keeps its `ses-` entity and neither
session overwrites the other.
"""

import shutil
import sys

import pytest

from fnirs_pipe.cli.workflows import run_participant_level
from tests._synth import make_bids_dataset

_ARGS = dict(
    analysis_level="participant", session_label=None, task_label=["tapping"],
    participant_label=["01"], bids_filter_file=None, work_dir=None, verbose=False,
    skip_bids_validation=True, ignore=None, n_jobs=1, dry_run=False, no_report=True,
    mode=None, config=None, dpf=[6.0, 6.0], sci_threshold=0.8, motion_correction="tddr",
    cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5,
)


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("sessions")
    bids = make_bids_dataset(root, subjects=("01",), tasks=("tapping",))
    flat = bids / "sub-01" / "nirs"
    for ses in ("a", "b"):
        dest = bids / "sub-01" / f"ses-{ses}" / "nirs"
        dest.mkdir(parents=True)
        for f in flat.iterdir():
            shutil.copy2(f, dest / f.name.replace("sub-01_", f"sub-01_ses-{ses}_", 1))
    shutil.rmtree(flat)

    out = root / "out"
    original = sys.argv
    sys.argv = ["fnirs-pipe", str(bids), str(out), "participant"]
    try:
        run_participant_level({**_ARGS, "bids_dir": bids, "output_dir": out})
    finally:
        sys.argv = original
    return out


@pytest.mark.parametrize("ses", ["a", "b"])
def test_each_session_keeps_its_own_outputs(out_dir, ses):
    nirs = out_dir / "sub-01" / f"ses-{ses}" / "nirs"
    stem = f"sub-01_ses-{ses}_task-tapping"
    assert (nirs / f"{stem}_desc-preproc_nirs.snirf").exists()
    assert (nirs / f"{stem}_desc-sqm_qc.json").exists()


def test_nothing_lands_in_a_session_less_folder(out_dir):
    assert not (out_dir / "sub-01" / "nirs").exists()


@pytest.mark.parametrize("ses", ["a", "b"])
def test_each_session_run_gets_its_provenance_graph(out_dir, ses):
    figures = out_dir / "sub-01" / "figures"
    assert (figures / f"sub-01_ses-{ses}_task-tapping_desc-provenance_nirs.png").exists()
