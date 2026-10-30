"""The dyad raw page draws each member's own prepared files rather than preparing the members again."""

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.cli import run as run_cli
from nirspipe.cli.hyper import _select_groups
from nirspipe.exceptions import StageError
from nirspipe.io.snirf import read_snirf
from nirspipe.qc.hyper import hyper_report

from tests._raw_qc import SCREENING
from tests._synth import BAD_PAIR, _write_dataset_root, _write_subject, synth_raw

# leaves a rejected channel as it was, so a correction run without the rejections differs
FLAGS = (*SCREENING, "--motion-correction", "spline_sg", "--skip-bids-validation")
REJECTED = f"S{BAD_PAIR + 1}_D{BAD_PAIR + 1}"


def _settings(bids, out):
    return SimpleNamespace(bids_dir=bids, output_dir=out, no_align=False, normalize=False,
                           tstart=None, tend=None, roi_map=None)


def _draw(bids, out, pairs, monkeypatch):
    drawn = {}
    monkeypatch.setattr(hyper_report, "build_hyper_report", lambda **kw: drawn.update(kw))
    ((gid, task, ses), members), = _select_groups(pairs, None, ["hold"], None).items()
    qc_cli._raw_one_group(_settings(bids, out), gid, task, ses, members)
    return drawn


def _artifact(x):
    """A half-second spike at 200 s and a step at 300 s, on the 10 Hz synthetic clock."""
    x = x.copy()
    x[2000:2005] *= 1.5
    x[3000:] *= 0.7
    return x


def _dyad(root):
    bids = root / "bids_hyper"
    _write_dataset_root(bids, ["11", "12"])
    for sid in ("11", "12"):
        raw = synth_raw(sid, "hold")
        # on the pair screening rejects, so only a correction that knows it leaves it alone
        raw.apply_function(_artifact, picks=[f"{REJECTED} 760", f"{REJECTED} 850"])
        _write_subject(bids, sid, "hold", raw)
    pairs = root / "pairs.csv"
    rows = "".join(f"G01,sub-{sid},hold\n" for sid in ("11", "12"))
    pairs.write_text("group_id,subject_id,task\n" + rows, encoding="utf-8")
    return bids, pairs


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    root = tmp_path_factory.mktemp("dyad_stages")
    bids, pairs = _dyad(root)
    run_cli.main([str(bids), str(root / "pipe"), "participant", *FLAGS, "--no-report"])
    qc_cli.main([str(bids), str(root / "qc"), "participant", *FLAGS])
    return bids, pairs, root


def _stage(root, sid, desc):
    (path,) = (root / "pipe").rglob(f"{sid}_task-hold_desc-{desc}_nirs.snirf")
    return read_snirf(path)


def test_the_corrected_side_is_each_member_s_own_correction(trees, monkeypatch):
    bids, pairs, root = trees
    drawn = _draw(bids, root / "qc", pairs, monkeypatch)
    assert set(drawn["full_after"]) == {"sub-11", "sub-12"}
    for sid, after in drawn["full_after"].items():
        expected = _stage(root, sid, "motcorrected")
        assert set(expected.info["bads"]) == {f"{REJECTED} 760", f"{REJECTED} 850"}
        assert after.ch_names == expected.ch_names
        assert np.array_equal(after.get_data(), expected.get_data())


def test_the_uncorrected_side_is_each_member_s_screened_recording(trees, monkeypatch):
    bids, pairs, root = trees
    drawn = _draw(bids, root / "qc", pairs, monkeypatch)
    for sid, before in drawn["full_raws"].items():
        expected = _stage(root, sid, "sci")
        assert np.array_equal(before.get_data(), expected.get_data())


def test_a_tree_without_its_stage_files_is_refused_with_the_rerun(trees, monkeypatch):
    bids, pairs, root = trees
    # beside the original, so the tree's relative link to the raw dataset still resolves
    copy = root / "qc_without_stages"
    shutil.copytree(root / "qc", copy)
    shutil.rmtree(copy / "stages")
    with pytest.raises(StageError, match="nirspipe-qc"):
        _draw(bids, copy, pairs, monkeypatch)
