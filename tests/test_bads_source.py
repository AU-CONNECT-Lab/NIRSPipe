"""Where the inter-brain metrics read their rejected channels from.

Rejection is decided in prep and written to the ``desc-sci`` sidecar. The channel-metrics
CSV carries the same set, but the *report* writes that one, so `--no-report` leaves a tree
with sidecars and no CSVs.

Reading the CSV alone was silent when it was absent. It did not matter for `--desc preproc`,
whose SNIRF sidecar restores ``info["bads"]`` on read, but `desc-errts` is written with an
empty ``info["bads"]`` and the CSV was its only other record: a `--no-report` prep followed
by an `errts` dyad analysis excluded nothing at all, and every coherence value on a dyad
with a rejected channel was wrong. `--bads-scope subject` was worse, since it reads every
run's CSV and so degraded to nothing whatever the stage.

These tests pin the source, the fallback, and the warning.
"""

import json

import pandas as pd
import pytest

from fnirs_pipe.pipeline.hyperscanning import GroupEntry, load_group_sqm

BADS_TAP = ["S6_D5 760", "S6_D5 850"]
BADS_REST = ["S7_D6 760"]


@pytest.fixture
def nirs_dir(tmp_path):
    d = tmp_path / "sub-01" / "nirs"
    d.mkdir(parents=True)
    return tmp_path


def _sidecar(root, task, bads):
    (root / "sub-01" / "nirs" / f"sub-01_task-{task}_desc-sci_nirs.json").write_text(
        json.dumps({"step": "sci", "bad_channels": bads,
                    "sci_scores": {"S1_D1 760": 0.9, "S6_D5 760": 0.2}}))


def _csv(root, task, bads):
    names = ["S1_D1 760", "S6_D5 760", "S6_D5 850", "S7_D6 760"]
    pd.DataFrame({"name": names, "sci": [0.9, 0.2, 0.2, 0.3],
                  "is_bad": [n in bads for n in names]}).to_csv(
        root / "sub-01" / "nirs" / f"sub-01_task-{task}_channel_metrics.csv", index=False)


def _load(root, scope="run", task="tap"):
    return load_group_sqm(root, [GroupEntry("G1", "sub-01", task)],
                          bads_scope=scope)["sub-01"]


def test_a_no_report_tree_still_rejects_channels(nirs_dir):
    """The failure this exists for: sidecars on disk, no CSV anywhere."""
    _sidecar(nirs_dir, "tap", BADS_TAP)
    assert _load(nirs_dir)["bad_channels"] == BADS_TAP


def test_the_per_channel_sci_comes_from_the_sidecar_too(nirs_dir):
    _sidecar(nirs_dir, "tap", BADS_TAP)
    assert _load(nirs_dir)["sci_per_channel"]["S6_D5 760"] == 0.2


def test_subject_scope_unions_over_runs_without_any_csv(nirs_dir):
    _sidecar(nirs_dir, "tap", BADS_TAP)
    _sidecar(nirs_dir, "rest", BADS_REST)
    sqm = _load(nirs_dir, scope="subject")
    assert sqm["bad_channels"] == sorted(BADS_TAP + BADS_REST)
    assert sqm["bad_channel_sources"]["S7_D6 760"] == ["rest"]


def test_the_csv_still_works_on_a_tree_that_has_no_sidecars(nirs_dir):
    """Older outputs, and anything written before prep recorded the sidecar."""
    _csv(nirs_dir, "tap", BADS_TAP)
    assert _load(nirs_dir)["bad_channels"] == BADS_TAP


def test_the_sidecar_wins_when_both_are_present(nirs_dir):
    """They agree in practice, so the test uses a disagreement to name the winner."""
    _sidecar(nirs_dir, "tap", BADS_TAP)
    _csv(nirs_dir, "tap", BADS_REST)
    assert _load(nirs_dir)["bad_channels"] == BADS_TAP


def test_a_subject_with_no_record_at_all_is_loud(nirs_dir, caplog):
    with caplog.at_level("WARNING"):
        sqm = _load(nirs_dir)
    assert not sqm.get("bad_channels")
    assert "every bad channel enters the inter-brain metrics" in caplog.text


def test_a_subject_missing_only_this_task_is_loud(nirs_dir, caplog):
    _sidecar(nirs_dir, "rest", BADS_REST)
    with caplog.at_level("WARNING"):
        sqm = _load(nirs_dir, task="tap")
    assert not sqm.get("bad_channels")
    assert "none for task-tap" in caplog.text


def test_the_log_names_the_source_it_read(nirs_dir, caplog):
    _sidecar(nirs_dir, "tap", BADS_TAP)
    with caplog.at_level("INFO"):
        _load(nirs_dir)
    assert "2 rejected channel(s) from desc-sci sidecar" in caplog.text
