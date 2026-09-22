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

from fnirs_pipe.pipeline.hyper import GroupEntry, load_group_sqm
from fnirs_pipe.qc.common.channel_table import CHANNEL_METRICS_SUFFIX

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
        root / "sub-01" / "nirs" / (f"sub-01_task-{task}" + CHANNEL_METRICS_SUFFIX),
        index=False, sep=chr(9))


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


# ---- the preflight summary ----

def _raw(n_pairs=3, bads=()):
    """A haemoglobin Raw with n_pairs S-D pairs, hbo and hbr each."""
    import mne
    import numpy as np

    names, types = [], []
    for i in range(n_pairs):
        names += [f"S{i+1}_D{i+1} hbo", f"S{i+1}_D{i+1} hbr"]
        types += ["hbo", "hbr"]
    info = mne.create_info(names, sfreq=10.0, ch_types=types)
    raw = mne.io.RawArray(np.zeros((len(names), 100)), info, verbose=False)
    raw.info["bads"] = list(bads)
    return raw


def test_the_summary_names_the_channel_budget(capsys):
    from fnirs_pipe.cli.hyper import _quality_summary

    raws = {"sub-01": _raw(3, ["S3_D3 hbo", "S3_D3 hbr"])}
    sqm = {"sub-01": {"sci_per_channel": {"a": 0.9, "b": 0.7},
                      "bad_channel_sources": {"S3_D3 760": ["tap", "rest"]}}}
    _quality_summary(raws, sqm)

    out = capsys.readouterr().out
    assert "sub-01" in out
    assert "bad 1" in out              # one S-D pair, not two chromophore entries
    assert "mean SCI 0.80" in out
    assert "from: rest, tap" in out


def test_the_summary_warns_when_most_of_the_montage_is_gone(capsys):
    from fnirs_pipe.cli.hyper import _quality_summary

    bads = [f"S{i}_D{i} {c}" for i in (1, 2, 3) for c in ("hbo", "hbr")]
    _quality_summary({"sub-02": _raw(4, bads)}, {})
    assert "loses 3 of 4 channel pairs" in capsys.readouterr().err


def test_the_summary_says_nothing_it_does_not_know(capsys):
    """No quality record at all still prints a line, rather than crashing on a missing key."""
    from fnirs_pipe.cli.hyper import _quality_summary

    _quality_summary({"sub-03": _raw(2)}, {})
    out = capsys.readouterr().out
    assert "mean SCI n/a" in out
    assert "from:" not in out


# ---- session trees ----
# A session goes in its own folder (`sub-01/ses-a/nirs`), and every reader that built that
# path by hand instead looked in `sub-01/nirs` and found nothing there. The recording itself
# raised "directory not found", so a dyad analysis on a session tree did not start at all;
# with the session folder reached but the quality record still read from the flat path, it
# started and rejected no channel. Both halves are pinned here.

def _ses_sidecar(root, session, task, bads):
    d = root / "sub-01" / f"ses-{session}" / "nirs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"sub-01_ses-{session}_task-{task}_desc-sci_nirs.json").write_text(
        json.dumps({"step": "sci", "bad_channels": bads,
                    "sci_scores": {"S1_D1 760": 0.9, "S6_D5 760": 0.2}}))


def test_a_session_tree_rejects_channels_when_the_entry_names_the_session(tmp_path):
    _ses_sidecar(tmp_path, "a", "tap", BADS_TAP)
    sqm = load_group_sqm(tmp_path, [GroupEntry("G1", "sub-01", "tap", session="a")])["sub-01"]
    assert sqm["bad_channels"] == BADS_TAP


def test_one_unnamed_session_is_still_found(tmp_path):
    """The group CSV has no session column, which is the common case: there is only one, and
    nothing about the study needs naming it. The record is still the subject's own."""
    _ses_sidecar(tmp_path, "a", "tap", BADS_TAP)
    sqm = load_group_sqm(tmp_path, [GroupEntry("G1", "sub-01", "tap")])["sub-01"]
    assert sqm["bad_channels"] == BADS_TAP


def test_a_named_session_does_not_read_the_other_ones_rejections(tmp_path):
    _ses_sidecar(tmp_path, "a", "tap", BADS_TAP)
    _ses_sidecar(tmp_path, "b", "tap", BADS_REST)
    for session, expected in (("a", BADS_TAP), ("b", BADS_REST)):
        sqm = load_group_sqm(
            tmp_path, [GroupEntry("G1", "sub-01", "tap", session=session)])["sub-01"]
        assert sqm["bad_channels"] == expected, session


def test_the_flat_tree_is_unchanged(nirs_dir):
    """The layout almost every dataset here uses. It must not have moved."""
    # nirs_dir, not tmp_path: `_sidecar` does not create the directory, `_ses_sidecar` does
    _sidecar(nirs_dir, "tap", BADS_TAP)
    assert _load(nirs_dir)["bad_channels"] == BADS_TAP
