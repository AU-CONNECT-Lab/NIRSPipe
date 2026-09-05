"""The null shares the hyper run but not its crossing.

`hyper-post --wtc-pseudo` used to pass `cross=wtc_channel_cross` straight through, so asking
for the exploratory 196-pair channel table also multiplied every surrogate iteration by 14.
That coupling was invisible on disk too: the pseudo sidecar recorded the band but neither the
iteration count nor the shape, so a 5-iteration probe and a 100-iteration null looked alike.

0.25.0 fixed it by splitting the null into its own command. The null now runs inside
`fnirs-hyper run` again, which is what keeps its band and its stage identical to the table
it sits beside, and the independence is carried by `--wtc-pseudo-cross` instead. These
tests hold that independence in place: the null is off unless asked for, its crossing is
its own decision, the sidecar says what was run, and a merge refuses to mix iteration
counts.
"""

import json

import pandas as pd
import pytest

from fnirs_pipe.cli.hyper import _build_parser
from fnirs_pipe.qc.wtc_aggregate import aggregate_wtc


def _hyper(*argv):
    return _build_parser().parse_args(["run", "/out", "--pairs-csv", "/p.csv", *argv])


# ---- the null is opt-in ----

def test_no_null_unless_asked():
    assert _hyper().wtc_pseudo is None


def test_the_iteration_count_is_what_asks_for_it():
    assert _hyper("--wtc-pseudo", "100").wtc_pseudo == 100


# ---- the crossing stays split ----

def test_the_null_is_homologous_unless_asked():
    assert _hyper("--wtc-pseudo", "100").wtc_pseudo_cross is False


def test_crossing_the_real_run_does_not_cross_the_null():
    args = _hyper("--wtc-pseudo", "100", "--wtc-channel-cross")
    assert args.wtc_channel_cross is True
    assert args.wtc_pseudo_cross is False


def test_the_null_can_be_crossed_on_its_own():
    args = _hyper("--wtc-pseudo", "100", "--wtc-pseudo-cross")
    assert args.wtc_pseudo_cross is True
    assert args.wtc_channel_cross is False


# ---- what lands on disk ----

def test_the_sidecar_records_the_iteration_count_and_the_shape(tmp_path, monkeypatch):
    import fnirs_pipe.pipeline.hyperscanning as hyper
    from fnirs_pipe.qc import wtc_null

    frame = pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                          "coherence": [0.3], "coherence_z": [0.31], "n_valid_frac": [1.0]})
    monkeypatch.setattr(hyper, "compute_wtc_pseudo", lambda *a, **k: frame)

    out = wtc_null.write_wtc_null(
        group_id="d01", task="baseline", aligned_raws={}, output_dir=tmp_path,
        n_iter=7, wtc_fmin=0.01, wtc_fmax=0.25, band_fmin=0.06, band_fmax=0.15,
        seed=1, cross=False, mask_coi=True)

    assert out.name == "group-d01_task-baseline_hyper-wtc-pseudo.tsv"
    params = json.loads(out.with_suffix(".json").read_text())["parameters"]
    assert params["n_iter"] == 7
    assert params["cross"] is False
    assert params["mask_coi"] is True
    assert params["band_fmin"] == 0.06


# ---- the merge guard ----

def _write_null(root, gid, task, n_iter):
    d = root / f"group-{gid}" / "nirs"
    d.mkdir(parents=True, exist_ok=True)
    tsv = d / f"group-{gid}_task-{task}_hyper-wtc-pseudo.tsv"
    pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                  "coherence": [0.3]}).to_csv(tsv, sep="\t", index=False)
    tsv.with_suffix(".json").write_text(json.dumps({"parameters": {
        "band_fmin": 0.06, "band_fmax": 0.15, "mask_coi": True, "n_iter": n_iter}}))


def test_nulls_of_different_lengths_refuse_to_merge(tmp_path):
    _write_null(tmp_path, "d01", "baseline", 100)
    _write_null(tmp_path, "d02", "baseline", 5)
    with pytest.raises(ValueError, match="n_iter"):
        aggregate_wtc(tmp_path, kind="wtc-pseudo")


def test_nulls_of_one_length_merge(tmp_path):
    _write_null(tmp_path, "d01", "baseline", 100)
    _write_null(tmp_path, "d02", "baseline", 100)
    merged = aggregate_wtc(tmp_path, kind="wtc-pseudo")
    assert sorted(merged["group_id"]) == ["d01", "d02"]


def test_merge_covers_every_kind_the_aggregator_has(tmp_path):
    """It asked for wtc-roi, gone since 0.24.0, and died before reaching the null."""
    from fnirs_pipe.cli.hyper import cmd_merge
    from fnirs_pipe.qc.wtc_aggregate import _KINDS

    _write_null(tmp_path, "d01", "baseline", 100)
    cmd_merge(tmp_path, verbose=False)                        # no kind raises

    assert (tmp_path / "group_hyper_wtc_pseudo.tsv").exists()
    for kind in _KINDS:
        aggregate_wtc(tmp_path, kind=kind)


# ---- the merge reminder ----

def _write_table(root, gid, task, kind="wtc"):
    d = root / f"group-{gid}" / "nirs"
    d.mkdir(parents=True, exist_ok=True)
    tsv = d / f"group-{gid}_task-{task}_hyper-{kind}.tsv"
    pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                  "coherence": [0.3]}).to_csv(tsv, sep="\t", index=False)
    return tsv


def test_a_run_says_when_nothing_has_been_merged(tmp_path, capsys):
    """A stale merged table reads like a finished result, so the run says which it is.
    It does not merge: a run often covers one dyad, and merging the whole tree after it
    would fail on a band a later run legitimately changed."""
    from fnirs_pipe.cli.hyper import _merge_reminder

    _write_table(tmp_path, "d01", "baseline")
    _write_table(tmp_path, "d02", "baseline")
    _merge_reminder(tmp_path)

    out = capsys.readouterr().out
    assert "2 wtc table(s) on disk, never merged" in out
    assert "fnirs-hyper merge" in out


def test_a_run_says_when_the_merged_table_is_behind(tmp_path, capsys):
    import os

    from fnirs_pipe.cli.hyper import _merge_reminder

    tsv = _write_table(tmp_path, "d01", "baseline")
    merged = tmp_path / "group_hyper_wtc.tsv"
    merged.write_text("stale")
    os.utime(merged, (1, 1))                       # older than the dyad table
    _merge_reminder(tmp_path)
    assert "1 newer than group_hyper_wtc.tsv" in capsys.readouterr().out

    os.utime(merged, (tsv.stat().st_mtime + 10,) * 2)
    _merge_reminder(tmp_path)
    assert capsys.readouterr().out == ""           # up to date, so nothing to say


def test_the_reminder_counts_what_the_merge_would_take(tmp_path, capsys):
    """Same glob as the aggregator, so the count cannot disagree with what merge does."""
    from fnirs_pipe.cli.hyper import _merge_reminder

    _write_table(tmp_path, "d01", "baseline", kind="wtc")
    _write_table(tmp_path, "d01", "baseline", kind="wtc-pseudo")
    _merge_reminder(tmp_path)

    out = capsys.readouterr().out
    assert "1 wtc table(s)" in out
    assert "1 wtc-pseudo table(s)" in out
