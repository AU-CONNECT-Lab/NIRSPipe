"""The null shares the hyper run but not its crossing.

`hyper-post --wtc-phase-null` used to pass `cross=wtc_channel_cross` straight through, so asking
for the exploratory 196-pair channel table also multiplied every surrogate iteration by 14.
That coupling was invisible on disk too: the null's sidecar recorded the band but neither the
iteration count nor the shape, so a 5-iteration probe and a 100-iteration null looked alike.

The null runs inside `fnirs-hyper run`, which is what keeps its band and its stage identical
to the table it sits beside, and the independence is carried by `--wtc-phase-null-cross`
instead. These tests hold that independence in place: the null is off unless asked for, its
crossing is its own decision, the sidecar says what was run, and a merge refuses to mix
iteration counts.
"""

import json

import pandas as pd
import logging

import pytest

from fnirs_pipe.cli.hyper import _parsers
from fnirs_pipe.pipeline.hyper.wtc_aggregate import aggregate_wtc


# The null is drawn and written in two phases, the report sitting between them: the level
# the phase arrows need has to exist before the figures, and the rank a null row carries is
# against the real tables the same report writes. These tests want both halves at once.
def _draw_and_write(wtc_null, **kwargs):
    run_only = ("limit_scales", "chroma", "sep_bands")
    nulls = wtc_null.run_wtc_null(**kwargs)
    return wtc_null.write_wtc_null(
        nulls, **{k: v for k, v in kwargs.items() if k not in run_only})


def _null(frame, cond_frames=(), levels=None):
    """A NullDraws around an already-made frame, for the tests that stub the draw away."""
    from fnirs_pipe.pipeline.hyper.surrogate import NullDraws

    keys = ["sub1", "sub2", "label"] + (["label2"] if "label2" in frame.columns else [])
    return NullDraws(draws=[frame], cond_draws=list(cond_frames), keys=keys,
                      levels=levels or {})


def _hyper(*argv):
    return _parsers()["fnirs-hyper"].parse_args(
        ["/deriv", "/out", "group", "--pairs-csv", "/p.csv", *argv])


# ---- the null is opt-in ----

def test_no_null_unless_asked():
    assert _hyper().wtc_phase_null is None


def test_the_iteration_count_is_what_asks_for_it():
    assert _hyper("--wtc-phase-null", "100").wtc_phase_null == 100


# ---- the crossing stays split ----

def test_the_null_is_homologous_unless_asked():
    assert _hyper("--wtc-phase-null", "100").wtc_phase_null_cross is False


def test_crossing_the_real_run_does_not_cross_the_null():
    args = _hyper("--wtc-phase-null", "100", "--wtc-channel-cross")
    assert args.wtc_channel_cross is True
    assert args.wtc_phase_null_cross is False


def test_the_null_can_be_crossed_on_its_own():
    args = _hyper("--wtc-phase-null", "100", "--wtc-phase-null-cross")
    assert args.wtc_phase_null_cross is True
    assert args.wtc_channel_cross is False


# ---- what lands on disk ----

def test_the_sidecar_records_the_iteration_count_and_the_shape(tmp_path, monkeypatch, make_raw):
    import fnirs_pipe.pipeline.hyper as hyper
    from fnirs_pipe.pipeline.hyper import wtc_null

    frame = pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                          "coherence": [0.3], "coherence_z": [0.31], "n_valid_frac": [1.0]})
    monkeypatch.setattr(hyper, "compute_wtc_phase_null", lambda *a, **k: _null(frame))

    # a real raw even though the WTC itself is stubbed: the sidecar reads the wavelet grid
    # off the recordings, so an empty map has no sampling rate to report
    raws = {"a": make_raw(n_ch=1), "b": make_raw(n_ch=1)}
    out = _draw_and_write(
        wtc_null,
        group_id="d01", task="baseline", aligned_raws=raws, output_dir=tmp_path,
        n_iter=7, wtc_fmin=0.01, wtc_fmax=0.25, band_fmin=0.06, band_fmax=0.15,
        seed=1, cross=False, mask_coi=True)

    assert out.name == "group-d01_task-baseline_hyper-wtc-phasenull.tsv"
    params = json.loads(out.with_suffix(".json").read_text())["parameters"]
    assert params["n_iter"] == 7
    assert params["cross"] is False
    assert params["mask_coi"] is True
    assert params["band_fmin"] == 0.06
    assert params["chroma"] == ["hbo", "hbr"]


def test_the_null_tags_each_chromophore_without_mutating_the_frame(tmp_path, monkeypatch, make_raw):
    """One pass per chromophore, and the frame each returns is not the null's to
    change. Inserting the column in place worked for HbO and raised on HbR as soon as
    two passes were handed the same object, which a cache or a stub does."""
    import fnirs_pipe.pipeline.hyper as hyper
    from fnirs_pipe.pipeline.hyper import wtc_null

    frame = pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                          "coherence": [0.3], "coherence_z": [0.31],
                          "n_valid_frac": [1.0]})
    monkeypatch.setattr(hyper, "compute_wtc_phase_null", lambda *a, **k: _null(frame))

    raws = {"a": make_raw(n_ch=1), "b": make_raw(n_ch=1)}
    out = _draw_and_write(
        wtc_null,
        group_id="d01", task="baseline", aligned_raws=raws, output_dir=tmp_path,
        n_iter=1, chroma=("hbo", "hbr"))

    df = pd.read_csv(out, sep="\t")
    assert list(df["chromophore"]) == ["hbo", "hbr"]
    assert "chromophore" not in frame.columns


def test_one_chromophore_writes_one_set_of_rows(tmp_path, monkeypatch, make_raw):
    import fnirs_pipe.pipeline.hyper as hyper
    from fnirs_pipe.pipeline.hyper import wtc_null

    frame = pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                          "coherence": [0.3], "coherence_z": [0.31],
                          "n_valid_frac": [1.0]})
    monkeypatch.setattr(hyper, "compute_wtc_phase_null", lambda *a, **k: _null(frame))

    raws = {"a": make_raw(n_ch=1), "b": make_raw(n_ch=1)}
    out = _draw_and_write(
        wtc_null,
        group_id="d01", task="baseline", aligned_raws=raws, output_dir=tmp_path,
        n_iter=1, chroma=("hbr",))

    df = pd.read_csv(out, sep="\t")
    assert list(df["chromophore"]) == ["hbr"]
    params = json.loads(out.with_suffix(".json").read_text())["parameters"]
    assert params["chroma"] == ["hbr"]


# ---- the merge guard ----

def _write_null(root, gid, task, n_iter, band_fmin=0.06):
    d = root / f"group-{gid}" / "nirs"
    d.mkdir(parents=True, exist_ok=True)
    tsv = d / f"group-{gid}_task-{task}_hyper-wtc-phasenull.tsv"
    # n_iter is a column as well as a sidecar field, which is what a merged table keeps
    pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                  "coherence": [0.3], "n_iter": [n_iter]}).to_csv(tsv, sep="\t", index=False)
    tsv.with_suffix(".json").write_text(json.dumps({"parameters": {
        "band_fmin": band_fmin, "band_fmax": 0.15, "mask_coi": True, "n_iter": n_iter}}))


def test_nulls_of_different_lengths_merge_but_say_so(tmp_path, caplog):
    """They used to be refused. n_iter is a column, so the rows survive the merge intact.

    A refusal here stopped the whole cohort because one group had a coarser null, on a
    dataset where the re-paired pool is finite and its size tracks recording length. What
    the reader needs is the count beside the percentile, and that is in the table.
    """
    _write_null(tmp_path, "d01", "baseline", 100)
    _write_null(tmp_path, "d02", "baseline", 5)
    with caplog.at_level(logging.WARNING, logger="fnirs_pipe.pipeline.hyper.wtc_aggregate"):
        merged = aggregate_wtc(tmp_path, kind="wtc-phasenull")
    assert sorted(merged["group_id"].unique()) == ["d01", "d02"]
    assert sorted(merged["n_iter"].unique()) == [5, 100]
    assert any("do not have one resolution" in r.getMessage() for r in caplog.records)


def test_a_band_still_refuses_to_merge(tmp_path):
    """Loosening n_iter must not loosen the band: that one changes what a column means."""
    _write_null(tmp_path, "d01", "baseline", 100)
    _write_null(tmp_path, "d02", "baseline", 100, band_fmin=0.02)
    with pytest.raises(ValueError, match="band_fmin"):
        aggregate_wtc(tmp_path, kind="wtc-phasenull")


def test_nulls_of_one_length_merge(tmp_path):
    _write_null(tmp_path, "d01", "baseline", 100)
    _write_null(tmp_path, "d02", "baseline", 100)
    merged = aggregate_wtc(tmp_path, kind="wtc-phasenull")
    assert sorted(merged["group_id"]) == ["d01", "d02"]


def test_merge_covers_every_kind_the_aggregator_has(tmp_path):
    """It asked for a kind the aggregator no longer has, and died before reaching the null."""
    from fnirs_pipe.cli.hyper import cmd_merge
    from fnirs_pipe.pipeline.hyper.wtc_aggregate import _KINDS

    _write_null(tmp_path, "d01", "baseline", 100)
    cmd_merge(tmp_path, verbose=False)                        # no kind raises

    assert (tmp_path / "group_hyper_wtc_phasenull.tsv").exists()
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
    _write_table(tmp_path, "d01", "baseline", kind="wtc-phasenull")
    _merge_reminder(tmp_path)

    out = capsys.readouterr().out
    assert "1 wtc table(s)" in out
    assert "1 wtc-phasenull table(s)" in out


# ---- the per-condition null ----

def _null_frames():
    whole = pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                          "coherence": [0.3], "coherence_z": [0.31], "n_valid_frac": [1.0]})
    by_cond = pd.DataFrame({"condition": ["rest", "talk"], "sub1": ["a", "a"],
                            "sub2": ["b", "b"], "label": ["S1_D1", "S1_D1"],
                            "coherence": [0.35, 0.28], "coherence_z": [0.36, 0.29],
                            "n_valid_frac": [1.0, 1.0]})
    return whole, by_cond


def test_windows_add_a_second_table_beside_the_whole_run_one(tmp_path, monkeypatch, make_raw):
    """Two files rather than one, mirroring the real side, where the whole-run and
    per-condition tables are also merged separately."""
    import fnirs_pipe.pipeline.hyper as hyper
    from fnirs_pipe.pipeline.hyper import wtc_null

    whole, by_cond = _null_frames()
    monkeypatch.setattr(hyper, "compute_wtc_phase_null",
                        lambda *a, **k: _null(whole, [by_cond]))

    raws = {"a": make_raw(n_ch=1), "b": make_raw(n_ch=1)}
    out = _draw_and_write(
        wtc_null,
        group_id="d01", task="full", aligned_raws=raws, output_dir=tmp_path,
        n_iter=2, chroma=("hbo",), windows=[("rest", 0.0, 30.0), ("talk", 30.0, 60.0)])

    assert out.name == "group-d01_task-full_hyper-wtc-phasenull.tsv"
    cond_path = out.with_name("group-d01_task-full_hyper-wtcbycond-phasenull.tsv")
    assert cond_path.exists()
    df = pd.read_csv(cond_path, sep="\t")
    assert list(df.columns[:2]) == ["chromophore", "condition"]
    assert set(df["condition"]) == {"rest", "talk"}


def test_the_windowed_sidecar_names_the_conditions(tmp_path, monkeypatch, make_raw):
    """Without them a table of five conditions and a table of two read the same on disk."""
    import fnirs_pipe.pipeline.hyper as hyper
    from fnirs_pipe.pipeline.hyper import wtc_null

    whole, by_cond = _null_frames()
    monkeypatch.setattr(hyper, "compute_wtc_phase_null",
                        lambda *a, **k: _null(whole, [by_cond]))

    raws = {"a": make_raw(n_ch=1), "b": make_raw(n_ch=1)}
    out = _draw_and_write(
        wtc_null,
        group_id="d01", task="full", aligned_raws=raws, output_dir=tmp_path,
        n_iter=2, chroma=("hbo",), windows=[("rest", 0.0, 30.0), ("talk", 30.0, 60.0)])

    cond_path = out.with_name("group-d01_task-full_hyper-wtcbycond-phasenull.tsv")
    params = json.loads(cond_path.with_suffix(".json").read_text())["parameters"]
    assert params["conditions"] == ["rest", "talk"]
    assert params["n_iter"] == 2


def test_no_windows_writes_only_the_whole_run_table(tmp_path, monkeypatch, make_raw):
    import fnirs_pipe.pipeline.hyper as hyper
    from fnirs_pipe.pipeline.hyper import wtc_null

    whole, _ = _null_frames()
    monkeypatch.setattr(hyper, "compute_wtc_phase_null", lambda *a, **k: _null(whole))

    raws = {"a": make_raw(n_ch=1), "b": make_raw(n_ch=1)}
    out = _draw_and_write(
        wtc_null,
        group_id="d01", task="full", aligned_raws=raws, output_dir=tmp_path,
        n_iter=1, chroma=("hbo",))

    assert not out.with_name("group-d01_task-full_hyper-wtcbycond-phasenull.tsv").exists()


def test_the_per_condition_null_is_its_own_merge_kind():
    """Merging it into the whole-run null would average five conditions into one row."""
    from fnirs_pipe.pipeline.hyper.wtc_aggregate import _KINDS

    assert _KINDS["wtcbycond-phasenull"] == "group_hyper_wtc_bycondition_phasenull"
    assert _KINDS["wtc-phasenull"] != _KINDS["wtcbycond-phasenull"]
