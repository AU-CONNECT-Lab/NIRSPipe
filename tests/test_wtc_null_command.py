"""The null is its own command now, and it no longer inherits the real run's crossing.

`hyper-post --wtc-pseudo` used to pass `cross=wtc_channel_cross` straight through, so asking
for the exploratory 196-pair channel table also multiplied every surrogate iteration by 14.
That coupling was invisible on disk too: the pseudo sidecar recorded the band but neither the
iteration count nor the shape, so a 5-iteration probe and a 100-iteration null looked alike.

These tests hold the split in place: the flag is gone from `hyper-post`, `hyper-null` carries
its own `--wtc-channel-cross` defaulting off, the sidecar says what was run, and a merge
refuses to mix iteration counts.
"""

import json
import pathlib

import pandas as pd
import pytest

from fnirs_pipe.cli.qc import _build_parser
from fnirs_pipe.qc.wtc_aggregate import aggregate_wtc


def _sub(name):
    action = next(a for a in _build_parser()._actions if getattr(a, "choices", None))
    return action.choices[name]


def _flags(name):
    return {f for a in _sub(name)._actions for f in a.option_strings if f.startswith("--")}


# ---- the split itself ----

def test_hyper_post_no_longer_runs_the_null():
    assert "--wtc-pseudo" not in _flags("hyper-post")


def test_hyper_null_exists_and_takes_the_iteration_count():
    assert "--wtc-pseudo" in _flags("hyper-null")


def test_the_null_is_homologous_unless_asked():
    args = _sub("hyper-null").parse_args(["/bids", "/out", "--pairs-csv", "/p.csv"])
    assert args.wtc_channel_cross is False
    assert args.wtc_pseudo == 100


def test_crossing_the_null_is_independent_of_crossing_the_real_run():
    crossed = _sub("hyper-null").parse_args(
        ["/bids", "/out", "--pairs-csv", "/p.csv", "--wtc-channel-cross"])
    assert crossed.wtc_channel_cross is True


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


def test_the_merge_command_covers_every_kind_the_aggregator_has():
    """It asked for wtc-roi, gone since 0.24.0, and died before reaching the null."""
    import inspect

    from fnirs_pipe.cli.qc import cmd_group_hyper_wtc
    from fnirs_pipe.qc.wtc_aggregate import _KINDS

    for kind in _KINDS:
        aggregate_wtc(pathlib.Path("."), kind=kind)          # every kind is accepted
    assert "for kind in _KINDS" in inspect.getsource(cmd_group_hyper_wtc)
