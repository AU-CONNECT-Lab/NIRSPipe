"""What the re-paired null reads before it draws, and what it says on disk afterwards.

Its band, its mask, its frequency range and its window come off the real table's sidecar
rather than off the command line. A null is only meaningful against the table it is
subtracted from, so a parameter the caller could retype differently is a way for the two to
disagree in silence; the same reasoning already keeps the alignment stamp out of the
caller's hands. These cover that read, the refusals around it, and the two keys that stop a
merge putting the two nulls in one file.
"""

import json

import pandas as pd
import pytest

from fnirs_pipe.cli.hyper import _build_parser
from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.pair_null import real_table_params
from fnirs_pipe.pipeline.wtc_aggregate import aggregate_wtc

_REAL = {"band_fmin": 0.06, "band_fmax": 0.15, "wtc_fmin": 0.004, "wtc_fmax": 0.20,
         "mask_coi": True, "aligned_duration_s": 900.0}


def _real_table(root, gid="d01", task="full", params=None, kind="wtc"):
    d = root / f"group-{gid}" / "nirs"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"group-{gid}_task-{task}_hyper"
    tsv = d / f"{stem}-{kind}.tsv"
    pd.DataFrame({"chromophore": ["hbo"], "sub1": ["a"], "sub2": ["b"],
                  "label": ["S1_D1"], "coherence": [0.3]}).to_csv(tsv, sep="\t", index=False)
    tsv.with_suffix(".json").write_text(
        json.dumps({"parameters": _REAL if params is None else params}))
    return d, stem


# ---- the parameters come off the real table ----

def test_the_band_and_the_mask_are_read_back_rather_than_retyped(tmp_path):
    d, stem = _real_table(tmp_path)
    got = real_table_params(d, stem)
    assert (got["band_fmin"], got["band_fmax"], got["mask_coi"]) == (0.06, 0.15, True)


def test_no_real_table_is_refused_by_name(tmp_path):
    with pytest.raises(StageError, match="no real WTC table"):
        real_table_params(tmp_path, "group-d01_task-full_hyper")


def test_a_sidecar_missing_the_band_is_refused_rather_than_defaulted(tmp_path):
    """A null built on a guess about the band is worse than none: nothing can tell them apart."""
    d, stem = _real_table(tmp_path, params={"wtc_fmin": 0.004, "wtc_fmax": 0.20})
    with pytest.raises(StageError, match="band_fmin"):
        real_table_params(d, stem)


def test_an_unreadable_sidecar_is_refused(tmp_path):
    d, stem = _real_table(tmp_path)
    (d / f"{stem}-wtc.json").write_text("{not json")
    with pytest.raises(StageError, match="unreadable"):
        real_table_params(d, stem)


# ---- the command surface ----

def _parse(*argv):
    return _build_parser().parse_args(["pair-null", "/out", "--pairs-csv", "/p.csv", *argv])


def test_the_pool_keeps_a_members_position_unless_told_otherwise():
    assert _parse().wtc_pair_pool == "position"


def test_the_pool_can_be_widened():
    assert _parse("--wtc-pair-pool", "any").wtc_pair_pool == "any"


def test_the_draw_count_is_unbounded_unless_capped():
    """The pool is finite, so a cap only ever loses resolution; it is not a default."""
    assert _parse().wtc_pair_max is None
    assert _parse("--wtc-pair-max", "5").wtc_pair_max == 5


def test_the_crossing_is_its_own_decision():
    """Same split as --wtc-phase-null-cross: crossing multiplies the cost per draw."""
    assert _parse().wtc_pair_cross is False
    assert _parse("--wtc-pair-cross").wtc_pair_cross is True


@pytest.mark.parametrize("flag", ["--wtc-band-fmin", "--wtc-band-fmax", "--wtc-mask-coi",
                                  "--wtc-fmin", "--tstart"])
def test_it_does_not_take_from_the_command_line_what_it_reads_off_the_table(flag):
    """Retyping any of these is how a null and its real table come to disagree."""
    with pytest.raises(SystemExit):
        _parse(flag, "0.06")


# ---- the merge guard ----

def _write_null(root, gid, task, kind, **params):
    d = root / f"group-{gid}" / "nirs"
    d.mkdir(parents=True, exist_ok=True)
    tsv = d / f"group-{gid}_task-{task}_hyper-{kind}.tsv"
    pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                  "null_mean": [0.3]}).to_csv(tsv, sep="\t", index=False)
    tsv.with_suffix(".json").write_text(json.dumps({"parameters": {
        "band_fmin": 0.06, "band_fmax": 0.15, "mask_coi": True, **params}}))
    return tsv


def test_the_two_nulls_merge_into_separate_files(tmp_path):
    _write_null(tmp_path, "d01", "full", "wtc-phasenull", n_iter=100, null_kind="phase")
    _write_null(tmp_path, "d01", "full", "wtc-pairnull", n_iter=22,
                null_kind="repaired", pair_pool="position")
    assert len(aggregate_wtc(tmp_path, kind="wtc-phasenull")) == 1
    assert len(aggregate_wtc(tmp_path, kind="wtc-pairnull")) == 1


def test_a_pair_null_renamed_onto_the_phase_null_path_is_refused(tmp_path):
    """Filenames already keep them apart; this catches one moved by hand."""
    _write_null(tmp_path, "d01", "full", "wtc-phasenull", n_iter=22, null_kind="phase")
    _write_null(tmp_path, "d02", "full", "wtc-phasenull", n_iter=22, null_kind="repaired")
    with pytest.raises(ValueError, match="null_kind"):
        aggregate_wtc(tmp_path, kind="wtc-phasenull")


def test_two_pools_refuse_to_merge(tmp_path):
    """`any` draws from twice the people, so its null is not the same null."""
    _write_null(tmp_path, "d01", "full", "wtc-pairnull", n_iter=22,
                null_kind="repaired", pair_pool="position")
    _write_null(tmp_path, "d02", "full", "wtc-pairnull", n_iter=44,
                null_kind="repaired", pair_pool="any")
    with pytest.raises(ValueError, match="n_iter|pair_pool"):
        aggregate_wtc(tmp_path, kind="wtc-pairnull")


def test_one_pool_merges(tmp_path):
    for gid in ("d01", "d02"):
        _write_null(tmp_path, gid, "full", "wtc-pairnull", n_iter=22,
                    null_kind="repaired", pair_pool="position")
    assert sorted(aggregate_wtc(tmp_path, kind="wtc-pairnull")["group_id"]) == ["d01", "d02"]


def test_merge_covers_the_new_kinds(tmp_path):
    from fnirs_pipe.cli.hyper import cmd_merge

    _write_null(tmp_path, "d01", "full", "wtc-pairnull", n_iter=22,
                null_kind="repaired", pair_pool="position")
    cmd_merge(tmp_path, verbose=False)
    assert (tmp_path / "group_hyper_wtc_pairnull.tsv").exists()
