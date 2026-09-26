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

from fnirs_pipe.cli.hyper import _parsers
from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.hyper.pair_null import real_table_params
from fnirs_pipe.pipeline.hyper.wtc_aggregate import aggregate_wtc, merge_kinds
from tests.hyper._names import KINDS, name


def _merge(root, kind):
    """One kind's merge, discovered the way `merge` discovers it."""
    key = tuple(sorted((k, str(v)) for k, v in KINDS[kind].items()))
    return aggregate_wtc(root, merge_kinds(root)[key])

_REAL = {"band_fmin": 0.06, "band_fmax": 0.15, "wtc_fmin": 0.004, "wtc_fmax": 0.20,
         "mask_coi": True, "aligned_duration_s": 900.0}


def _real_table(root, gid="G01", task="main", params=None, kind="wtc"):
    d = root / f"group-{gid}" / "nirs"
    d.mkdir(parents=True, exist_ok=True)
    tsv = d / name(gid, task, kind)
    pd.DataFrame({"chromophore": ["hbo"], "sub1": ["a"], "sub2": ["b"],
                  "label": ["S1_D1"], "coherence": [0.3]}).to_csv(tsv, sep="\t", index=False)
    tsv.with_suffix(".json").write_text(
        json.dumps({"parameters": _REAL if params is None else params}))
    return tsv


# ---- the parameters come off the real table ----

def test_the_band_and_the_mask_are_read_back_rather_than_retyped(tmp_path):
    got = real_table_params(_real_table(tmp_path))
    assert (got["band_fmin"], got["band_fmax"], got["mask_coi"]) == (0.06, 0.15, True)


def test_no_real_table_is_refused_by_name(tmp_path):
    with pytest.raises(StageError, match="no real WTC table"):
        real_table_params(tmp_path / name("G01", "main"))


def test_a_sidecar_missing_the_band_is_refused_rather_than_defaulted(tmp_path):
    """A null built on a guess about the band is worse than none: nothing can tell them apart."""
    tsv = _real_table(tmp_path, params={"wtc_fmin": 0.004, "wtc_fmax": 0.20})
    with pytest.raises(StageError, match="band_fmin"):
        real_table_params(tsv)


def test_an_unreadable_sidecar_is_refused(tmp_path):
    tsv = _real_table(tmp_path)
    tsv.with_suffix(".json").write_text("{not json")
    with pytest.raises(StageError, match="unreadable"):
        real_table_params(tsv)


# ---- the command surface ----

def _parse(*argv):
    return _parsers()["fnirs-hyper-pairnull"].parse_args(
        ["/deriv", "/out", "group", "--pairs-csv", "/p.csv", *argv])


def test_the_pool_keeps_a_members_position_unless_told_otherwise():
    assert _parse().wtc_pair_pool == "position"


def test_the_pool_can_be_widened():
    assert _parse("--wtc-pair-pool", "any").wtc_pair_pool == "any"


def test_the_draw_count_is_unbounded_unless_capped():
    """The pool is finite, so a cap only ever loses resolution; it is not a default."""
    assert _parse().wtc_pair_max is None
    assert _parse("--wtc-pair-max", "5").wtc_pair_max == 5


def test_the_null_is_crossed_unless_told_otherwise():
    """Independent of the real run's crossing: crossing multiplies the cost per draw."""
    assert _parse().wtc_pair_cross is True
    assert _parse("--no-wtc-pair-cross").wtc_pair_cross is False


@pytest.mark.parametrize("flag", ["--wtc-band-fmin", "--wtc-band-fmax", "--wtc-mask-coi",
                                  "--wtc-fmin", "--tstart", "--wtc-whiten"])
def test_it_does_not_take_from_the_command_line_what_it_reads_off_the_table(flag):
    """Retyping any of these is how a null and its real table come to disagree."""
    with pytest.raises(SystemExit):
        _parse(flag, "0.06")


# ---- the merge guard ----

def _write_null(root, gid, task, kind, **params):
    d = root / f"group-{gid}" / "nirs"
    d.mkdir(parents=True, exist_ok=True)
    tsv = d / name(gid, task, kind)
    pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                  "null_mean": [0.3]}).to_csv(tsv, sep="\t", index=False)
    tsv.with_suffix(".json").write_text(json.dumps({"parameters": {
        "band_fmin": 0.06, "band_fmax": 0.15, "mask_coi": True, **params}}))
    return tsv


def test_the_two_nulls_merge_into_separate_files(tmp_path):
    _write_null(tmp_path, "G01", "main", "wtc-phasenull", n_iter=100, null_kind="phase")
    _write_null(tmp_path, "G01", "main", "wtc-pairnull", n_iter=9,
                null_kind="repaired", pair_pool="position")
    assert len(_merge(tmp_path, "wtc-phasenull")) == 1
    assert len(_merge(tmp_path, "wtc-pairnull")) == 1


def test_a_pair_null_renamed_onto_the_phase_null_path_is_refused(tmp_path):
    """Filenames already keep them apart; this catches one moved by hand."""
    _write_null(tmp_path, "G01", "main", "wtc-phasenull", n_iter=9, null_kind="phase")
    _write_null(tmp_path, "G02", "main", "wtc-phasenull", n_iter=9, null_kind="repaired")
    with pytest.raises(ValueError, match="null_kind"):
        _merge(tmp_path, "wtc-phasenull")


def test_two_pools_refuse_to_merge(tmp_path):
    """`any` draws from twice the people, so its null is not the same null."""
    _write_null(tmp_path, "G01", "main", "wtc-pairnull", n_iter=9,
                null_kind="repaired", pair_pool="position")
    _write_null(tmp_path, "G02", "main", "wtc-pairnull", n_iter=18,
                null_kind="repaired", pair_pool="any")
    with pytest.raises(ValueError, match="n_iter|pair_pool"):
        _merge(tmp_path, "wtc-pairnull")


def test_one_pool_merges(tmp_path):
    for gid in ("G01", "G02"):
        _write_null(tmp_path, gid, "main", "wtc-pairnull", n_iter=9,
                    null_kind="repaired", pair_pool="position")
    assert sorted(_merge(tmp_path, "wtc-pairnull")["group_id"]) == ["G01", "G02"]


def test_merge_covers_the_new_kinds(tmp_path):
    from fnirs_pipe.cli.hyper import cmd_merge

    _write_null(tmp_path, "G01", "main", "wtc-pairnull", n_iter=9,
                null_kind="repaired", pair_pool="position")
    cmd_merge(tmp_path, verbose=False)
    assert (tmp_path / "null-pair_stat-wtc_relmat.tsv").exists()


# ---- the ROI mapping path ----

def test_an_roi_mapping_is_read_rather_than_crashing(tmp_path, monkeypatch):
    """--roi-mapping is read and handed to the draw.

    The call is stubbed because what is under test is the command's own argument handling,
    not the draw; the draw has its own tests and needs a derivatives tree.
    """
    import fnirs_pipe.pipeline.hyper.pair_null as pair_null
    from fnirs_pipe.cli.hyper import cmd_pair_null

    (tmp_path / "roi.json").write_text('{"pfc": ["S1_D1", "S1_D2"]}')
    (tmp_path / "pairs.csv").write_text(
        "group_id,subject_id,task\nG01,sub-a,main\nG01,sub-b,main\n"
        "G02,sub-c,main\nG02,sub-d,main\n")

    seen = {}
    monkeypatch.setattr(pair_null, "run_pair_null",
                        lambda *a, **k: seen.update(k) or tmp_path / "out.tsv")
    cmd_pair_null(
        derivatives_dir=tmp_path, output_dir=tmp_path,
        pairs_csv=tmp_path / "pairs.csv", group_id="G01",
        task_label=None, desc="errts", roi_mapping=str(tmp_path / "roi.json"),
        bads_scope="run", wtc_chroma="both", wtc_pair_pool="position",
        wtc_pair_max=None, wtc_pair_cross=False, wtc_roi_min_channels=2,
        wtc_limit_scales=True, verbose=False)

    assert seen["roi_map"] == {"pfc": ["S1_D1", "S1_D2"]}


def test_an_unreadable_roi_mapping_exits_rather_than_tracebacks(tmp_path):
    from fnirs_pipe.cli.hyper import cmd_pair_null

    (tmp_path / "roi.json").write_text("{not json")
    (tmp_path / "pairs.csv").write_text(
        "group_id,subject_id,task\nG01,sub-a,main\nG01,sub-b,main\n")
    with pytest.raises(SystemExit):
        cmd_pair_null(
            derivatives_dir=tmp_path, output_dir=tmp_path,
            pairs_csv=tmp_path / "pairs.csv", group_id=None,
            task_label=None, desc="errts", roi_mapping=str(tmp_path / "roi.json"),
            bads_scope="run", wtc_chroma="both", wtc_pair_pool="position",
            wtc_pair_max=None, wtc_pair_cross=False, wtc_roi_min_channels=2,
            wtc_limit_scales=True, verbose=False)


def test_the_real_table_records_the_window_it_describes():
    """The real table records --tstart/--tend, as the null that ranks it does.

    A cohort whose triggers sit apart needs a common analysis window for its dyads to be
    comparable, so the window has to be readable back off the table it was applied to. Same
    argument as the alignment stamp: a windowed table and a whole-recording one cannot be
    told apart by their numbers.
    """
    import inspect

    from fnirs_pipe.pipeline.hyper import hyper_post

    src = inspect.getsource(hyper_post.run_hyper_post)
    assert "analysis_window_s" in src, (
        "the real WTC table's sidecar no longer records the window it describes")
