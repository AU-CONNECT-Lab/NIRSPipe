"""The re-paired null of a group of more than two members, end to end on synthetic triads.

The target's first two members are copies of one recording, so its pairings with the third
hold the same real side: drawn against one stand-in for the third member, as they must be
within a draw, their draws are equal under each draw's id; the pairing of the two copies
replaces the second member with someone else and is not.
"""

import json

import numpy as np
import pandas as pd
import pytest

from nirspipe.exceptions import StageError
from nirspipe.io.groups import GroupEntry
from nirspipe.qc.hyper.hyper_report import build_hyper_post_report
from tests.hyper.test_hyper_page_contract import DURATION, SFREQ, _raw

BAND = (0.03, 0.10)
TARGET = ("sub-01", "sub-02", "sub-03")


def _recordings() -> dict:
    rng = np.random.default_rng(0)
    n = int(SFREQ * DURATION)
    out = {f"sub-{k:02d}": _raw(k, rng.standard_normal(n)) for k in range(1, 13)}
    out["sub-02"] = out["sub-01"].copy()
    return out


def _entries(gid, subs):
    return [GroupEntry(gid, s, "tap") for s in subs]


@pytest.fixture
def run(tmp_path, monkeypatch):
    return _real_then_null(tmp_path, monkeypatch)


def _real_then_null(tmp_path, monkeypatch, **real):
    """Write the target's real tables, then return a function drawing its re-paired null."""
    from nirspipe.pipeline.hyper import alignment, group_io, group_quality

    raws = _recordings()
    target = {s: raws[s] for s in TARGET}
    build_hyper_post_report(
        group_id="G1", task="tap", group=_entries("G1", TARGET), aligned_raws=target,
        offsets={s: 0.0 for s in TARGET}, output_dir=tmp_path, wtc_fmin=0.02, wtc_fmax=0.2,
        wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1], wtc_chroma=("hbo",),
        wtc_by_condition=True, no_report=True, desc="preproc", bads_scope="run", **real)
    monkeypatch.setattr(group_io, "load_group_haemo", lambda out, entries, desc="preproc": {
        e.subject_id: raws[e.subject_id].copy() for e in entries})
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    monkeypatch.setattr(alignment, "align_recordings",
                        lambda raws, task: (raws, {s: 0.0 for s in raws}))

    def draw(donors: dict, **kwargs):
        from nirspipe.pipeline.hyper.pair_null import run_pair_null
        groups = {("G1", "tap", None): _entries("G1", TARGET),
                  **{(gid, "tap", None): _entries(gid, subs) for gid, subs in donors.items()}}
        run_pair_null("G1", "tap", _entries("G1", TARGET), groups, tmp_path, tmp_path,
                      chroma=("hbo",), **kwargs)
        return tmp_path
    return draw


def _file(root, *parts):
    return next(p for p in root.rglob("*_relmat.tsv") if all(part in p.name for part in parts))


def _by_draw(table, a, b, value):
    rows = table[(table["sub1"] == a) & (table["sub2"] == b)]
    return rows.set_index(["condition", "label", "label2", "draw"])[value].sort_index()


TRIADS = {"G2": ("sub-04", "sub-05", "sub-06"), "G3": ("sub-07", "sub-08", "sub-09")}


def test_the_null_splits_channels_by_the_bands_the_real_table_was_split_by(tmp_path,
                                                                          monkeypatch):
    """Bands given to the real run decide which channels are long, so which rows the null has."""
    from nirspipe.pipeline.hyper import _helpers

    draw = _real_then_null(tmp_path, monkeypatch, sep_bands=(0.012, 0.020, None))
    seen, axis = [], _helpers.long_axis_over

    def _spy(raws, ch_type="hbo", sep_bands=None):
        seen.append(tuple(sep_bands))
        return axis(raws, ch_type, sep_bands)

    monkeypatch.setattr(_helpers, "long_axis_over", _spy)
    draw(TRIADS)
    assert seen and {(round(a, 6), round(b, 6), c) for a, b, c in seen} == {(0.012, 0.02, None)}


def test_every_pairing_is_drawn_under_the_draws_it_shares(run):
    draws = pd.read_csv(_file(run(TRIADS), "null-pair_stat-wtc_desc-draws"), sep="\t")

    assert {(a, b): set(part["draw"]) for (a, b), part in draws.groupby(["sub1", "sub2"])} == {
        pair: {"sub-05+sub-06", "sub-08+sub-09"}
        for pair in (("sub-01", "sub-02"), ("sub-01", "sub-03"), ("sub-02", "sub-03"))}


@pytest.mark.parametrize("statistic, value", [("wtc", "coherence"), ("isc", "r")])
def test_the_later_member_has_one_stand_in_per_draw_across_its_pairings(run, statistic, value):
    draws = pd.read_csv(_file(run(TRIADS), f"null-pair_stat-{statistic}_desc-draws"), sep="\t")
    if "label2" not in draws.columns:
        draws["label2"] = draws["label"]

    first, second = (_by_draw(draws, a, "sub-03", value) for a in ("sub-01", "sub-02"))
    assert first.notna().any()
    assert np.allclose(first, second, rtol=0, atol=1e-12, equal_nan=True)
    copies = _by_draw(draws, "sub-01", "sub-02", value)
    assert not np.allclose(copies, first, equal_nan=True)


def test_a_stand_ins_rejected_channel_leaves_its_pairings_out_of_that_draw(run, monkeypatch):
    """sub-06 stands in for sub-03 in draw sub-05+sub-06 with S1_D1 rejected: those pairings
    have no value there, so the draws hold no row for them and every row they hold has one."""
    from nirspipe.pipeline.hyper import group_io
    load = group_io.load_group_haemo

    def rejecting(out, entries, desc="preproc"):
        raws = load(out, entries, desc)
        if "sub-06" in raws:
            raws["sub-06"].info["bads"] = ["S1_D1 hbo", "S1_D1 hbr"]
        return raws

    monkeypatch.setattr(group_io, "load_group_haemo", rejecting)
    root = run(TRIADS)
    for statistic, value in (("wtc", "coherence"), ("isc", "r")):
        draws = pd.read_csv(_file(root, f"null-pair_stat-{statistic}_desc-draws"), sep="	")
        side = draws["label2"] if "label2" in draws.columns else draws["label"]
        stand_in = (draws["sub2"] == "sub-03") & (draws["draw"] == "sub-05+sub-06")
        assert draws[value].notna().all(), statistic
        assert not (stand_in & (side == "S1_D1")).any(), statistic
        # the same pairings in the other draw, and the rest of this one, stay
        assert ((draws["sub2"] == "sub-03") & (draws["draw"] == "sub-08+sub-09")
                & (side == "S1_D1")).any(), statistic
        assert (stand_in & (side == "S2_D2")).any(), statistic


def test_the_correlation_draws_carry_r_as_the_real_table_and_the_phase_draws_do(run):
    draws = pd.read_csv(_file(run(TRIADS), "null-pair_stat-isc_desc-draws"), sep="	")
    assert {"r", "r_z"} <= set(draws.columns) and "coherence" not in draws.columns


def test_the_sidecar_counts_draws_not_stand_ins(run):
    table = _file(run(TRIADS), "task-tap_cond-all_null-pair_stat-wtc_relmat")
    params = json.loads(table.with_suffix(".json").read_text(encoding="utf-8"))["parameters"]

    assert params["n_iter"] == 2 and params["pair_candidates"] == 2
    assert params["pair_partners"] == ["sub-05+sub-06", "sub-08+sub-09"]
    summary = pd.read_csv(table, sep="\t")
    assert set(summary["n_iter"].dropna()) == {2}


def test_a_triad_among_dyads_is_refused_naming_the_pairings_no_draw_fills(run):
    """By row order a dyad has no third member, so nobody stands in for sub-03."""
    with pytest.raises(StageError, match="sub-01-sub-03, sub-02-sub-03"):
        run({"G2": ("sub-04", "sub-05"), "G3": ("sub-07", "sub-08")})


def test_the_any_pool_lets_dyads_stand_in_for_a_triad(run):
    draws = pd.read_csv(_file(run({"G2": ("sub-04", "sub-05")}, pool="any"),
                              "null-pair_stat-wtc_desc-draws"), sep="\t")
    assert set(draws.loc[draws["sub2"] == "sub-03", "draw"]) == {"sub-04+sub-05",
                                                                  "sub-05+sub-04"}


def test_a_cap_counts_draws_across_the_pairings_so_each_stops_on_the_same_donor(run, monkeypatch):
    """G2's stand-in for sub-03 is unreadable: a cap of 2 per pairing gave sub-01-sub-02 G2 and
    G3 but its other pairings G3 and G4, three draws of which the group mean had one whole."""
    from nirspipe.pipeline.hyper import group_io
    load = group_io.load_group_haemo

    def unreadable_sub06(out, entries, desc="preproc"):
        if any(e.subject_id == "sub-06" for e in entries):
            raise OSError("unreadable")
        return load(out, entries, desc=desc)
    monkeypatch.setattr(group_io, "load_group_haemo", unreadable_sub06)

    root = run({**TRIADS, "G4": ("sub-10", "sub-11", "sub-12")}, n_max=2)
    draws = pd.read_csv(_file(root, "null-pair_stat-wtc_desc-draws"), sep="	")
    assert {(a, b): set(part["draw"]) for (a, b), part in draws.groupby(["sub1", "sub2"])} == {
        pair: {"sub-08+sub-09", "sub-11+sub-12"}
        for pair in (("sub-01", "sub-02"), ("sub-01", "sub-03"), ("sub-02", "sub-03"))}
    table = _file(root, "task-tap_cond-all_null-pair_stat-wtc_relmat")
    params = json.loads(table.with_suffix(".json").read_text(encoding="utf-8"))["parameters"]
    assert params["n_iter"] == 2 and params["pair_candidates"] == 3
    assert params["pair_refused"] == {"unreadable": ["sub-06"]}
