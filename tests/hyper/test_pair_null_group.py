"""The two levels above the cell, and the resolution each one can express.

The point of averaging before ranking is that a pool of k cannot express a p under 1/(k+1) per
cell, so these hold the arithmetic that makes a cohort verdict possible at all.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nirspipe.pipeline.hyper.pair_null_group import (
    _exact_p, _tailed_p, _variants, by_cell, by_cohort, by_occasion, correct_cohort,
    magnitude_against_unwhitened_phase, matched_pools, write_group_null)
from tests.hyper._names import cohort as _cohort, name


@pytest.fixture(autouse=True)
def _hyper_tree(tmp_path):
    """tmp_path is the nirspipe-hyper tree, which every command stamps before it writes."""
    from nirspipe.io.derivatives import write_dataset_description
    write_dataset_description(tmp_path, name="nirspipe-hyper output", generated_by="nirspipe-hyper")


SEP = chr(9)

CHANNELS = ["S1_D1", "S1_D2", "S2_D1", "S2_D2"]
OCCASIONS = ["G01", "G03", "G04"]
DRAWS = [f"sub-02G{n:02d}" for n in (5, 6, 7, 8)]


def _draws(level=0.30, spread=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"chromophore": "hbo", "condition": "game", "occasion": o, "draw": s,
             "label": c, "coherence": level + rng.normal(0, spread)}
            for o in OCCASIONS for s in DRAWS for c in CHANNELS]
    return pd.DataFrame(rows)


def _resample_rows(frame):
    """The resample read alone, `by_cohort` writing one row per read per condition."""
    return frame[frame.test == "resample"]


def _real(level):
    return pd.DataFrame([{"chromophore": "hbo", "condition": "game", "occasion": o,
                          "label": c, "coherence": level}
                         for o in OCCASIONS for c in CHANNELS])


# ---- the exact p ----

def test_beating_every_draw_gives_the_finest_p_the_pool_allows():
    assert _exact_p(9, 9) == pytest.approx(1 / 10)


def test_beating_none_of_them_gives_one():
    assert _exact_p(0, 9) == pytest.approx(1.0)


# ---- per occasion ----

def test_each_occasion_is_ranked_in_its_own_draws():
    out = by_occasion(_draws(), _real(0.40))
    assert set(out.occasion) == set(OCCASIONS)
    assert (out.percentile == 100).all()
    assert out.p.unique().tolist() == [pytest.approx(1 / 5)]


def test_a_real_value_below_its_draws_is_not_ranked_above_them():
    out = by_occasion(_draws(), _real(0.10))
    assert (out.percentile == 0).all()
    assert out.p.to_numpy() == pytest.approx(1.0)


def test_the_lift_is_against_that_occasions_own_draw_mean():
    # draws centred on 0.30, 0.31 and 0.29: a lift off their pooled 0.30 would read 0.10 thrice
    out = by_occasion(_draws_with_spread(0.02), _real(0.40)).set_index("occasion")
    assert out.lift.to_dict() == pytest.approx({"G01": 0.10, "G03": 0.09, "G04": 0.11},
                                               abs=1e-12)


# ---- each draw on the real pairing's channels ----

def _one_occasion(real_cells, draw_cells, occasion="G01"):
    """Real and draw rows for one occasion of `game`; a missing or None cell is a rejected channel."""
    real = pd.DataFrame([{"chromophore": "hbo", "condition": "game", "occasion": occasion,
                          "label": c, "coherence": v} for c, v in real_cells.items()])
    draws = pd.DataFrame([{"chromophore": "hbo", "condition": "game", "occasion": occasion,
                           "draw": d, "label": c, "coherence": np.nan if v is None else v}
                          for d, cells in draw_cells.items() for c, v in cells.items()])
    return draws, real


def test_a_draw_missing_a_channel_is_compared_on_the_channels_it_shares():
    # real .30 over a b c; d1 is .05 under on a b, d2 level on all three, d3 .10 under on c
    draws, real = _one_occasion(
        {"a": 0.40, "b": 0.30, "c": 0.20},
        {"d1": {"a": 0.35, "b": 0.25, "c": None},
         "d2": {"a": 0.30, "b": 0.30, "c": 0.30},
         "d3": {"a": None, "b": None, "c": 0.10}})
    row = by_occasion(draws, real).iloc[0]
    assert row.coherence == pytest.approx(0.30)
    assert row.null_mean == pytest.approx(0.25)
    assert row.lift == pytest.approx(0.05)
    assert row.percentile == pytest.approx(200 / 3)
    assert row.p == pytest.approx(2 / 4)


def test_a_channel_only_the_draw_kept_does_not_enter_its_mean():
    draws, real = _one_occasion({"a": 0.40, "b": 0.20},
                                {"d1": {"a": 0.30, "b": 0.10, "c": 0.90}})
    (value, pool), = matched_pools(draws, real).values()
    assert value == pytest.approx(0.30)
    assert pool.tolist() == pytest.approx([0.20])


def test_a_draw_sharing_no_channel_is_left_out_of_the_pool():
    draws, real = _one_occasion({"a": 0.40, "b": 0.20},
                                {"d1": {"a": 0.30, "b": 0.30}, "d2": {"a": None, "b": None, "c": 0.5},
                                 "d3": {"a": 0.20, "b": 0.20}})
    row = by_occasion(draws, real).iloc[0]
    assert row.n_iter == 2
    assert row.null_mean == pytest.approx(0.25)


def test_with_nothing_rejected_the_pool_is_the_draws_own_means():
    draws, real = _one_occasion({"a": 0.40, "b": 0.20},
                                {"d1": {"a": 0.30, "b": 0.10}, "d2": {"a": 0.50, "b": 0.30}})
    (_, pool), = matched_pools(draws, real).values()
    assert sorted(pool.tolist()) == pytest.approx([0.20, 0.40])


# ---- a group of more than two: one value, the mean over its member pairings ----

TRIAD = [("sub-01", "sub-02"), ("sub-01", "sub-03"), ("sub-02", "sub-03")]


def _triad(real: dict, draws: dict, occasion="G02"):
    """``real`` {pairing: {label: value}}, ``draws`` {pairing: {draw: {label: value}}}."""
    base = {"chromophore": "hbo", "condition": "game", "occasion": occasion}
    real_rows = [{**base, "sub1": a, "sub2": b, "label": c, "coherence": v}
                 for (a, b), cells in real.items() for c, v in cells.items()]
    draw_rows = [{**base, "sub1": a, "sub2": b, "draw": k, "label": c,
                  "coherence": np.nan if v is None else v}
                 for (a, b), by_draw in draws.items() for k, cells in by_draw.items()
                 for c, v in cells.items()]
    return pd.DataFrame(draw_rows), pd.DataFrame(real_rows)


def test_a_group_is_its_pairings_mean_against_the_mean_of_each_draws_pairings():
    """Real .6 .3 .3, mean .4; each draw the mean of the same draw id across the three."""
    draws, real = _triad(
        {pair: {"a": v} for pair, v in zip(TRIAD, (0.6, 0.3, 0.3))},
        {pair: {k: {"a": v} for k, v in enumerate(values)}
         for pair, values in zip(TRIAD, ((0.1, 0.2, 0.3), (0.2, 0.2, 0.2), (0.3, 0.5, 0.1)))})
    (value, pool), = matched_pools(draws, real).values()
    assert value == pytest.approx(0.4)
    assert pool.tolist() == pytest.approx([0.2, 0.3, 0.2])


def test_a_draw_missing_one_of_the_pairings_is_left_out():
    draws, real = _triad({pair: {"a": 0.4} for pair in TRIAD},
                         {pair: {k: {"a": 0.2} for k in (range(2) if pair == TRIAD[2]
                                                         else range(3))} for pair in TRIAD})
    (_, pool), = matched_pools(draws, real).values()
    assert len(pool) == 2


def test_each_pairing_is_matched_on_its_own_channels():
    """Pairing 2's draw lost b: on a alone it is .3 under its real .5, so it counts as
    .4 - .3 = .1, and pairing 1's as .2. Pooling the cells across pairings would give .18."""
    draws, real = _triad(
        {TRIAD[0]: {"a": 0.4, "b": 0.2}, TRIAD[1]: {"a": 0.5, "b": 0.3}},
        {TRIAD[0]: {"d1": {"a": 0.3, "b": 0.1}}, TRIAD[1]: {"d1": {"a": 0.2, "b": None}}})
    (value, pool), = matched_pools(draws, real).values()
    assert value == pytest.approx(0.35)
    assert pool.tolist() == pytest.approx([0.15])


def test_a_pairing_without_draws_leaves_its_group_unranked(caplog):
    draws, real = _triad({pair: {"a": 0.4} for pair in TRIAD},
                         {pair: {0: {"a": 0.2}} for pair in TRIAD[:2]})
    assert matched_pools(draws, real) == {}
    assert "no draws for sub-02-sub-03" in caplog.text


def test_a_region_thin_in_one_pairing_leaves_out_that_group():
    """Its mean is over every pairing, so a region one pairing cannot fill is not its own."""
    cells = {"S1_D1": 0.4, "S1_D2": 0.4}
    draws, real = _triad({pair: (cells if pair != TRIAD[2] else {"S1_D1": 0.4})
                          for pair in TRIAD}, {pair: {0: cells} for pair in TRIAD})
    levels = {lv: r for g, lv, _, _, r in _variants(draws, real, {"front": list(cells)},
                                                    min_channels=2) if g == "roi"}
    assert "front" not in levels


def test_an_occasion_of_one_pairing_still_reads_with_its_member_columns():
    base = {"chromophore": "hbo", "condition": "game", "occasion": "G01", "label": "a",
            "sub1": "sub-01", "sub2": "sub-02"}
    real = pd.DataFrame([{**base, "coherence": 0.4}])
    draws = pd.DataFrame([{**base, "draw": k, "coherence": c} for k, c in enumerate((0.1, 0.3))])
    (value, pool), = matched_pools(draws, real).values()
    assert value == pytest.approx(0.4)
    assert sorted(pool.tolist()) == pytest.approx([0.1, 0.3])


def _write_class_tree(root, n_occasions=3):
    """Classes of a teacher and two students, teacher-student pairings at .5 and .3 and the
    students' at .2, every draw .1, the roles in the real tables' sidecars."""
    members = ("t", "s1", "s2")
    roles = {"t": "teacher", "s1": "student", "s2": "student"}
    real_values = {("t", "s1"): 0.5, ("t", "s2"): 0.3, ("s1", "s2"): 0.2}
    for k in range(1, n_occasions + 1):
        occ = f"G{k:02d}"
        d = root / f"group-{occ}" / "nirs"
        d.mkdir(parents=True)
        ids = {m: f"sub-{m}{occ}" for m in members}
        base = {"chromophore": "hbo", "condition": "game"}
        real = pd.DataFrame([{**base, "sub1": ids[a], "sub2": ids[b], "label": c,
                              "coherence": v} for (a, b), v in real_values.items()
                             for c in CHANNELS])
        draws = pd.DataFrame([{**base, "sub1": ids[a], "sub2": ids[b], "draw": f"x{j}",
                               "label": c, "coherence": 0.1 + 0.01 * j}
                              for (a, b) in real_values for j in range(5) for c in CHANNELS])
        draws.to_csv(d / name(occ, "main", "wtcbycond-pairnull-draws"), sep="\t", index=False)
        real_tsv = d / name(occ, "main", "wtc")
        real.to_csv(real_tsv, sep="\t", index=False)
        real_tsv.with_suffix(".json").write_text(json.dumps({"parameters": {
            "roi_min_channels": 2, "member_roles": {ids[m]: roles[m] for m in members}}}))


def test_with_roles_each_role_pair_is_read_on_its_own(tmp_path):
    _write_class_tree(tmp_path)
    write_group_null(tmp_path, "main", n_resample=200, seed=3)
    occ = pd.read_csv(tmp_path / _cohort("pair", "byoccasion"), sep="\t")

    means = (occ[occ.granularity != "channel"]
             .groupby(["granularity", "level"])["coherence"].first())
    assert means.to_dict() == pytest.approx({("whole", "whole"): (0.5 + 0.3 + 0.2) / 3,
                                             ("rolepair", "student-student"): 0.2,
                                             ("rolepair", "student-teacher"): 0.4})
    cohort = pd.read_csv(tmp_path / _cohort("pair", "cohort"), sep="\t")
    assert set(cohort.loc[cohort.granularity == "rolepair", "family"]) == {2}


def test_one_role_pair_in_the_whole_cohort_adds_no_family(tmp_path):
    """One role for everyone: its one role pair is the whole brain over again."""
    _write_class_tree(tmp_path)
    for side in tmp_path.rglob("*_stat-wtc_relmat.json"):
        params = json.loads(side.read_text())["parameters"]
        params["member_roles"] = {sub: "member" for sub in params["member_roles"]}
        side.write_text(json.dumps({"parameters": params}))
    write_group_null(tmp_path, "main", n_resample=200, seed=3)
    occ = pd.read_csv(tmp_path / _cohort("pair", "byoccasion"), sep="\t")
    assert "rolepair" not in set(occ.granularity)


def test_the_cohort_lift_is_the_mean_of_the_occasions_matched_lifts():
    # lifts .05, .10 and .00 on three occasions, each with one draw missing channel c
    parts = [_one_occasion({"a": 0.40, "b": 0.30, "c": 0.20},
                           {"d1": {"a": 0.40 - lift, "b": 0.30 - lift, "c": None},
                            "d2": {"a": 0.40 - lift, "b": 0.30 - lift, "c": 0.20 - lift}},
                           occasion=occ)
             for occ, lift in zip(OCCASIONS, (0.05, 0.10, 0.00))]
    draws = pd.concat([d for d, _ in parts], ignore_index=True)
    real = pd.concat([r for _, r in parts], ignore_index=True)
    coh = by_cohort(draws, real, n_resample=500, seed=1).set_index("test")
    assert coh.loc["paired", "lift"] == pytest.approx(0.05)
    assert coh.loc["paired", "null_mean"] == pytest.approx(0.25)
    assert coh.loc["resample", "lift"] == pytest.approx(0.05)


# ---- the cohort ----

def test_the_cohort_null_is_tighter_than_one_occasions():
    """Averaging occasions is the whole reason the cohort level can reject where a cell cannot."""
    draws, real = _draws(), _real(0.40)
    occ = by_occasion(draws, real)
    coh = _resample_rows(by_cohort(draws, real, n_resample=2000, seed=1))
    assert coh.null_sd.iloc[0] < occ.null_sd.mean()


def test_a_real_value_inside_the_draws_does_not_clear_the_cohort_null():
    draws = _draws()
    middle = draws.coherence.mean()
    coh = _resample_rows(by_cohort(draws, _real(middle), n_resample=2000, seed=1))
    assert coh.p.iloc[0] > 0.05


def test_a_real_value_far_above_them_clears_it():
    coh = _resample_rows(by_cohort(_draws(), _real(0.40), n_resample=2000, seed=1))
    assert coh.p.iloc[0] == pytest.approx(1 / 2001)


def test_the_cohort_p_is_reproducible_from_the_seed():
    a = by_cohort(_draws(), _real(0.305), n_resample=2000, seed=7).p.iloc[0]
    b = by_cohort(_draws(), _real(0.305), n_resample=2000, seed=7).p.iloc[0]
    assert a == b


# ---- reading a tree ----

def _write_tree(root, crossed=False):
    for occ in OCCASIONS:
        d = root / f"group-{occ}" / "nirs"
        d.mkdir(parents=True)
        draws = _draws()[lambda f: f.occasion == occ].drop(columns=["occasion"])
        real = _real(0.40)[lambda f: f.occasion == occ].drop(columns=["occasion"])
        if crossed:
            # a crossed real table carries the cells the homologous null never drew
            real = pd.concat([real.assign(label2=real.label),
                              real.assign(label2="S9_D9", coherence=0.9)])
        draws.to_csv(d / name(occ, "main", "wtcbycond-pairnull-draws"),
                     sep="\t", index=False)
        _with_whole_run(real).to_csv(d / name(occ, "main", "wtc"), sep="\t", index=False,
                                     na_rep="n/a")
        _record_minimum(d / name(occ, "main", "wtc"))


def _with_whole_run(real):
    """The whole run's rows ahead of the conditions', as the real table holds them; never read."""
    whole = real.drop(columns=["condition"]).assign(coherence=0.99)
    return pd.concat([whole, real], ignore_index=True)[list(real.columns)]


def _record_minimum(real_tsv, minimum=2):
    """The sidecar a real run writes, carrying the ROI minimum the cohort reads back."""
    real_tsv.with_suffix(".json").write_text(
        json.dumps({"parameters": {"roi_min_channels": minimum}}))


def test_both_tables_and_their_sidecars_are_written(tmp_path):
    _write_tree(tmp_path)
    written = write_group_null(tmp_path, "main", n_resample=500, seed=3)
    assert [p.name for p in written] == [
        _cohort("pair", "byoccasion"), _cohort("pair", "cohort")]
    for path in written:
        assert path.exists() and path.with_suffix(".json").exists()


def test_a_crossed_real_table_contributes_only_its_homologous_cells(tmp_path):
    """The null draws homologous pairings, so the 0.9 crossed cells have nothing to rank against."""
    _write_tree(tmp_path, crossed=True)
    write_group_null(tmp_path, "main", n_resample=500, seed=3)
    out = pd.read_csv(tmp_path / _cohort("pair", "cohort"), sep="\t")
    assert out.coherence.iloc[0] == pytest.approx(0.40)


def test_no_occasion_with_both_a_real_value_and_draws_gives_no_rows():
    real = _real(0.40)[lambda f: f.occasion != "G01"]
    draws = _draws()[lambda f: f.occasion == "G01"]
    assert by_occasion(draws, real).empty
    assert by_cohort(draws, real, n_resample=100, seed=0).empty


def test_a_channel_no_occasion_can_rank_is_left_out_rather_than_raised(tmp_path):
    """Rejected in the real member where it survives in the stand-ins, and the reverse elsewhere."""
    _write_tree(tmp_path)
    for occ in OCCASIONS:
        d = tmp_path / f"group-{occ}" / "nirs"
        kind = "wtc" if occ == "G01" else "wtcbycond-pairnull-draws"
        path = d / name(occ, "main", kind)
        table = pd.read_csv(path, sep="\t")
        table[table.label != "S1_D1"].to_csv(path, sep="\t", index=False)
    write_group_null(tmp_path, "main", n_resample=500, seed=3)
    out = pd.read_csv(tmp_path / _cohort("pair", "byoccasion"), sep="\t")
    assert "S1_D1" not in set(out.level)
    assert {"S1_D2", "S2_D1", "S2_D2"} <= set(out.level)


def test_a_tree_with_no_draws_says_what_has_to_run(tmp_path):
    (tmp_path / "group-G01" / "nirs").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="pair-null"):
        write_group_null(tmp_path, "main")


# ---- the same reading, whichever null drew the draws ----

def _as_phase_tree(root, shared=(False, False)):
    """The re-paired tree's draws renamed into the phase null's, each with the sidecar it writes."""
    _write_tree(root)
    for occ, each in zip(OCCASIONS, shared):
        d = root / f"group-{occ}" / "nirs"
        dst = d / name(occ, "main", "wtcbycond-phasenull-draws")
        (d / name(occ, "main", "wtcbycond-pairnull-draws")).rename(dst)
        if each is not None:
            dst.with_suffix(".json").write_text(
                json.dumps({"parameters": {"phase_null_shared": each}}))


def test_the_phase_nulls_draws_are_read_the_same_way(tmp_path):
    """Both nulls differ in what a draw is and in nothing above the cell."""
    _as_phase_tree(tmp_path)
    written = write_group_null(tmp_path, "main", null="phase", n_resample=500, seed=3)
    assert [p.name for p in written] == [
        _cohort("phase", "byoccasion"), _cohort("phase", "cohort")]
    side = json.loads(written[1].with_suffix(".json").read_text())
    assert side["parameters"]["null_kind"] == "phase"
    assert side["parameters"]["phase_null_shared"] is False


def test_coherence_phase_draws_that_do_not_say_how_they_scrambled_are_refused(tmp_path):
    _as_phase_tree(tmp_path, shared=(False, None))
    with pytest.raises(ValueError, match="whether its phase null shared its phases"):
        write_group_null(tmp_path, "main", null="phase", n_resample=200, seed=3)


def test_coherence_draws_scrambled_both_ways_are_not_pooled(tmp_path):
    _as_phase_tree(tmp_path, shared=(False, True))
    with pytest.raises(ValueError, match="scrambled both ways"):
        write_group_null(tmp_path, "main", null="phase", n_resample=200, seed=3)


# ---- every level the draws support, without being asked for one ----

ROI = {"front": ["S1_D1", "S1_D2"], "back": ["S2_D1", "S2_D2"], "thin": ["S1_D1"]}


def test_a_homologous_draw_offers_the_whole_brain_level_and_its_channels():
    got = [(g, lv) for g, lv, _, _, _ in _variants(_draws(), _real(0.4), None)]
    assert got[0] == ("whole", "whole")
    assert [lv for g, lv in got if g == "channel"] == CHANNELS


def test_a_crossed_null_tests_every_pairing_on_its_own():
    """The literature's family: every crossed pairing across occasions, apart from the diagonal's."""
    d = _draws().assign(label2=lambda f: f.label)
    crossed = pd.concat([d, d.assign(label2="S9_D9")])
    got = [lv for g, lv, pr, _, _ in _variants(crossed, crossed, None) if g == "channel" and pr == "all"]
    assert sorted(got) == sorted([f"{c}>{c}" for c in CHANNELS] + [f"{c}>S9_D9" for c in CHANNELS])


def test_a_homologous_null_offers_no_crossed_channel_family():
    d = _draws().assign(label2=lambda f: f.label)
    assert not [lv for g, lv, pr, _, _ in _variants(d, d, None) if g == "channel" and pr == "all"]


def test_a_crossed_draw_also_offers_every_pairing():
    d = _draws().assign(label2=lambda f: f.label)
    crossed = pd.concat([d, d.assign(label2="S9_D9", coherence=0.9)])
    got = [(g, lv, pr) for g, lv, pr, _, _ in _variants(crossed, crossed, None)]
    assert got[:2] == [("whole", "whole", "homologous"), ("whole", "whole", "all")]


def test_a_region_map_adds_one_level_per_region_that_has_the_channels():
    got = [lv for g, lv, _, _, _ in _variants(_draws(), _real(0.4), ROI, min_channels=2)
           if g == "roi"]
    assert got == ["front", "back"]          # "thin" has one channel, below min_channels


def test_the_written_tables_carry_the_granularity_and_the_pairings(tmp_path):
    _write_tree(tmp_path)
    written = write_group_null(tmp_path, "main", roi_map=ROI, n_resample=500, seed=3)
    cohort = pd.read_csv(written[1], sep="	")
    assert list(cohort.columns[:4]) == ["granularity", "level", "pairings", "condition"]
    assert set(cohort.granularity) == {"whole", "roi", "channel"}
    assert set(cohort[cohort.granularity == "channel"].level) == set(CHANNELS)
    assert set(cohort.pairings) == {"homologous"}


def test_a_region_is_the_mean_of_its_own_channels(tmp_path):
    _write_tree(tmp_path)
    written = write_group_null(tmp_path, "main", roi_map=ROI, n_resample=500, seed=3)
    cohort = _resample_rows(pd.read_csv(written[1], sep="	")).set_index("level")
    # every real value is 0.40 here, so each region reports it and so does the whole brain
    for level in ("whole", "front", "back"):
        assert cohort.loc[level, "coherence"] == pytest.approx(0.40)


# ---- the crossed region matrix ----


def _full_cross(frame, value=None):
    """Every channel against every channel, as a crossed run writes it."""
    out = pd.concat([frame.assign(label2=c) for c in CHANNELS], ignore_index=True)
    if value is not None:
        out["coherence"] = [value(a, b) for a, b in zip(out.label, out.label2)]
    return out


def _write_crossed_tree(root, value):
    for occ in OCCASIONS:
        d = root / f"group-{occ}" / "nirs"
        d.mkdir(parents=True)
        draws = _full_cross(_draws()[lambda f: f.occasion == occ]).drop(columns=["occasion"])
        real = _full_cross(_real(0.40)[lambda f: f.occasion == occ], value)
        draws.to_csv(d / name(occ, "main", "wtcbycond-pairnull-draws"), sep=SEP, index=False)
        _with_whole_run(real.drop(columns=["occasion"])).to_csv(
            d / name(occ, "main", "wtc"), sep=SEP, index=False, na_rep="n/a")
        _record_minimum(d / name(occ, "main", "wtc"))


def test_a_crossed_null_with_a_region_map_tests_every_ordered_region_pair():
    got = [lv for g, lv, pr, _, _ in _variants(_full_cross(_draws()), _full_cross(_real(0.4)), ROI,
                                               min_channels=2)
           if g == "roi" and pr == "all"]
    # "thin" has one channel, so it pairs with nothing
    assert got == ["front>front", "front>back", "back>front", "back>back"]


def test_a_homologous_null_offers_no_region_pairs():
    d = _draws().assign(label2=lambda f: f.label)
    assert not [lv for g, lv, pr, _, _ in _variants(d, d, ROI) if g == "roi" and pr == "all"]


def test_a_region_pair_needs_channels_on_both_sides_not_just_pairings():
    roi = {"front": ["S1_D1", "S1_D2"], "one": ["S2_D1"]}
    got = [lv for g, lv, pr, _, _ in _variants(_full_cross(_draws()), _full_cross(_real(0.4)), roi,
                                               min_channels=2)
           if g == "roi" and pr == "all"]
    # front>one rests on two pairings, but on one channel of the second member
    assert got == ["front>front"]


def test_a_region_pair_is_the_mean_of_every_pairing_between_them(tmp_path):
    front = set(ROI["front"])
    _write_crossed_tree(tmp_path, lambda a, b: 0.6 if a in front and b not in front else 0.4)
    written = write_group_null(tmp_path, "main", roi_map=ROI, n_resample=500, seed=3)
    cohort = _resample_rows(pd.read_csv(written[1], sep=SEP))
    pairs = cohort[(cohort.granularity == "roi") & (cohort.pairings == "all")].set_index("level")
    assert pairs.loc["front>back", "coherence"] == pytest.approx(0.6)
    assert pairs.loc["back>front", "coherence"] == pytest.approx(0.4)
    # the region pairs are one family, corrected apart from the homologous regions
    assert set(pairs["family"]) == {4}
    homologous = cohort[(cohort.granularity == "roi") & (cohort.pairings == "homologous")]
    assert set(homologous["family"]) == {2}


# ---- what the guards refuse ----

def test_a_table_predating_the_draw_column_is_refused_not_dropped(tmp_path):
    """Concatenating it gives NaN and the occasion leaves the groupby without a word."""
    _write_tree(tmp_path)
    stale = (tmp_path / "group-G03" / "nirs"
             / name("G03", "main", "wtcbycond-pairnull-draws"))
    old = pd.read_csv(stale, sep="\t").rename(columns={"draw": "stand_in"})
    old.to_csv(stale, sep="\t", index=False)
    with pytest.raises(ValueError, match="no draw column"):
        write_group_null(tmp_path, "main", n_resample=200, seed=3)


def test_a_real_table_holding_the_whole_run_only_is_refused(tmp_path):
    """What a tree from before the whole run and the conditions shared one table holds."""
    _write_tree(tmp_path)
    path = tmp_path / "group-G03" / "nirs" / name("G03", "main", "wtc")
    old = pd.read_csv(path, sep=SEP)
    old[old.condition.isna()].drop(columns=["condition"]).to_csv(path, sep=SEP, index=False)
    with pytest.raises(ValueError, match="no condition column"):
        write_group_null(tmp_path, "main", n_resample=200, seed=3)


def test_a_numeric_condition_label_meets_its_draws(tmp_path):
    """Beside the whole run's n/a a label "1" would read as 1.0 and match no draw."""
    _write_tree(tmp_path)
    for occ in OCCASIONS:
        for path in _paths(tmp_path, occ):
            table = pd.read_csv(path, sep=SEP, dtype={"condition": str})
            table["condition"] = table["condition"].replace("game", "1")
            table.to_csv(path, sep=SEP, index=False, na_rep="n/a")
    written = write_group_null(tmp_path, "main", n_resample=500, seed=3)
    out = _resample_rows(pd.read_csv(written[1], sep=SEP, dtype={"condition": str}))
    assert not out.empty and set(out.condition) == {"1"}
    assert out.coherence.iloc[0] == pytest.approx(0.40)


def _cross(path):
    """Turn a homologous table on disk into a crossed one, as a crossed run writes it."""
    d = pd.read_csv(path, sep=SEP)
    pd.concat([d.assign(label2=d.label),
               d.assign(label2="S9_D9", coherence=0.9)]).to_csv(path, sep=SEP, index=False)


def _paths(root, occ):
    nirs = root / f"group-{occ}" / "nirs"
    return (nirs / name(occ, "main", "wtcbycond-pairnull-draws"),
            nirs / name(occ, "main", "wtc"))


def test_a_part_crossed_tree_gets_no_all_pairings_level(tmp_path, caplog):
    """Crossed pairings for one occasion and homologous ones for the next are not one statistic."""
    _write_tree(tmp_path)
    for occ in OCCASIONS:
        _cross(_paths(tmp_path, occ)[1])
    _cross(_paths(tmp_path, "G03")[0])

    written = write_group_null(tmp_path, "main", n_resample=200, seed=3)
    assert set(pd.read_csv(written[1], sep=SEP).pairings) == {"homologous"}
    assert "homologous only" in caplog.text


def test_crossed_draws_against_a_diagonal_real_table_are_refused(tmp_path, caplog):
    """It would rank the homologous cells inside a null built from every crossed pairing."""
    _write_tree(tmp_path)
    for occ in OCCASIONS:
        _cross(_paths(tmp_path, occ)[0])

    written = write_group_null(tmp_path, "main", n_resample=200, seed=3)
    assert set(pd.read_csv(written[1], sep=SEP).pairings) == {"homologous"}
    assert "the real table is not" in caplog.text


def test_a_fully_crossed_tree_does_get_it(tmp_path):
    _write_tree(tmp_path)
    for occ in OCCASIONS:
        for f in _paths(tmp_path, occ):
            _cross(f)
    written = write_group_null(tmp_path, "main", n_resample=200, seed=3)
    assert set(pd.read_csv(written[1], sep=SEP).pairings) == {"homologous", "all"}


def test_two_bands_in_one_tree_are_refused(tmp_path):
    """A tree part way through a re-band has one band on the draws and another on the real."""
    import json

    _write_tree(tmp_path)
    for occ, band in zip(OCCASIONS, [(0.06, 0.15), (0.02, 0.10), (0.06, 0.15)]):
        for f in _paths(tmp_path, occ):
            f.with_suffix(".json").write_text(json.dumps(
                {"parameters": {"band_fmin": band[0], "band_fmax": band[1]}}))
    with pytest.raises(ValueError, match="not on one band"):
        write_group_null(tmp_path, "main", n_resample=200, seed=3)



# ---- the per-cell tables, corrected ----

def _write_cells(root, percentiles, n_iter=9):
    """One per-cell null table per occasion, the shape `nirspipe-hyper-pairnull` writes."""
    for occ, pcts in zip(OCCASIONS, percentiles):
        d = root / f"group-{occ}" / "nirs"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"chromophore": "hbo", "condition": "game", "sub1": "a", "sub2": "b",
                       "label": c, "coherence": 0.3, "percentile": pct, "n_iter": n_iter}
                      for c, pct in zip(CHANNELS, pcts)]).to_csv(
            d / name(occ, "main", "wtcbycond-pairnull"), sep="\t", index=False)


def test_the_percentile_becomes_an_exact_p(tmp_path):
    """Beating all k draws is rank 1 of k + 1, which is the finest p the pool can express."""
    _write_cells(tmp_path, [[100, 50, 0, 100]] * 3)
    out = by_cell(tmp_path, "main", "hbo", "repaired")
    assert out.p.min() == pytest.approx(1 / 10)
    assert out.p.max() == pytest.approx(1.0)


def test_a_percentile_over_three_draws_is_read_back_as_a_count(tmp_path):
    """Two of three beaten is 66.67 on disk and rank 2 of 4 back, p 0.5 exactly."""
    _write_cells(tmp_path, [[200 / 3] * 4] * 3, n_iter=3)
    out = by_cell(tmp_path, "main", "hbo", "repaired")
    assert set(out.p) == {0.5}


def test_nothing_clears_the_correction_when_the_cells_are_middling(tmp_path):
    _write_cells(tmp_path, [[50, 55, 45, 50]] * 3)
    out = by_cell(tmp_path, "main", "hbo", "repaired", method="fdr_bh")
    assert int((out.p_fdr_bh < 0.05).sum()) == 0


def test_the_cell_table_carries_only_the_raw_p_by_default(tmp_path):
    _write_cells(tmp_path, [[100, 50, 0, 100]] * 3)
    out = by_cell(tmp_path, "main", "hbo", "repaired")
    assert "p" in out.columns and "family" in out.columns
    assert not [c for c in out.columns if c.startswith("p_")]


def test_the_family_is_one_condition_and_is_reported(tmp_path):
    _write_cells(tmp_path, [[100, 100, 100, 100]] * 3)
    out = by_cell(tmp_path, "main", "hbo", "repaired")
    assert set(out.family) == {12}          # 3 occasions x 4 channels, one condition
    assert set(out.level) == {"channel"}


def test_the_cell_table_is_written_beside_the_others(tmp_path):
    _write_tree(tmp_path)
    _write_cells(tmp_path, [[100, 50, 0, 100]] * 3)
    written = write_group_null(tmp_path, "main", n_resample=200, seed=3)
    assert [p.name for p in written][0] == (
        _cohort("pair", "bycell"))


def test_no_cell_tables_is_not_an_error(tmp_path):
    """A tree with draws but no per-cell summary still gets the aggregate levels."""
    _write_tree(tmp_path)
    written = write_group_null(tmp_path, "main", n_resample=200, seed=3)
    assert all("bycell" not in p.name for p in written)
    assert len(written) == 2


# ---- the paired read ----

def test_both_reads_report_the_same_lift():
    """They divide it by different things; the numerator is one number."""
    coh = by_cohort(_draws(), _real(0.40), n_resample=2000, seed=1)
    assert set(coh.test) == {"resample", "paired"}
    lifts = coh.groupby("test").lift.first()
    assert lifts["paired"] == pytest.approx(lifts["resample"], abs=2e-3)


def test_the_paired_read_counts_the_occasions_not_the_resamples():
    coh = by_cohort(_draws(), _real(0.40), n_resample=2000, seed=1)
    paired = coh[coh.test == "paired"].iloc[0]
    assert paired.n_occasions == len(OCCASIONS)
    assert paired.df == len(OCCASIONS) - 1
    assert paired.n_positive == len(OCCASIONS)


def test_a_real_value_inside_the_draws_does_not_clear_the_paired_read_either():
    draws = _draws()
    coh = by_cohort(draws, _real(draws.coherence.mean()), n_resample=2000, seed=1)
    assert coh[coh.test == "paired"].p.iloc[0] > 0.05


def _draws_with_spread(delta, means=(0.30, 0.31, 0.29)):
    """Draws whose occasion means are fixed and whose within-occasion spread is `delta`.

    Each occasion gets its draws symmetrically either side of its mean, so the mean is exact
    whatever delta is. That isolates the one thing the two reads disagree about.
    """
    rows = []
    for occ, mean in zip(OCCASIONS, means):
        for k, s in enumerate((-1.5, -0.5, 0.5, 1.5)):
            for c in CHANNELS:
                rows.append({"chromophore": "hbo", "condition": "game", "occasion": occ,
                             "draw": DRAWS[k], "label": c, "coherence": mean + s * delta})
    return pd.DataFrame(rows)


def test_the_paired_read_is_blind_to_the_spread_between_an_occasions_own_draws():
    """The point of it: a surrogate whose draws agree too well narrows only the resample null.

    Phase randomisation applied one channel at a time does exactly that, which is why its
    aggregate cannot be ranked inside its own iterations.
    """
    tight, wide = _draws_with_spread(0.0005), _draws_with_spread(0.02)
    means = [d.groupby("occasion").coherence.mean() for d in (tight, wide)]
    assert means[0].sub(means[1]).abs().max() == pytest.approx(0.0, abs=1e-12)
    real = _real(0.32)
    out = [by_cohort(d, real, n_resample=4000, seed=2) for d in (tight, wide)]
    resample = [o[o.test == "resample"].p.iloc[0] for o in out]
    paired = [o[o.test == "paired"].p.iloc[0] for o in out]
    assert resample[0] < resample[1]              # the tighter draws make the null narrower
    assert paired[0] == pytest.approx(paired[1], abs=1e-9)


def test_two_occasions_carry_a_lift_but_no_t():
    """A t over two differences has no spread to estimate, so the row says so by omission."""
    draws = _draws()[lambda d: d.occasion.isin(OCCASIONS[:2])]
    real = _real(0.40)[lambda d: d.occasion.isin(OCCASIONS[:2])]
    paired = by_cohort(draws, real, n_resample=500, seed=1).query("test == 'paired'").iloc[0]
    assert np.isfinite(paired.lift)
    assert pd.isna(paired.get("p", np.nan))


# ---- the corrections on the cohort table ----

def _cohort_frame():
    """A cohort table shaped the way `write_group_null` assembles one."""
    rows = []
    for gran, levels in (("whole", ["whole"]), ("roi", ["front", "back"]),
                         ("channel", CHANNELS)):
        for level in levels:
            for cond, p in (("game", 0.001), ("rest", 0.30)):
                rows.append({"granularity": gran, "level": level, "pairings": "homologous",
                             "condition": cond, "test": "paired", "p": p, "lift": 0.01})
    return pd.DataFrame(rows)


def test_the_family_is_the_cells_of_one_condition_at_one_level():
    out = correct_cohort(_cohort_frame())
    fam = out.groupby(["granularity", "condition"]).family.first()
    assert fam[("channel", "game")] == len(CHANNELS)
    assert fam[("roi", "game")] == 2
    # one cell is no family, and the column has to say so rather than look corrected
    assert fam[("whole", "game")] == 1


@pytest.mark.parametrize("method", ["fdr_bh", "fdr_by", "holm", "bonferroni"])
def test_a_family_of_one_leaves_the_corrected_p_equal_to_p(method):
    out = correct_cohort(_cohort_frame(), method)
    whole = out[out.granularity == "whole"]
    assert whole[f"p_{method}"].to_numpy() == pytest.approx(whole.p.to_numpy())


def test_no_correction_is_the_default_and_adds_no_column():
    out = correct_cohort(_cohort_frame())
    assert not [c for c in out.columns if c.startswith("p_")]
    assert out["p"].to_numpy() == pytest.approx(_cohort_frame()["p"].to_numpy())


@pytest.mark.parametrize("method", ["fdr_bh", "fdr_by", "holm", "bonferroni"])
def test_a_chosen_method_is_the_multipletests_one_and_leaves_p_alone(method):
    from statsmodels.stats.multitest import multipletests
    out = correct_cohort(_cohort_frame(), method)
    part = out[(out.granularity == "channel") & (out.condition == "game")]
    assert part[f"p_{method}"].to_numpy() == pytest.approx(
        multipletests(part["p"], method=method)[1])
    assert out["p"].to_numpy() == pytest.approx(_cohort_frame()["p"].to_numpy())


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown p correction"):
        correct_cohort(_cohort_frame(), "fdr")


def test_bh_is_no_stricter_than_by_or_bonferroni():
    frame = _cohort_frame()
    # a family with a spread of p values, so the methods can differ
    ch = frame[(frame.granularity == "channel") & (frame.condition == "game")].index
    frame.loc[ch, "p"] = [0.001, 0.02, 0.2, 0.6]
    bh, by, bonf = (correct_cohort(frame, m).loc[ch, f"p_{m}"]
                    for m in ("fdr_bh", "fdr_by", "bonferroni"))
    assert (bh <= by + 1e-12).all()
    assert (bh <= bonf + 1e-12).all()


def test_a_row_without_a_p_takes_no_part_in_its_family():
    frame = _cohort_frame()
    ch = frame[(frame.granularity == "channel") & (frame.condition == "game")].index
    frame.loc[ch[0], "p"] = np.nan
    out = correct_cohort(frame, "fdr_bh")
    part = out.loc[ch]
    assert pd.isna(part.loc[ch[0], "p_fdr_bh"])
    assert (part["family"] == len(CHANNELS) - 1).all()


def test_the_written_cohort_table_is_uncorrected_by_default(tmp_path):
    _write_tree(tmp_path)
    written = write_group_null(tmp_path, "main", roi_map=ROI, n_resample=500, seed=3)
    cohort = pd.read_csv(written[1], sep="	")
    assert {"p", "family"} <= set(cohort.columns)
    assert not [c for c in cohort.columns if c.startswith("p_")]
    # the sidecar has to name the family, the count being uninterpretable without it
    side = json.loads(Path(str(written[1]).replace(".tsv", ".json")).read_text())
    assert side["parameters"]["p_correction"] == "none"
    assert "cohort_correction_family" in side["parameters"]


def test_a_chosen_correction_reaches_both_tables_beside_the_raw_p(tmp_path):
    _write_tree(tmp_path)
    _write_cells(tmp_path, [[100, 50, 0, 100]] * 3)
    written = write_group_null(tmp_path, "main", n_resample=200, seed=3,
                               p_correction="holm")
    cells, cohort = (pd.read_csv(p, sep="	") for p in (written[0], written[2]))
    for table in (cells, cohort):
        assert {"p", "p_holm", "family"} <= set(table.columns)
    side = json.loads(Path(str(written[2]).replace(".tsv", ".json")).read_text())
    assert side["parameters"]["p_correction"] == "holm"


def test_a_suffixed_region_map_gives_the_same_levels_as_a_bare_one():
    """Called straight, not through the CLI's loader: " hbo" on an entry must not hide its pair."""
    suffixed = {roi: [f"{ch} hbo" for ch in chans] for roi, chans in ROI.items()}
    draws, real = _full_cross(_draws()), _full_cross(_real(0.4))
    levels = lambda m: [(g, lv, pr) for g, lv, pr, _, _ in _variants(draws, real, m)]
    assert levels(suffixed) == levels(ROI)


# ---- the region minimum comes off the real tables ----

def _recorded(root: Path, gid: str, **params) -> str:
    tsv = root / f"{gid}.tsv"
    tsv.write_text("label\tcoherence\nS1_D1\t0.3\n")
    tsv.with_suffix(".json").write_text(json.dumps({"parameters": params}))
    return str(tsv)


def test_the_region_minimum_is_the_one_the_real_tables_recorded(tmp_path):
    from nirspipe.pipeline.hyper.pair_null_group import _roi_min_of

    paths = {_recorded(tmp_path, g, roi_min_channels=3) for g in OCCASIONS}
    assert _roi_min_of(paths) == 3


def test_real_tables_grouped_under_two_minimums_are_refused(tmp_path):
    from nirspipe.pipeline.hyper.pair_null_group import _roi_min_of

    paths = {_recorded(tmp_path, "G01", roi_min_channels=2),
             _recorded(tmp_path, "G03", roi_min_channels=3)}
    with pytest.raises(ValueError, match="different minimums"):
        _roi_min_of(paths)


def test_a_real_table_that_does_not_record_the_minimum_is_refused(tmp_path):
    from nirspipe.pipeline.hyper.pair_null_group import _roi_min_of

    paths = {_recorded(tmp_path, "G01", roi_min_channels=2),
             _recorded(tmp_path, "G03", band_fmin=0.06, band_fmax=0.15)}
    with pytest.raises(ValueError, match="do not record the ROI minimum"):
        _roi_min_of(paths)


# ---- the region gate, one occasion at a time ----

def test_by_default_one_surviving_channel_keeps_a_region():
    got = [lv for g, lv, _, _, _ in _variants(_draws(), _real(0.4), ROI) if g == "roi"]
    assert got == ["front", "back", "thin"]


def test_a_region_thin_in_one_occasion_leaves_out_that_occasion_only():
    """Its own ROI table blanks the region there, so the cohort must not rank it there."""
    real = _real(0.4)
    real = real[~((real.occasion == "G03") & (real.label == "S1_D2"))]

    levels = {lv: (d, r) for g, lv, _, d, r in _variants(_draws(), real, ROI, min_channels=2)
              if g == "roi"}

    d, r = levels["front"]
    assert set(r.occasion) == set(d.occasion) == {"G01", "G04"}
    assert set(levels["back"][1].occasion) == set(OCCASIONS)


def test_the_crossed_roi_cells_are_corrected_as_their_own_family(tmp_path):
    """Its (front, front) cell holds the cross pairings, so it is not the homologous front."""
    _write_cells(tmp_path, [[100, 100, 100, 100]] * 3)
    for occ in OCCASIONS:
        rows = [{"chromophore": "hbo", "condition": "game", "sub1": "a", "sub2": "b",
                 "label": a, "label2": b, "coherence": 0.3, "percentile": 100.0, "n_iter": 9}
                for a in ("front", "back") for b in ("front", "back")]
        pd.DataFrame(rows).to_csv(tmp_path / f"group-{occ}" / "nirs"
                                  / name(occ, "main", "wtcbycond-roichan-pairnull"),
                                  sep="	", index=False)

    out = by_cell(tmp_path, "main", "hbo", "repaired")

    crossed = out[out.level == "roi_crossed"]
    assert len(crossed) == 12 and set(crossed.family) == {12}     # 3 occasions x 4 pairs
    assert set(out[out.level == "channel"].family) == {12}
    assert set(crossed.label2) == {"front", "back"}


def test_the_command_writes_the_cohort_tests_methods_into_logs(tmp_path):
    """No report carries a cohort test, so its paragraph lands beside the tables."""
    from nirspipe.cli.hyper import cmd_group_null

    _write_tree(tmp_path)
    cmd_group_null(tmp_path, task="main", chroma=["hbo"], null=["repaired"],
                   roi_mapping=None, n_resample=500, seed=3, verbose=False)

    written = list((tmp_path / "logs").glob("*_desc-cohort_relmat_methods.md"))
    assert len(written) == 1
    text = written[0].read_text(encoding="utf-8")
    assert "Cohort-level coupling was tested against the re-paired null" in text
    assert "among 500 cohort means" in text
    assert "P values were not corrected for multiple comparisons." in text


# ---- the correlation, read above the cell ----

ISC_SETTINGS = {"isc_band_hz": None, "isc_whiten_s": 0.0, "isc_max_lag_s": 0.0,
                "phase_null_shared": False, "roi_min_channels": 2}


def _isc_rows(rs, condition="game"):
    """One occasion's homologous ISC rows, ``rs`` one r per channel."""
    rs = np.asarray(rs, dtype=float)
    return pd.DataFrame({"chromophore": "hbo", "condition": condition, "sub1": "sub-a",
                         "sub2": "sub-b", "label": CHANNELS, "label2": CHANNELS,
                         "r": rs, "r_z": np.arctanh(rs)})


def _write_isc_tree(root, rs, null="pair", whole_run_r=0.99, settings=ISC_SETTINGS):
    """Real ISC at ``rs`` per channel, and nine draws per cell scattered round zero."""
    rng = np.random.default_rng(1)
    for occ in OCCASIONS:
        d = root / f"group-{occ}" / "nirs"
        d.mkdir(parents=True, exist_ok=True)
        # the whole-run row has no condition; a cohort test over conditions must not read it
        real = pd.concat([_isc_rows(rs), _isc_rows([whole_run_r] * 4, condition=np.nan)])
        real_tsv = d / name(occ, "main", "iscpairs")
        real.to_csv(real_tsv, sep="\t", index=False)
        draws = pd.concat([_isc_rows(rng.normal(0, 0.03, 4)).assign(draw=f"x{i}")
                           for i in range(9)])
        draws_tsv = d / name(occ, "main", f"iscbycond-{null}null-draws")
        draws.to_csv(draws_tsv, sep="\t", index=False)
        for tsv in (real_tsv, draws_tsv):
            tsv.with_suffix(".json").write_text(json.dumps({"parameters": settings}))


def _isc_cohort(tmp_path, test, null="pair"):
    path = tmp_path / _cohort(null, "cohort", statistic="isc")
    return pd.read_csv(path, sep="\t"), json.loads(path.with_suffix(".json").read_text())


def test_the_two_sided_p_doubles_the_smaller_tail():
    pool = np.arange(19, dtype=float)
    assert _tailed_p(100.0, pool, two_sided=False) == pytest.approx(1 / 20)
    assert _tailed_p(100.0, pool, two_sided=True) == pytest.approx(2 / 20)
    assert _tailed_p(-100.0, pool, two_sided=True) == pytest.approx(2 / 20)
    assert _tailed_p(-100.0, pool, two_sided=False) == pytest.approx(1.0)


def test_a_negative_correlation_is_found_by_the_signed_test(tmp_path):
    """The case a one-tailed read misses: every channel anticorrelated."""
    _write_isc_tree(tmp_path, [-0.4] * 4)
    write_group_null(tmp_path, "main", n_resample=2000, seed=1)
    out, side = _isc_cohort(tmp_path, "signed")
    whole = out[(out.granularity == "whole") & (out.test == "resample")].iloc[0]
    assert whole.r_z == pytest.approx(-0.42364893019360184)            # arctanh(-0.4)
    assert whole.p < 0.01
    assert side["parameters"]["isc_test"] == "signed"
    assert side["parameters"]["two_sided"] is True
    assert side["parameters"]["measure"] == "isc"


def test_opposite_signs_cancel_when_signed_and_count_by_magnitude(tmp_path):
    """The trade between the two reads, at the whole-brain mean."""
    _write_isc_tree(tmp_path, [0.4, -0.4, 0.4, -0.4])
    write_group_null(tmp_path, "main", n_resample=2000, seed=1)
    signed, _ = _isc_cohort(tmp_path, "signed")
    write_group_null(tmp_path, "main", n_resample=2000, seed=1, isc_test="magnitude")
    magnitude, side = _isc_cohort(tmp_path, "magnitude")

    def whole(t):
        return t[(t.granularity == "whole") & (t.test == "resample")].iloc[0]
    assert whole(signed).p > 0.2
    assert whole(magnitude).abs_r_z == pytest.approx(0.42364893019360184)   # arctanh(0.4)
    assert whole(magnitude).p < 0.01
    assert side["parameters"]["two_sided"] is False


def test_the_whole_run_rows_take_no_part(tmp_path):
    _write_isc_tree(tmp_path, [0.2] * 4, whole_run_r=0.99)
    write_group_null(tmp_path, "main", n_resample=500, seed=1)
    out, _ = _isc_cohort(tmp_path, "signed")
    assert out.r_z.iloc[0] == pytest.approx(0.2027325540540822)         # arctanh(0.2)


def test_each_statistic_gets_its_own_methods_paragraph(tmp_path):
    from nirspipe.cli.hyper import cmd_group_null

    _write_tree(tmp_path)
    _write_isc_tree(tmp_path, [0.3] * 4)
    cmd_group_null(tmp_path, task="main", chroma=["hbo"], null=["repaired"], roi_mapping=None,
                   n_resample=200, seed=3, verbose=False)
    texts = {p.name: p.read_text(encoding="utf-8")
             for p in (tmp_path / "logs").glob("*_desc-cohort_relmat_methods.md")}
    assert len(texts) == 2
    isc = next(t for n, t in texts.items() if "stat-isc" in n)
    wtc = next(t for n, t in texts.items() if "stat-wtc" in n)
    assert "the inter-subject correlation, Fisher z-transformed, was averaged" in isc
    assert "a two-tailed paired t test" in isc
    assert "coherence was averaged" in wtc and "a one-tailed paired t test" in wtc


def test_the_isc_cells_are_ranked_from_the_draws_with_the_same_tail(tmp_path):
    _write_isc_tree(tmp_path, [-0.4] * 4)
    write_group_null(tmp_path, "main", n_resample=500, seed=1, p_correction="fdr_bh")
    cells = pd.read_csv(tmp_path / _cohort("pair", "bycell", statistic="isc"), sep="\t")
    assert len(cells) == len(OCCASIONS) * len(CHANNELS)
    # beating every one of nine draws, two-sided: 2/10
    assert cells.p.to_numpy() == pytest.approx(0.2)
    assert {"r_z", "p_fdr_bh", "family"} <= set(cells.columns)
    assert set(cells.family) == {12}


def test_both_statistics_are_read_in_one_call_when_both_drew(tmp_path):
    _write_tree(tmp_path)
    _write_isc_tree(tmp_path, [0.3] * 4)
    written = write_group_null(tmp_path, "main", n_resample=200, seed=3)
    assert [p.name for p in written] == [
        _cohort("pair", "byoccasion"), _cohort("pair", "cohort"),
        _cohort("pair", "bycell", statistic="isc"),
        _cohort("pair", "byoccasion", statistic="isc"),
        _cohort("pair", "cohort", statistic="isc")]


def test_the_phase_nulls_isc_draws_are_read_too(tmp_path):
    _write_isc_tree(tmp_path, [0.3] * 4, null="phase")
    written = write_group_null(tmp_path, "main", null="phase", n_resample=200, seed=3)
    assert _cohort("phase", "cohort", statistic="isc") in [p.name for p in written]


def test_the_cohort_tables_record_whether_the_phase_null_shared_its_phases(tmp_path):
    _write_isc_tree(tmp_path, [0.3] * 4, null="phase",
                    settings={**ISC_SETTINGS, "phase_null_shared": True})
    write_group_null(tmp_path, "main", null="phase", n_resample=200, seed=3)
    _, side = _isc_cohort(tmp_path, "signed", null="phase")
    assert side["parameters"]["phase_null_shared"] is True


def test_phase_draws_that_do_not_say_how_they_scrambled_are_refused(tmp_path):
    settings = {k: v for k, v in ISC_SETTINGS.items() if k != "phase_null_shared"}
    _write_isc_tree(tmp_path, [0.3] * 4, null="phase", settings=settings)
    with pytest.raises(ValueError, match="whether its phase null shared its phases"):
        write_group_null(tmp_path, "main", null="phase", n_resample=200, seed=3)


def test_dyads_scrambled_both_ways_are_not_pooled(tmp_path):
    _write_isc_tree(tmp_path, [0.3] * 4, null="phase")
    side = next(tmp_path.rglob("*_stat-isc_relmat.json"))
    side.write_text(json.dumps({"parameters": {**ISC_SETTINGS, "phase_null_shared": True}}))
    with pytest.raises(ValueError, match="not computed alike.*shared phases"):
        write_group_null(tmp_path, "main", null="phase", n_resample=200, seed=3)


def test_the_re_paired_null_does_not_ask_how_a_phase_null_scrambled(tmp_path):
    settings = {k: v for k, v in ISC_SETTINGS.items() if k != "phase_null_shared"}
    _write_isc_tree(tmp_path, [0.3] * 4, settings=settings)
    assert write_group_null(tmp_path, "main", n_resample=200, seed=3)


def test_isc_tables_that_do_not_record_their_settings_are_refused(tmp_path):
    _write_isc_tree(tmp_path, [0.3] * 4, settings={})
    with pytest.raises(ValueError, match="does not record how its correlation"):
        write_group_null(tmp_path, "main", n_resample=200, seed=3)


def test_isc_draws_and_real_tables_computed_differently_are_refused(tmp_path):
    _write_isc_tree(tmp_path, [0.3] * 4)
    side = next(tmp_path.rglob("*_stat-isc_relmat.json"))
    side.write_text(json.dumps({"parameters": {**ISC_SETTINGS, "isc_max_lag_s": 2.0}}))
    with pytest.raises(ValueError, match="not computed alike"):
        write_group_null(tmp_path, "main", n_resample=200, seed=3)


@pytest.mark.parametrize("test, null, whiten_s, flagged", [
    ("magnitude", "phase", 0.0, True), ("magnitude", "phase", None, True),
    ("magnitude", "phase", 10.0, False), ("magnitude", "repaired", 0.0, False),
    ("signed", "phase", 0.0, False)])
def test_only_unwhitened_magnitude_against_the_phase_null_is_flagged(test, null, whiten_s,
                                                                      flagged):
    assert magnitude_against_unwhitened_phase(test, null, whiten_s) is flagged


def test_unwhitened_magnitude_against_the_phase_null_warns_and_says_so_on_the_page(
        tmp_path, caplog):
    from nirspipe.qc.hyper.groupnull_report import write_groupnull_report

    _write_isc_tree(tmp_path, [0.3] * 4, null="phase")
    write_group_null(tmp_path, "main", null="phase", n_resample=200, seed=3,
                     isc_test="magnitude")
    assert "|r| on unwhitened correlations" in caplog.text
    _, side = _isc_cohort(tmp_path, "magnitude", null="phase")
    assert side["parameters"]["isc_whiten_s"] == 0.0
    page = write_groupnull_report(tmp_path, "main").read_text(encoding="utf-8")
    assert "|r| was tested against the phase-scrambled null" in page


def test_the_signed_test_against_the_phase_null_says_nothing(tmp_path, caplog):
    from nirspipe.qc.hyper.groupnull_report import write_groupnull_report

    _write_isc_tree(tmp_path, [0.3] * 4, null="phase")
    write_group_null(tmp_path, "main", null="phase", n_resample=200, seed=3)
    assert "|r| on unwhitened correlations" not in caplog.text
    page = write_groupnull_report(tmp_path, "main").read_text(encoding="utf-8")
    assert "|r| was tested against the phase-scrambled null" not in page


def test_an_unknown_isc_test_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown ISC test"):
        write_group_null(tmp_path, "main", isc_test="two-sided")


def test_the_paired_read_is_two_sided_when_asked():
    draws = _draws(level=0.30)
    real = _real(0.10)
    one = by_cohort(draws, real, n_resample=200, seed=1).query("test == 'paired'").iloc[0]
    two = by_cohort(draws, real, n_resample=200, seed=1,
                    two_sided=True).query("test == 'paired'").iloc[0]
    assert one.p > 0.99
    assert two.p < 0.01


def _write_crossed_isc_tree(root, r=0.3):
    """Every ordered channel pairing, real and drawn, as a crossed run and null write them."""
    rng = np.random.default_rng(2)
    pairs = [(a, b) for a in CHANNELS for b in CHANNELS]

    def rows(values):
        values = np.asarray(values, dtype=float)
        return pd.DataFrame({"chromophore": "hbo", "condition": "game", "sub1": "sub-a",
                             "sub2": "sub-b", "label": [a for a, _ in pairs],
                             "label2": [b for _, b in pairs], "r": values,
                             "r_z": np.arctanh(values)})
    for occ in OCCASIONS:
        d = root / f"group-{occ}" / "nirs"
        d.mkdir(parents=True, exist_ok=True)
        real_tsv = d / name(occ, "main", "iscpairs")
        rows([r] * len(pairs)).to_csv(real_tsv, sep="\t", index=False)
        draws_tsv = d / name(occ, "main", "iscbycond-pairnull-draws")
        pd.concat([rows(rng.normal(0, 0.03, len(pairs))).assign(draw=f"x{i}")
                   for i in range(9)]).to_csv(draws_tsv, sep="\t", index=False)
        for tsv in (real_tsv, draws_tsv):
            tsv.with_suffix(".json").write_text(json.dumps({"parameters": ISC_SETTINGS}))


def test_the_isc_cells_include_each_region_corrected_apart_from_the_channels(tmp_path):
    _write_isc_tree(tmp_path, [-0.4] * 4)
    write_group_null(tmp_path, "main", roi_map=ROI, n_resample=200, seed=3,
                     p_correction="holm")
    cells = pd.read_csv(tmp_path / _cohort("pair", "bycell", statistic="isc"), sep="\t")
    roi = cells[cells.level == "roi"]
    # "thin" has one channel per member, under the recorded minimum of two
    assert set(roi.label) == {"front", "back"}
    assert len(roi) == 2 * len(OCCASIONS) and set(roi.family) == {6}
    assert roi.r_z.to_numpy() == pytest.approx(-0.42364893019360184)    # arctanh(-0.4)
    # beating all nine draws, two-sided
    assert roi.p.to_numpy() == pytest.approx(0.2)
    assert set(cells[cells.level == "channel"].family) == {12}


def test_a_crossed_isc_null_gives_the_region_matrix_its_own_family(tmp_path):
    _write_crossed_isc_tree(tmp_path)
    write_group_null(tmp_path, "main", roi_map=ROI, n_resample=200, seed=3)
    cells = pd.read_csv(tmp_path / _cohort("pair", "bycell", statistic="isc"), sep="\t")
    crossed = cells[cells.level == "roi_crossed"]
    assert {(a, b) for a, b in zip(crossed.label, crossed.label2)} == {
        ("front", "front"), ("front", "back"), ("back", "front"), ("back", "back")}
    assert len(crossed) == 4 * len(OCCASIONS) and set(crossed.family) == {12}
    assert set(cells[cells.level == "channel"].family) == {16 * len(OCCASIONS)}


# ---- one command over several nulls and chromophores ----

def _add_phase_draws(root):
    """The re-paired draws again under the phase null's name, so both nulls are on disk."""
    for occ in OCCASIONS:
        d = root / f"group-{occ}" / "nirs"
        src = d / name(occ, "main", "wtcbycond-pairnull-draws")
        dst = d / name(occ, "main", "wtcbycond-phasenull-draws")
        dst.write_bytes(src.read_bytes())
        dst.with_suffix(".json").write_text(json.dumps({"parameters": {"phase_null_shared": False}}))


def test_groupnull_takes_several_nulls_and_chromophores():
    from nirspipe.cli.hyper import _parsers
    parser = _parsers()["nirspipe-hyper-groupnull"]
    args = parser.parse_args(["out", "group", "--task-label", "main",
                              "--null", "repaired", "phase", "--chroma", "hbo", "hbr"])
    assert args.null == ["repaired", "phase"] and args.chroma == ["hbo", "hbr"]
    args = parser.parse_args(["out", "group", "--task-label", "main"])
    assert args.null is None and args.chroma is None


def test_without_null_every_null_on_disk_is_read_as_if_asked_alone(tmp_path):
    from nirspipe.cli.hyper import cmd_group_null

    _write_tree(tmp_path)
    _add_phase_draws(tmp_path)
    cmd_group_null(tmp_path, task="main", chroma=["hbo"], null=None, roi_mapping=None,
                   n_resample=300, seed=3, verbose=False)
    together = {k: pd.read_csv(tmp_path / _cohort(k, "cohort"), sep=SEP) for k in ("pair", "phase")}

    alone = tmp_path / "alone"
    alone.mkdir()
    _write_tree(alone)
    write_group_null(alone, "main", null="repaired", n_resample=300, seed=3)
    pd.testing.assert_frame_equal(together["pair"],
                                  pd.read_csv(alone / _cohort("pair", "cohort"), sep=SEP))
    assert (tmp_path / "task-main_desc-groupnull_report.html").exists()


def test_groupnull_without_any_draws_stops(tmp_path, capsys):
    from nirspipe.cli.hyper import cmd_group_null

    with pytest.raises(SystemExit):
        cmd_group_null(tmp_path, task="main", chroma=["hbo"], null=None, roi_mapping=None,
                       n_resample=300, seed=3, verbose=False)
    assert "no null draws for task main" in capsys.readouterr().err


def _add_hbr_rows(root):
    """Every table again under HbR, a little lower, so the chromophores differ."""
    for path in root.rglob("group-*_task-main_*.tsv"):
        frame = pd.read_csv(path, sep=SEP)
        hbr = frame.assign(chromophore="hbr")
        if "coherence" in hbr:
            hbr["coherence"] = hbr["coherence"] - 0.05
        pd.concat([frame, hbr]).to_csv(path, sep=SEP, index=False)


def test_without_chroma_every_chromophore_the_draws_hold_is_read_on_its_own(tmp_path):
    from nirspipe.cli.hyper import cmd_group_null

    _write_tree(tmp_path)
    _add_hbr_rows(tmp_path)
    cmd_group_null(tmp_path, task="main", chroma=None, null=["repaired"], roi_mapping=None,
                   n_resample=300, seed=3, verbose=False)
    hbo = pd.read_csv(tmp_path / _cohort("pair", "cohort", chromophore="hbo"), sep=SEP)
    hbr = pd.read_csv(tmp_path / _cohort("pair", "cohort", chromophore="hbr"), sep=SEP)
    # each read from its own rows: the HbR coherence sits 0.05 under the HbO one
    assert hbr.coherence.to_numpy() == pytest.approx(hbo.coherence.to_numpy() - 0.05)


def test_a_chromophore_the_draws_do_not_hold_is_refused(tmp_path):
    _write_tree(tmp_path)
    with pytest.raises(ValueError, match="no hbr draws from the repaired null.*they hold hbo"):
        write_group_null(tmp_path, "main", chroma="hbr", n_resample=300, seed=3)
