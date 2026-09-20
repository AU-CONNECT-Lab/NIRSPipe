"""The two levels above the cell, and the resolution each one can express.

The point of averaging before ranking is that a pool of 22 cannot express a p under 1/23 per
cell, so these hold the arithmetic that makes a cohort verdict possible at all.
"""

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.pipeline.pair_null_group import (
    _exact_p, _variants, by_cell, by_cohort, by_occasion, write_group_null)

CHANNELS = ["S1_D1", "S1_D2", "S2_D1", "S2_D2"]
OCCASIONS = ["d01", "d03", "d04"]
DRAWS = [f"sub-p2d{n:02d}" for n in (5, 6, 7, 8)]


def _draws(level=0.30, spread=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"chromophore": "hbo", "condition": "game", "occasion": o, "draw": s,
             "label": c, "coherence": level + rng.normal(0, spread)}
            for o in OCCASIONS for s in DRAWS for c in CHANNELS]
    return pd.DataFrame(rows)


def _real(level):
    return pd.DataFrame([{"chromophore": "hbo", "condition": "game", "occasion": o,
                          "label": c, "coherence": level}
                         for o in OCCASIONS for c in CHANNELS])


# ---- the exact p ----

def test_beating_every_draw_gives_the_finest_p_the_pool_allows():
    assert _exact_p(22, 22) == pytest.approx(1 / 23)
    assert _exact_p(4, 4) == pytest.approx(1 / 5)


def test_beating_none_of_them_gives_one():
    assert _exact_p(0, 22) == pytest.approx(1.0)


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
    draws = _draws()
    out = by_occasion(draws, _real(0.40)).set_index("occasion")
    for occ in OCCASIONS:
        own = draws[draws.occasion == occ].groupby("draw").coherence.mean().mean()
        assert out.loc[occ, "lift"] == pytest.approx(0.40 - own, abs=1e-9)


# ---- the cohort ----

def test_the_cohort_null_is_tighter_than_one_occasions():
    """Averaging occasions is the whole reason the cohort level can reject where a cell cannot."""
    draws, real = _draws(), _real(0.40)
    occ = by_occasion(draws, real)
    coh = by_cohort(draws, real, n_resample=2000, seed=1)
    assert coh.null_sd.iloc[0] < occ.null_sd.mean()


def test_a_real_value_inside_the_draws_does_not_clear_the_cohort_null():
    draws = _draws()
    middle = draws.coherence.mean()
    coh = by_cohort(draws, _real(middle), n_resample=2000, seed=1)
    assert coh.p.iloc[0] > 0.05


def test_a_real_value_far_above_them_clears_it():
    coh = by_cohort(_draws(), _real(0.40), n_resample=2000, seed=1)
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
        draws.to_csv(d / f"group-{occ}_task-full_hyper-wtcbycond-pairnull-draws.tsv",
                     sep="\t", index=False)
        real.to_csv(d / f"group-{occ}_task-full_hyper-wtcbycond.tsv", sep="\t", index=False)


def test_both_tables_and_their_sidecars_are_written(tmp_path):
    _write_tree(tmp_path)
    written = write_group_null(tmp_path, n_resample=500, seed=3)
    assert [p.name for p in written] == [
        "group_hyper_wtc_bycondition_pairnull_byoccasion.tsv",
        "group_hyper_wtc_bycondition_pairnull_cohort.tsv"]
    for path in written:
        assert path.exists() and path.with_suffix(".json").exists()


def test_a_crossed_real_table_contributes_only_its_homologous_cells(tmp_path):
    """The null draws homologous pairings, so the 0.9 crossed cells have nothing to rank against."""
    _write_tree(tmp_path, crossed=True)
    write_group_null(tmp_path, n_resample=500, seed=3)
    out = pd.read_csv(tmp_path / "group_hyper_wtc_bycondition_pairnull_cohort.tsv", sep="\t")
    assert out.coherence.iloc[0] == pytest.approx(0.40)


def test_a_tree_with_no_draws_says_what_has_to_run(tmp_path):
    (tmp_path / "group-d01" / "nirs").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="pair-null"):
        write_group_null(tmp_path)


# ---- the same reading, whichever null drew the draws ----

def test_the_phase_nulls_draws_are_read_the_same_way(tmp_path):
    """Both nulls differ in what a draw is and in nothing above the cell."""
    _write_tree(tmp_path)
    for occ in OCCASIONS:
        d = tmp_path / f"group-{occ}" / "nirs"
        src = d / f"group-{occ}_task-full_hyper-wtcbycond-pairnull-draws.tsv"
        src.rename(d / f"group-{occ}_task-full_hyper-wtcbycond-phasenull-draws.tsv")
    written = write_group_null(tmp_path, null="phase", n_resample=500, seed=3)
    assert [p.name for p in written] == [
        "group_hyper_wtc_bycondition_phasenull_byoccasion.tsv",
        "group_hyper_wtc_bycondition_phasenull_cohort.tsv"]
    import json
    side = json.loads(written[1].with_suffix(".json").read_text())
    assert side["parameters"]["null_kind"] == "phase"


# ---- every level the draws support, without being asked for one ----

ROI = {"front": ["S1_D1", "S1_D2"], "back": ["S2_D1", "S2_D2"], "thin": ["S1_D1"]}


def test_a_homologous_draw_offers_the_whole_brain_level_only():
    got = [(lv, pr) for lv, pr, _, _ in _variants(_draws(), _real(0.4), None)]
    assert got == [("whole", "homologous")]


def test_a_crossed_draw_also_offers_every_pairing():
    d = _draws().assign(label2=lambda f: f.label)
    crossed = pd.concat([d, d.assign(label2="S9_D9", coherence=0.9)])
    got = [(lv, pr) for lv, pr, _, _ in _variants(crossed, crossed, None)]
    assert got == [("whole", "homologous"), ("whole", "all")]


def test_a_region_map_adds_one_level_per_region_that_has_the_channels():
    got = [(lv, pr) for lv, pr, _, _ in _variants(_draws(), _real(0.4), ROI)]
    assert got == [("whole", "homologous"), ("front", "homologous"), ("back", "homologous")]


def test_the_written_tables_carry_the_level_and_the_pairings(tmp_path):
    _write_tree(tmp_path)
    written = write_group_null(tmp_path, roi_map=ROI, n_resample=500, seed=3)
    cohort = pd.read_csv(written[1], sep="	")
    assert list(cohort.columns[:3]) == ["level", "pairings", "condition"]
    assert set(cohort.level) == {"whole", "front", "back"}
    assert set(cohort.pairings) == {"homologous"}


def test_a_region_is_the_mean_of_its_own_channels(tmp_path):
    _write_tree(tmp_path)
    written = write_group_null(tmp_path, roi_map=ROI, n_resample=500, seed=3)
    cohort = pd.read_csv(written[1], sep="	").set_index("level")
    # every real value is 0.40 here, so each region reports it and so does the whole brain
    for level in ("whole", "front", "back"):
        assert cohort.loc[level, "coherence"] == pytest.approx(0.40)


# ---- what the guards refuse ----

def test_a_table_predating_the_draw_column_is_refused_not_dropped(tmp_path):
    """Concatenating it gives NaN and the occasion leaves the groupby without a word."""
    _write_tree(tmp_path)
    stale = (tmp_path / "group-d03" / "nirs"
             / "group-d03_task-full_hyper-wtcbycond-pairnull-draws.tsv")
    old = pd.read_csv(stale, sep="\t").rename(columns={"draw": "stand_in"})
    old.to_csv(stale, sep="\t", index=False)
    with pytest.raises(ValueError, match="no draw column"):
        write_group_null(tmp_path, n_resample=200, seed=3)


def test_a_part_crossed_tree_gets_no_all_pairings_level(tmp_path, caplog):
    """196 pairings for one occasion and 14 for the next is not one statistic."""
    _write_tree(tmp_path)
    one = (tmp_path / "group-d03" / "nirs"
           / "group-d03_task-full_hyper-wtcbycond-pairnull-draws.tsv")
    d = pd.read_csv(one, sep="\t")
    crossed = pd.concat([d.assign(label2=d.label),
                         d.assign(label2="S9_D9", coherence=0.9)])
    crossed.to_csv(one, sep="\t", index=False)

    written = write_group_null(tmp_path, n_resample=200, seed=3)
    cohort = pd.read_csv(written[1], sep="\t")
    assert set(cohort.pairings) == {"homologous"}
    assert "homologous only" in caplog.text


def test_a_fully_crossed_tree_does_get_it(tmp_path):
    _write_tree(tmp_path)
    for occ in OCCASIONS:
        f = (tmp_path / f"group-{occ}" / "nirs"
             / f"group-{occ}_task-full_hyper-wtcbycond-pairnull-draws.tsv")
        d = pd.read_csv(f, sep="\t")
        pd.concat([d.assign(label2=d.label),
                   d.assign(label2="S9_D9", coherence=0.9)]).to_csv(f, sep="\t", index=False)
    written = write_group_null(tmp_path, n_resample=200, seed=3)
    cohort = pd.read_csv(written[1], sep="\t")
    assert set(cohort.pairings) == {"homologous", "all"}


# ---- the per-cell tables, corrected ----

def _write_cells(root, percentiles, n_iter=22):
    """One per-cell null table per occasion, the shape `pair-null` writes."""
    for occ, pcts in zip(OCCASIONS, percentiles):
        d = root / f"group-{occ}" / "nirs"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"chromophore": "hbo", "condition": "game", "sub1": "a", "sub2": "b",
                       "label": c, "coherence": 0.3, "percentile": pct, "n_iter": n_iter}
                      for c, pct in zip(CHANNELS, pcts)]).to_csv(
            d / f"group-{occ}_task-full_hyper-wtcbycond-pairnull.tsv", sep="\t", index=False)


def test_the_percentile_becomes_an_exact_p(tmp_path):
    """Beating all 22 is rank 1 of 23, which is the finest p the pool can express."""
    _write_cells(tmp_path, [[100, 50, 0, 100]] * 3)
    out = by_cell(tmp_path, "full", "hbo", "repaired")
    assert out.p.min() == pytest.approx(1 / 23)
    assert out.p.max() == pytest.approx(1.0)


def test_nothing_clears_the_correction_when_the_cells_are_middling(tmp_path):
    _write_cells(tmp_path, [[50, 55, 45, 50]] * 3)
    out = by_cell(tmp_path, "full", "hbo", "repaired")
    assert int((out.q < 0.05).sum()) == 0


def test_the_family_is_one_condition_and_is_reported(tmp_path):
    _write_cells(tmp_path, [[100, 100, 100, 100]] * 3)
    out = by_cell(tmp_path, "full", "hbo", "repaired")
    assert set(out.family) == {12}          # 3 occasions x 4 channels, one condition
    assert set(out.level) == {"channel"}


def test_the_cell_table_is_written_beside_the_others(tmp_path):
    _write_tree(tmp_path)
    _write_cells(tmp_path, [[100, 50, 0, 100]] * 3)
    written = write_group_null(tmp_path, n_resample=200, seed=3)
    assert [p.name for p in written][0] == (
        "group_hyper_wtc_bycondition_pairnull_bycell.tsv")


def test_no_cell_tables_is_not_an_error(tmp_path):
    """A tree with draws but no per-cell summary still gets the aggregate levels."""
    _write_tree(tmp_path)
    written = write_group_null(tmp_path, n_resample=200, seed=3)
    assert all("bycell" not in p.name for p in written)
    assert len(written) == 2
