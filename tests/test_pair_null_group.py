"""The two levels above the cell, and the resolution each one can express.

The point of averaging before ranking is that a pool of 22 cannot express a p under 1/23 per
cell, so these hold the arithmetic that makes a cohort verdict possible at all.
"""

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.pipeline.pair_null_group import (
    _exact_p, by_cohort, by_occasion, write_group_null)

CHANNELS = ["S1_D1", "S1_D2", "S2_D1", "S2_D2"]
OCCASIONS = ["d01", "d03", "d04"]
STAND_INS = [f"sub-p2d{n:02d}" for n in (5, 6, 7, 8)]


def _draws(level=0.30, spread=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"chromophore": "hbo", "condition": "game", "occasion": o, "stand_in": s,
             "label": c, "coherence": level + rng.normal(0, spread)}
            for o in OCCASIONS for s in STAND_INS for c in CHANNELS]
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
        own = draws[draws.occasion == occ].groupby("stand_in").coherence.mean().mean()
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
