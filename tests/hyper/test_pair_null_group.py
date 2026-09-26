"""The two levels above the cell, and the resolution each one can express.

The point of averaging before ranking is that a pool of k cannot express a p under 1/(k+1) per
cell, so these hold the arithmetic that makes a cohort verdict possible at all.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.pipeline.hyper.pair_null_group import (
    _exact_p, _variants, by_cell, by_cohort, by_occasion, correct_cohort,
    write_group_null)
from tests.hyper._names import cohort as _cohort, name

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
        real.to_csv(d / name(occ, "main", "wtcbycond"), sep="\t", index=False)


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
        kind = "wtcbycond" if occ == "G01" else "wtcbycond-pairnull-draws"
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

def test_the_phase_nulls_draws_are_read_the_same_way(tmp_path):
    """Both nulls differ in what a draw is and in nothing above the cell."""
    _write_tree(tmp_path)
    for occ in OCCASIONS:
        d = tmp_path / f"group-{occ}" / "nirs"
        src = d / name(occ, "main", "wtcbycond-pairnull-draws")
        src.rename(d / name(occ, "main", "wtcbycond-phasenull-draws"))
    written = write_group_null(tmp_path, "main", null="phase", n_resample=500, seed=3)
    assert [p.name for p in written] == [
        _cohort("phase", "byoccasion"), _cohort("phase", "cohort")]
    import json
    side = json.loads(written[1].with_suffix(".json").read_text())
    assert side["parameters"]["null_kind"] == "phase"


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
    got = [lv for g, lv, _, _, _ in _variants(_draws(), _real(0.4), ROI) if g == "roi"]
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


def _cross(path):
    """Turn a homologous table on disk into a crossed one, as a crossed run writes it."""
    d = pd.read_csv(path, sep=SEP)
    pd.concat([d.assign(label2=d.label),
               d.assign(label2="S9_D9", coherence=0.9)]).to_csv(path, sep=SEP, index=False)


def _paths(root, occ):
    nirs = root / f"group-{occ}" / "nirs"
    return (nirs / name(occ, "main", "wtcbycond-pairnull-draws"),
            nirs / name(occ, "main", "wtcbycond"))


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
    """One per-cell null table per occasion, the shape `fnirs-hyper-pairnull` writes."""
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


def test_nothing_clears_the_correction_when_the_cells_are_middling(tmp_path):
    _write_cells(tmp_path, [[50, 55, 45, 50]] * 3)
    out = by_cell(tmp_path, "main", "hbo", "repaired")
    assert int((out.q < 0.05).sum()) == 0


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


def test_a_family_of_one_leaves_q_equal_to_p():
    out = correct_cohort(_cohort_frame())
    whole = out[out.granularity == "whole"]
    for col in ("q", "q_by", "q_holm", "q_bonferroni"):
        assert whole[col].to_numpy() == pytest.approx(whole.p.to_numpy())


def test_q_stays_benjamini_hochberg_so_an_existing_reader_is_unaffected():
    from statsmodels.stats.multitest import multipletests
    out = correct_cohort(_cohort_frame())
    part = out[(out.granularity == "channel") & (out.condition == "game")]
    assert part["q"].to_numpy() == pytest.approx(
        multipletests(part["p"], method="fdr_bh")[1])


def test_the_four_methods_are_ordered_bh_then_the_stricter_ones():
    """BH is the most permissive of the four, which is why it is the one reported."""
    frame = _cohort_frame()
    # a family with a spread of p values, so the methods can differ
    ch = frame[(frame.granularity == "channel") & (frame.condition == "game")].index
    frame.loc[ch, "p"] = [0.001, 0.02, 0.2, 0.6]
    out = correct_cohort(frame)
    part = out.loc[ch]
    assert (part["q"] <= part["q_by"] + 1e-12).all()
    assert (part["q"] <= part["q_bonferroni"] + 1e-12).all()


def test_a_row_without_a_p_takes_no_part_in_its_family():
    frame = _cohort_frame()
    ch = frame[(frame.granularity == "channel") & (frame.condition == "game")].index
    frame.loc[ch[0], "p"] = np.nan
    out = correct_cohort(frame)
    part = out.loc[ch]
    assert pd.isna(part.loc[ch[0], "q"])
    assert (part["family"] == len(CHANNELS) - 1).all()


def test_the_written_cohort_table_carries_the_corrections(tmp_path):
    _write_tree(tmp_path)
    written = write_group_null(tmp_path, "main", roi_map=ROI, n_resample=500, seed=3)
    cohort = pd.read_csv(written[1], sep="	")
    for col in ("q", "q_by", "q_holm", "q_bonferroni", "family"):
        assert col in cohort.columns
    # the sidecar has to name the family, the count being uninterpretable without it
    side = json.loads(Path(str(written[1]).replace(".tsv", ".json")).read_text())
    assert "cohort_correction_family" in side["parameters"]
