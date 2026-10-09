"""The null keeps its spread, and the arrows are drawn against it per frequency.

`--wtc-phase-null` keeps its iterations, so a table says how wide the null is as well as
where it sits, and a real value can be ranked inside it. The phase arrows are drawn against
a per-frequency level rather than a flat display threshold, which is not a test: surrogate
coherence is not flat in frequency, it rises at both ends of the computed range, so one
number over the whole map would draw arrows preferentially where the data is least
trustworthy.

Both come from keeping what the surrogates already produce. The draws survive to be
ranked against; each surrogate map is counted into a per-frequency histogram on the way past.
"""

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.pipeline.hyper.surrogate import (
    NULL_ARROW_QUANTILE,
    NullDraws,
    _accumulate_null_hist,
    _null_level,
)
from fnirs_pipe.pipeline.hyper.wtc import WTCResult

FREQS = np.array([0.02, 0.06, 0.15])
KEYS = ["sub1", "sub2", "label"]


def _draw(coherence, n_valid_frac=1.0):
    return pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                         "coherence": [coherence], "n_valid_frac": [n_valid_frac]})


def _map(rows, n_times=40, coi=1e6):
    """One pair's surrogate map, each frequency row held at its own constant."""
    wtc = np.repeat(np.asarray(rows, dtype=float)[:, None], n_times, axis=1)
    return WTCResult(pairs={("a", "b"): {"S1_D1": {"wtc": wtc,
                                                   "coi": np.full(n_times, coi),
                                                   "phase": np.zeros_like(wtc),
                                                   "sig": None}}},
                     freqs=FREQS, times=np.arange(n_times, dtype=float))


# ---- the draws survive the averaging ----

def test_the_table_reports_the_spread_the_mean_came_out_of():
    """A null summarised by its mean alone cannot say whether a real value above it is
    anywhere near unusual, which is the only question a null is drawn to answer."""
    null = NullDraws(draws=[_draw(0.2), _draw(0.4), _draw(0.6)],
                      cond_draws=[], keys=KEYS, levels={})
    table, _ = null.summarise()

    assert table["null_mean"].iloc[0] == pytest.approx(0.4)
    assert table["null_sd"].iloc[0] == pytest.approx(0.2)
    assert table["n_iter"].iloc[0] == 3


def test_the_percentile_is_counted_not_interpolated():
    """Three of four draws beat by the real value is the 75th percentile exactly. At the
    iteration counts a null is affordable at, a quantile-interpolated rank disagrees with
    the counted one by more than the number is worth."""
    null = NullDraws(draws=[_draw(v) for v in (0.10, 0.20, 0.30, 0.90)],
                      cond_draws=[], keys=KEYS, levels={})
    table, _ = null.summarise(real=_draw(0.5))

    assert table["percentile"].iloc[0] == pytest.approx(75.0)


def test_a_cell_the_real_table_has_no_row_for_is_not_ranked():
    """NaN rather than a rank against nothing: a 0 would read as a cell the real data lost
    to, which is a claim, and an absent row is not one."""
    null = NullDraws(draws=[_draw(0.2), _draw(0.4)], cond_draws=[], keys=KEYS, levels={})
    other = _draw(0.5)
    other["label"] = ["S9_D9"]
    table, _ = null.summarise(real=other)

    assert np.isnan(table["percentile"].iloc[0])


def test_a_pair_blank_in_every_iteration_is_not_ranked_either():
    """A NaN draw compares False, so counting would put a real value at the 0th percentile:
    "it beat none of them", said about draws that were never taken."""
    null = NullDraws(draws=[_draw(float("nan")), _draw(float("nan"))],
                      cond_draws=[], keys=KEYS, levels={})
    table, _ = null.summarise(real=_draw(0.5))

    assert np.isnan(table["percentile"].iloc[0])


def test_no_real_table_leaves_the_percentile_column_off_rather_than_empty():
    null = NullDraws(draws=[_draw(0.2)], cond_draws=[], keys=KEYS, levels={})
    table, _ = null.summarise()

    assert "percentile" not in table.columns
    assert "null_p95" in table.columns


# ---- a homologous null against a crossed real table ----

LABELS = ["S1_D1", "S2_D2", "S3_D3"]


def _homologous_draw(coherence):
    return pd.DataFrame({"sub1": "a", "sub2": "b", "label": LABELS,
                         "coherence": coherence, "n_valid_frac": 1.0})


def _crossed_real(diagonal, off_diagonal):
    """The real table a crossed run writes: label pairs in axis1 x axis2 order."""
    return pd.DataFrame([
        {"sub1": "a", "sub2": "b", "label": a, "label2": b,
         "coherence": diagonal if a == b else off_diagonal, "n_valid_frac": 1.0}
        for a in LABELS for b in LABELS
    ])


def test_a_homologous_null_ranks_the_crossed_table_s_diagonal():
    """A crossed real table beside `--no-wtc-phase-null-cross`: the homologous null is the
    null for the pairings it was drawn for. Keying on `label` alone would take
    the row that label came first in, which is its pairing with the *first* channel of the other
    montage: right for the first label by coincidence and wrong for every other."""
    null = NullDraws(draws=[_homologous_draw(v) for v in (0.10, 0.20, 0.30)],
                      cond_draws=[], keys=KEYS, levels={})
    table, _ = null.summarise(real=_crossed_real(diagonal=0.9, off_diagonal=0.05))

    assert list(table["label"]) == LABELS
    assert list(table["percentile"]) == [100.0, 100.0, 100.0]


def test_the_off_diagonal_is_not_what_the_homologous_null_is_ranked_against():
    """The complement of the above: were the crossed rows still being keyed by `label`,
    a diagonal below the null and an off-diagonal above it would come back as 100."""
    null = NullDraws(draws=[_homologous_draw(v) for v in (0.40, 0.50, 0.60)],
                      cond_draws=[], keys=KEYS, levels={})
    table, _ = null.summarise(real=_crossed_real(diagonal=0.05, off_diagonal=0.9))

    assert list(table["percentile"]) == [0.0, 0.0, 0.0]


# ---- the level is per frequency ----

def test_the_level_is_read_per_frequency_not_over_the_whole_map():
    """Unlike the flat --wtc-arrow-min: a map whose rows sit at different levels
    gets one threshold per row, so a cell is judged against its own frequency's null."""
    hists: dict = {}
    _accumulate_null_hist(hists, _map([0.9, 0.2, 0.7]), mask_coi=True)
    level = _null_level(hists[("a", "b", "S1_D1")])

    assert level[0] == pytest.approx(0.9, abs=0.002)
    assert level[1] == pytest.approx(0.2, abs=0.002)
    assert level[2] == pytest.approx(0.7, abs=0.002)


def test_the_quantile_is_taken_over_the_draws_pooled_across_iterations():
    """Every in-cone cell of every surrogate map is a draw from that frequency's null, so
    the quantile is over all of them and not over one iteration's summary."""
    hists: dict = {}
    for value in np.linspace(0.0, 1.0, 101):
        _accumulate_null_hist(hists, _map([value, value, value]), mask_coi=True)
    level = _null_level(hists[("a", "b", "S1_D1")])

    assert level[0] == pytest.approx(NULL_ARROW_QUANTILE, abs=0.01)


def test_cells_outside_the_cone_are_not_draws_from_the_null():
    """They are coefficients built against the padding. Counting them would put the level
    near 1 at the low-frequency end, which is where the padding reaches furthest in."""
    hists: dict = {}
    # coi 20 s keeps only the rows whose frequency clears 1/20 Hz, so 0.02 drops out
    _accumulate_null_hist(hists, _map([0.9, 0.2, 0.7], coi=20.0), mask_coi=True)
    level = _null_level(hists[("a", "b", "S1_D1")])

    assert np.isnan(level[0])
    assert level[1] == pytest.approx(0.2, abs=0.002)


def test_a_pair_with_nothing_inside_the_cone_gets_no_level_at_all():
    """Not an all-NaN level but no entry, which is what the caller keys on to leave the
    map's `sig` alone and fall back to the flat threshold."""
    hists: dict = {}
    _accumulate_null_hist(hists, _map([0.5, 0.5, 0.5], coi=1.0), mask_coi=True)

    assert hists == {}


def test_a_frequency_with_no_level_draws_no_arrow():
    """NaN, not 0: a 0 would pass every cell at that frequency as beating the null. A
    comparison against NaN is False, so the row simply goes unmarked."""
    from fnirs_pipe.qc.figures.hyper.hyper_post_figures import _arrow_mask

    wtc = np.array([[0.99, 0.99], [0.30, 0.60]])
    coi = np.full(2, 1e6)
    mask = _arrow_mask(wtc, np.array([np.nan, 0.5]), FREQS[:2], 1.0 / coi)

    assert mask.tolist() == [[False, False], [False, True]]


# ---- what the maps do with it ----

def test_the_arrow_mask_prefers_the_level_over_the_flat_threshold():
    """The level lands in `sig`, the slot pycwt's Monte Carlo level already used, so the
    figures do not have to choose between two thresholds."""
    from fnirs_pipe.qc.figures.hyper.hyper_post_figures import _arrow_mask

    wtc = np.array([[0.30, 0.60], [0.30, 0.60]])
    coi = np.full(2, 1e6)
    flat = _arrow_mask(wtc, None, FREQS[:2], 1.0 / coi, arrow_min=0.5)
    against_null = _arrow_mask(wtc, np.array([0.2, 0.8]), FREQS[:2], 1.0 / coi, arrow_min=0.5)

    assert flat.tolist() == [[False, True], [False, True]]
    # row 0's null is low, so both its cells clear it; row 1's is high and neither does
    assert against_null.tolist() == [[True, True], [False, False]]


def test_the_caption_names_the_level_it_actually_used():
    """Three sources and three wordings. A caption reading "the Monte Carlo level" over
    arrows drawn against the phase-scrambled null, or against a display threshold, claims a test
    nobody ran."""
    from fnirs_pipe.qc.figures.hyper.hyper_post_figures import _clears

    level = np.array([0.4, 0.5, 0.6])
    assert _clears({"sig": None}, arrow_min=0.5) == "0.5"
    assert _clears({"sig": level}) == "the Monte Carlo level"
    assert _clears({"sig": level, "sig_source": "null"}) == "the phase-scrambled null"


def test_a_condition_window_keeps_the_source_of_its_level():
    """Every condition page is drawn from a window of the whole-record map; losing the source
    there would caption arrows drawn against the phase-scrambled null as the Monte Carlo level."""
    from fnirs_pipe.pipeline.hyper.wtc import window_result
    from fnirs_pipe.qc.figures.hyper.hyper_post_figures import _clears

    whole = _map([0.5, 0.5, 0.5])
    whole.pairs[("a", "b")]["S1_D1"].update(sig=np.full(len(FREQS), 0.4), sig_source="null")
    windowed = window_result(whole, 5.0, 30.0)
    assert _clears(windowed.pairs[("a", "b")]["S1_D1"]) == "the phase-scrambled null"


# ---- what lands on disk ----

def test_the_level_survives_a_round_trip_to_disk(tmp_path):
    """It is what the arrows are drawn against, and redrawing a report must not cost the
    hours the null took to produce it."""
    from fnirs_pipe.pipeline.hyper.wtc_store import load_null_levels, save_null_levels

    levels = {("a", "b", "S1_D1"): np.array([0.31, 0.42, 0.53]),
              ("a", "b", ("S1_D1", "S2_D2")): np.array([0.11, 0.22, 0.33])}
    path = save_null_levels(levels, tmp_path / "g_hyper-wtc-nulllevel-hbo.npz")
    back = load_null_levels(path)

    assert back[("a", "b")]["S1_D1"] == pytest.approx(levels[("a", "b", "S1_D1")])
    assert back[("a", "b")][("S1_D1", "S2_D2")] == pytest.approx([0.11, 0.22, 0.33])


def test_reband_leaves_the_level_archive_alone(tmp_path):
    """It sits under the same `*_hyper-wtc*.npz` prefix `fnirs-hyper-band` globs, but holds
    one row per pair rather than a map, so opening it only produces a warning."""
    from fnirs_pipe.pipeline.hyper.wtc_store import reband_tree, save_null_levels

    save_null_levels({("a", "b", "S1_D1"): np.array([0.3, 0.4, 0.5])},
                     tmp_path / "group-G01_task-main_hyper-wtc-nulllevel-hbo.npz")

    assert reband_tree(tmp_path, 0.06, 0.15) == []


# ---- the per-condition levels and what decides whether a report may use one ----

def test_per_condition_levels_survive_a_round_trip_to_disk(tmp_path):
    from fnirs_pipe.pipeline.hyper.wtc_store import load_cond_null_levels, save_cond_null_levels

    levels = {"game1#2": {("a", "b", "S1_D1"): np.array([0.3, 0.4, 0.5])},
              "talk": {("a", "b", ("S1_D1", "S2_D2")): np.array([0.6, 0.7, 0.8])}}
    back = load_cond_null_levels(save_cond_null_levels(levels, tmp_path / "level.npz"))
    assert np.allclose(back["game1#2"][("a", "b")]["S1_D1"], [0.3, 0.4, 0.5])
    assert np.allclose(back["talk"][("a", "b")][("S1_D1", "S2_D2")], [0.6, 0.7, 0.8])


def _stamped(tmp_path, **params):
    import json
    path = tmp_path / "level.npz"
    path.with_suffix(".json").write_text(json.dumps({"Sources": [], "parameters": params}))
    return path


def test_a_level_matching_this_run_is_usable(tmp_path):
    from fnirs_pipe.pipeline.hyper.wtc_store import level_mismatch

    path = _stamped(tmp_path, wtc_fmin=0.02, align_offset_s={"a": 0.0}, n_iter=5)
    assert level_mismatch(path, {"wtc_fmin": 0.02, "align_offset_s": {"a": 0.0}}, {}) is None


def test_a_level_without_its_sidecar_is_not_usable(tmp_path):
    from fnirs_pipe.pipeline.hyper.wtc_store import level_mismatch

    assert "missing" in level_mismatch(tmp_path / "level.npz", {}, {})


def test_a_setting_the_writer_recorded_and_this_run_lacks_counts_as_a_difference(tmp_path):
    """A level drawn on trigger-aligned recordings does not fit a run aligned on none."""
    from fnirs_pipe.pipeline.hyper.wtc_store import level_mismatch

    path = _stamped(tmp_path, wtc_fmin=0.02, align_trigger={"a": "start"})
    assert "align_trigger" in level_mismatch(path, {"wtc_fmin": 0.02}, {})


def test_the_re_paired_isc_null_carries_a_level_for_the_size_of_r(tmp_path):
    """The chords compare |r| with a level, so a p95 over signed r is the wrong number:
    a null whose draws swing both ways has a low signed p95 and a high |r| one."""
    from fnirs_pipe.pipeline.hyper.pair_null import _write_isc_null

    draws = [pd.DataFrame({"chromophore": ["hbo"], "condition": ["talk"], "sub1": ["a"],
                           "sub2": ["b"], "label": ["S1_D1"], "label2": ["S1_D1"],
                           "coherence": [r], "n_valid_frac": [1.0]})
             for r in (-0.9, -0.8, 0.1, 0.2, 0.1, -0.85)]

    def path_of(entities):
        return tmp_path / ("_".join(f"{k}-{v}" for k, v in sorted(entities.items())) + ".tsv")

    _write_isc_null([], draws, [], path_of, [], {}, [("talk", 0.0, 60.0)], 0, 0.0, None)
    table = pd.read_csv(path_of({"condition": "all", "nulldist": "pair", "statistic": "isc"}),
                        sep="\t")
    assert table["null_p95"].iloc[0] < 0.3
    assert table["null_abs_p95"].iloc[0] > 0.8


def test_the_re_paired_isc_percentile_ranks_the_size_of_r(tmp_path):
    """The phase null ranks |r| among |draws|; the re-paired one has to say the same thing,
    or a strongly negative pair reads as beating none of its stand-ins."""
    from fnirs_pipe.pipeline.hyper.pair_null import _write_isc_null

    keys = {"chromophore": ["hbo"], "condition": ["talk"], "sub1": ["a"], "sub2": ["b"],
            "label": ["S1_D1"], "label2": ["S1_D1"]}
    draws = [pd.DataFrame({**keys, "coherence": [r], "n_valid_frac": [1.0]})
             for r in (-0.9, -0.8, 0.1, 0.2, 0.1, -0.85)]

    def path_of(entities):
        return tmp_path / ("_".join(f"{k}-{v}" for k, v in sorted(entities.items())) + ".tsv")

    pd.DataFrame({**keys, "r": [-0.95]}).to_csv(path_of({"statistic": "isc"}), sep="\t",
                                                index=False)
    _write_isc_null([], draws, [], path_of, [], {}, [("talk", 0.0, 60.0)], 0, 0.0, None)
    table = pd.read_csv(path_of({"condition": "all", "nulldist": "pair", "statistic": "isc"}),
                        sep="\t")
    assert table["percentile"].iloc[0] == pytest.approx(100.0)
