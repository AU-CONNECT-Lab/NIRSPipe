"""Hyper analyses leave out the union of both members' BAD_ spans, and change nothing without one."""

import json

import numpy as np
import pandas as pd
import pytest

from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import group_output_path
from nirspipe.pipeline.hyper import GroupEntry
from nirspipe.pipeline.hyper.isc import _isc_from_rows, compute_isc_pairs
from nirspipe.pipeline.hyper.surrogate import _accumulate_null_hist
from nirspipe.pipeline.hyper.whiten import ar_whiten_fixed, whiten_raws
from nirspipe.pipeline.hyper.wtc import (WTCResult, _outside_coi, compute_wtc, window_result,
                                         wtc_band_mean)
from nirspipe.pipeline.hyper.wtc_store import load_wtc, save_wtc
from nirspipe.qc.hyper.hyper_report import build_hyper_post_report
from nirspipe.utils.spans import bad_spans, frames_under
from tests.hyper.test_hyper_page_contract import DURATION, SFREQ, _raw

SUBS = ("sub-01", "sub-02")
BAND = (0.03, 0.10)
# one corrupted stretch on the first member, one unselected on the second
SPANS = {"sub-01": [(100.0, 30.0, "BAD_gvtd")], "sub-02": [(250.0, 50.0, "BAD_unselected")]}


def _dyad(spans=None, garbage=None):
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    shared = np.sin(2 * np.pi * 0.05 * t)
    out = {}
    for i, sid in enumerate(SUBS, start=1):
        raw = _raw(i, shared)
        for onset, duration, desc in (spans or {}).get(sid, []):
            raw.annotations.append(onset, duration, desc)
            if garbage is not None:
                a, b = int(onset * SFREQ), int((onset + duration) * SFREQ)
                raw._data[:, a:b] = garbage * np.random.default_rng(i).standard_normal(
                    (raw._data.shape[0], b - a))
        out[sid] = raw
    return out


def _union(dyad):
    return frames_under(dyad["sub-01"], bad_spans(dyad["sub-01"])) | \
        frames_under(dyad["sub-02"], bad_spans(dyad["sub-02"]))


# ---- without spans ----

def test_without_a_span_nothing_is_excluded_and_every_average_is_unchanged():
    result = compute_wtc(_dyad(), fmin=0.02, fmax=0.2)
    assert result.excluded == {}
    # the masked route with nothing masked is the same expression, value for value
    blank = WTCResult(result.pairs, result.freqs, result.times,
                      excluded={key: np.zeros(len(result.times), bool) for key in result.pairs})
    pd.testing.assert_frame_equal(wtc_band_mean(result, *BAND), wtc_band_mean(blank, *BAND))

    x = _dyad()["sub-01"].get_data(picks=[0])[0]
    np.testing.assert_array_equal(ar_whiten_fixed(x, 20, np.ones(len(x), bool)),
                                  ar_whiten_fixed(x, 20))


def test_without_a_span_the_correlation_takes_todays_route():
    dyad = _dyad()
    mat, _, _, _ = compute_isc_pairs(dyad, list(SUBS), "hbo")
    a = dyad["sub-01"].get_data(picks="hbo")
    b = dyad["sub-02"].get_data(picks="hbo")
    np.testing.assert_array_equal(mat, _isc_from_rows(a, b)[0])


# ---- the coherence ----

def test_a_pairing_excludes_the_union_of_its_two_members_spans():
    dyad = _dyad(SPANS)
    result = compute_wtc(dyad, fmin=0.02, fmax=0.2)
    (mask,) = result.excluded.values()
    step = int(round((result.times[1] - result.times[0]) * SFREQ))
    np.testing.assert_array_equal(mask, _union(dyad)[::step][:len(result.times)])
    under = result.times[mask]
    assert ((under >= 100) & (under < 130) | (under >= 250) & (under < 300)).all()
    assert mask.sum() == pytest.approx(80, abs=2)


def test_the_band_mean_leaves_out_excluded_columns():
    result = compute_wtc(_dyad(SPANS), fmin=0.02, fmax=0.2)
    (pair_key, labels), = result.pairs.items()
    kept = ~result.excluded[pair_key]
    band = (result.freqs >= BAND[0]) & (result.freqs <= BAND[1])
    table = wtc_band_mean(result, *BAND).set_index("label")
    for label, data in labels.items():
        cells = _outside_coi(result.freqs[band], data["coi"]) & kept[None, :]
        wtc = np.asarray(data["wtc"], float)[band]
        assert table.loc[label, "coherence"] == pytest.approx(wtc[cells].mean(), rel=1e-12)
        assert table.loc[label, "n_valid_frac"] == pytest.approx(cells.mean(), rel=1e-12)
        assert table.loc[label, "phase_n"] == cells.sum()


def test_a_condition_window_and_saved_maps_keep_the_mask(tmp_path):
    result = compute_wtc(_dyad(SPANS), fmin=0.02, fmax=0.2)
    part = window_result(result, 90.0, 200.0)
    (mask,) = part.excluded.values()
    keep = (result.times >= 90.0) & (result.times <= 200.0)
    np.testing.assert_array_equal(mask, next(iter(result.excluded.values()))[keep])

    path = save_wtc(result, tmp_path / "maps.npz")
    pd.testing.assert_frame_equal(wtc_band_mean(load_wtc(path), *BAND),
                                  wtc_band_mean(result, *BAND), rtol=1e-5)


def test_the_null_histogram_counts_only_kept_cells():
    result = compute_wtc(_dyad(SPANS), fmin=0.02, fmax=0.2)
    hists: dict = {}
    _accumulate_null_hist(hists, result, mask_coi=True)
    (pair_key, labels), = result.pairs.items()
    kept = ~result.excluded[pair_key]
    for label, data in labels.items():
        cells = _outside_coi(result.freqs, data["coi"]) & kept[None, :]
        assert hists[(*pair_key, label)].sum() == cells.sum()


def test_a_condition_wholly_excluded_is_nan():
    result = compute_wtc(_dyad(SPANS), fmin=0.02, fmax=0.2)
    table = wtc_band_mean(window_result(result, 255.0, 295.0), *BAND)
    assert table["coherence"].isna().all() and (table["n_valid_frac"] == 0).all()


# ---- the correlation ----

def test_the_correlation_is_pearson_over_kept_samples_whatever_lies_in_the_spans():
    dyad = _dyad(SPANS)
    keep = ~_union(dyad)
    mat, names, _, _ = compute_isc_pairs(dyad, list(SUBS), "hbo")
    a = dyad["sub-01"].get_data(picks="hbo")
    b = dyad["sub-02"].get_data(picks="hbo")
    for i in range(len(names)):
        for j in range(len(names)):
            assert mat[i, j] == pytest.approx(np.corrcoef(a[i, keep], b[j, keep])[0, 1],
                                              abs=1e-12)
    loud, _, _, _ = compute_isc_pairs(_dyad(SPANS, garbage=1.0), list(SUBS), "hbo")
    np.testing.assert_allclose(loud, mat, atol=1e-12)


def test_a_lagged_pair_of_samples_drops_when_either_position_is_excluded():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((2, 400)), rng.standard_normal((2, 400))
    keep = np.ones(400, bool)
    keep[120:160] = keep[300:310] = False
    best, lag = _isc_from_rows(a, b, max_lag=5, keep=keep)
    for i in range(2):
        for j in range(2):
            by_shift = {}
            for s in range(-5, 6):
                ia = np.arange(max(0, -s), 400 - max(0, s))
                both = keep[ia] & keep[ia + s]
                by_shift[s] = np.corrcoef(a[i, ia[both]], b[j, ia[both] + s])[0, 1]
            s_best = max(by_shift, key=lambda s: abs(by_shift[s]))
            assert best[i, j] == pytest.approx(by_shift[s_best], abs=1e-12)
            assert lag[i, j] == s_best


# ---- whitening ----

def test_whitening_is_fitted_on_the_members_own_kept_time():
    order = 10
    x = np.random.default_rng(5).standard_normal(2000).cumsum() * 0.01
    keep = np.ones(2000, bool)
    keep[800:900] = False
    y = x.copy()
    y[800:900] += 50.0 * np.random.default_rng(6).standard_normal(100)
    rx, ry = ar_whiten_fixed(x, order, keep), ar_whiten_fixed(y, order, keep)
    # the coefficients ignore the gap; the filter reaches order samples past it
    np.testing.assert_allclose(rx[900 + order:] - rx[900 + order:].mean(),
                               ry[900 + order:] - ry[900 + order:].mean(), atol=1e-9)
    np.testing.assert_allclose(rx[order:800] - rx[order:800].mean(),
                               ry[order:800] - ry[order:800].mean(), atol=1e-9)


def test_whitening_refuses_a_member_whose_kept_time_is_too_fragmented():
    spans = {"sub-01": [(float(t), 1.5, "BAD_gvtd") for t in range(10, 390, 2)]}
    with pytest.raises(StageError, match="too few stretches"):
        whiten_raws(_dyad(spans), 10.0)


# ---- the dyad tables ----

def _table(out, **entities):
    return pd.read_csv(group_output_path(out, "G1", {"task": "tap", **entities},
                                         "relmat", ".tsv"), sep="\t")


def test_the_dyad_tables_record_kept_seconds_and_a_wholly_excluded_condition_is_nan(tmp_path):
    # talk runs 210-390 s; the second member's unselected time covers all of it
    spans = {"sub-01": [(100.0, 30.0, "BAD_gvtd")],
             "sub-02": [(205.0, 195.0, "BAD_unselected")]}
    build_hyper_post_report(
        group_id="G1", task="tap", group=[GroupEntry("G1", s, "tap") for s in SUBS],
        aligned_raws=_dyad(spans), offsets={s: 0.0 for s in SUBS}, output_dir=tmp_path,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1],
        wtc_chroma=("hbo",), wtc_by_condition=True, no_report=True)

    whole = _table(tmp_path, statistic="wtc")
    assert (whole["kept_s"] == DURATION - 30.0 - 195.0).all()
    by_cond = _table(tmp_path, condition="all", statistic="wtc").set_index("condition")
    assert (by_cond.loc["rest", "kept_s"] == 150.0).all()
    assert by_cond.loc["rest", "coherence"].notna().all()
    assert (by_cond.loc["talk", "kept_s"] == 0.0).all()
    assert by_cond.loc["talk", "coherence"].isna().all()

    isc = _table(tmp_path, statistic="isc")
    talk = isc[isc["condition"] == "talk"]
    assert (talk["kept_s"] == 0.0).all() and talk["r"].isna().all()
    assert (isc[isc["condition"].isna()]["kept_s"] == DURATION - 30.0 - 195.0).all()


def test_a_level_drawn_under_other_spans_is_not_reused():
    from nirspipe.pipeline.hyper.wtc_store import level_params
    plain = level_params(_dyad(), wtc_fmin=0.02, wtc_fmax=0.2, mask_coi=True)
    spanned = level_params(_dyad(SPANS), wtc_fmin=0.02, wtc_fmax=0.2, mask_coi=True)
    assert plain["excluded_spans_s"] == {"sub-01": [], "sub-02": []}
    assert json.dumps(spanned["excluded_spans_s"]) != json.dumps(plain["excluded_spans_s"])


# ---- the re-paired null ----

def test_a_re_paired_draw_excludes_the_real_union_and_the_stand_ins_corrupted_time(monkeypatch):
    from nirspipe.pipeline.hyper import group_io, group_quality, pair_null

    real = _dyad(SPANS)
    stand_in = _dyad({"sub-02": [(60.0, 20.0, "BAD_unselected"), (150.0, 10.0, "BAD_gvtd")]})
    monkeypatch.setattr(group_io, "load_group_haemo",
                        lambda out, entries, desc="preproc": {entries[0].subject_id:
                                                              stand_in["sub-02"].copy()})
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    (pid, _, pair, _, _, _, excluded), = pair_null._draw_condition_pairs(
        "/out", "tap", "sub-01", real["sub-01"], [GroupEntry("GXX", "sub-09", "tap")],
        desc="preproc", bads_scope="run", scope_tasks=["tap"], windows=[("rest", 20.0, 200.0)],
        band_fmin=0.03, n_max=None, refused={}, real_partner_raw=real["sub-02"])

    def seconds(mask, raw):
        return raw.first_time - real["sub-01"].first_time + np.flatnonzero(mask) / SFREQ

    fixed = seconds(excluded["sub-01"], pair["sub-01"])
    assert ((fixed >= 100) & (fixed < 130) | (fixed >= 250) & (fixed < 300)).all()
    # the cut reaches past the condition by the cone margin, into the real partner's span
    end = seconds(np.ones(pair["sub-01"].n_times, bool), pair["sub-01"])[-1]
    assert excluded["sub-01"].sum() == pytest.approx((30 + max(0.0, min(end, 300) - 250))
                                                     * SFREQ, abs=2)
    partner = np.flatnonzero(excluded[pid]) / SFREQ + pair[pid].first_time
    assert ((partner >= 150) & (partner < 160)).all()                 # its unselected time is not
    assert excluded[pid].sum() == pytest.approx(10 * SFREQ, abs=1)


# ---- the dyad report ----

def test_excluded_columns_are_washed_out_and_carry_no_arrow():
    from nirspipe.qc.figures.hyper.hyper_post_figures import (_excluded_runs,
                                                             build_wtc_map_interactive)
    times = np.arange(5.0)
    assert _excluded_runs(times, np.array([0, 1, 1, 0, 1], bool)) == [(0.5, 2.5), (3.5, 4.5)]
    assert _excluded_runs(times, None) == [] and _excluded_runs(times, np.zeros(5, bool)) == []

    result = compute_wtc(_dyad(SPANS), fmin=0.02, fmax=0.2)
    (pair_key, labels), = result.pairs.items()
    data = next(d for d in labels.values() if d is not None)
    plain = build_wtc_map_interactive(data, result.freqs, result.times, "a × b", [], {}, "S1_D1",
                                      arrow_min=0.0)
    marked = build_wtc_map_interactive({**data, "excluded": result.excluded[pair_key]},
                                       result.freqs, result.times, "a × b", [], {}, "S1_D1",
                                       arrow_min=0.0)
    runs = _excluded_runs(result.times, result.excluded[pair_key])
    washed = [s for s in marked.layout.shapes if s.type == "rect"]
    assert [(s.x0, s.x1) for s in washed] == runs and not [s for s in plain.layout.shapes
                                                          if s.type == "rect"]
    arrows = [a.x for a in marked.layout.annotations if a.name == "wtcarrow"]
    assert arrows and not any(a0 <= x <= a1 for x in arrows for a0, a1 in runs)
    assert any(a0 <= a.x <= a1 for a in plain.layout.annotations if a.name == "wtcarrow"
               for a0, a1 in runs)


def test_a_scope_header_leads_with_the_kept_seconds_only_when_spans_exist():
    from nirspipe.qc.hyper.hyper_report import _number_table
    band = pd.DataFrame({"sub1": ["a"], "sub2": ["b"], "label": ["S1_D1"],
                         "coherence": [0.3], "n_valid_frac": [0.25], "kept_s": [75.6]})
    plain = _number_table({"hbo": band}, None, ["S1_D1"], "channel", ("hbo",))
    assert plain["valid"] == "25% outside COI"
    spanned = _number_table({"hbo": band}, None, ["S1_D1"], "channel", ("hbo",), spans=True)
    assert spanned["valid"] == "kept 75.6 s · 25% outside COI and BAD_ spans"


def test_the_dyad_report_builds_with_spans_and_says_what_it_kept(tmp_path):
    build_hyper_post_report(
        group_id="G1", task="tap", group=[GroupEntry("G1", s, "tap") for s in SUBS],
        aligned_raws=_dyad(SPANS), offsets={s: 0.0 for s in SUBS}, output_dir=tmp_path,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1],
        wtc_chroma=("hbo",), wtc_by_condition=True)
    pages = {p.name: p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.html")}
    assert any("outside COI and BAD_ spans" in text and "kept 320.0 s" in text
               for text in pages.values())
