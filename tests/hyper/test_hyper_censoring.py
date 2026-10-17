"""Hyper analyses leave out the union of both members' BAD_ spans, and change nothing without one."""

import json

import numpy as np
import pandas as pd
import pytest

from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import group_output_path
from nirspipe.io.groups import GroupEntry
from nirspipe.pipeline.hyper.isc import _isc_from_rows, compute_isc_pairs
from nirspipe.pipeline.hyper.surrogate import _accumulate_null_hist
from nirspipe.pipeline.hyper.whiten import ar_whiten_fixed, whiten_raws
from nirspipe.pipeline.hyper.wtc import (WTC_DJ, WTCResult, _morlet, _pair_from_prepared,
                                         _pairwise_wtc, _prepare_channel, _scale_range,
                                         _trim_pair, _wavelet_grid, compute_wtc,
                                         window_result, wtc_band_mean)
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


def _masked_map():
    """A map at 0.5 whose excluded columns 3 to 5 hold 1.0, every cell clear of the cone."""
    wtc = np.full((3, 10), 0.5)
    excluded = np.zeros(10, bool)
    excluded[3:6] = True
    wtc[:, excluded] = 1.0
    return WTCResult(pairs={("a", "b"): {"S1_D1": {"wtc": wtc, "coi": np.full(10, 1e6),
                                                   "phase": np.zeros_like(wtc)}}},
                     freqs=np.array([0.05, 0.10, 0.20]), times=np.arange(10.0),
                     excluded={("a", "b"): excluded})


def test_the_band_mean_leaves_out_excluded_columns():
    row = wtc_band_mean(_masked_map(), *BAND).iloc[0]
    assert row["coherence"] == pytest.approx(0.5)
    # 7 of 10 columns on the band's 2 rows
    assert row["n_valid_frac"] == pytest.approx(0.7)
    assert row["phase_n"] == 14


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
    hists: dict = {}
    _accumulate_null_hist(hists, _masked_map(), mask_coi=True)
    assert hists[("a", "b", "S1_D1")].sum() == 21           # 3 rows by 7 kept columns


def test_a_condition_wholly_excluded_is_nan():
    result = compute_wtc(_dyad(SPANS), fmin=0.02, fmax=0.2)
    table = wtc_band_mean(window_result(result, 255.0, 295.0), *BAND)
    assert table["coherence"].isna().all() and (table["n_valid_frac"] == 0).all()


# ---- a member with no power ----

def _silent_dyad(fill: str):
    """Both members filled over 100-300 s, flat or linear, as a censored span is."""
    dyad = _dyad({sid: [(100.0, 200.0, "BAD_gvtd")] for sid in SUBS})
    a, b = int(100.0 * SFREQ), int(300.0 * SFREQ)
    for raw in dyad.values():
        for row in raw._data:
            row[a:b] = np.linspace(row[a - 1], row[b], b - a) if fill == "linear" else row[a - 1]
    return dyad


@pytest.mark.parametrize("significance", [False, True])
@pytest.mark.parametrize("fill", ["flat", "linear"])
def test_where_a_member_has_no_power_the_coherence_is_undefined(fill, significance):
    result = compute_wtc(_silent_dyad(fill), fmin=0.02, fmax=0.2, significance=significance,
                         seed=0, mc_count=5)
    for labels in result.pairs.values():
        for data in labels.values():
            wtc = np.asarray(data["wtc"], float)
            assert not np.isinf(wtc).any()
            assert np.nanmin(wtc) >= 0.0 and np.nanmax(wtc) <= 1.0 + 1e-9


@pytest.mark.parametrize("fill", ["flat", "linear"])
def test_a_silent_stretch_blanks_only_its_own_columns(fill):
    result = compute_wtc(_silent_dyad(fill), fmin=0.02, fmax=0.2)
    (pair_key, labels), = result.pairs.items()
    kept = ~result.excluded[pair_key]
    for data in labels.values():
        assert np.isfinite(np.asarray(data["wtc"], float)[:, kept]).all()


def test_without_a_silent_stretch_both_routes_return_pycwts_ratio_bit_for_bit():
    pycwt = pytest.importorskip("pycwt")
    dyad = _dyad()
    step = int(SFREQ)
    x, y = (dyad[sid].get_data(picks=[0])[0] for sid in SUBS)
    dt, fmin, fmax = 1 / SFREQ, 0.02, 0.2
    s0, J = _scale_range(dt, WTC_DJ, fmin, fmax, len(x))
    plain = pycwt.wct(x, y, dt=dt, dj=WTC_DJ, sig=False, normalize=True, wavelet=_morlet(),
                      s0=s0, J=J, cache=False)
    expected = _trim_pair(*plain[:4], len(x), step, fmin, fmax)

    grid = _wavelet_grid(dt, len(x), fmin, fmax, limit_scales=True)
    p1, p2 = _prepare_channel(x, dt, grid), _prepare_channel(y, dt, grid)
    fast = _trim_pair(*_pair_from_prepared(p1, p2, dt, grid), p1.coi, p1.freqs, len(x), step,
                      fmin, fmax)
    slow = _pairwise_wtc(x, y, dt, step, fmin, fmax, cache=False)
    for got in (fast, slow):
        for a, b in zip(expected, got):
            if a is None:
                assert b is None
            else:
                assert np.array_equal(a, b)


def test_the_null_histogram_skips_an_undefined_cell():
    result = _masked_map()
    wtc = result.pairs[("a", "b")]["S1_D1"]["wtc"]
    wtc[0, 0], wtc[1, 1] = np.nan, np.inf                   # two of the 21 kept cells
    hists: dict = {}
    _accumulate_null_hist(hists, result, mask_coi=True)
    assert hists[("a", "b", "S1_D1")].sum() == 19


def test_an_undefined_cell_is_neither_valid_nor_a_phase_cell():
    result = compute_wtc(_dyad(), fmin=0.02, fmax=0.2)
    before = wtc_band_mean(result, *BAND).set_index("label")
    band = np.flatnonzero((result.freqs >= BAND[0]) & (result.freqs <= BAND[1]))
    middle = len(result.times) // 2
    for labels in result.pairs.values():
        for data in labels.values():
            data["wtc"] = np.array(data["wtc"], float)
            data["wtc"][band, middle] = np.nan
    after = wtc_band_mean(result, *BAND).set_index("label")
    n_cells = len(band) * len(result.times)
    assert (after["phase_n"] == before["phase_n"] - len(band)).all()
    np.testing.assert_allclose(after["n_valid_frac"],
                               before["n_valid_frac"] - len(band) / n_cells, rtol=1e-12)


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
    # b trails a by 3 samples, and both carry garbage at every excluded frame
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 400))
    b = np.roll(a, 3, axis=1)
    keep = np.ones(400, bool)
    keep[120:160] = keep[300:310] = False
    for x in (a, b):
        x[:, ~keep] = 50.0 * rng.standard_normal((2, (~keep).sum()))
    best, lag = _isc_from_rows(a, b, max_lag=5, keep=keep)
    assert np.diag(best) == pytest.approx([1.0, 1.0], abs=1e-12)
    assert np.diag(lag).tolist() == [3.0, 3.0]


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

    table = _table(tmp_path, statistic="wtc")
    whole = table[table["condition"].isna()]
    assert (whole["kept_s"] == DURATION - 30.0 - 195.0).all()
    by_cond = table[table["condition"].notna()].set_index("condition")
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


def _re_paired_isc_shares(tmp_path, monkeypatch, real_spans, stand_in_spans) -> dict:
    """One re-paired null against one stand-in: each condition's ISC ``n_valid_frac`` values."""
    from nirspipe.pipeline.hyper import alignment, group_io, group_quality
    from nirspipe.pipeline.hyper.pair_null import run_pair_null

    real = _dyad(real_spans)
    stand_in = _dyad(stand_in_spans)["sub-02"]
    group = [GroupEntry("G1", s, "tap") for s in SUBS]
    others = [GroupEntry("G2", s, "tap") for s in ("sub-03", "sub-04")]
    build_hyper_post_report(
        group_id="G1", task="tap", group=group, aligned_raws=real,
        offsets={s: 0.0 for s in SUBS}, output_dir=tmp_path, wtc_fmin=0.02, wtc_fmax=0.2,
        wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1], wtc_chroma=("hbo",),
        wtc_by_condition=True, no_report=True, desc="preproc", bads_scope="run")
    monkeypatch.setattr(group_io, "load_group_haemo", lambda out, entries, desc="preproc": {
        e.subject_id: (real.get(e.subject_id) or stand_in).copy() for e in entries})
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    monkeypatch.setattr(alignment, "align_recordings",
                        lambda raws, task: (raws, {s: 0.0 for s in raws}))
    run_pair_null("G1", "tap", group, {("G1", "tap", None): group, ("G2", "tap", None): others},
                  tmp_path, tmp_path, chroma=("hbo",))

    tables = [p for p in tmp_path.rglob("*_relmat.tsv") if "null-pair" in p.name
              and "stat-isc" in p.name and "cond-all" in p.name]
    assert len(tables) == 2                                            # the summary and the draws
    shares: dict = {}
    for path in tables:
        df = pd.read_csv(path, sep="\t")
        for cond, values in df.groupby("condition")["n_valid_frac"]:
            shares.setdefault(cond, set()).update(values.round(6))
    return shares


def test_the_re_paired_correlation_records_the_share_of_the_window_it_kept(tmp_path, monkeypatch):
    # real: sub-01 loses 30 s of rest, its partner 50 s of talk; the stand-in 10 s of its rest
    shares = _re_paired_isc_shares(tmp_path, monkeypatch, SPANS,
                                   {"sub-02": [(60.0, 10.0, "BAD_gvtd")]})
    assert shares == {"rest": {round(140 / 180, 6)}, "talk": {round(130 / 180, 6)}}


def test_without_a_span_the_re_paired_correlation_keeps_every_sample(tmp_path, monkeypatch):
    assert _re_paired_isc_shares(tmp_path, monkeypatch, None, None) == {"rest": {1.0},
                                                                        "talk": {1.0}}


# ---- the dyad raw page's spectrum ----

def _spectra(dyad) -> dict:
    from nirspipe.qc.figures.hyper.hyper_figures import build_psd
    fig = build_psd(dyad, "S1_D1", list(SUBS))
    return {trace.name: np.asarray(trace.y) for trace in fig.data}


def test_the_dyad_raw_spectrum_reads_only_frames_outside_bad_spans():
    quiet, loud = _spectra(_dyad(SPANS)), _spectra(_dyad(SPANS, garbage=1.0))
    assert quiet.keys() == loud.keys() == set(SUBS)
    for sid in SUBS:
        np.testing.assert_array_equal(loud[sid], quiet[sid])


def test_a_member_under_spans_end_to_end_gets_no_dyad_raw_spectrum():
    assert set(_spectra(_dyad({"sub-01": [(0.0, DURATION, "BAD_gvtd")]}))) == {"sub-02"}


def test_without_a_span_the_dyad_raw_spectrum_is_the_plain_welch():
    import mne
    from nirspipe.qc.figures.common._utils import PSD_NFFT

    dyad = _dyad()
    got = _spectra(dyad)
    for sid in SUBS:
        raw = dyad[sid]
        data = raw.get_data(picks=[raw.ch_names.index("S1_D1 hbo")])
        psds, freqs = mne.time_frequency.psd_array_welch(
            data, raw.info["sfreq"], n_fft=min(PSD_NFFT, data.shape[1]), verbose=False)
        np.testing.assert_array_equal(got[sid], psds[0][freqs <= min(2.0, SFREQ / 2)])


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
