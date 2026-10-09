"""Prewhitening before the coherence: one order for everybody, and every route agrees.

The order is fixed in seconds and shared by every channel of both members, the record keeps
its length, and the filter's startup transient is zeroed rather than cut. The real table and
both nulls read the same whitened signals, the correlation reads none of them, and a table,
a level or a merge that disagrees on whitening is refused rather than mixed in.
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import group_output_path
from nirspipe.pipeline.hyper import GroupEntry
from nirspipe.pipeline.hyper.whiten import ar_whiten_fixed, whiten_order, whiten_raws
from nirspipe.qc.hyper.hyper_report import build_hyper_post_report
from tests.hyper.test_hyper_page_contract import DURATION, SFREQ, _raw

SUBS = ("sub-01", "sub-02")


@pytest.fixture(scope="module")
def dyad():
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    shared = np.sin(2 * np.pi * 0.05 * t)
    return {sid: _raw(i, shared).crop(tmin=20.0) for i, sid in enumerate(SUBS, start=1)}


# ---- the filter ----

def test_the_residual_keeps_the_length_and_zeroes_the_transient():
    rng = np.random.default_rng(0)
    x = lfilter([1.0], [1.0, -0.9], rng.standard_normal(2000))
    resid = ar_whiten_fixed(x, 20)
    assert resid.shape == x.shape
    assert np.all(resid[:20] == 0.0) and np.any(resid[20:] != 0.0)


def test_an_autocorrelated_series_comes_out_close_to_white():
    rng = np.random.default_rng(1)
    x = lfilter([1.0], [1.0, -0.95], rng.standard_normal(20000))
    resid = ar_whiten_fixed(x, 10)[10:]
    lag1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    assert np.corrcoef(x[:-1], x[1:])[0, 1] > 0.9
    assert abs(lag1) < 0.05


@pytest.mark.parametrize("row", [np.full(500, np.nan), np.full(500, 3.0), np.zeros(5)])
def test_a_row_that_cannot_be_fitted_is_reported_rather_than_returned(row):
    assert ar_whiten_fixed(row, 10) is None


# ---- the recordings ----

def test_both_members_get_the_order_asked_for_in_seconds(dyad):
    assert whiten_order(dyad, 10.0) == int(round(10.0 * SFREQ))


def test_every_long_channel_of_both_members_is_whitened_and_the_originals_are_not(dyad):
    before = {sid: raw.get_data().copy() for sid, raw in dyad.items()}
    white = whiten_raws(dyad, 10.0)
    order = whiten_order(dyad, 10.0)
    for sid, raw in white.items():
        assert raw is not dyad[sid]
        assert np.array_equal(dyad[sid].get_data(), before[sid])
        data = raw.get_data()
        assert data.shape == before[sid].shape
        assert np.all(data[:, :order] == 0.0)
        assert not np.allclose(data, before[sid])


def test_a_record_too_short_for_the_order_is_refused_rather_than_fitted_lower(dyad):
    short = {sid: raw.copy().crop(tmax=30.0) for sid, raw in dyad.items()}
    with pytest.raises(StageError, match="fewer than four times"):
        whiten_raws(short, 10.0)


def test_a_channel_that_cannot_be_fitted_is_left_out_on_the_copy(dyad):
    broken = {sid: raw.copy() for sid, raw in dyad.items()}
    broken["sub-01"]._data[0] = 1.0          # constant: no autocovariance to fit
    white = whiten_raws(broken, 10.0)
    assert broken["sub-01"].ch_names[0] in white["sub-01"].info["bads"]
    assert broken["sub-01"].ch_names[0] not in broken["sub-01"].info["bads"]


def test_off_is_the_recordings_themselves(dyad):
    assert all(a is b for a, b in zip(whiten_raws(dyad, 0.0).values(), dyad.values()))


# ---- the report ----

def _build(dyad, out, whiten_s=0.0, **kwargs):
    build_hyper_post_report(
        group_id="G1", task="tap", group=[GroupEntry("G1", s, "tap") for s in SUBS],
        aligned_raws=dyad, offsets={s: 0.0 for s in SUBS}, output_dir=out,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=0.03, wtc_band_fmax=0.10,
        wtc_chroma=("hbo",), wtc_by_condition=True, wtc_whiten_s=whiten_s, no_report=True,
        **kwargs)


def _table(out, **entities):
    return pd.read_csv(group_output_path(out, "G1", {"task": "tap", **entities},
                                         "relmat", ".tsv"), sep="\t")


def _params(out, **entities):
    path = group_output_path(out, "G1", {"task": "tap", **entities}, "relmat", ".json")
    return json.loads(path.read_text(encoding="utf-8"))["parameters"]


@pytest.fixture(scope="module")
def plain_and_white(dyad, tmp_path_factory):
    plain, white = tmp_path_factory.mktemp("plain"), tmp_path_factory.mktemp("white")
    _build(dyad, plain)
    _build(dyad, white, whiten_s=10.0)
    return plain, white


def test_the_coherence_is_computed_on_the_whitened_signals(plain_and_white):
    plain, white = plain_and_white
    assert not np.allclose(_table(plain, statistic="wtc")["coherence"],
                           _table(white, statistic="wtc")["coherence"])


def test_the_correlation_is_not_whitened_by_it(plain_and_white):
    plain, white = plain_and_white
    pd.testing.assert_frame_equal(_table(plain, pairing=None, statistic="isc"),
                                  _table(white, pairing=None, statistic="isc"))


def test_every_wtc_table_records_the_whitening(plain_and_white):
    plain, white = plain_and_white
    assert "wtc_whiten_s" not in _params(plain, statistic="wtc")
    for entities in ({"statistic": "wtc"}, {"condition": "all", "statistic": "wtc"}):
        params = _params(white, **entities)
        assert params["wtc_whiten_s"] == 10.0
        assert params["wtc_whiten_order"] == int(round(10.0 * SFREQ))


def test_a_phase_level_drawn_unwhitened_is_refused_by_a_whitened_report(dyad, tmp_path):
    from nirspipe.pipeline.hyper.wtc_null import run_wtc_null

    run_wtc_null("G1", "tap", dyad, tmp_path, n_iter=2, wtc_fmin=0.02, wtc_fmax=0.2,
                 band_fmin=0.03, band_fmax=0.10, seed=1, chroma=("hbo",))
    level = group_output_path(tmp_path, "G1", {"task": "tap", "chromophore": "hbo",
                                               "nulldist": "phase", "statistic": "wtc",
                                               "desc": "level"}, "relmat", ".json")
    assert json.loads(level.read_text(encoding="utf-8"))["parameters"]["wtc_whiten_s"] == 0.0
    _build(dyad, tmp_path, whiten_s=10.0)
    assert _params(tmp_path, statistic="wtc")["phase_level_source"] == {"hbo": "none"}


def test_the_phase_null_is_drawn_on_the_whitened_signals(dyad, tmp_path):
    from nirspipe.pipeline.hyper.wtc_null import run_wtc_null

    run_wtc_null("G1", "tap", dyad, tmp_path, n_iter=2, wtc_fmin=0.02, wtc_fmax=0.2,
                 band_fmin=0.03, band_fmax=0.10, seed=1, chroma=("hbo",), whiten_s=10.0)
    _build(dyad, tmp_path, whiten_s=10.0)
    assert _params(tmp_path, statistic="wtc")["phase_level_source"] == {"hbo": "phase"}


# ---- the re-paired null reads it off the real table ----

def test_a_re_paired_draw_hands_the_coherence_whitened_cuts_and_the_correlation_plain(
        dyad, monkeypatch):
    from nirspipe.pipeline.hyper import group_io, group_quality, pair_null

    monkeypatch.setattr(group_io, "load_group_haemo",
                        lambda out, entries, desc="preproc": {entries[0].subject_id:
                                                              dyad["sub-02"].copy()})
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    drawn = list(pair_null._draw_condition_pairs(
        "/out", "tap", "sub-01", dyad["sub-01"], [GroupEntry("GXX", "sub-09", "tap")],
        desc="preproc", bads_scope="run", scope_tasks=["tap"],
        windows=[("talk", 190.0, 370.0)], band_fmin=0.03, n_max=None, refused={},
        whiten_s=10.0))
    assert len(drawn) == 1
    _, _, pair, _, white, _ = drawn[0]
    for sid in pair:
        assert pair[sid].n_times == white[sid].n_times
        assert not np.allclose(pair[sid].get_data(), white[sid].get_data())
    # the fixed member's whitened cut is the real table's whole-record whitening, cut
    whole = whiten_raws({"sub-01": dyad["sub-01"]}, 10.0)["sub-01"]
    t0 = white["sub-01"].first_time - dyad["sub-01"].first_time
    expected = whole.copy().crop(tmin=t0, tmax=t0 + white["sub-01"].times[-1]).get_data()
    assert np.allclose(white["sub-01"].get_data(), expected)


# ---- nothing merges across it ----

def test_tables_whitened_differently_refuse_to_merge(tmp_path):
    from nirspipe.pipeline.hyper.wtc_aggregate import aggregate_wtc

    paths = []
    for gid, extra in (("G01", {}), ("G02", {"wtc_whiten_s": 10.0})):
        tsv = tmp_path / f"group-{gid}" / "nirs" / f"group-{gid}_task-tap_stat-wtc_relmat.tsv"
        tsv.parent.mkdir(parents=True)
        pd.DataFrame({"chromophore": ["hbo"], "sub1": ["a"], "sub2": ["b"],
                      "label": ["S1_D1"], "coherence": [0.3]}).to_csv(tsv, sep="\t",
                                                                      index=False)
        tsv.with_suffix(".json").write_text(json.dumps({"parameters": {
            "band_fmin": 0.03, "band_fmax": 0.1, "mask_coi": True, **extra}}))
        paths.append(tsv)
    with pytest.raises(ValueError, match="wtc_whiten_s"):
        aggregate_wtc(tmp_path, paths)


def test_a_re_banded_table_keeps_what_its_maps_were_computed_with(dyad, tmp_path):
    from nirspipe.pipeline.hyper.wtc_store import reband_tree

    _build(dyad, tmp_path, whiten_s=10.0, wtc_save_maps=True)
    written = reband_tree(tmp_path, 0.05, 0.15)
    assert written
    params = json.loads(written[0].with_suffix(".json").read_text(encoding="utf-8"))["parameters"]
    assert params["wtc_whiten_s"] == 10.0
    assert (params["band_fmin"], params["band_fmax"]) == (0.05, 0.15)
    assert params["wtc_fmin"] == 0.02


def test_a_stand_in_too_short_to_whiten_is_counted_rather_than_raised(dyad, monkeypatch):
    from nirspipe.pipeline.hyper import group_io, group_quality, pair_null

    short = dyad["sub-02"].copy().crop(tmax=35.0)      # 175 samples against AR(50)
    monkeypatch.setattr(group_io, "load_group_haemo",
                        lambda out, entries, desc="preproc": {entries[0].subject_id: short})
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    refused: dict = {}
    drawn = list(pair_null._draw_condition_pairs(
        "/out", "tap", "sub-01", dyad["sub-01"], [GroupEntry("GXX", "sub-09", "tap")],
        desc="preproc", bads_scope="run", scope_tasks=["tap"],
        windows=[("rest", 0.0, 30.0)], band_fmin=0.03, n_max=None, refused=refused,
        whiten_s=10.0))
    assert drawn == []
    assert refused == {"too_short_to_whiten": ["sub-09"]}
