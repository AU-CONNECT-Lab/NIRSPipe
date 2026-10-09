"""Prewhitening before the correlation: the coherence's rule, its own order.

One order in seconds for every channel of both members, fitted on the whole aligned record and
then cut, the zeroed transient left out of every window. The real table and the re-paired
null read the same whitening; the coherence reads none of it.
"""

import json

import numpy as np
import pandas as pd
import pytest

from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import group_output_path
from nirspipe.io.snirf import long_channel_picks
from nirspipe.pipeline.hyper import GroupEntry
from nirspipe.pipeline.hyper.isc import compute_isc_pairs
from nirspipe.pipeline.hyper.whiten import whiten_order, whiten_raws
from nirspipe.qc.hyper.hyper_report import build_hyper_post_report
from tests.hyper.test_hyper_page_contract import DURATION, SFREQ, _raw

SUBS = ("sub-01", "sub-02")
WHITEN_S = 10.0


@pytest.fixture(scope="module")
def dyad():
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    shared = np.sin(2 * np.pi * 0.05 * t)
    return {sid: _raw(i, shared).crop(tmin=20.0) for i, sid in enumerate(SUBS, start=1)}


def _build(dyad, out, isc_whiten_s=0.0):
    build_hyper_post_report(
        group_id="G1", task="tap", group=[GroupEntry("G1", s, "tap") for s in SUBS],
        aligned_raws=dyad, offsets={s: 0.0 for s in SUBS}, output_dir=out,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=0.03, wtc_band_fmax=0.10,
        wtc_chroma=("hbo",), wtc_by_condition=True, isc_whiten_s=isc_whiten_s,
        no_report=True)


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
    _build(dyad, white, isc_whiten_s=WHITEN_S)
    return plain, white


# ---- the real table ----

def test_the_correlation_is_the_whole_record_whitening_cut_past_its_transient(
        dyad, plain_and_white):
    _, white = plain_and_white
    whole = whiten_raws(dyad, WHITEN_S)
    # the report's default is uncrossed, so its table holds the same-channel pairs
    _, _, expected, _ = compute_isc_pairs(whole, list(SUBS), "hbo", skip_s=WHITEN_S,
                                          cross=False)
    written = _table(white, statistic="isc")
    written = written[(written["chromophore"] == "hbo") & written["condition"].isna()]
    assert written["r"].to_numpy() == pytest.approx(expected["r"].to_numpy(), abs=1e-9,
                                                    nan_ok=True)


def test_the_transient_is_left_out_of_the_window(dyad):
    whole = whiten_raws(dyad, WHITEN_S)
    end = float(whole["sub-01"].times[-1])
    skipped, _, _, _ = compute_isc_pairs(whole, list(SUBS), "hbo", skip_s=WHITEN_S)
    cut, _, _, _ = compute_isc_pairs(whole, list(SUBS), "hbo", window=(WHITEN_S, end + 1.0))
    assert skipped == pytest.approx(cut, nan_ok=True)


def test_the_coherence_is_not_whitened_by_it(plain_and_white):
    plain, white = plain_and_white
    pd.testing.assert_frame_equal(_table(plain, statistic="wtc"), _table(white, statistic="wtc"))


def test_every_isc_table_records_the_whitening(plain_and_white):
    plain, white = plain_and_white
    off = _params(plain, statistic="isc")
    assert off["isc_whiten_s"] == 0.0 and "isc_whiten_order" not in off
    on = _params(white, statistic="isc")
    assert on["isc_whiten_s"] == WHITEN_S
    assert on["isc_whiten_order"] == int(round(WHITEN_S * SFREQ))


def test_no_table_carries_a_per_channel_order_any_more(plain_and_white):
    _, white = plain_and_white
    assert not {"ar_order", "ar_order2"} & set(_table(white, statistic="isc").columns)


# ---- the re-paired null ----

def _draws(dyad, monkeypatch, windows, **kwargs):
    from nirspipe.pipeline.hyper import group_io, group_quality, pair_null

    monkeypatch.setattr(group_io, "load_group_haemo",
                        lambda out, entries, desc="preproc": {entries[0].subject_id:
                                                              dyad["sub-02"].copy()})
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    return list(pair_null._draw_condition_pairs(
        "/out", "tap", "sub-01", dyad["sub-01"], [GroupEntry("GXX", "sub-09", "tap")],
        desc="preproc", bads_scope="run", scope_tasks=["tap"], windows=windows,
        band_fmin=0.03, n_max=None, refused={}, **kwargs))


def test_a_re_paired_draw_hands_the_correlation_its_own_whitened_cuts(dyad, monkeypatch):
    (_, _, pair, inner, white, isc_window), = _draws(
        dyad, monkeypatch, [("talk", 190.0, 370.0)], isc_whiten_s=WHITEN_S)
    # the coherence was not whitened, so its cut is the plain one
    whole = whiten_raws({"sub-01": dyad["sub-01"]}, WHITEN_S)["sub-01"]
    t0 = pair["sub-01"].first_time - dyad["sub-01"].first_time
    expected = whole.copy().crop(tmin=t0, tmax=t0 + pair["sub-01"].times[-1]).get_data()
    assert np.allclose(pair["sub-01"].get_data(), expected)
    assert not np.allclose(pair["sub-01"].get_data(), white["sub-01"].get_data())
    # far from either record's start, so nothing is skipped
    assert isc_window == pytest.approx(inner)


def test_a_condition_at_the_record_start_is_correlated_past_the_transient(dyad, monkeypatch):
    (_, _, pair, inner, _, isc_window), = _draws(
        dyad, monkeypatch, [("rest", 0.0, 180.0)], isc_whiten_s=WHITEN_S)
    assert inner[0] == pytest.approx(0.0)
    assert isc_window == pytest.approx((WHITEN_S, inner[1]))
    order = whiten_order(dyad, WHITEN_S)
    for raw in pair.values():
        data = raw.get_data(picks=long_channel_picks(raw, "hbo"))
        assert np.all(data[:, :order] == 0.0) and np.all(data[:, order] != 0.0)


def test_without_correlation_whitening_the_window_is_the_condition_s(dyad, monkeypatch):
    (_, _, _, inner, _, isc_window), = _draws(dyad, monkeypatch, [("rest", 0.0, 180.0)])
    assert isc_window == inner


def test_a_real_table_from_before_the_seconds_setting_is_refused(tmp_path):
    from nirspipe.pipeline.hyper.pair_null import _isc_settings_of

    sidecar = tmp_path / "isc.json"
    sidecar.write_text(json.dumps({"parameters": {"isc_whiten_max_order": 0}}))
    with pytest.raises(StageError, match="isc_whiten_s"):
        _isc_settings_of(sidecar, 0.0, 0.0, None)
