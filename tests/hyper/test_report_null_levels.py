"""Which null a dyad page's phase arrows and chords were drawn against, and saying so.

A level on disk is used only where its sidecar says it was drawn on these recordings with
these settings; otherwise the page falls back and says why. The re-paired null is drawn one
condition at a time, so it thresholds condition pages and never the whole run.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.io.derivatives import group_output_path
from fnirs_pipe.pipeline.hyper import GroupEntry, _hyper_sidecar, alignment_params
from fnirs_pipe.pipeline.hyper.wtc import compute_wtc, wtc_grid_params
from fnirs_pipe.pipeline.hyper.wtc_store import (level_params, save_cond_null_levels,
                                                 save_null_levels)
from fnirs_pipe.qc.common.windows import condition_windows
from fnirs_pipe.qc.hyper.hyper_report import build_hyper_post_report
from tests.hyper.test_hyper_page_contract import LABELS, SFREQ, DURATION, _raw

FMIN, FMAX = 0.02, 0.2
SUBS = ("sub-01", "sub-02")


@pytest.fixture(scope="module")
def dyad():
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    shared = np.sin(2 * np.pi * 0.05 * t)
    return {sid: _raw(i, shared).crop(tmin=20.0) for i, sid in enumerate(SUBS, start=1)}


@pytest.fixture(scope="module")
def n_freqs(dyad):
    return len(compute_wtc(dyad, fmin=FMIN, fmax=FMAX, ch_type="hbo").freqs)


def _level_file(out, nulldist, **extra):
    entities = {"task": "tap", "chromophore": "hbo", "nulldist": nulldist,
                "statistic": "wtc", "desc": "level", **extra}
    return group_output_path(out, "G1", entities, "relmat", ".npz")


def _stamp(path, dyad, **override):
    params = {**level_params(dyad, wtc_fmin=FMIN, wtc_fmax=FMAX, mask_coi=True), **override}
    _hyper_sidecar(path, "test", [], **params)


def _spans(dyad):
    return {label: [round(a, 3), round(b, 3)]
            for label, a, b in condition_windows(dyad["sub-01"], min_duration=1.0 / FMIN)}


def _build(dyad, out):
    build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", s, "tap") for s in SUBS],
        aligned_raws=dyad, offsets={s: 0.0 for s in SUBS}, output_dir=out,
        wtc_fmin=FMIN, wtc_fmax=FMAX, wtc_band_fmin=0.03, wtc_band_fmax=0.10,
        wtc_chroma=("hbo",), wtc_by_condition=True)


def _page(out, condition=None):
    pages = [p for p in (out / "group-G1").rglob("*_report.html")
             if ("cond-" in p.name) == (condition is not None)
             and (condition is None or f"cond-{condition}_" in p.name)]
    assert len(pages) == 1, pages
    return pages[0].read_text(encoding="utf-8")


def _cell(html, heading):
    return re.search(rf"<th>{heading}</th><td>(.*?)</td>", html, re.S).group(1).strip()


def _sidecar(out, entities):
    path = group_output_path(out, "G1", {"task": "tap", **entities}, "relmat", ".json")
    return json.loads(path.read_text(encoding="utf-8"))["parameters"]


def _phase_level(dyad, out, n, value=0.0, **override):
    path = save_null_levels({(*SUBS, label): np.full(n, value) for label in LABELS},
                            _level_file(out, "phase"))
    _stamp(path, dyad, **override)


def _phase_n(out, entities, condition=None) -> int:
    """How many cells the phase columns averaged: none when a level of 2.0 was applied."""
    path = group_output_path(out, "G1", {"task": "tap", **entities}, "relmat", ".tsv")
    table = pd.read_csv(path, sep="\t")
    if condition is not None:
        table = table[table["condition"] == condition]
    return int(table["phase_n"].sum())


def _pair_level(dyad, out, n, condition="talk"):
    path = save_cond_null_levels({condition: {(*SUBS, label): np.full(n, 2.0)
                                              for label in LABELS}},
                                 _level_file(out, "pair", condition="all"))
    _stamp(path, dyad, condition_windows_s=_spans(dyad), n_iter=5)


def test_a_fitting_phase_level_thresholds_every_page(dyad, n_freqs, tmp_path):
    _phase_level(dyad, tmp_path, n_freqs)
    _build(dyad, tmp_path)
    assert _cell(_page(tmp_path), "Arrow threshold") == "the phase-scrambled null"
    assert _cell(_page(tmp_path, "talk"), "Arrow threshold") == "the phase-scrambled null"
    assert _sidecar(tmp_path, {"statistic": "wtc"})["phase_level_source"] == {"hbo": "phase"}


def test_a_phase_level_drawn_with_other_settings_is_not_used(dyad, n_freqs, tmp_path):
    _phase_level(dyad, tmp_path, n_freqs, value=2.0, wtc_fmin=0.01)
    _build(dyad, tmp_path)
    html = _page(tmp_path)
    assert _phase_n(tmp_path, {"statistic": "wtc"}) > 0
    assert _cell(html, "Arrow threshold") == "0.5"
    assert "drawn with different wtc_fmin" in html


def test_a_phase_level_with_no_sidecar_is_not_used(dyad, n_freqs, tmp_path):
    save_null_levels({(*SUBS, label): np.full(n_freqs, 2.0) for label in LABELS},
                     _level_file(tmp_path, "phase"))
    _build(dyad, tmp_path)
    assert _phase_n(tmp_path, {"statistic": "wtc"}) > 0


def test_a_level_of_the_wrong_length_is_not_named_as_the_threshold(dyad, n_freqs, tmp_path):
    """The arrows already ignore a level of the wrong length, so nothing may name it."""
    _phase_level(dyad, tmp_path, n_freqs - 1)
    _build(dyad, tmp_path)
    assert _cell(_page(tmp_path), "Arrow threshold") == "0.5"


def test_the_re_paired_level_thresholds_its_condition_and_nothing_else(dyad, n_freqs,
                                                                         tmp_path):
    _phase_level(dyad, tmp_path, n_freqs)
    _pair_level(dyad, tmp_path, n_freqs, condition="talk")
    _build(dyad, tmp_path)
    assert _cell(_page(tmp_path, "talk"), "Arrow threshold") == "the re-paired null"
    assert _cell(_page(tmp_path, "rest"), "Arrow threshold") == "the phase-scrambled null"
    assert _cell(_page(tmp_path), "Arrow threshold") == "the phase-scrambled null"
    by_cond = _sidecar(tmp_path, {"condition": "all", "statistic": "wtc"})
    assert by_cond["phase_level_source"] == {"hbo": {"rest": "phase", "talk": "pair"}}


def test_a_re_paired_level_for_another_span_is_not_used(dyad, n_freqs, tmp_path):
    path = save_cond_null_levels({"talk": {(*SUBS, label): np.full(n_freqs, 2.0)
                                           for label in LABELS}},
                                 _level_file(tmp_path, "pair", condition="all"))
    spans = {k: [a + 5.0, b] for k, (a, b) in _spans(dyad).items()}
    _stamp(path, dyad, condition_windows_s=spans, n_iter=5)
    _build(dyad, tmp_path)
    assert _phase_n(tmp_path, {"condition": "all", "statistic": "wtc"}, "talk") > 0


def test_the_chords_name_the_re_paired_null_where_it_drew_them(dyad, tmp_path):
    rows = [{"chromophore": c, "condition": "talk", "sub1": SUBS[0], "sub2": SUBS[1],
             "label": a, "label2": b, "null_mean": 0.0, "null_p95": 0.1,
             "null_abs_p95": 0.1, "n_iter": 5}
            for c in ("hbo", "hbr") for a in LABELS for b in LABELS]
    path = group_output_path(tmp_path, "G1", {"task": "tap", "condition": "all",
                                              "nulldist": "pair", "statistic": "isc"},
                             "relmat", ".tsv")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    _hyper_sidecar(path, "test", [], **wtc_grid_params(dyad), **alignment_params(dyad),
                   isc_whiten_max_order=0, isc_max_lag_s=0.0, isc_band_hz=None,
                   condition_windows_s=_spans(dyad), n_iter=5)
    _build(dyad, tmp_path)
    assert _cell(_page(tmp_path, "talk"), "ISC chords") == (
        "above each pairing's re-paired null, 5 partners")
    assert "re-paired" not in _cell(_page(tmp_path), "ISC chords")


def _drawn_null(dyad, out):
    from fnirs_pipe.pipeline.hyper.wtc_null import run_wtc_null
    return run_wtc_null(group_id="G1", task="tap", aligned_raws=dyad, output_dir=out,
                        n_iter=2, wtc_fmin=FMIN, wtc_fmax=FMAX, band_fmin=0.03,
                        band_fmax=0.10, seed=0, chroma=("hbo",))


def _build_with_null(dyad, out, **extra):
    build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", s, "tap") for s in SUBS],
        aligned_raws=dyad, offsets={s: 0.0 for s in SUBS}, output_dir=out,
        wtc_fmin=FMIN, wtc_fmax=FMAX, wtc_band_fmin=0.03, wtc_band_fmax=0.10,
        wtc_seed=0, wtc_chroma=("hbo",), wtc_nulls=_drawn_null(dyad, out),
        wtc_phase_null=2, **extra)


def _null_table(out):
    return group_output_path(out, "G1", {"task": "tap", "nulldist": "phase",
                                         "statistic": "wtc"}, "relmat", ".tsv")


def test_the_null_table_is_on_the_first_runs_provenance_diagram(dyad, tmp_path):
    # the diagram is drawn from the sidecars on disk, so a table written after the report
    # would only reach it on the next run
    _build_with_null(dyad, tmp_path)
    assert _null_table(tmp_path).exists()
    (mmd,) = (tmp_path / "group-G1").rglob("*provenance*.mmd")
    assert _null_table(tmp_path).stem.replace("-", "_") in mmd.read_text(encoding="utf-8")


def test_the_null_table_is_written_without_a_report(dyad, tmp_path):
    _build_with_null(dyad, tmp_path, no_report=True)
    assert _null_table(tmp_path).exists()
    assert not list((tmp_path / "group-G1").rglob("*provenance*.mmd"))


def test_the_isc_phase_null_keeps_its_draws_per_condition(dyad, tmp_path):
    build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", s, "tap") for s in SUBS],
        aligned_raws=dyad, offsets={s: 0.0 for s in SUBS}, output_dir=tmp_path,
        wtc_fmin=FMIN, wtc_fmax=FMAX, wtc_band_fmin=0.03, wtc_band_fmax=0.10,
        wtc_chroma=("hbo",), wtc_by_condition=True, isc_phase_null=4, wtc_seed=1,
        no_report=True)
    entities = {"condition": "all", "nulldist": "phase", "statistic": "isc", "desc": "draws"}
    path = group_output_path(tmp_path, "G1", {"task": "tap", **entities}, "relmat", ".tsv")
    draws = pd.read_csv(path, sep="\t")
    assert list(draws.columns[:2]) == ["chromophore", "condition"]
    # the correlation runs both chromophores whatever the coherence was asked for
    assert set(draws["chromophore"]) == {"hbo", "hbr"}
    assert set(draws["condition"]) == set(_spans(dyad))
    assert set(draws["draw"]) == set(range(4))
    assert {"r", "r_z", "sub1", "sub2", "label", "label2"} <= set(draws.columns)
    side = _sidecar(tmp_path, entities)
    assert side["isc_phase_null_iter"] == 4
    assert side["conditions"] == list(_spans(dyad))
