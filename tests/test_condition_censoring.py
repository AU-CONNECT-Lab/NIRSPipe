"""Per-condition shares and assessments leave out BAD_ windows; the dyad screening page too."""

import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.pipeline.hyper.group_io import GroupEntry
from nirspipe.pipeline.hyper.group_quality import compute_group_sqm_raw
from nirspipe.qc.metrics.windowed import (condition_window_means, coupled_mask_from_matrices,
                                          window_centers, windows_touching)
from nirspipe.qc.subject.record_io import read_record
from nirspipe.qc.subject.sqm_record import _kept_seconds
from nirspipe.utils.spans import add_bad_spans

from tests._fingerprint import BLOCKS, CLI_ARGS, fingerprint_raw, make_fingerprint_dataset

SCI_CUTOFF, PSP_CUTOFF = 0.75, 0.12          # the fingerprint's CLI_ARGS


# ---- pieces ----

def test_a_window_touching_a_span_at_all_is_flagged():
    times = [[0, 10], [10, 20], [20, 30]]
    assert windows_touching(times, [(18.0, 19.0)]).tolist() == [False, True, False]
    assert windows_touching(times, [(10.0, 10.0)]).tolist() == [False, False, False]
    assert windows_touching(times, []).tolist() == [False, False, False]


def test_kept_seconds_count_overlapping_spans_once():
    assert _kept_seconds(100.0, 200.0, []) == 100.0
    assert _kept_seconds(100.0, 200.0, [(90.0, 120.0), (110.0, 130.0), (190.0, 300.0)]) == 60.0
    assert _kept_seconds(100.0, 200.0, [(0.0, 400.0)]) == 0.0


# ---- the whole run ----

def _by_condition(tmp_path, keep_until, *extra):
    bids, _ = make_fingerprint_dataset(tmp_path, blocks=True)
    table = tmp_path / "keep.tsv"
    table.write_text(f"onset\tduration\n0\t{keep_until}\n", encoding="utf-8")
    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--keep-spans", str(table),
                  *extra, "--skip-bids-validation"])
    (record_path,) = out.rglob("*desc-sqm_qc.json")
    return out, read_record(record_path)


def test_a_condition_counts_only_its_windows_clear_of_every_span(tmp_path):
    _, record = _by_condition(tmp_path, 330, "--no-report")
    entry = record["by_condition"]["cb"]
    _, t0, duration, _ = next(b for b in BLOCKS if b[0] == "cb")
    t1 = t0 + duration
    assert entry["kept_s"] == pytest.approx(330 - t0)

    win = record["windowed"]
    channels = win["sci_channels"]
    centers = window_centers(win["sci_times"])
    inside = (centers >= t0) & (centers <= t1)
    counted = inside & ~windows_touching(win["sci_times"], [(330.0, 400.0)])
    assert 0 < counted.sum() < inside.sum()
    assert entry["counted_windows"] == counted.sum()

    mask = coupled_mask_from_matrices(win["sci_matrix"], win["psp_matrix"],
                                      SCI_CUTOFF, PSP_CUTOFF)
    share = np.asarray(mask, dtype=float)[:, counted].mean(axis=1)
    got = entry["per_channel"]["good_frac_per_channel"]
    np.testing.assert_allclose([got[ch] for ch in channels], share)

    # the SCI mean keeps every window of the condition, as the run's own mean does
    sci = condition_window_means(win["sci_matrix"], win["sci_times"], [("cb", t0, t1)])["cb"]
    got_sci = entry["per_channel"]["sci_win_per_channel"]
    np.testing.assert_allclose([got_sci[ch] for ch in channels], sci)


def test_a_condition_wholly_inside_spans_is_kept_but_not_assessed(tmp_path):
    out, record = _by_condition(tmp_path, 200, "--by-condition")
    entry = record["by_condition"]["cb"]
    assert entry["kept_s"] == 0.0 and entry["counted_windows"] == 0
    assert entry["bad_channels"] == []
    assert entry["per_channel"]["good_frac_per_channel"] == {}
    assert record["by_condition"]["ca"]["counted_windows"] > 0

    (page,) = out.rglob("*_cond-cb_report.html")
    assert "so it is not assessed" in page.read_text(encoding="utf-8")


# ---- the dyad screening page ----

def test_the_dyad_screening_page_leaves_out_an_input_span(tmp_path):
    raw, truth = fingerprint_raw("01", "tapping", blocks=True)
    decoupled = truth.block_bad[0]        # decoupled 290-370 s, inside the second block

    def rejected(spans):
        marked = raw.copy()
        add_bad_spans(marked, spans, "BAD_artifact")
        sqm = compute_group_sqm_raw([GroupEntry("G01", "01", "tapping")], {"01": marked},
                                    SCI_CUTOFF, tmp_path, 0.8, 1.9, psp_threshold=PSP_CUTOFF,
                                    min_good_frac=0.85)
        return {b.split()[0] for b in sqm["01"]["bad_channels"]}

    assert decoupled in rejected([])
    assert decoupled not in rejected([(280.0, 120.0)])


def test_without_a_span_touching_it_a_condition_is_unchanged_bit_for_bit():
    from nirspipe.qc.subject.condition_views import condition_slices_from_record

    rng = np.random.default_rng(5)
    times = [[10.0 * i, 10.0 * (i + 1)] for i in range(30)]
    record = {"windowed": {"sci_matrix": rng.uniform(0.3, 1.0, (4, 30)).tolist(),
                           "psp_matrix": rng.uniform(0.0, 0.3, (4, 30)).tolist(),
                           "sci_times": times, "psp_times": times}}
    names, windows = ["a", "b", "c", "d"], [("x", 20.0, 120.0)]
    plain = condition_slices_from_record(record, names, windows, SCI_CUTOFF, PSP_CUTOFF)
    assert condition_slices_from_record(record, names, windows, SCI_CUTOFF, PSP_CUTOFF,
                                        spans=[(200.0, 260.0)]) == plain
    moved = condition_slices_from_record(record, names, windows, SCI_CUTOFF, PSP_CUTOFF,
                                         spans=[(55.0, 56.0)])
    assert moved["x"]["good_frac_per_channel"] != plain["x"]["good_frac_per_channel"]
    assert moved["x"]["sci_win_per_channel"] == plain["x"]["sci_win_per_channel"]
