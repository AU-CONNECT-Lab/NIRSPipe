"""Per-condition shares and assessments leave out BAD_ windows; the dyad screening page too."""

import json
import re

import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.qc.metrics.windowed import (condition_window_means, passing_mask_from_matrices,
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

    mask = passing_mask_from_matrices(win["sci_matrix"], win["psp_matrix"],
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


def test_an_unassessed_condition_claims_no_failing_count(tmp_path):
    out, _ = _by_condition(tmp_path, 200, "--by-condition")
    (page,) = out.rglob("*_cond-cb_report.html")
    text = page.read_text(encoding="utf-8")
    assert not re.search(r'class="badge-\w+">0/\d+ \(0%\)<', text)
    assert re.search(r"Failing in this condition</th>\s*<td>n/a</td>", text)
    (assessed,) = out.rglob("*_cond-ca_report.html")
    assert re.search(r'class="badge-\w+">\d+/\d+ \(\d+%\)<', assessed.read_text(encoding="utf-8"))


# ---- raw QC's verdict, which the dyad page reads ----

def test_raw_qc_leaves_out_an_input_span(tmp_path):
    from nirspipe.cli import qc as qc_cli
    from tests._synth import _write_dataset_root, _write_subject

    _raw, truth = fingerprint_raw("01", "tapping", blocks=True)
    decoupled = truth.block_bad[0]        # decoupled 290-370 s, inside the second block

    def rejected(name, spans):
        bids = tmp_path / name / "bids"
        _write_dataset_root(bids, ["01"])
        raw, _ = fingerprint_raw("01", "tapping", blocks=True)
        add_bad_spans(raw, spans, "BAD_artifact")
        _write_subject(bids, "01", "tapping", raw)
        out = tmp_path / name / "out"
        qc_cli.main([str(bids), str(out), "participant", "--dpf", "6",
                     "--cardiac-l-freq", "0.8", "--cardiac-h-freq", "1.9",
                     "--resp-l-freq", "0.1", "--resp-h-freq", "0.5",
                     "--sci-threshold", str(SCI_CUTOFF), "--psp-threshold", str(PSP_CUTOFF),
                     "--min-good-frac", "0.85", "--skip-bids-validation"])
        (record,) = out.rglob("*_desc-sqmraw_qc.json")
        return {b.split()[0] for b in json.loads(record.read_text())["bad_channels"]}

    assert decoupled in rejected("whole", [])
    assert decoupled not in rejected("spanned", [(280.0, 120.0)])


# ---- the raw viewer's condition pages ----

def _raw_condition_summaries(tmp_path, spans) -> dict:
    """Raw QC with `--by-condition` on the fingerprint with ``spans`` in its input file."""
    from nirspipe.cli import qc as qc_cli
    from tests._synth import _write_dataset_root, _write_subject

    bids = tmp_path / "bids"
    _write_dataset_root(bids, ["01"])
    raw, _ = fingerprint_raw("01", "tapping", blocks=True)
    add_bad_spans(raw, spans, "BAD_artifact")
    _write_subject(bids, "01", "tapping", raw)
    out = tmp_path / "out"
    qc_cli.main([str(bids), str(out), "participant", "--participant-label", "01", "--dpf", "6",
                 "--cardiac-l-freq", "0.8", "--cardiac-h-freq", "1.9",
                 "--resp-l-freq", "0.1", "--resp-h-freq", "0.5",
                 "--sci-threshold", str(SCI_CUTOFF), "--by-condition", "--skip-bids-validation"])
    summaries = {}
    for page in out.rglob("*_cond-*_desc-raw_report.html"):
        text = page.read_text(encoding="utf-8")
        start = text.index("_STATIC_DATA = ") + len("_STATIC_DATA = ")
        (view,), _ = json.JSONDecoder().raw_decode(text, start)
        summaries[view["condition_label"]] = {**view["summary"], "_page": text}
    return summaries


def test_a_raw_condition_page_states_its_kept_time_and_an_unassessed_one_claims_no_count(
        tmp_path):
    # the span covers all of cb (210-370 s) and none of ca (30-190 s)
    summaries = _raw_condition_summaries(tmp_path, [(200.0, 200.0)])
    ca, cb = summaries["ca"], summaries["cb"]
    assert cb["unassessed"] is True and "unassessed" not in ca
    assert "so it is not assessed" in cb["scope"] and "not assessed" not in ca["scope"]
    assert ca["kept"].startswith("160.0 of 160.0 s outside every BAD_ span")
    assert cb["kept"] == "0.0 of 160.0 s outside every BAD_ span; 0 screening windows counted"
    assert 'id="sum-kept"' in cb["_page"]


def test_without_a_span_a_raw_condition_page_carries_what_it_did(tmp_path):
    for summary in _raw_condition_summaries(tmp_path, []).values():
        assert "kept" not in summary and "unassessed" not in summary
        assert "not assessed" not in summary["scope"]


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
