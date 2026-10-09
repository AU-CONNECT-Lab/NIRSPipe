"""The PSD figures draw each stage on its own row, and the filter they draw is the filter that ran."""

import numpy as np
import pytest

from nirspipe.cli.run import mode_defaults
from tests._fingerprint import CARDIAC_FREQ, CLI_ARGS, SYSTEMIC_FREQ
from tests.figure_accuracy._payload import one_figure
from tests.figure_accuracy._read import traces, xy


def _arg(flag):
    return float(CLI_ARGS[CLI_ARGS.index(flag) + 1])


def _db_at(trace, freq):
    x, y = xy(trace)
    return y[np.argmin(np.abs(x - freq))]


def _peak(trace, band=(0.02, 0.45)):
    x, y = xy(trace)
    keep = (x > band[0]) & (x < band[1])
    return x[keep][np.argmax(y[keep])]


def _channel_lines(fig, axis, colour="#e74c3c"):
    return [t for t in traces(fig, axis=axis) if t.get("name") is None and t["line"]["color"] == colour]


@pytest.fixture(scope="module")
def psd(denoise_run):
    return one_figure(denoise_run.figure("psd"))


def test_each_row_says_which_stage_it_is(psd):
    for axis, label in (("x", "Before bandpass"), ("x2", "After bandpass"), ("x3", "desc-errts")):
        assert traces(psd, "HbO", axis)[0]["hovertemplate"].startswith(label)


def test_the_rows_hold_the_kept_channels_and_not_the_rejected_one(denoise_run, psd):
    truth = denoise_run.truth
    lines = _channel_lines(psd, "x3")
    assert len(lines) == sum(not p.bad for p in truth.pairs)
    found = set()
    for line in lines:
        peak = _peak(line)
        nearest = min(truth.long_pairs, key=lambda p: abs(p.hbo_freq - peak))
        if abs(nearest.hbo_freq - peak) < 0.005:
            found.add(nearest.name)
    assert found == {p.name for p in truth.long_pairs if not p.bad}


def test_the_systemic_peak_is_on_the_rows_before_regression_and_not_after(psd):
    before = _db_at(traces(psd, "HbO", "x")[0], SYSTEMIC_FREQ)
    after = _db_at(traces(psd, "HbO", "x3")[0], SYSTEMIC_FREQ)
    assert before - after > 15


def test_the_drawn_filter_response_is_the_attenuation_the_filter_produced(psd):
    before, after = traces(psd, "HbO", "x")[0], traces(psd, "HbO", "x2")[0]
    response = traces(psd, "Filter response")[0]
    fx = xy(response)[0]
    drawn = np.asarray(response["text"], dtype=float)
    low_pass = mode_defaults("denoise")["low_pass"]
    # deeper in the stopband the measured ratio stops at the Welch window's leakage floor
    for freq, tolerance in ((0.3, 1.0), (0.9 * low_pass, 3.0), (low_pass, 3.0),
                            (1.2 * low_pass, 3.0), (CARDIAC_FREQ, 10.0)):
        measured = _db_at(after, freq) - _db_at(before, freq)
        assert measured == pytest.approx(drawn[np.argmin(np.abs(fx - freq))], abs=tolerance), freq


def test_the_bands_are_the_ones_the_run_was_given(psd):
    spans = {(s["x0"], s["x1"]) for s in psd["layout"]["shapes"] if s["type"] == "rect"}
    assert (_arg("--cardiac-l-freq"), _arg("--cardiac-h-freq")) in spans
    assert (_arg("--resp-l-freq"), _arg("--resp-h-freq")) in spans


def test_each_pair_page_draws_its_own_pair(denoise_run):
    for pair in denoise_run.truth.long_pairs:
        if pair.bad:
            continue
        fig = one_figure(denoise_run.figure("psddetail", chan=pair.name.replace("_", "")))
        # before regression the shared systemic peak can outweigh a small pair's own
        assert _peak(_channel_lines(fig, "x3")[0]) == pytest.approx(pair.hbo_freq, abs=0.005)
        before = _db_at(_channel_lines(fig, "x2")[0], SYSTEMIC_FREQ)
        after = _db_at(_channel_lines(fig, "x3")[0], SYSTEMIC_FREQ)
        assert before - after > 15
