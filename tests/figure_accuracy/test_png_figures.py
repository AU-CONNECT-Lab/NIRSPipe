"""Figures saved as images cannot be read back, so these check what each one was handed."""

import numpy as np
import pytest

from tests.figure_accuracy._run import Run
from tests.figure_accuracy.conftest import run_capturing


@pytest.fixture(scope="module")
def censored(tmp_path_factory):
    """A prep run that censors on GVTD, with the bad-segment zoom's arguments captured."""
    root = tmp_path_factory.mktemp("fingerprint_censored")
    done = run_capturing(root, ["--gvtd-censor", "long", "--gvtd-censor-n-std", "3",
                                "--gvtd-min-epoch-s", "5"], ("bad_segment_zoom_figure",))
    return Run(root / "out", done["truth"], captured=done["captured"])


def _zoom(run):
    return run.captured["bad_segment_zoom_figure"][1]


def test_the_zoom_covers_both_artefacts(censored):
    for _, when in (censored.truth.spike, censored.truth.step):
        assert any(onset - 1 <= when <= onset + duration + 1
                   for onset, duration in _zoom(censored)["bad_segments"]), when


@pytest.mark.parametrize("row, desc", [("raw_before", "sci"), ("raw_after", "motcorrected")])
def test_the_zoom_rows_are_the_od_either_side_of_the_correction(censored, row, desc):
    raw = _zoom(censored)[row]
    expected = censored.read(desc).get_data(picks=raw.ch_names)
    np.testing.assert_allclose(raw.get_data(), expected, rtol=1e-6)


def _jump(raw, pair, when):
    """The level change across ``when`` on the pair's 760 row, in units of that row's own spread."""
    y = raw.get_data(picks=[f"{pair} 760"])[0]
    t = raw.times
    before, after = y[(t > when - 6) & (t < when - 1)], y[(t > when + 1) & (t < when + 6)]
    return abs(np.median(after) - np.median(before)) / np.std(y)


def test_the_zoom_s_after_row_carries_less_of_the_planted_step_than_its_before_row(censored):
    pair, when = censored.truth.step
    zoom = _zoom(censored)
    # an uncorrected row, whatever its units, keeps the whole step relative to its own spread
    assert _jump(zoom["raw_before"], pair, when) > 1.0
    assert _jump(zoom["raw_after"], pair, when) < 0.5 * _jump(zoom["raw_before"], pair, when)


def test_the_brain_views_grade_by_the_windowed_sci(denoise_run):
    _, kwargs = denoise_run.captured["quality_brain_views"]
    raw = denoise_run.record()["per_channel"]["raw"]
    assert kwargs["sci_scores"] == pytest.approx(raw["sci_win_per_channel"])
    assert kwargs["sci_scores"] != pytest.approx(raw["sci_per_channel"])


def test_the_flat_map_beside_them_grades_the_same_sci_over_the_run_s_rejections(denoise_run):
    args, kwargs = denoise_run.captured["optode_layout_static"]
    raw = denoise_run.record()["per_channel"]["raw"]
    assert args[1] == pytest.approx(raw["sci_win_per_channel"])
    assert {c.rsplit(" ", 1)[0] for c in args[2]} == {p.name for p in denoise_run.truth.pairs if p.bad}
    # which of an optode's channels are long is the run's call; this run kept the defaults
    from nirspipe.qc.metrics import separation_bands
    assert kwargs["sep_bands"] == separation_bands()


def test_the_brain_views_place_each_channel_at_its_own_position_with_its_own_verdict(denoise_run):
    args, _ = denoise_run.captured["quality_brain_views"]
    names, coords, good = args[0], np.asarray(args[1], float), np.asarray(args[2], bool)
    for name, xyz, ok in zip(names, coords, good):
        pair = denoise_run.truth.pair(name.rsplit(" ", 1)[0])
        np.testing.assert_allclose(xyz, pair.position, atol=1e-6)
        assert ok == (not pair.bad), name
