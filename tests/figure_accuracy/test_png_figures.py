"""Figures saved as images cannot be read back, so these check what each one was handed."""

import numpy as np
import pytest

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset
from tests.figure_accuracy._run import Run


@pytest.fixture(scope="module")
def zoom_inputs(tmp_path_factory):
    """A prep run that censors on GVTD, with the bad-segment zoom's arguments captured."""
    from fnirs_pipe.cli.run import main
    from fnirs_pipe.qc.subject import report

    seen = {}
    original = report.bad_segment_zoom_figure

    def spy(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    root = tmp_path_factory.mktemp("fingerprint_censored")
    bids, truth = make_fingerprint_dataset(root)
    report.bad_segment_zoom_figure = spy
    try:
        main([str(bids), str(root / "out"), "participant", *CLI_ARGS,
              "--gvtd-censor", "long", "--gvtd-censor-n-std", "3", "--gvtd-min-epoch-s", "5",
              "--skip-bids-validation"])
    finally:
        report.bad_segment_zoom_figure = original
    return seen, Run(root / "out", truth)


def test_the_zoom_covers_both_artefacts(zoom_inputs):
    seen, run = zoom_inputs
    for _, when in (run.truth.spike, run.truth.step):
        assert any(onset - 1 <= when <= onset + duration + 1 for onset, duration in seen["bad_segments"]), when


def test_the_zoom_before_row_is_the_od_before_correction(zoom_inputs):
    seen, run = zoom_inputs
    raw = seen["raw_before"]
    expected = run.read("sci").get_data(picks=raw.ch_names)
    np.testing.assert_allclose(raw.get_data(), expected, rtol=1e-6)


@pytest.mark.xfail(strict=True, reason="report.py hands the zoom the raw intensity as its After row")
def test_the_zoom_after_row_is_the_od_after_correction(zoom_inputs):
    seen, run = zoom_inputs
    raw = seen["raw_after"]
    expected = run.read("motcorrected").get_data(picks=raw.ch_names)
    np.testing.assert_allclose(raw.get_data(), expected, rtol=1e-6)
