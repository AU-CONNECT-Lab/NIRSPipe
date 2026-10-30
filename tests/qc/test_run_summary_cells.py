"""The run summary's cells are formatted once, so the run report and the raw page round a half alike."""

from nirspipe.qc.common.channel_table import summary_cells


def test_a_share_on_a_half_rounds_as_python_formats_it():
    # 5 of 40 is 12.5%; JavaScript's Math.round printed 13% where the run report printed 12%
    assert summary_cells(5, 40, None, None, None)["bad"] == "5/40 (12%)"


def test_every_cell_of_a_measured_run():
    assert summary_cells(3, 40, 0.85, 1.2, 391.4) == {
        "bad": "3/40 (8%)", "frac": "85% of windows", "heart_rate": "1.20 Hz (72 bpm)",
        "duration": "391 s"}


def test_what_was_not_measured_has_no_cell():
    assert summary_cells(0, 0, None, None, None) == {
        "bad": None, "frac": None, "heart_rate": None, "duration": None}
