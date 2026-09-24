"""Every row of the haemoglobin panel names the channel set its caption claims.

The panel says "measured on long channels" and prints eight rows. Seven came off the
record's ``*_long`` sections; the GCOR pair did not. It was measured a second time inside
``run_post``, over every channel, and handed to the report as a function argument, so the
two sat in one list under one caption and disagreed. The two readings can move in opposite
directions across the confound regression: the short channels are what the regression
removes, so including them in the measure of what the regression did makes it look like it
worked.

Nothing fails when this drifts. Both numbers are plausible, both render, and the row label
is the same either way, which is why the invariant is pinned here as a property of the
assembled panel rather than as an expected value.
"""

import json

from fnirs_pipe.qc.subject.report import _section_sqm

# a record whose three channel sets disagree on every metric the panel reads, so a row that
# picked the wrong one cannot coincide with the right one
RECORD = {
    "raw":       {"sci_mean": 0.90, "n_long_channels": 4, "n_short_channels": 2},
    "raw_long":  {"sci_mean": 0.95},
    "raw_short": {"sci_mean": 0.80},
    "motion":    {"gvtd_mean": 1.0e-3},
    "preproc":       {"hbo_hbr_corr_mean": -0.03, "gcor_hbo": 0.46, "gcor_hbr": 0.15},
    "preproc_long":  {"hbo_hbr_corr_mean": -0.23, "gcor_hbo": 0.52, "gcor_hbr": 0.21},
    "preproc_short": {"hbo_hbr_corr_mean": 0.36, "gcor_hbo": 0.47, "gcor_hbr": 0.35},
    "filtered":       {"gcor_hbo": 0.66, "gcor_hbr": 0.19},
    "filtered_long":  {"gcor_hbo": 0.71, "gcor_hbr": 0.33},
    "filtered_short": {"gcor_hbo": 0.58, "gcor_hbr": 0.12},
    "errts":       {"hbo_hbr_corr_mean": -0.24, "gcor_hbo": 0.17, "gcor_hbr": 0.15},
    "errts_long":  {"hbo_hbr_corr_mean": -0.46, "gcor_hbo": 0.38, "gcor_hbr": 0.35},
    "errts_short": {"hbo_hbr_corr_mean": -0.11, "gcor_hbo": 0.22, "gcor_hbr": 0.09},
    "per_channel": {},
}


def _panel(tmp_path, record=None) -> dict:
    nirs = tmp_path / "nirs"
    nirs.mkdir(parents=True, exist_ok=True)
    (nirs / "sub-01_task-main_desc-sqm_qc.json").write_text(
        json.dumps(RECORD if record is None else record), encoding="utf-8")
    errors: list = []
    out = _section_sqm({}, [], "01", errors, nirs, sqm_label="sub-01_task-main")
    assert not errors, errors
    return out["sqm"]


def test_the_regression_pair_is_the_same_channel_set_as_the_rows_beside_it(tmp_path):
    sqm = _panel(tmp_path)
    # the caption's set, shown by a row that was always right
    assert sqm["hbo_hbr_corr_mean"] == RECORD["preproc_long"]["hbo_hbr_corr_mean"]
    for chroma in ("hbo", "hbr"):
        assert sqm[f"gcor_{chroma}_prereg"] == RECORD["filtered_long"][f"gcor_{chroma}"]
        assert sqm[f"gcor_{chroma}_postreg"] == RECORD["errts_long"][f"gcor_{chroma}"]


def test_the_before_side_is_the_bandpassed_stage_not_the_unfiltered_one(tmp_path):
    """The bandpass alone raises GCOR, so `preproc` -> `errts` would price in the filter."""
    sqm = _panel(tmp_path)
    assert sqm["gcor_hbo_prereg"] != RECORD["preproc_long"]["gcor_hbo"]


def test_a_montage_with_no_split_falls_back_to_the_whole_file_sections(tmp_path):
    unsplit = {k: v for k, v in RECORD.items() if not k.endswith(("_long", "_short"))}
    sqm = _panel(tmp_path, unsplit)
    assert sqm["gcor_hbo_prereg"] == RECORD["filtered"]["gcor_hbo"]
    assert sqm["gcor_hbo_postreg"] == RECORD["errts"]["gcor_hbo"]


def test_a_run_that_regressed_nothing_prints_no_pair(tmp_path):
    """Without an errts stage the template falls back to a lone number, so the pair must
    not be half-filled: a prereg with no postreg renders as neither."""
    no_errts = {k: v for k, v in RECORD.items() if not k.startswith("errts")}
    sqm = _panel(tmp_path, no_errts)
    assert "gcor_hbo_prereg" not in sqm and "gcor_hbo_postreg" not in sqm
