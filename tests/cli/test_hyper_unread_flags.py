"""A flag typed for a dyad run that does not read it is named in the log and on the dyad page,
as nirspipe names its own. One table says what a dyad run reads."""

import logging

import pytest

from nirspipe.cli import hyper as hyper_cli

# a run that draws nothing at random, reads conditions, maps no regions and writes its report
PLAIN = {"wtc_significance": False, "wtc_phase_null": None, "isc_phase_null": 0,
         "wtc_cond_transform": False, "by_condition": True, "roi_mapping": None,
         "no_report": False}


# ---- the table ----

def test_a_plain_run_reads_no_draw_setting_no_pad_and_no_region_floor():
    assert hyper_cli.unread_settings(PLAIN) == {
        "wtc_mc_count": "only --wtc-significance reads it",
        "wtc_phase_null_cross": "only --wtc-phase-null reads it",
        "phase_null_shared": "no phase null is drawn",
        "seed": "nothing is drawn at random",
        "wtc_cond_pad": "only --wtc-cond-transform reads it",
        "wtc_roi_min_channels": "no region is averaged without --roi-mapping"}


def test_each_draw_reads_its_own_settings_and_the_seed():
    significance = set(hyper_cli.unread_settings({**PLAIN, "wtc_significance": True}))
    assert not {"wtc_mc_count", "seed"} & significance
    assert {"phase_null_shared", "wtc_phase_null_cross"} <= significance
    isc = set(hyper_cli.unread_settings({**PLAIN, "isc_phase_null": 100}))
    assert not {"seed", "phase_null_shared"} & isc
    assert "wtc_phase_null_cross" in isc
    wtc = set(hyper_cli.unread_settings({**PLAIN, "wtc_phase_null": 10}))
    assert not {"seed", "phase_null_shared", "wtc_phase_null_cross"} & wtc


def test_the_cone_needs_conditions_and_the_display_levels_need_the_report():
    found = hyper_cli.unread_settings({**PLAIN, "by_condition": False, "no_report": True})
    assert found["wtc_cond_cone"] == "only --by-condition reads it"
    assert found["wtc_arrow_min"] == found["isc_threshold"] == (
        "it only draws the report, which --no-report skips")
    read = hyper_cli.unread_settings({**PLAIN, "wtc_cond_transform": True,
                                      "roi_mapping": "rois.json"})
    assert not {"wtc_cond_pad", "wtc_roi_min_channels", "wtc_cond_cone", "wtc_arrow_min",
                "isc_threshold"} & set(read)


def test_the_shared_band_edge_is_unread_only_when_both_metrics_set_their_own():
    assert "band_fmin" not in hyper_cli.unread_settings({**PLAIN, "wtc_band_fmin": 0.03})
    found = hyper_cli.unread_settings({**PLAIN, "wtc_band_fmin": 0.03, "isc_fmin": 0.02,
                                       "isc_fmax": 0.10})
    assert found["band_fmin"] == "--wtc-band-fmin and --isc-fmin override it"
    assert "band_fmax" not in found


# ---- through the command ----

def _warned(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "typed but not applied" in r.getMessage()]


def _run(tmp_path, *extra):
    deriv = tmp_path / "deriv"
    deriv.mkdir(exist_ok=True)
    pairs = tmp_path / "pairs.csv"
    # a table with no rows ends the run once the settings are checked, before any dyad loads
    pairs.write_text("group_id,subject_id,task\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        hyper_cli.main([str(deriv), str(tmp_path / "out"), "group", "--pairs-csv", str(pairs),
                        *extra])


def test_a_dyad_run_names_the_typed_flags_it_will_not_apply(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _run(tmp_path, "--wtc-mc-count", "50", "--seed", "3", "--wtc-roi-min-channels", "2",
             "--wtc-cond-cone")
    assert _warned(caplog) == [
        "--wtc-mc-count is typed but not applied: only --wtc-significance reads it",
        "--seed is typed but not applied: nothing is drawn at random",
        "--wtc-roi-min-channels is typed but not applied: no region is averaged without "
        "--roi-mapping"]


def test_a_dyad_run_that_applies_everything_typed_says_nothing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _run(tmp_path, "--wtc-significance", "--wtc-mc-count", "50", "--seed", "3",
             "--wtc-phase-null", "5", "--no-wtc-phase-null-cross", "--phase-null-shared")
    assert _warned(caplog) == []


def test_a_setting_a_config_file_carries_is_not_called_typed(tmp_path, caplog):
    config = tmp_path / "record.toml"
    config.write_text("[hyper]\nwtc_mc_count = 50\nseed = 3\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        _run(tmp_path, "--config", str(config))
    assert _warned(caplog) == []
