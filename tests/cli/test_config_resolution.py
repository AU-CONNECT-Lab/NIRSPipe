"""Resolving CLI flags and a --config TOML into PrepConfig / PostConfig.

This is the layer between argparse and the pipeline: test_cli.py stops at
parse_args, and everything above these builders needs a BIDS dataset. The two
builders are private, but they are the only seam here that does not require
running a subject, so they are addressed directly.

The precedence rules matter because values supplied through TOML never appear in
argv.
"""

from enum import Enum

import pytest

from fnirs_pipe.cli.run import _MODE_CHOICES, mode_defaults
from fnirs_pipe.cli.workflows import (
    _build_post_config,
    _make_prep_config,
    _refuse_cosine_without_cutoff,
    _resolve_post_settings,
)


class _Model(Enum):
    """Stand-in for the enum values the CLI layer may still hand over."""
    spm = "spm"


def _post(args=None, toml=None):
    return _build_post_config("01", None, args or {}, toml or {})


# ---- CLI vs TOML precedence ----

def test_cli_value_wins_over_toml():
    assert _post({"hrf_model": "glover"}, {"hrf_model": "spm"}).hrf_model == "glover"


def test_toml_supplies_what_the_cli_omitted():
    assert _post({}, {"hrf_model": "spm"}).hrf_model == "spm"


def test_absent_in_both_is_none():
    assert _post().hrf_model is None


def test_cli_override_does_not_discard_the_rest_of_the_toml():
    # The shape of a real invocation: a config file plus one flag overriding it.
    cfg = _post(
        {"noise_model": "ar2"},
        {"hrf_model": "spm", "drift_model": "cosine", "drift_high_pass": 0.01,
         "drift_order": 2},
    )
    assert cfg.noise_model == "ar2"
    assert cfg.hrf_model == "spm"
    assert cfg.drift_model == "cosine"
    assert cfg.drift_order == 2


def test_a_false_cli_value_still_beats_the_toml():
    # Precedence is decided on "is not None", not on truthiness: --no-combine-runs has
    # to win over combine_runs = true in the file.
    assert _post({"combine_runs": False}, {"combine_runs": True}).combine_runs is False


def test_a_zero_cli_value_still_beats_the_toml():
    assert _post({"high_pass": 0.0}, {"high_pass": 0.01}).high_pass == 0.0


def test_enum_values_are_unwrapped_to_plain_strings():
    assert _post({"hrf_model": _Model.spm}).hrf_model == "spm"


# ---- the small parsers inside the builders ----

@pytest.mark.parametrize("raw, expected", [
    ("0,1,2", (0, 1, 2)),
    ("3", (3,)),
    (None, None),
])
def test_fir_delays_string_becomes_a_tuple_of_ints(raw, expected):
    assert _post({"fir_delays": raw}).fir_delays == expected


def test_fir_delays_can_come_from_the_toml():
    assert _post({}, {"fir_delays": "0,5"}).fir_delays == (0, 5)


@pytest.mark.parametrize("raw, expected", [
    ("all", "all"),
    ("none", None),   # the literal string a user types, not Python's None
    ("", None),
    (None, None),
])
def test_short_channel_none_is_normalized_away(raw, expected):
    assert _post({"short_channel": raw}).short_channel == expected


def test_events_path_is_stored_as_a_string():
    assert _post({"events_path": "/data/events.tsv"}).events_path == "/data/events.tsv"


def test_events_path_absent_is_none():
    assert _post().events_path is None


# ---- PrepConfig ----

_PREP_ARGS = {
    "dpf": [6.0, 6.0],
    "sci_threshold": 0.8,
    "motion_correction": "tddr",
    "cardiac_l_freq": 0.7,
    "cardiac_h_freq": 1.5,
    "resp_l_freq": 0.2,
    "resp_h_freq": 0.5,
}


def _prep(**overrides):
    return _make_prep_config("01", None, {**_PREP_ARGS, **overrides})


@pytest.mark.parametrize("raw, expected", [
    ("S1_D1", ["S1_D1"]),
    ("S1_D1,S2_D2", ["S1_D1", "S2_D2"]),
    ("S1_D1, S2_D2", ["S1_D1", "S2_D2"]),   # whitespace after the comma is stripped
    ("", []),
    (None, []),
])
def test_bad_channels_string_is_split_and_stripped(raw, expected):
    assert _prep(bad_channels=raw).bad_channels == expected


def test_qc_window_defaults_to_ten_seconds():
    assert _prep().qc_window_s == 10.0


def test_qc_window_is_taken_from_window_length():
    assert _prep(window_length=5.0).qc_window_s == 5.0


def test_ignore_values_are_unwrapped():
    assert _prep(ignore=[_Model.spm, "bids-validation"]).ignore == ["spm", "bids-validation"]


def test_ignore_absent_is_empty_list():
    assert _prep().ignore == []


def test_motion_correction_enum_is_unwrapped():
    assert _prep(motion_correction=_Model.spm).motion_correction == "spm"


# ---- drift model needs its cutoff ----

def test_cosine_drift_without_a_cutoff_is_rejected():
    # nilearn multiplies the cutoff by the frame times, so without this the run dies
    # several minutes in with "unsupported operand type(s) for *: 'NoneType' and 'float'"
    from fnirs_pipe.pipeline.post_pipeline import PostConfig

    with pytest.raises(ValueError, match="drift-high-pass"):
        PostConfig(subject="01", cardiac_l_freq=0.7, cardiac_h_freq=1.5,
                   resp_l_freq=0.1, resp_h_freq=0.5, drift_model="cosine")


def test_the_other_drift_models_need_no_cutoff():
    from fnirs_pipe.pipeline.post_pipeline import PostConfig

    bands = dict(subject="01", cardiac_l_freq=0.7, cardiac_h_freq=1.5,
                 resp_l_freq=0.1, resp_h_freq=0.5)
    assert PostConfig(**bands, drift_model="polynomial", drift_order=3).drift_order == 3
    assert PostConfig(**bands).drift_model is None


# ---- the mode's defaults, under --config and the command line ----

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.5)


def _resolved(args, config_toml=None):
    args = dict(args)
    toml, sources = _resolve_post_settings(args, config_toml or {})
    return args, toml, sources


@pytest.mark.parametrize("mode", _MODE_CHOICES)
def test_every_mode_ships_defaults_that_build_a_config(mode):
    # glm leaves the cosine cutoff to the design, so the test supplies one
    extra = {"drift_high_pass": 0.008} if mode == "glm" else {}
    args, toml, _ = _resolved({"mode": mode, **_BANDS, **extra})
    config = _build_post_config("01", None, args, toml)
    for key, value in mode_defaults(mode).items():
        assert getattr(config, key) == value


def test_the_mode_fills_what_nothing_else_set():
    args, _, sources = _resolved({"mode": "rest", **_BANDS})
    assert args["low_pass"] == mode_defaults("rest")["low_pass"]
    assert sources["low_pass"] == "mode"
    assert sources["resample_sfreq"] == "default"


def test_config_beats_the_mode_and_the_cli_beats_both():
    args, _, sources = _resolved({"mode": "rest", "high_pass": 0.02, **_BANDS},
                                 {"high_pass": 0.05, "low_pass": 0.08})
    assert (args["high_pass"], sources["high_pass"]) == (0.02, "cli")
    assert (args["low_pass"], sources["low_pass"]) == (0.08, "config")
    assert sources["drift_model"] == "mode"


def test_a_mode_in_the_config_file_is_not_read():
    args, _, _ = _resolved({"mode": None, **_BANDS}, {"mode": "glm"})
    assert args["mode"] is None


def test_no_mode_means_no_defaults():
    args, _, sources = _resolved({"mode": None, **_BANDS})
    assert args.get("drift_model") is None
    assert "mode" not in sources.values()


def test_a_config_drift_order_is_no_longer_shadowed():
    # --drift-order used to default to 1 in argparse, which beat any TOML value
    args, toml, _ = _resolved({"mode": "denoise", "drift_order": None, **_BANDS},
                              {"drift_model": "polynomial", "drift_order": 3})
    assert _build_post_config("01", None, args, toml).drift_order == 3


def test_glm_without_a_cutoff_stops_and_names_the_mode():
    args, _, sources = _resolved({"mode": "glm", **_BANDS})
    with pytest.raises(SystemExit, match="--mode glm uses a cosine drift model"):
        _refuse_cosine_without_cutoff(args, sources)


def test_rest_needs_no_cutoff_from_the_user():
    args, _, sources = _resolved({"mode": "rest", **_BANDS})
    _refuse_cosine_without_cutoff(args, sources)
