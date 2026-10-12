"""A run record loaded into the Analysis page regenerates the run it records.

The page reads the record with the command line's own ``--config`` reader, so these tests
only have to show that what the reader returns lands in the controls and comes back out of
the command builder unchanged.
"""

from __future__ import annotations

import inspect

import pytest

from nirspipe import __version__
from nirspipe.cli.run import _build_parser, config_keys, read_config
from nirspipe.interface.callbacks.analysis_callbacks import (
    _RECORD_BY_HAND,
    _RECORD_CONTROLS,
    generate_command,
    record_to_form,
)

# record settings no control on the page holds; record_to_form names them when they are set
_NOT_ON_PAGE = {"epoch_single_trial", "gvtd_min_epoch_s", "fir_delays", "aux_regressors",
                "aux_channels", "bids_filter_file", "ignore", "allow_cropped_input"}

_RECORD = f"""
[environment]
nirspipe_version = "{__version__}"

[prep]
dpf = [6.0]
sci_threshold = 0.8
motion_correction = "spline"
motion_std_thresh = 12.0
cardiac_l_freq = 0.7
cardiac_h_freq = 1.5
resp_l_freq = 0.1
resp_h_freq = 0.5
psp_threshold = 0.2
min_good_frac = 0.6
screen_scope = "task"
short_max_dist = 12.0
window_length = 8.0
epoch_tmin = -2.0
epoch_tmax = 20.0
epoch_chunk_duration = 20.0
by_condition = true
gvtd_censor = "short"
gvtd_censor_n_std = 8.0
keep_spans = "/tables/keep.tsv"
keep_spans_from = "start"
bad_spans = "/tables/bad.tsv"
bad_channels = "/tables/rated.tsv"

[post]
mode = "glm"
high_pass = "none"
low_pass = 0.4
filter_method = "fir"
filter_order = 4
censor_fill = "spline"
resample_sfreq = 4.0
stim_dur = 10.0
hrf_model = "glover"
noise_model = "ar2"
drift_model = "cosine"
drift_high_pass = 0.005
short_channel = "mean"
fc = true
combine_runs = true
roi_mapping = "/tables/roi.json"

[execution]
participant_label = ["01"]
task_label = ["tapping", "rest"]
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "sub-01.toml"
    path.write_text(_RECORD, encoding="utf-8")
    return read_config(path)


# the callback's parameters are its controls' ids without the prefix, but for these two
_PARAM_OF = {"an-subjects-store": "subjects", "an-shell-select": "shell"}


def _generated_argv(values: dict, flags: list[str]) -> list[str]:
    """The page's own command builder, fed the controls through its callback's parameters."""
    form = {**values, "an-bids-dir": "/bids", "an-output-dir": "/out",
            "an-subjects-store": ["01"], "an-n-jobs": 1, "an-flags": flags,
            "an-shell-select": "bash"}
    given = {_PARAM_OF.get(cid, cid[3:].replace("-", "_")): v for cid, v in form.items()}
    params = [p for p in inspect.signature(generate_command).parameters if p != "n_clicks"]
    assert not set(given) - set(params), "a control the record fills reaches no parameter"
    _, stored = generate_command(1, *[given.get(p) for p in params])
    return stored["argv"]


def test_a_loaded_record_regenerates_every_setting_it_holds(config):
    values, not_held = record_to_form(config)
    assert not_held == []
    flags = ["combine_runs"] if config["post"].get("combine_runs") else []
    parsed = vars(_build_parser().parse_args(_generated_argv(values, flags)[1:]))
    for section in ("prep", "post", "execution"):
        for key, recorded in config[section].items():
            if key == "participant_label":
                continue
            assert parsed[key] == recorded, f"[{section}] {key}"


def test_a_setting_the_record_leaves_out_empties_its_control(config):
    config["post"].pop("stim_dur")
    config["prep"].pop("gvtd_censor")
    values, _ = record_to_form(config)
    assert values["an-stim-dur"] is None
    assert values["an-gvtd-censor"] == "off"


def test_what_the_page_cannot_hold_is_named_as_flags(config):
    config["prep"].update(dpf=[6.0, 5.5], epoch_single_trial=True, gvtd_min_epoch_s=15.0)
    config["execution"]["ignore"] = ["channel-status"]
    values, not_held = record_to_form(config)
    assert values["an-dpf"] is None
    assert set(not_held) == {"--dpf 6.0 5.5", "--epoch-single-trial",
                             "--gvtd-min-epoch-s 15.0", "--ignore channel-status"}


def test_a_default_the_record_writes_out_is_not_reported(config):
    default = next(a.default for a in _build_parser()._actions if a.dest == "gvtd_min_epoch_s")
    config["prep"]["gvtd_min_epoch_s"] = default
    assert record_to_form(config)[1] == []


def test_every_setting_a_record_can_carry_has_a_home():
    offered = {key for keys in config_keys().values() for key in keys}
    homeless = offered - set(_RECORD_CONTROLS) - _RECORD_BY_HAND - _NOT_ON_PAGE
    assert not homeless, (
        f"a run record can carry {sorted(homeless)} and the page neither fills a control from "
        f"it nor lists it in _NOT_ON_PAGE"
    )
