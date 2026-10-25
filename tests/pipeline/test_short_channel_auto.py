"""`--short-channel auto`, the default: pca on a montage with short channels, none on one without."""

import json
import tomllib

import mne
import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _build_post_config
from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.post_pipeline import PostConfig, run_post
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep
from nirspipe.qc.boilerplate.vocabulary import _regressor_phrase

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset
from tests._synth import SHORT_DISTANCE, synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)


def _haemo(root, short):
    config = PrepConfig(subject="01", dpf=[6.0, 6.0], sci_threshold=0.8,
                        motion_correction="tddr", **_BANDS)
    return run_prep(synth_raw("01", "rest", short_channels=short), config, output_dir=root,
                    source_entities={"task": "rest"}).raw_haemo


@pytest.fixture(scope="module")
def with_short(tmp_path_factory):
    return _haemo(tmp_path_factory.mktemp("auto_short"), True)


@pytest.fixture(scope="module")
def without_short(tmp_path_factory):
    return _haemo(tmp_path_factory.mktemp("auto_noshort"), False)


def _denoise(haemo, out_dir, **overrides):
    config = PostConfig(subject="01", high_pass=0.01, low_pass=0.1, noise_model="ols",
                        **overrides, **_BANDS)
    return run_post(haemo.copy(), config, output_dir=out_dir, mode="denoise",
                    source_entities={"task": "rest"})


def _errts(out_dir):
    return sorted(out_dir.rglob("*desc-errts_nirs.snirf"))


def _sidecar(out_dir):
    sidecar = _errts(out_dir)[0].with_suffix(".json")
    return json.loads(sidecar.read_text(encoding="utf-8"))["parameters"]


def _short_names(raw):
    picks = mne.pick_types(raw.info, fnirs=True, exclude=[])
    dists = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    return [raw.ch_names[i] for i, d in zip(picks, dists) if 0 < d <= SHORT_DISTANCE + 1e-6]


# ---- what auto chooses ----

def test_a_montage_with_short_channels_gets_every_good_one_regressed_out(with_short, tmp_path):
    """The residual is each channel's OLS residual on a constant and every good short channel."""
    _denoise(with_short, tmp_path)  # no short_channel: the library default is still none
    assert _errts(tmp_path) == []
    _denoise(with_short, tmp_path, short_channel="auto")

    filtered = read_snirf(next(tmp_path.rglob("*desc-filtered_nirs.snirf")))
    shorts = [n for n in _short_names(filtered) if n not in filtered.info["bads"]]
    assert shorts
    y = filtered.get_data()
    X = np.column_stack([np.ones(y.shape[1]), filtered.get_data(picks=shorts).T])
    beta, *_ = np.linalg.lstsq(X, y.T, rcond=None)
    expected = y - (X @ beta).T

    got = read_snirf(_errts(tmp_path)[0]).get_data()
    long_rows = [i for i, n in enumerate(filtered.ch_names) if n not in shorts]
    scale = np.abs(y[long_rows]).max()
    assert np.abs(got[long_rows] - expected[long_rows]).max() < 1e-6 * scale


def test_its_sidecar_names_pca_and_that_it_was_chosen(with_short, tmp_path):
    _denoise(with_short, tmp_path / "auto", short_channel="auto")
    _denoise(with_short, tmp_path / "typed", short_channel="pca")
    auto, typed = _sidecar(tmp_path / "auto"), _sidecar(tmp_path / "typed")
    assert (auto["short_channel"], auto["short_channel_auto"]) == ("pca", True)
    assert typed["short_channel"] == "pca" and "short_channel_auto" not in typed


def test_a_montage_without_short_channels_is_not_regressed(without_short, tmp_path):
    _denoise(without_short, tmp_path, short_channel="auto")
    assert _errts(tmp_path) == []


def test_with_a_drift_model_it_regresses_the_drift_alone(without_short, tmp_path):
    _, _, dm, *_ = _denoise(without_short, tmp_path, short_channel="auto",
                            drift_model="polynomial", drift_order=1)
    assert not [c for c in dm.columns if c.startswith("short_ch_")]
    sidecar = _sidecar(tmp_path)
    assert sidecar.get("short_channel") is None and "short_channel_auto" not in sidecar


def test_every_short_channel_rejected_keeps_the_rule_of_no_regressor(with_short, tmp_path):
    haemo = with_short.copy()
    haemo.info["bads"] = sorted(set(haemo.info["bads"]) | set(_short_names(haemo)))
    _, _, dm, *_ = _denoise(haemo, tmp_path, short_channel="auto")
    assert not [c for c in dm.columns if c.startswith("short_ch_")]
    sidecar = _sidecar(tmp_path)
    assert sidecar.get("short_channel") is None and "short_channel_auto" not in sidecar


# ---- the command line and the record ----

def test_the_command_line_default_is_auto():
    assert _build_post_config("01", None, {}, {}).short_channel == "auto"
    assert _build_post_config("01", None, {"short_channel": "none"}, {}).short_channel is None


@pytest.fixture(scope="module")
def bids(tmp_path_factory):
    return make_fingerprint_dataset(tmp_path_factory.mktemp("auto_record"))[0]


def _record(bids, out, *flags):
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--mode", "denoise", *flags,
                  "--participant-label", "01", "--skip-bids-validation", "--dry-run"])
    return tomllib.loads((out / "sub-01" / "logs" / "sub-01.toml").read_text(encoding="utf-8"))


def test_the_record_writes_auto_where_nothing_was_typed(bids, tmp_path):
    record = _record(bids, tmp_path)
    assert record["post"]["short_channel"] == "auto"
    assert record["post_sources"]["short_channel"] == "default"


def test_a_run_told_none_reruns_as_none(bids, tmp_path):
    first = _record(bids, tmp_path / "first", "--short-channel", "none")
    assert first["post"]["short_channel"] == "none"
    rerun = _record(bids, tmp_path / "rerun",
                    "--config", str(tmp_path / "first" / "sub-01" / "logs" / "sub-01.toml"))
    assert rerun["post"]["short_channel"] == "none"


# ---- the Methods ----

def test_the_methods_say_the_regressor_was_the_default():
    auto = _regressor_phrase({"short_channel": "pca", "short_channel_auto": True})
    typed = _regressor_phrase({"short_channel": "pca"})
    assert auto == ("an orthogonal basis spanning the retained short channels (HbO and HbR "
                    "decomposed together; used by default, since the montage has short channels)")
    assert "default" not in typed
