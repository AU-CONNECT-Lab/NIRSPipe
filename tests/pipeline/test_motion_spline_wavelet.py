"""Spline + wavelet, and the two settings some methods read.

spline_wavelet is spline at its own detection thresholds followed by the wavelet at its own
fence, both steps otherwise unchanged, so the chain is held equal to the two functions called
in turn. --motion-std-thresh and --wavelet-iqr reach only the methods that read them; left
unset, every method runs exactly as before.
"""

import json

import mne
import numpy as np
import pytest

from nirspipe.cli import _app_options
from nirspipe.pipeline import motion
from nirspipe.pipeline.motion import (
    _spline_motion_correct,
    _spline_motion_mask,
    _wavelet_motion_correct,
    correct_motion,
    motion_settings,
    overmarked,
)
from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep
from nirspipe.utils.lineage import lineage_of, stage_of

from tests._synth import synth_raw

pywt = pytest.importorskip("pywt")

SFREQ = 10.0
N_T = 6000
N_CH = 4
NAMES = [f"S{i // 2 + 1}_D1 {760 if i % 2 == 0 else 850}" for i in range(N_CH)]
STEP_AT = 3000


def _stepped(amp: float, seed: int = 0) -> np.ndarray:
    """Time x channels white noise, a step on the first pair."""
    d = 0.01 * np.random.default_rng(seed).normal(size=(N_T, N_CH))
    d[STEP_AT:, :2] += amp
    return d


def _raw(d: np.ndarray, sfreq: float = SFREQ) -> mne.io.Raw:
    return mne.io.RawArray(d.T.copy(), mne.create_info(NAMES, sfreq, "fnirs_od"), verbose="error")


# ---- The method ----

def test_it_takes_the_papers_detection_and_fence():
    assert (motion.SPLINE_WAVELET_T_MOTION, motion.SPLINE_WAVELET_AMP_THRESH) == (1.0, 0.4)
    assert (motion.SPLINE_WAVELET_STD_THRESH, motion.SPLINE_WAVELET_IQR_FACTOR) == (13.0, 0.8)
    # the mask margin and the smoothing are spline's own
    assert (motion.SPLINE_T_MASK, motion.SPLINE_P) == (1.0, 0.99)


def test_it_is_spline_at_its_thresholds_then_the_wavelet():
    raw = _raw(_stepped(2.0))
    splined, _ = _spline_motion_correct(raw, t_motion=1.0, std_thresh=13.0, amp_thresh=0.4)
    expected = _wavelet_motion_correct(splined, iqr_factor=0.8).get_data()
    out = correct_motion(raw, method="spline_wavelet")
    assert stage_of(out) == "motcorrected"
    assert (out.get_data() == expected).all()
    # the order matters: the wavelet first gives a different signal
    reverse, _ = _spline_motion_correct(_wavelet_motion_correct(raw, iqr_factor=0.8),
                                        t_motion=1.0, std_thresh=13.0, amp_thresh=0.4)
    assert not np.allclose(out.get_data(), reverse.get_data())


def test_its_detection_marks_what_splines_own_misses():
    good = np.ones(N_CH, bool)
    # a 0.3 step is about 18 SDs of the first difference, under 0.4 in OD
    d = _stepped(0.3)
    assert not _spline_motion_mask(d, SFREQ, NAMES, good).any()
    assert _spline_motion_mask(d, SFREQ, NAMES, good, t_motion=1.0, std_thresh=13.0,
                               amp_thresh=0.4)[:, :2].any()
    # and 0.4 in OD marks on its own, whatever the SD threshold
    assert _spline_motion_mask(_stepped(0.5), SFREQ, NAMES, good, t_motion=1.0,
                               std_thresh=1e9, amp_thresh=0.4)[:, :2].any()


def test_the_window_is_one_second_either_way():
    # 10 lags at 10 Hz see the step from 10 rows before it; 10 rows either side, then +1
    mask = _spline_motion_mask(_stepped(2.0), SFREQ, NAMES, np.ones(N_CH, bool), t_motion=1.0,
                               std_thresh=13.0, amp_thresh=0.4)
    assert (np.flatnonzero(mask[:, 0]) == np.arange(2981, 3011)).all()


def test_it_records_its_spline_step():
    out = correct_motion(_raw(_stepped(2.0)), method="spline_wavelet")
    params = lineage_of(out).params
    assert params["method"] == "spline_wavelet"
    assert params["spline_wavelet"]["segments"] == 2 and params["spline_wavelet"]["motion_share"] > 0


def test_it_refuses_below_the_splines_rate():
    with pytest.raises(ValueError, match="at least 3.333 Hz"):
        correct_motion(_raw(_stepped(2.0)[:600], sfreq=3.3), method="spline_wavelet")


# ---- The two settings ----

@pytest.mark.parametrize("method, expected", [
    ("spline_wavelet", {"motion_std_thresh": 13.0, "wavelet_iqr": 0.8}),
    ("spline", {"motion_std_thresh": 50.0}),
    ("wavelet", {"wavelet_iqr": 1.5}),
    ("wavelet_tddr", {"wavelet_iqr": 1.5}),
    ("tddr", {}), ("tpca", {}), ("spline_sg", {}), ("none", {}),
])
def test_unset_settings_take_each_methods_default(method, expected):
    assert motion_settings(method) == expected


def test_a_set_value_replaces_the_default():
    assert motion_settings("spline_wavelet", 20, 0.5) == {"motion_std_thresh": 20.0,
                                                          "wavelet_iqr": 0.5}


@pytest.mark.parametrize("method, kwargs, match", [
    ("tddr", {"iqr_factor": 0.8}, "--wavelet-iqr applies to wavelet, wavelet_tddr, spline_wavelet"),
    ("tpca", {"std_thresh": 20.0}, "--motion-std-thresh applies to spline, spline_wavelet"),
    ("spline", {"iqr_factor": 0.8}, "--motion-correction spline does not read it"),
    ("spline_wavelet", {"std_thresh": 0.0}, "must be positive"),
    ("wavelet", {"iqr_factor": -1.0}, "must be positive"),
])
def test_a_setting_the_method_cannot_use_is_refused(method, kwargs, match):
    with pytest.raises(ValueError, match=match):
        motion_settings(method, **kwargs)
    with pytest.raises(ValueError, match=match):
        correct_motion(_raw(_stepped(2.0)), method=method, **kwargs)


def test_the_settings_reach_the_steps_that_read_them():
    raw = _raw(_stepped(0.3))
    assert (correct_motion(raw, method="wavelet", iqr_factor=0.8).get_data()
            == _wavelet_motion_correct(raw, iqr_factor=0.8).get_data()).all()
    assert (correct_motion(raw, method="spline", std_thresh=13.0).get_data()
            == _spline_motion_correct(raw, std_thresh=13.0)[0].get_data()).all()
    assert not np.allclose(correct_motion(raw, method="spline_wavelet", std_thresh=60.0).get_data(),
                           correct_motion(raw, method="spline_wavelet").get_data())


def test_left_unset_the_existing_methods_are_unchanged():
    raw = _raw(_stepped(2.0))
    assert (correct_motion(raw, method="wavelet").get_data()
            == _wavelet_motion_correct(raw).get_data()).all()
    assert (correct_motion(raw, method="spline").get_data()
            == _spline_motion_correct(raw)[0].get_data()).all()


def test_the_cli_check_exits_naming_the_flag(capsys):
    with pytest.raises(SystemExit) as stop:
        _app_options.check_motion_settings("tddr", None, 0.8)
    assert stop.value.code == 1
    assert "--wavelet-iqr" in capsys.readouterr().err
    _app_options.check_motion_settings("spline_wavelet", 15.0, 0.5)


# ---- The sidecar ----

def test_prep_records_the_values_the_method_ran_with(tmp_path):
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.8,
                        motion_correction="spline_wavelet", wavelet_iqr=0.5,
                        cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)
    run_prep(synth_raw("01", "rest"), config, tmp_path, source_entities={"task": "rest"})
    sidecar = json.loads(next(tmp_path.rglob("*desc-motcorrected_nirs.json")).read_text())
    assert sidecar["parameters"]["motion_correction"] == "spline_wavelet"
    assert sidecar["parameters"]["motion_std_thresh"] == 13.0
    assert sidecar["parameters"]["wavelet_iqr"] == 0.5
    assert set(sidecar["spline_wavelet"]) == {"motion_share", "segments"}


# ---- The warning when the detector marks most of a run ----

def test_the_sd_rule_tightens_as_the_sampling_rate_rises():
    # a 1 Hz pulse, 0.01 in OD: its 1 s change is about 0.45 * rate first-difference SDs,
    # so 13 SDs is not reached at 10 Hz and is passed at 50 Hz
    def mask(sfreq):
        t = np.arange(int(600 * sfreq)) / sfreq
        d = np.tile(0.01 * np.sin(2 * np.pi * 1.0 * t)[:, None], (1, N_CH))
        return _spline_motion_mask(d, sfreq, NAMES, np.ones(N_CH, bool), t_motion=1.0,
                                   std_thresh=13.0, amp_thresh=0.4)
    assert not mask(10.0).any()
    assert mask(50.0).mean() > 0.9


@pytest.mark.parametrize("method, summary, expected", [
    ("spline_wavelet", {"motion_share": 0.11}, True),
    ("spline", {"motion_share": 0.5}, True),
    ("spline_wavelet", {"motion_share": motion.MOTION_SHARE_WARN}, False),
    ("spline_sg", {"motion_share": 0.5}, False),        # its detector reads no SD threshold
    ("spline", None, False),
    ("tddr", None, False),
])
def test_overmarked_reads_the_spline_detectors_share(method, summary, expected):
    assert overmarked(method, summary) is expected


def test_the_note_names_the_share_the_rate_and_the_setting():
    line = section_note("caveat.motion_overmarked", method="spline_wavelet", share=0.67,
                        thresh=13.0, sfreq=50.0)
    for part in ("spline_wavelet", "67%", "13 standard deviations", "50 Hz", "--motion-std-thresh"):
        assert part in line, part


def test_prep_warns_and_still_writes_the_correction(tmp_path, caplog):
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.8,
                        motion_correction="spline_wavelet", motion_std_thresh=0.5,
                        cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)
    with caplog.at_level("WARNING", logger="nirspipe"):
        run_prep(synth_raw("01", "rest"), config, tmp_path, source_entities={"task": "rest"})
    warned = [r.getMessage() for r in caplog.records if "--motion-std-thresh" in r.getMessage()]
    assert len(warned) == 1 and "sub-01" in warned[0]
    assert next(tmp_path.rglob("*desc-motcorrected_nirs.snirf")).exists()


def test_prep_says_nothing_when_the_share_is_low(tmp_path, caplog):
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.8,
                        motion_correction="spline_wavelet",
                        cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)
    with caplog.at_level("WARNING", logger="nirspipe"):
        run_prep(synth_raw("01", "rest"), config, tmp_path, source_entities={"task": "rest"})
    assert not [r for r in caplog.records if "--motion-std-thresh" in r.getMessage()]
