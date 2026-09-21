"""The drift basis empties a band, so the seam that checks bands has to see it."""

from __future__ import annotations

import mne
import numpy as np

from fnirs_pipe.io.snirf import _DRIFT_KEYS
from fnirs_pipe.pipeline.hyper.group_io import (
    _low_edge,
    unfiltered_stage_note,
    warn_outside_passband,
)
from fnirs_pipe.pipeline.post_pipeline import _ANALYSIS_KEYS
from fnirs_pipe.utils.lineage import stamp


def _raw_stamped(**params):
    info = mne.create_info(["S1_D1 hbo"], sfreq=2.0, ch_types="hbo")
    raw = mne.io.RawArray(np.zeros((1, 200)), info, verbose="ERROR")
    return stamp(raw, stage="errts", step="load", **params)


# ---- which setting owns the low edge ----

def test_the_bandpass_owns_it_when_nothing_else_is_recorded():
    assert _low_edge({"high_pass": 0.01})[0] == 0.01


def test_a_cosine_basis_owns_it_when_there_is_no_bandpass():
    edge, label = _low_edge({"drift_model": "cosine", "drift_high_pass": 0.01})
    assert edge == 0.01 and "drift" in label


def test_the_higher_of_the_two_binds():
    assert _low_edge({"high_pass": 0.01, "drift_model": "cosine",
                      "drift_high_pass": 0.02})[0] == 0.02
    assert _low_edge({"high_pass": 0.02, "drift_model": "cosine",
                      "drift_high_pass": 0.01})[0] == 0.02


def test_a_polynomial_basis_has_no_cutoff_to_offer():
    assert _low_edge({"drift_model": "polynomial", "drift_order": 3})[0] is None


def test_nothing_recorded_is_no_edge():
    assert _low_edge({})[0] is None


# ---- what the hyper seam does with it ----

def test_a_band_below_the_drift_cutoff_is_warned_about(caplog):
    raws = {"01": _raw_stamped(drift_model="cosine", drift_high_pass=0.02)}
    with caplog.at_level("WARNING"):
        warn_outside_passband(raws, 0.01, 0.2)
    assert "drift basis" in caplog.text


def test_a_band_inside_the_drift_cutoff_is_quiet(caplog):
    raws = {"01": _raw_stamped(drift_model="cosine", drift_high_pass=0.01)}
    with caplog.at_level("WARNING"):
        warn_outside_passband(raws, 0.01, 0.2)
    assert caplog.text == ""


def test_a_drift_basis_alone_is_not_called_unfiltered():
    raws = {"01": _raw_stamped(drift_model="cosine", drift_high_pass=0.01)}
    assert unfiltered_stage_note(raws) is None


def test_a_polynomial_basis_alone_still_is():
    raws = {"01": _raw_stamped(drift_model="polynomial", drift_order=3)}
    assert "no bandpass" in (unfiltered_stage_note(raws) or "")


# ---- the record survives to where those read it ----

def test_the_reader_restores_the_drift_basis():
    assert "drift_high_pass" in _DRIFT_KEYS and "drift_model" in _DRIFT_KEYS


def test_changing_the_cutoff_counts_as_a_different_analysis():
    assert "drift_high_pass" in _ANALYSIS_KEYS


# ---- and it survives an actual run and the SNIRF round trip ----

def test_a_denoise_run_records_its_drift_cutoff(tmp_path):
    import json

    from fnirs_pipe.io.snirf import read_snirf
    from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
    from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep
    from fnirs_pipe.utils.lineage import lineage_of

    from tests._synth import synth_raw

    bands = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)
    prepped = run_prep(synth_raw("01", "rest"),
                       PrepConfig(subject="01", dpf=[6.0, 6.0], sci_threshold=0.8,
                                  motion_correction="tddr", **bands),
                       output_dir=tmp_path / "prep", source_entities={"task": "rest"})
    out = tmp_path / "post"
    run_post(prepped.raw_haemo.copy(),
             PostConfig(subject="01", high_pass=0.01, low_pass=0.1,
                        drift_model="cosine", drift_high_pass=0.02, **bands),
             output_dir=out, mode="denoise", source_entities={"task": "rest"})

    errts = next(out.rglob("*_desc-errts_nirs.snirf"))
    params = json.loads(errts.with_suffix(".json").read_text())["parameters"]
    assert params["drift_model"] == "cosine" and params["drift_high_pass"] == 0.02
    # the bandpass is still its own number, not overwritten by the drift cutoff
    assert params["high_pass"] == 0.01

    restored = (lineage_of(read_snirf(errts)).params or {})
    assert _low_edge(restored) == (0.02, "0.02 Hz cosine drift basis")
