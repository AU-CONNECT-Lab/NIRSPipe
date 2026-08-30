"""Wiring: does a setting reach the computation, and does a file hold what its name says.

The rest of the suite checks that a setting is parsed correctly and that it is recorded
correctly. Both can pass while the value never reaches the function that uses it: the
sidecar would still read `dpf: 6.0` and every concentration in the run would be wrong,
silently, with no test red. These are the tests for that gap, so they all have the same
shape: run the pipeline twice with one setting changed, and require the output to move.

Where the physics fixes the direction, the assertion checks the direction rather than mere
inequality. Beer-Lambert concentration goes as 1/ppf, so doubling the DPF must halve the
output exactly; "differs" would also pass if the two runs were wired to different things
entirely.

The last section covers the two `except Exception` paths that degrade quietly. Nothing
downstream fails when they trip, the report just loses a panel, so only a test that
demands success will ever notice.
"""

import json

import numpy as np
import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.io.snirf import read_snirf
from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, compute_sci, run_prep
from fnirs_pipe.qc.sqm_record import build_sqm_records
from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.lineage import stage_of

from ._synth import SFREQ, synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)
_BASE = dict(dpf=[6.0, 6.0], sci_threshold=0.8, motion_correction="tddr", **_BANDS)


def _prep(tmp_dir, **overrides):
    """One prep run on synthetic data. Returns (PrepResult, the nirs output directory)."""
    config = PrepConfig(subject="01", **{**_BASE, **overrides})
    result = run_prep(synth_raw("01", "tapping"), config, output_dir=tmp_dir,
                      source_entities={"task": "tapping"})
    return result, tmp_dir / "sub-01" / "nirs"


def _stage(nirs_dir, desc):
    """One of the run's stage files. Prep stopped carrying these in memory on 2026-08-29."""
    return read_snirf(next(nirs_dir.glob(f"*_desc-{desc}_nirs.snirf")))


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return _prep(tmp_path_factory.mktemp("wiring_base"))


# ---- Does the setting reach the computation ----

def test_dpf_reaches_beer_lambert(tmp_path_factory):
    """Concentration goes as 1/ppf, so twice the DPF must give exactly half the output."""
    six, _ = _prep(tmp_path_factory.mktemp("dpf6"), dpf=[6.0, 6.0])
    twelve, _ = _prep(tmp_path_factory.mktemp("dpf12"), dpf=[12.0, 12.0])
    assert np.allclose(twelve.raw_haemo.get_data() * 2, six.raw_haemo.get_data(), rtol=1e-6)


def test_sci_threshold_reaches_the_comparison(tmp_path_factory):
    """Raising the threshold can only add channels, and on this data it must add some.

    0.9 rather than 0.99: the good pairs sit at about 0.976, and a threshold above that
    marks every channel, which Beer-Lambert cannot run on at all.
    """
    lenient, _ = _prep(tmp_path_factory.mktemp("sci_low"), sci_threshold=0.01)
    strict, _ = _prep(tmp_path_factory.mktemp("sci_high"), sci_threshold=0.9)
    assert set(lenient.bad_channels) <= set(strict.bad_channels)
    assert len(strict.bad_channels) > len(lenient.bad_channels)


def test_the_cardiac_band_reaches_the_sci_computation(baseline):
    """SCI measured off the cardiac peak must collapse; the good pairs only share cardiac."""
    raw_od = _stage(baseline[1], "sci")
    in_band = list(compute_sci(raw_od, 0.7, 1.5).values())
    off_band = list(compute_sci(raw_od, 2.0, 3.0).values())
    assert np.median(in_band) > np.median(off_band) + 0.3


def test_the_motion_method_reaches_the_signal(tmp_path_factory):
    _, none_dir = _prep(tmp_path_factory.mktemp("mot_none"), motion_correction="none")
    _, tddr_dir = _prep(tmp_path_factory.mktemp("mot_tddr"), motion_correction="tddr")
    before = _stage(none_dir, "sci").get_data()
    assert np.array_equal(_stage(none_dir, "motcorrected").get_data(), before)
    assert not np.allclose(_stage(tddr_dir, "motcorrected").get_data(), before)


def test_the_bandpass_cutoffs_reach_the_filter(baseline, tmp_path_factory):
    """A passband under the cardiac peak has to take the cardiac power out."""
    haemo = baseline[0].raw_haemo
    out = tmp_path_factory.mktemp("bandpass")
    wide = run_post(haemo.copy(), PostConfig(subject="01", high_pass=0.01, low_pass=2.0, **_BANDS),
                    output_dir=out / "wide", mode="denoise", source_entities={"task": "tapping"})[0]
    narrow = run_post(haemo.copy(), PostConfig(subject="01", high_pass=0.01, low_pass=0.1, **_BANDS),
                      output_dir=out / "narrow", mode="denoise", source_entities={"task": "tapping"})[0]
    assert narrow.get_data().var() < 0.5 * wide.get_data().var()


def test_the_resample_rate_reaches_the_data(baseline, tmp_path_factory):
    out = tmp_path_factory.mktemp("resample")
    result = run_post(baseline[0].raw_haemo.copy(),
                      PostConfig(subject="01", high_pass=0.01, low_pass=0.5,
                                 resample_sfreq=SFREQ / 2, **_BANDS),
                      output_dir=out, mode="denoise", source_entities={"task": "tapping"})[0]
    assert result.info["sfreq"] == pytest.approx(SFREQ / 2)


# ---- Does a file hold what its name says ----

_EXPECTED_ON_DISK = {
    "od": "od", "sci": "od", "motcorrected": "od", "preproc": "haemo",
}


@pytest.mark.parametrize("desc, kind", sorted(_EXPECTED_ON_DISK.items()))
def test_the_file_holds_the_stage_its_name_claims(baseline, desc, kind):
    """desc- is what read_snirf re-stamps from, so a mislabelled file poisons everything after."""
    path = next(baseline[1].glob(f"*desc-{desc}_nirs.snirf"))
    raw = read_snirf(path)
    assert stage_of(raw) == desc
    if kind == "od":
        assert is_optical_density(raw)
    else:
        assert set(raw.get_channel_types()) == {"hbo", "hbr"}


def test_the_bad_channels_survive_to_the_final_output(baseline):
    """Beer-Lambert renames the channels, so the marks have to survive the rename.

    SCI marks wavelength channels (S3_D3 760/850); the final output holds chromophore
    channels (S3_D3 hbo/hbr). Only the source-detector pair is comparable across the two.
    """
    result, _ = baseline
    assert result.bad_channels                                  # the synthetic bad pair
    marked = {ch.rsplit(" ", 1)[0] for ch in result.bad_channels}
    survived = {ch.rsplit(" ", 1)[0] for ch in result.raw_haemo.info["bads"]}
    assert survived >= marked


# ---- Failing where the cause is ----

def test_a_threshold_that_rejects_everything_says_so(tmp_path_factory):
    """Without this guard the run dies inside Beer-Lambert, which never names the threshold."""
    with pytest.raises(StageError, match="SCI threshold"):
        _prep(tmp_path_factory.mktemp("sci_all_bad"), sci_threshold=1.1)


def test_marking_every_channel_by_hand_says_so(tmp_path_factory):
    every_pair = sorted({ch.rsplit(" ", 1)[0] for ch in synth_raw("01", "tapping").ch_names})
    with pytest.raises(StageError, match="no usable channel"):
        _prep(tmp_path_factory.mktemp("manual_all_bad"), bad_channels=every_pair)


# ---- The paths that degrade quietly ----

def test_the_windowed_metrics_actually_ran(baseline):
    """sqm_record swallows any failure here, leaving the report's per-window panel empty.

    Prep stopped computing these on 2026-08-29; the record is the only place they exist, and
    the report reads them back from it.
    """
    _, nirs_dir = baseline
    written = build_sqm_records(nirs_dir)
    record = json.loads(written[0].read_text(encoding="utf-8"))
    windowed = record.get("windowed") or {}
    for name in ("sci_matrix", "sci_times", "psp_matrix", "psp_times", "qc_window_s",
                 "spike_spans_s", "motion_corrected_spans_s"):
        assert name in windowed, name
    n_ch = len(_stage(nirs_dir, "motcorrected").ch_names)
    assert len(windowed["sci_matrix"]) == n_ch
    assert len(windowed["sci_matrix"][0]) == len(windowed["sci_times"])


def test_the_regression_gcor_actually_ran(baseline, tmp_path_factory):
    """post_pipeline swallows any failure here and the SQM record loses four metrics."""
    config = PostConfig(subject="01", high_pass=0.01, low_pass=0.1, short_channel=True,
                        drift_model="cosine", drift_high_pass=0.01, drift_order=1, **_BANDS)
    gcor_reg = run_post(baseline[0].raw_haemo.copy(), config,
                        output_dir=tmp_path_factory.mktemp("gcor"), mode="rest",
                        source_entities={"task": "tapping"})[6]
    assert gcor_reg is not None
    assert set(gcor_reg) == {"gcor_hbo_prereg", "gcor_hbr_prereg",
                             "gcor_hbo_postreg", "gcor_hbr_postreg"}
    assert all(np.isfinite(v) for v in gcor_reg.values())
