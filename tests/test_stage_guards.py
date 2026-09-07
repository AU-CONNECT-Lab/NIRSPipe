"""Stage guards: the places that refuse to proceed on data from the wrong stage.

Three of them, all enforcing the same rule from different angles. Two check that
the desc- entity a caller asked for matches the stamp the object carries, which
matters because desc- is what read_snirf re-stamps from: a filename that disagrees
with its contents makes every downstream stamp wrong. The third refuses to compute
spectral metrics on filtered data, where they would describe the filter.

Each is a hard raise, so a test that only ran the happy path would not notice if
the guard were deleted.
"""

import mne
import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.post_pipeline import PostConfig, _write_step_snirf
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep
from fnirs_pipe.qc.metrics import compute_prep_haemo_sqm
from fnirs_pipe.utils.lineage import Recorder, stamp

from ._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)

_SPECTRAL_KEYS = {
    "cardiac_band_power_hbo", "cardiac_band_power_hbr",
    "cardiac_band_frac_hbo", "cardiac_band_frac_hbr",
    "resp_band_power_hbo", "resp_band_power_hbr",
    "resp_band_frac_hbo", "resp_band_frac_hbr",
}


@pytest.fixture(scope="module")
def preproc_haemo():
    """A Beer-Lambert output, stamped as the pipeline would stamp it."""
    raw = synth_raw("01", "tapping", duration=60.0)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    haemo = mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)
    return stamp(haemo, stage="preproc", step="beer_lambert")


# ---- post: _write_step_snirf(desc) must agree with the stamp ----

def test_post_write_rejects_a_desc_that_contradicts_the_stamp(preproc_haemo, tmp_path):
    filtered = stamp(preproc_haemo.copy(), stage="filtered", step="bandpass")
    config = PostConfig(subject="01", **_BANDS)

    with pytest.raises(StageError, match="resampled"):
        _write_step_snirf(filtered, config, tmp_path, desc="resampled", rec=Recorder())


def test_post_write_rejects_an_unstamped_object(preproc_haemo, tmp_path):
    bare = preproc_haemo.copy()
    bare.info["temp"] = None
    config = PostConfig(subject="01", **_BANDS)

    with pytest.raises(StageError):
        _write_step_snirf(bare, config, tmp_path, desc="filtered", rec=Recorder())


# ---- prep: _save(desc) must agree with the stamp ----

def test_prep_save_rejects_a_step_that_stamped_the_wrong_stage(tmp_path, monkeypatch):
    # The guard cannot be reached through normal input, since the pipeline stamps
    # correctly. Break one transformation instead: that is the mistake it exists for.
    monkeypatch.setattr(
        "fnirs_pipe.pipeline.prep_pipeline.intensity_to_od",
        lambda raw: stamp(raw.copy(), stage="bogus", step="od_conversion", source=raw),
    )
    config = PrepConfig(subject="01", dpf=[6.0, 6.0], sci_threshold=0.8,
                        motion_correction="tddr", **_BANDS)

    with pytest.raises(StageError, match="od"):
        run_prep(synth_raw("01", "tapping", duration=60.0), config, output_dir=tmp_path)


# ---- QC: spectral metrics are only defined on the Beer-Lambert output ----

def test_prep_haemo_sqm_rejects_filtered_input(preproc_haemo):
    # On band-passed data these metrics describe the filter, not the recording.
    filtered = stamp(preproc_haemo.copy(), stage="filtered", step="bandpass")

    with pytest.raises(StageError):
        compute_prep_haemo_sqm(filtered, **_BANDS)


def test_prep_haemo_sqm_rejects_unstamped_input(preproc_haemo):
    bare = preproc_haemo.copy()
    bare.info["temp"] = None

    with pytest.raises(StageError):
        compute_prep_haemo_sqm(bare, **_BANDS)


@pytest.mark.parametrize("bads", [
    pytest.param([], id="clean"),
    pytest.param(["S3_D3 hbo", "S3_D3 hbr"], id="with-bad-channels"),
])
def test_prep_haemo_sqm_accepts_the_preproc_stage(preproc_haemo, bads):
    # A guard that rejects everything would pass the rejection tests above, so this
    # one has to admit valid input. The spectral keys are what distinguish this from
    # the stage-insensitive compute_haemo_sqm.
    #
    # Values, not just keys: @_safe_metrics inserts every key it was given as None
    # when the function raises, so key presence alone is satisfied by total failure.
    # That is how eight of these went silently missing once, and only when bad
    # channels were present, hence the second case.
    haemo = preproc_haemo.copy()
    haemo.info["bads"] = bads

    record = compute_prep_haemo_sqm(haemo, **_BANDS)

    assert _SPECTRAL_KEYS <= set(record)
    assert not [key for key in _SPECTRAL_KEYS if record[key] is None]
