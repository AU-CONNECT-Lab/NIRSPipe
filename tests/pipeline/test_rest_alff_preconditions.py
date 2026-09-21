"""ALFF's input is the only rest-mode path with no bandpass, so the drift model is its
only detrend.

FC reads the bandpassed residual, whose high-pass edge removes drift whatever the drift
model does. ALFF reads the broadband one, because fALFF's denominator has to span the full
spectrum. Nothing else on that path removes a linear trend, and a ramp's spectral leakage
lands inside the ALFF band, so a drift model that does not detrend produces ALFF values that
are wrong rather than noisy: measured against a linear ramp, by fifteen orders of magnitude.

`--drift-model none` stays legal, since it is a defensible choice for an FC-only run. What
these tests pin is that it costs the ALFF outputs instead of silently corrupting them.
"""

import pytest

from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep

from tests._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)

_ALFF_DF, _FC_DF = 3, 4  # (result, glm_est, dm, alff_df, fc_df, fc_hbr_df, fc_seed, fc_roi)


@pytest.fixture(scope="module")
def haemo(tmp_path_factory):
    config = PrepConfig(subject="01", dpf=[6.0, 6.0], sci_threshold=0.8,
                        motion_correction="tddr", **_BANDS)
    result = run_prep(synth_raw("01", "tapping"), config,
                      output_dir=tmp_path_factory.mktemp("rest_prec"),
                      source_entities={"task": "tapping"})
    return result.raw_haemo


def _rest(haemo, tmp_path, **drift):
    config = PostConfig(subject="01", high_pass=0.01, low_pass=0.1,
                        drift_high_pass=0.01, **drift, **_BANDS)
    out = run_post(haemo.copy(), config, output_dir=tmp_path, mode="rest",
                   source_entities={"task": "tapping"})
    return out[_ALFF_DF], out[_FC_DF], sorted(tmp_path.rglob("*_alff.tsv"))


# ---- drift models that do remove a linear trend ----

@pytest.mark.parametrize("drift", [
    dict(drift_model="cosine"),
    dict(drift_model="polynomial", drift_order=1),
    dict(drift_model="polynomial", drift_order=3),
])
def test_alff_is_produced_when_the_drift_model_detrends(haemo, tmp_path, drift):
    alff_df, _, tsvs = _rest(haemo, tmp_path, **drift)
    assert alff_df is not None
    assert len(tsvs) == 1


# ---- drift models that do not ----

@pytest.mark.parametrize("drift", [
    dict(drift_model="none"),
    dict(drift_model="polynomial", drift_order=0),   # intercept only, the linear term survives
])
def test_alff_is_skipped_when_the_drift_model_leaves_the_trend_in(haemo, tmp_path, drift):
    alff_df, _, tsvs = _rest(haemo, tmp_path, **drift)
    assert alff_df is None
    assert tsvs == [], "a wrong ALFF file on disk is worse than no file"


@pytest.mark.parametrize("drift", [
    dict(drift_model="none"),
    dict(drift_model="polynomial", drift_order=0),
])
def test_fc_still_runs_when_alff_is_skipped(haemo, tmp_path, drift):
    # FC reads the bandpassed residual, so it is unaffected: the guard must cost ALFF only
    _, fc_df, _ = _rest(haemo, tmp_path, **drift)
    assert fc_df is not None


def test_the_run_does_not_raise_on_a_drift_model_that_does_not_detrend(haemo, tmp_path):
    # rejecting the flag outright would break a legitimate FC-only configuration
    _rest(haemo, tmp_path, drift_model="none")
