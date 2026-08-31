"""The denoise mode's two independent halves: the confound regression, and `--fc`.

Either can run without the other, which is what separates this mode from rest. rest forces
a drift model and always writes ALFF, so "bandpass, then correlate" -- the most common FC
recipe there is -- had no route through the package that did not also run a GLM nobody asked
for. These tests pin that route, and they pin which signal the correlation was taken on,
because that is the only thing distinguishing the two ways of reaching it.

The regression half had no coverage at all before this file; it arrived in 0.22.0 as
`_has_confounds` and nothing reached it.
"""

import json

import pytest

from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep

from ._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)
# run_post: (result, glm_est, dm, alff_df, fc_df, fc_hbr_df, gcor_reg, fc_seed, fc_roi)
_ALFF_DF, _FC_DF, _GCOR_REG, _FC_SEED, _FC_ROI = 3, 4, 6, 7, 8


@pytest.fixture(scope="module")
def haemo(tmp_path_factory):
    config = PrepConfig(subject="01", dpf=[6.0, 6.0], sci_threshold=0.8,
                        motion_correction="tddr", **_BANDS)
    result = run_prep(synth_raw("01", "rest"), config,
                      output_dir=tmp_path_factory.mktemp("denoisefc_prep"),
                      source_entities={"task": "rest"})
    return result.raw_haemo


@pytest.fixture(scope="module")
def roi_map(haemo):
    pairs = sorted({c.rsplit(" ", 1)[0] for c in haemo.ch_names})
    half = len(pairs) // 2
    return {"ROI_A": [f"{p} hbo" for p in pairs[:half]],
            "ROI_B": [f"{p} hbo" for p in pairs[half:]]}


def _denoise(haemo, out_dir, **overrides):
    config = PostConfig(subject="01", high_pass=0.01, low_pass=0.1,
                        **overrides, **_BANDS)
    return run_post(haemo.copy(), config, output_dir=out_dir, mode="denoise",
                    source_entities={"task": "rest"})


def _names(out_dir, pattern):
    return sorted(p.name for p in out_dir.rglob(pattern))


def _fc_source(out_dir):
    """The file the FC sidecar credits the correlation to."""
    sidecar = next(out_dir.rglob("*_desc-hbo_fc.json"))
    return json.loads(sidecar.read_text())["Sources"][0]


# ---- the regression half: which flags reach it ----

def test_neither_flag_runs_no_regression(haemo, tmp_path):
    _denoise(haemo, tmp_path)
    assert _names(tmp_path, "*desc-errts*.snirf") == []


def test_a_short_channel_strategy_is_enough_on_its_own(haemo, tmp_path):
    _denoise(haemo, tmp_path, short_channel="mean")
    assert _names(tmp_path, "*desc-errts*.snirf") == ["sub-01_task-rest_desc-errts_nirs.snirf"]


def test_a_drift_model_is_enough_on_its_own(haemo, tmp_path):
    """No short channels involved, so this is the flag that has to carry it alone."""
    _denoise(haemo, tmp_path, drift_model="polynomial", drift_order=1)
    assert _names(tmp_path, "*desc-errts*.snirf") == ["sub-01_task-rest_desc-errts_nirs.snirf"]


def test_an_explicit_none_drift_model_is_not_a_reason_to_regress(haemo, tmp_path):
    _denoise(haemo, tmp_path, drift_model="none")
    assert _names(tmp_path, "*desc-errts*.snirf") == []


def test_regression_gcor_arrives_only_with_a_short_channel_strategy(haemo, tmp_path):
    """It measures what the short-channel regression removed, so a drift-only run has none."""
    with_sc = _denoise(haemo, tmp_path / "sc", short_channel="mean")
    drift   = _denoise(haemo, tmp_path / "drift", drift_model="polynomial", drift_order=1)
    assert sorted(with_sc[_GCOR_REG]) == [
        "gcor_hbo_postreg", "gcor_hbo_prereg", "gcor_hbr_postreg", "gcor_hbr_prereg"]
    assert drift[_GCOR_REG] is None


# ---- the --fc half ----

def test_without_the_flag_nothing_connectivity_is_written(haemo, tmp_path):
    out = _denoise(haemo, tmp_path, short_channel="mean")
    assert _names(tmp_path, "*fc*.tsv") == []
    assert out[_FC_DF] is None and out[_FC_ROI] == {} and out[_FC_SEED] == {}


def test_the_flag_writes_the_same_family_rest_writes(haemo, tmp_path, roi_map):
    out = _denoise(haemo, tmp_path, short_channel="mean", roi_map=roi_map, fc=True)
    written = _names(tmp_path, "*fc*.tsv")
    for chromo in ("hbo", "hbr"):
        for suffix in ("fc", "fcz", "fcroi", "fcroiz", "fcseed", "fcseedz"):
            assert f"sub-01_task-rest_desc-{chromo}_{suffix}.tsv" in written
    assert out[_FC_DF] is not None
    assert sorted(out[_FC_ROI]) == ["hbo", "hbr"]
    assert sorted(out[_FC_SEED]) == ["hbo", "hbr"]


def test_the_roi_products_need_a_roi_map(haemo, tmp_path):
    out = _denoise(haemo, tmp_path, short_channel="mean", fc=True)
    assert _names(tmp_path, "*fcroi*.tsv") == []
    assert _names(tmp_path, "*fcseed*.tsv") == []
    assert out[_FC_DF] is not None


# ---- which signal the correlation was taken on ----

def test_fc_without_any_regression_runs_on_the_bandpassed_data(haemo, tmp_path):
    """This is the route rest mode cannot express: no drift model, no GLM, no ALFF."""
    out = _denoise(haemo, tmp_path, fc=True)
    assert out[_FC_DF] is not None
    assert _names(tmp_path, "*desc-errts*.snirf") == []
    assert _fc_source(tmp_path).endswith("_desc-filtered_nirs.snirf")


def test_fc_after_a_regression_runs_on_the_residual(haemo, tmp_path):
    out = _denoise(haemo, tmp_path, short_channel="mean", fc=True)
    assert out[_FC_DF] is not None
    assert _fc_source(tmp_path).endswith("_desc-errts_nirs.snirf")


def test_the_regression_actually_changes_the_correlations(haemo, tmp_path):
    """The source-picking above is only meaningful if the two signals differ.

    Without this, both tests would pass on a branch that always read the bandpassed data
    and merely labelled the sidecar differently.
    """
    plain = _denoise(haemo, tmp_path / "plain", fc=True)[_FC_DF]
    regressed = _denoise(haemo, tmp_path / "regressed",
                         short_channel="mean", fc=True)[_FC_DF]
    assert list(plain.columns) == list(regressed.columns)
    assert not plain.equals(regressed)


# ---- what the mode cannot produce ----

def test_alff_is_not_written_by_the_denoise_mode(haemo, tmp_path):
    """Same reason glm does not write it: the source is bandpassed, so fALFF would be ~1."""
    out = _denoise(haemo, tmp_path, short_channel="mean", fc=True)
    assert out[_ALFF_DF] is None
    assert _names(tmp_path, "*_alff.tsv") == []
