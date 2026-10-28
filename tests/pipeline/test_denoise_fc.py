"""The denoise mode's two independent halves: the confound regression, and `--fc`.

Either can run without the other, which is what separates this mode from rest. rest forces
a drift model and always writes ALFF, so "bandpass, then correlate" (the most common FC
recipe there is) needs a route through the package that does not also run a GLM nobody asked
for. These tests pin that route, and they pin which signal the correlation was taken on,
because that is the only thing distinguishing the two ways of reaching it.
"""

import json

import pytest

from nirspipe.io.naming import derivative_path

from nirspipe.pipeline.post_pipeline import PostConfig, run_post
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep

from tests._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)
# run_post: (result, glm_est, dm, alff_df, fc, fc_seed, fc_roi)
_ALFF_DF, _FC, _FC_SEED, _FC_ROI = 3, 4, 5, 6


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
    sidecar = next(out_dir.rglob("*_chromo-hbo_stat-pearson_relmat.json"))
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


# ---- the --fc half ----

def test_without_the_flag_nothing_connectivity_is_written(haemo, tmp_path):
    out = _denoise(haemo, tmp_path, short_channel="mean")
    assert _names(tmp_path, "*_relmat.tsv") == []
    assert out[_FC] == {} and out[_FC_ROI] == {} and out[_FC_SEED] == {}


def test_the_flag_writes_the_same_family_rest_writes(haemo, tmp_path, roi_map):
    out = _denoise(haemo, tmp_path, short_channel="mean", roi_map=roi_map, fc=True)
    written = _names(tmp_path, "*_relmat.tsv")
    for chromo in ("hbo", "hbr"):
        for entities in (dict(statistic="pearson"), dict(statistic="fisherz"),
                         dict(segmentation="custom", aggregation="roi", statistic="pearson"),
                         dict(segmentation="custom", aggregation="roi", statistic="fisherz"),
                         dict(segmentation="custom", aggregation="seed", statistic="pearson"),
                         dict(segmentation="custom", aggregation="seed", statistic="fisherz")):
            expected = derivative_path(tmp_path, "relmat", ".tsv", subject="01",
                                       task="rest", chromophore=chromo, **entities)
            assert expected.name in written, expected.name
    assert "hbo" in out[_FC]
    assert sorted(out[_FC_ROI]) == ["hbo", "hbr"]
    assert sorted(out[_FC_SEED]) == ["hbo", "hbr"]


def test_the_roi_products_need_a_roi_map(haemo, tmp_path):
    out = _denoise(haemo, tmp_path, short_channel="mean", fc=True)
    assert _names(tmp_path, "*fcroi*.tsv") == []
    assert _names(tmp_path, "*fcseed*.tsv") == []
    assert "hbo" in out[_FC]


# ---- which signal the correlation was taken on ----

def test_fc_without_any_regression_runs_on_the_bandpassed_data(haemo, tmp_path):
    """This is the route rest mode cannot express: no drift model, no GLM, no ALFF."""
    out = _denoise(haemo, tmp_path, fc=True)
    assert "hbo" in out[_FC]
    assert _names(tmp_path, "*desc-errts*.snirf") == []
    assert _fc_source(tmp_path).endswith("_desc-filtered_nirs.snirf")


def test_fc_after_a_regression_runs_on_the_residual(haemo, tmp_path):
    out = _denoise(haemo, tmp_path, short_channel="mean", fc=True)
    assert "hbo" in out[_FC]
    assert _fc_source(tmp_path).endswith("_desc-errts_nirs.snirf")


def test_the_regression_actually_changes_the_correlations(haemo, tmp_path):
    """The source-picking above is only meaningful if the two signals differ.

    Without this, both tests would pass on a branch that always read the bandpassed data
    and merely labelled the sidecar differently.
    """
    plain = _denoise(haemo, tmp_path / "plain", fc=True)[_FC]["hbo"]
    regressed = _denoise(haemo, tmp_path / "regressed",
                         short_channel="mean", fc=True)[_FC]["hbo"]
    assert list(plain.columns) == list(regressed.columns)
    assert not plain.equals(regressed)


# ---- what the mode cannot produce ----

def test_alff_is_not_written_by_the_denoise_mode(haemo, tmp_path):
    """Same reason glm does not write it: the source is bandpassed, so fALFF would be ~1."""
    out = _denoise(haemo, tmp_path, short_channel="mean", fc=True)
    assert out[_ALFF_DF] is None
    assert _names(tmp_path, "*_stat-alff_nirsmap.tsv") == []


# ---- the matrix before the regression ----

def _matrix(path):
    import pandas as pd
    return pd.read_csv(path, sep="\t", index_col=0)


def test_after_a_regression_the_matrix_before_it_is_written_beside_the_residual_s(haemo, tmp_path):
    _denoise(haemo, tmp_path / "plain", fc=True)
    _denoise(haemo, tmp_path / "regressed", short_channel="mean", fc=True)
    for chromo in ("hbo", "hbr"):
        before = next((tmp_path / "regressed").rglob(
            f"sub-01_task-rest_chromo-{chromo}_stat-pearson_desc-filtered_relmat.tsv"))
        after = next((tmp_path / "regressed").rglob(
            f"sub-01_task-rest_chromo-{chromo}_stat-pearson_relmat.tsv"))
        # the bandpassed data with no regression is the same series, correlated the same way
        plain = next((tmp_path / "plain").rglob(
            f"sub-01_task-rest_chromo-{chromo}_stat-pearson_relmat.tsv"))
        b, a, p = _matrix(before), _matrix(after), _matrix(plain)
        assert list(b.index) == list(a.index)
        assert b.isna().equals(a.isna()), "blank where the residual's matrix is"
        assert (b.notna() & ~p.reindex_like(b).isna()).any().any()
        assert ((b - p.reindex_like(b)).abs().max().max()) < 1e-9
        sidecar = json.loads(before.with_suffix(".json").read_text())
        assert sidecar["Sources"][0].endswith("_desc-filtered_nirs.snirf")


def test_without_a_regression_there_is_no_before(haemo, tmp_path):
    _denoise(haemo, tmp_path, fc=True)
    assert _names(tmp_path, "*_desc-*_relmat.tsv") == []


def test_rest_mode_writes_it_too(haemo, tmp_path):
    config = PostConfig(subject="01", high_pass=0.01, low_pass=0.1, drift_model="cosine",
                        drift_high_pass=0.01, short_channel="mean", **_BANDS)
    run_post(haemo.copy(), config, output_dir=tmp_path, mode="rest",
             source_entities={"task": "rest"})
    assert _names(tmp_path, "*_desc-filtered_relmat.tsv") == [
        "sub-01_task-rest_chromo-hbo_stat-pearson_desc-filtered_relmat.tsv",
        "sub-01_task-rest_chromo-hbr_stat-pearson_desc-filtered_relmat.tsv"]
