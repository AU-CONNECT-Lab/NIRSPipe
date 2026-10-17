"""A short channel that is the whole regressor is fitted against a copy of itself.

The `mean` regressor is the mean of the good short channels of a chromophore, and the design
matrix is applied to every channel including the ones it was built from. With one surviving
short channel that mean *is* that channel sample for sample, so its own residual is
numerically zero and every measure read off the residual is an artefact of the method rather
than a measurement.

What is pinned here is the line, at one channel and not at some share, and that the
correction reaches the tables without touching any other channel.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nirspipe.pipeline.glm import _short_channel_regressors, sole_regressor_channels


def _haemo(separations_mm, bads=()):
    """One hbo/hbr pair per separation, positions set so the distances are real."""
    names, locs = [], []
    for i, mm in enumerate(separations_mm):
        src = np.array([i * 0.05, 0.0, 0.0])
        det = src + np.array([mm / 1000.0, 0.0, 0.0])
        for chroma in ("hbo", "hbr"):
            loc = np.zeros(12)
            loc[0:3], loc[3:6], loc[6:9] = (src + det) / 2, src, det
            names.append(f"S{i + 1}_D{i + 1} {chroma}")
            locs.append(loc)
    info = mne.create_info(names, sfreq=10.0, ch_types=["hbo", "hbr"] * len(separations_mm))
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc
    raw = mne.io.RawArray(np.random.default_rng(0).normal(size=(len(names), 600)),
                          info, verbose="ERROR")
    raw.info["bads"] = list(bads)
    return raw


def test_one_surviving_short_channel_is_named():
    raw = _haemo([8.0, 8.5, 30.0], bads=["S2_D2 hbo", "S2_D2 hbr"])
    assert sole_regressor_channels(raw, "mean") == ["S1_D1 hbo", "S1_D1 hbr"]


def test_two_surviving_short_channels_are_not():
    """Their residual is the channel minus the mean of the two, which is a real if small
    quantity; only at one is it identically zero."""
    raw = _haemo([8.0, 8.5, 30.0])
    assert sole_regressor_channels(raw, "mean") == []


def test_the_pca_strategy_falls_the_same_way():
    """A single channel's basis is that channel, however it is decomposed."""
    raw = _haemo([8.0, 8.5, 30.0], bads=["S2_D2 hbo", "S2_D2 hbr"])
    assert sole_regressor_channels(raw, "pca") == ["S1_D1 hbo", "S1_D1 hbr"]


def test_nothing_is_claimed_without_the_regression():
    raw = _haemo([8.0, 30.0])
    assert sole_regressor_channels(raw, None) == []
    assert sole_regressor_channels(raw, False) == []


def test_a_montage_with_no_short_channel_claims_nothing_rather_than_raising():
    """The refusal is `_short_channel_regressors`' to make, and it carries the message that
    names the flag; this one is asked in places that must not raise."""
    assert sole_regressor_channels(_haemo([30.0, 35.0]), "mean") == []


def test_the_regressor_is_still_built_and_still_serves_the_long_channels():
    """The long channels are regressed correctly by a one-channel regressor: it is a fine
    estimate of the systemic signal. Only its own source loses a residual, so the run must
    not be refused."""
    raw = _haemo([8.0, 8.5, 30.0], bads=["S2_D2 hbo", "S2_D2 hbr"])
    out = _short_channel_regressors(raw, "mean")
    assert set(out) == {"short_ch_hbo_mean", "short_ch_hbr_mean"}
    kept = raw.get_data(picks=["S1_D1 hbo"])[0]
    assert np.allclose(out["short_ch_hbo_mean"], kept)


def test_the_warning_names_the_channel_and_says_what_is_blanked(caplog):
    raw = _haemo([8.0, 8.5, 30.0], bads=["S2_D2 hbo", "S2_D2 hbr"])
    with caplog.at_level("WARNING"):
        _short_channel_regressors(raw, "mean")
    said = caplog.text
    assert "S1_D1 hbo" in said and "GLM fit, ALFF and FC are left blank" in said


def test_pca_names_every_good_short_channel_however_many_survive():
    raw = _haemo([8.0, 8.5, 30.0])
    assert sole_regressor_channels(raw, "pca") == ["S1_D1 hbo", "S2_D2 hbo",
                                                   "S1_D1 hbr", "S2_D2 hbr"]


def test_each_channel_pca_names_is_left_with_no_residual():
    raw = _haemo([8.0, 8.5, 30.0])
    basis = _short_channel_regressors(raw, "pca")
    design = np.column_stack([np.ones(raw.n_times), *basis.values()])
    for name in sole_regressor_channels(raw, "pca"):
        y = raw.get_data(picks=[name])[0]
        resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        assert np.linalg.norm(resid) < 1e-10 * np.linalg.norm(y), name
    long = raw.get_data(picks=["S3_D3 hbo"])[0]
    resid = long - design @ np.linalg.lstsq(design, long, rcond=None)[0]
    assert np.linalg.norm(resid) > 0.5 * np.linalg.norm(long)


def test_the_pca_warning_names_every_channel_once(caplog):
    raw = _haemo([8.0, 8.5, 30.0])
    with caplog.at_level("WARNING"):
        _short_channel_regressors(raw, "pca")
    warned = [r.getMessage() for r in caplog.records if "left blank" in r.getMessage()]
    assert len(warned) == 1
    assert all(ch in warned[0] for ch in ("S1_D1 hbo", "S2_D2 hbo", "S1_D1 hbr", "S2_D2 hbr"))


@pytest.mark.parametrize("strategy", ["mean", "pca"])
def test_a_chromophore_with_no_good_short_channel_builds_no_regressor_and_names_nothing(strategy):
    raw = _haemo([8.0, 30.0], bads=["S1_D1 hbr"])
    assert _short_channel_regressors(raw, strategy) == {}
    assert sole_regressor_channels(raw, strategy) == []


# ---- the GLM tables ----

def _glm_tables(tmp_path, strategy, bads):
    from nirspipe.io.tables import read_table
    from nirspipe.pipeline.glm import run_glm_pipeline

    raw = _haemo([8.0, 8.5, 30.0], bads=bads)
    raw.set_annotations(mne.Annotations([5.0, 25.0, 45.0], [5.0] * 3, ["tap"] * 3))
    out = tmp_path / "sub-01" / "nirs"
    out.mkdir(parents=True)
    source = out / "sub-01_task-x_desc-filtered_nirs.snirf"
    run_glm_pipeline(raw, stim_dur=5.0, hrf_model="spm", noise_model="ols",
                     drift_model="polynomial", high_pass=None, drift_order=1,
                     fir_delays=None, short_channel=strategy,
                     contrast_def={"tap": {"tap": 1}},
                     output_dir=str(out), source_path=str(source))
    fit = read_table(out / "sub-01_task-x_desc-glm_nirsmap.tsv")
    con = read_table(out / "sub-01_task-x_desc-contrast_nirsmap.tsv")
    return fit, con


def test_the_glm_tables_blank_a_channel_fitted_against_itself(tmp_path):
    """One good short channel a chromophore: theta 1 on its own regressor, mse at round-off,
    t in the billions. Its rows stay, so every subject's table has the same shape, with
    every statistic n/a."""
    fit, con = _glm_tables(tmp_path, "mean", ["S2_D2 hbo", "S2_D2 hbr"])
    stats_fit = ["df", "mse", "p_value", "se", "t", "theta", "Significant"]
    stats_con = ["effect", "p_value", "stat", "z_score", "Significant"]
    for table, stats in ((fit, stats_fit), (con, stats_con)):
        lone = table[table["ch_name"].isin(["S1_D1 hbo", "S1_D1 hbr"])]
        assert len(lone) and lone[stats].isna().all().all()
        # a rejected channel is flagged, not blanked; a long channel keeps its fit
        for kept in ("S2_D2 hbo", "S3_D3 hbo"):
            rows = table[table["ch_name"] == kept]
            assert len(rows) and rows[stats].notna().all().all(), kept
    assert set(fit["Condition"][fit["ch_name"] == "S1_D1 hbo"]) >= {"tap", "short_ch_hbo_mean"}


def test_two_good_short_channels_keep_their_glm_rows(tmp_path):
    fit, _ = _glm_tables(tmp_path, "mean", [])
    assert fit[["theta", "t", "p_value"]].notna().all().all()
