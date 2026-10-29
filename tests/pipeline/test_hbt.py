"""HbT through post: the pair's sum, what --chromophores puts in each table, and what it leaves alone.

Expected values come from identities that need no package code: HbT is HbO + HbR sample by
sample, so under OLS its beta and its block average are the pair's sums; mALFF and zALFF of
one chromophore average to 1 and 0 over the channels they were scaled by; an FC cell is
numpy's correlation of two summed residual rows. Adding HbT must leave every HbO and HbR value
as it was, so those are compared with a run that did not ask for it.
"""

import json

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.chromophores import hbt_names, hbt_of, in_order
from nirspipe.pipeline.post_pipeline import PostConfig, run_post
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep

from tests._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)
# one half of a good pair marked bad, so its HbT is bad while its HbO is not
HALF_BAD = "S2_D2 hbr"


# ---- the helpers, on arrays small enough to check by eye ----

def _pair_raw(hbo, hbr, bads=()):
    names = ["S1_D1 hbo", "S1_D1 hbr", "S2_D2 hbo", "S2_D2 hbr"]
    raw = mne.io.RawArray(np.array([hbo[0], hbr[0], hbo[1], hbr[1]], float),
                          mne.create_info(names, 5.0, ["hbo", "hbr", "hbo", "hbr"]),
                          verbose="error")
    raw.info["bads"] = list(bads)
    raw.set_annotations(mne.Annotations([0.2], [0.2], ["BAD_manual"]))
    return raw


def test_hbt_is_the_pairs_sum_and_bad_where_either_half_is():
    raw = _pair_raw(hbo=[[1, 2, 3], [10, 20, 30]], hbr=[[-1, 0, 1], [5, 5, 5]],
                    bads=["S2_D2 hbr"])
    hbt = hbt_of(raw)
    assert hbt.ch_names == ["S1_D1 hbt", "S2_D2 hbt"]
    np.testing.assert_array_equal(hbt.get_data(), [[0, 2, 4], [15, 25, 35]])
    assert hbt.info["bads"] == ["S2_D2 hbt"]
    assert list(hbt.annotations.description) == ["BAD_manual"]
    # the source is untouched
    np.testing.assert_array_equal(raw.get_data(picks=["S1_D1 hbo"])[0], [1, 2, 3])


def test_an_hbt_channel_needs_both_halves_listed():
    assert hbt_names(["S1_D1 hbo", "S1_D1 hbr", "S2_D2 hbo"]) == ["S1_D1 hbt"]


def test_the_chromophores_come_back_once_each_in_one_order():
    assert in_order(["hbt", "hbo", "hbt"]) == ("hbo", "hbt")
    for bad in (["hbx"], []):
        with pytest.raises(ValueError):
            in_order(bad)


# ---- through run_post ----

@pytest.fixture(scope="module")
def haemo(tmp_path_factory):
    config = PrepConfig(subject="01", dpf=[6.0, 6.0], sci_threshold=0.8,
                        motion_correction="tddr", **_BANDS)
    result = run_prep(synth_raw("01", "tapping"), config,
                      output_dir=tmp_path_factory.mktemp("hbt_prep"),
                      source_entities={"task": "tapping"})
    raw = result.raw_haemo
    raw.info["bads"] = [*raw.info["bads"], HALF_BAD]
    return raw


def _post(haemo, out, mode, **overrides):
    settings = dict(high_pass=0.01, low_pass=0.1, drift_model="cosine", drift_high_pass=0.01,
                    noise_model="ols")
    if mode == "glm":
        # no low-pass under an AR model; the high-pass gives the fit a stage file to name
        settings.update(hrf_model="spm", stim_dur=5.0, low_pass=None)
    config = PostConfig(subject="01", **{**settings, **overrides}, **_BANDS)
    run_post(haemo.copy(), config, output_dir=out, mode=mode, source_entities={"task": "tapping"})
    return out / "sub-01" / "nirs"


@pytest.fixture(scope="module")
def runs(haemo, tmp_path_factory):
    """ols and ar1 GLMs with and without HbT, and rest with and without it."""
    roi = {"A": ["S1_D1 hbo", "S2_D2 hbo"], "B": ["S3_D3 hbo", "S4_D4 hbo"]}
    out = {}
    for key, mode, overrides in (
            ("ols", "glm", {}),
            ("ols_hbt", "glm", {"chromophores": ("hbo", "hbr", "hbt")}),
            ("ar1", "glm", {"noise_model": "ar1"}),
            ("ar1_hbr_hbt", "glm", {"noise_model": "ar1", "chromophores": ("hbr", "hbt")}),
            ("rest", "rest", {"short_channel": "mean", "roi_map": roi}),
            ("rest_hbt", "rest", {"short_channel": "mean", "roi_map": roi,
                                  "chromophores": ("hbo", "hbr", "hbt")})):
        out[key] = _post(haemo, tmp_path_factory.mktemp(key), mode, **overrides)
    return out


def _table(nirs, pattern):
    (path,) = nirs.glob(pattern)
    return pd.read_csv(path, sep="\t", keep_default_na=True, na_values=["n/a"])


def _sidecar(nirs, pattern):
    (path,) = nirs.glob(pattern)
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def test_under_ols_the_hbt_beta_is_the_pairs_sum(runs):
    glm = _table(runs["ols_hbt"], "*desc-glm_nirsmap.tsv")
    glm["pair"] = glm["ch_name"].str.rsplit(" ", n=1).str[0]
    theta = glm.pivot_table(index=["pair", "Condition"], columns="Chroma", values="theta")
    assert len(theta) and theta["hbt"].notna().any()
    np.testing.assert_allclose(theta["hbt"], theta["hbo"] + theta["hbr"], rtol=1e-9, atol=1e-18)


def test_adding_hbt_leaves_every_hbo_and_hbr_row_as_it_was(runs):
    for base, with_hbt, pattern in (("ols", "ols_hbt", "*desc-glm_nirsmap.tsv"),
                                    ("ols", "ols_hbt", "*desc-whitening_nirsmap.tsv"),
                                    ("ols", "ols_hbt", "*desc-blockavg_timeseries.tsv"),
                                    ("ar1", "ar1_hbr_hbt", "*desc-glm_nirsmap.tsv")):
        old, new = _table(runs[base], pattern), _table(runs[with_hbt], pattern)
        chroma = "Chroma" if "Chroma" in old.columns else "chroma"
        kept = new[chroma].unique()
        pd.testing.assert_frame_equal(
            new[new[chroma] != "hbt"].reset_index(drop=True),
            old[old[chroma].isin(kept)].reset_index(drop=True), check_exact=True)
    errts = [read_snirf(next(runs[k].glob("*desc-errts_nirs.snirf"))) for k in ("ols", "ols_hbt")]
    assert errts[1].ch_names == errts[0].ch_names
    assert not any(ch.endswith(" hbt") for ch in errts[1].ch_names)
    np.testing.assert_array_equal(errts[1].get_data(), errts[0].get_data())


def test_the_tables_carry_only_the_chromophores_asked_for(runs):
    nirs = runs["ar1_hbr_hbt"]
    assert set(_table(nirs, "*desc-glm_nirsmap.tsv")["Chroma"]) == {"hbr", "hbt"}
    assert set(_table(nirs, "*desc-blockavg_timeseries.tsv")["chroma"]) == {"hbr", "hbt"}
    spectrum = _table(nirs, "*stat-psd_desc-whitening_timeseries.tsv")
    assert not [c for c in spectrum.columns if c.startswith("hbo")]
    assert "hbt_whitened" in spectrum.columns
    assert _sidecar(nirs, "*desc-glm_nirsmap.tsv")["parameters"]["chromophores"] == ["hbr", "hbt"]
    assert set(_table(runs["ols"], "*desc-glm_nirsmap.tsv")["Chroma"]) == {"hbo", "hbr"}


def test_hbt_is_bad_where_either_half_is(runs):
    glm = _table(runs["ols_hbt"], "*desc-glm_nirsmap.tsv")
    bad = glm.groupby("ch_name")["bad"].first()
    assert bad["S2_D2 hbt"] and not bad["S2_D2 hbo"]


def test_the_block_average_of_hbt_is_the_pairs_sum(runs):
    table = _table(runs["ols_hbt"], "*desc-blockavg_timeseries.tsv")
    table["pair"] = table["ch_name"].str.rsplit(" ", n=1).str[0]
    mean = table.pivot_table(index=["condition", "pair", "time"], columns="chroma", values="mean")
    np.testing.assert_allclose(mean["hbt"], mean["hbo"] + mean["hbr"], rtol=1e-9, atol=1e-18)


def test_the_hbt_fc_is_the_correlation_of_the_summed_residuals(runs):
    nirs = runs["rest_hbt"]
    errts = read_snirf(next(nirs.glob("*desc-errts_nirs.snirf")))
    fc = _table(nirs, "*chromo-hbt_stat-pearson_relmat.tsv").set_index("channel")
    a, b = "S1_D1", "S4_D4"
    summed = [errts.get_data(picks=[f"{p} hbo"])[0] + errts.get_data(picks=[f"{p} hbr"])[0]
              for p in (a, b)]
    assert fc.loc[f"{a} hbt", f"{b} hbt"] == pytest.approx(np.corrcoef(*summed)[0, 1], abs=1e-12)
    # bad by its half; the short pair is left out, as every FC product leaves short channels
    assert fc.loc["S2_D2 hbt"].isna().all()
    assert "S5_D5 hbt" not in fc.index and "S5_D5 hbt" not in fc.columns


def test_hbt_alff_is_scaled_within_hbt_and_the_rest_is_unchanged(runs):
    alff = _table(runs["rest_hbt"], "sub-01_task-tapping_stat-alff_nirsmap.tsv")
    # S1_D1 and S4_D4: S2_D2 is bad by its half, S3_D3 by screening, S5_D5 fitted against itself
    hbt = alff[alff["channel"].str.endswith(" hbt") & alff["malff"].notna()]
    assert sorted(hbt["channel"]) == ["S1_D1 hbt", "S4_D4 hbt"]
    assert hbt["malff"].mean() == pytest.approx(1.0)
    assert hbt["zalff"].mean() == pytest.approx(0.0, abs=1e-12)
    assert hbt["zalff"].std(ddof=0) == pytest.approx(1.0)
    old = _table(runs["rest"], "sub-01_task-tapping_stat-alff_nirsmap.tsv")
    pd.testing.assert_frame_equal(alff[~alff["channel"].str.endswith(" hbt")].reset_index(drop=True),
                                  old, check_exact=True)
    for chroma in ("hbo", "hbr"):
        pattern = f"*chromo-{chroma}_stat-pearson_relmat.tsv"
        pd.testing.assert_frame_equal(_table(runs["rest_hbt"], pattern),
                                      _table(runs["rest"], pattern), check_exact=True)


def test_the_roi_tables_name_their_hbt_members(runs):
    nirs = runs["rest_hbt"]
    assert list(nirs.glob("*chromo-hbt_seg-*_agg-roi_stat-pearson_relmat.tsv"))
    roi = _table(nirs, "*agg-roi_stat-alff_nirsmap.tsv")
    assert set(roi["chromophore"]) == {"hbo", "hbr", "hbt"}
    members = _sidecar(nirs, "*agg-roi_stat-alff_nirsmap.tsv")["parameters"]["roi_channels"]
    # S2_D2 hbt is bad through its HbR half, so it is no member
    assert members["hbt"]["A"] == ["S1_D1 hbt"]


# ---- the Methods ----

def test_the_methods_define_hbt_once_before_the_glm_that_fits_it(runs):
    from nirspipe.qc.boilerplate.vocabulary import steps_from_sidecars

    keys = [k for k, _ in steps_from_sidecars(runs["ols_hbt"])]
    assert keys.count("hbt") == 1 and keys.index("hbt") < keys.index("glm")
    assert "fitted as a series of its own" in dict(steps_from_sidecars(runs["ols_hbt"]))["glm"]["hbt"]
    plain = steps_from_sidecars(runs["ols"])
    assert "hbt" not in dict(plain) and dict(plain)["glm"]["hbt"] == ""


def test_the_rest_methods_name_each_analysis_s_chromophores(runs):
    from nirspipe.qc.boilerplate.generate import step_sentence
    from nirspipe.qc.boilerplate.vocabulary import steps_from_sidecars

    steps = steps_from_sidecars(runs["rest_hbt"])
    keys = [k for k, _ in steps]
    assert keys.index("hbt") < min(keys.index("alff"), keys.index("fc"))
    assert dict(steps)["fc"]["chroma"] == dict(steps)["alff"]["chroma"] == "HbO, HbR and HbT"
    assert dict(steps_from_sidecars(runs["rest"]))["fc"]["chroma"] == "HbO and HbR"
    fc = _sidecar(runs["rest_hbt"], "*chromo-hbo_stat-pearson_relmat.tsv")["parameters"]
    assert "Fisher z transform, z = arctanh(r)" in step_sentence("fc", fc)
