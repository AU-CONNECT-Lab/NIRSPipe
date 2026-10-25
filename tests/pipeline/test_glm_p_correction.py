"""--p-correction on the GLM and contrast tables: which rows form a family, and what lands beside p_value.

Expected corrected values are worked by hand from the raw p (Bonferroni p x m capped at 1,
Benjamini-Hochberg p x m / rank with the running minimum from the largest p down), not from
the package's own helper.
"""

from __future__ import annotations

import json

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.io.tables import read_table
from nirspipe.pipeline.glm import run_glm_pipeline
from nirspipe.qc.boilerplate.vocabulary import template_slots
from nirspipe.utils import correct_p

# one short pair, then four long ones; S3_D3's HbR half is rejected, so its HbT is too
SEPARATIONS_MM = (8.0, 30.0, 30.0, 30.0, 30.0)
BADS = ("S3_D3 hbr",)
LONG_GOOD = {"hbo": {"S2_D2", "S3_D3", "S4_D4", "S5_D5"},
             "hbr": {"S2_D2", "S4_D4", "S5_D5"},
             "hbt": {"S2_D2", "S4_D4", "S5_D5"}}


def _bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    out = np.empty_like(p)
    out[order] = np.minimum.accumulate(scaled[::-1])[::-1].clip(max=1.0)
    return out


# ---- the helper ----

def test_bonferroni_and_bh_on_one_family_match_the_hand_values():
    frame = pd.DataFrame({"test": ["a"] * 4, "p": [0.01, 0.02, 0.03, 0.5]})
    bonf = correct_p(frame, ["test"], "bonferroni")
    np.testing.assert_allclose(bonf["p_bonferroni"], [0.04, 0.08, 0.12, 1.0])
    np.testing.assert_allclose(correct_p(frame, ["test"], "fdr_bh")["p_fdr_bh"],
                               [0.04, 0.04, 0.04, 0.5])
    assert bonf["family"].tolist() == [4] * 4


def test_a_row_left_out_keeps_a_blank_p_and_shrinks_the_family():
    frame = pd.DataFrame({"test": ["a"] * 4, "p": [0.01, 0.02, 0.03, 0.5]})
    out = correct_p(frame, ["test"], "bonferroni", eligible=pd.Series([True, False, True, True]))
    assert np.isnan(out["p_bonferroni"][1])
    np.testing.assert_allclose(out["p_bonferroni"][[0, 2, 3]], [0.03, 0.09, 1.0])
    assert out["family"].tolist() == [3] * 4


def test_none_writes_the_family_and_no_corrected_column():
    out = correct_p(pd.DataFrame({"test": ["a", "b"], "p": [0.1, 0.2]}), ["test"], "none")
    assert "p_none" not in out.columns and out["family"].tolist() == [1, 1]


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown p correction"):
        correct_p(pd.DataFrame({"p": [0.1]}), [], "fdr_tsbh")


# ---- through the GLM ----

def _haemo() -> mne.io.RawArray:
    names, locs = [], []
    for i, mm in enumerate(SEPARATIONS_MM):
        src = np.array([i * 0.05, 0.0, 0.0])
        det = src + np.array([mm / 1000.0, 0.0, 0.0])
        for chroma in ("hbo", "hbr"):
            loc = np.zeros(12)
            loc[0:3], loc[3:6], loc[6:9] = (src + det) / 2, src, det
            names.append(f"S{i + 1}_D{i + 1} {chroma}")
            locs.append(loc)
    info = mne.create_info(names, sfreq=10.0, ch_types=["hbo", "hbr"] * len(SEPARATIONS_MM))
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc
    raw = mne.io.RawArray(np.random.default_rng(3).normal(size=(len(names), 900)) * 1e-6,
                          info, verbose="ERROR")
    raw.info["bads"] = list(BADS)
    raw.set_annotations(mne.Annotations([5.0, 30.0, 55.0], [5.0] * 3, ["tap"] * 3))
    return raw


def _tables(tmp_path, method):
    out = tmp_path / method / "sub-01" / "nirs"
    out.mkdir(parents=True)
    run_glm_pipeline(_haemo(), stim_dur=5.0, hrf_model="spm", noise_model="ols",
                     drift_model="polynomial", high_pass=None, drift_order=1, fir_delays=None,
                     contrast_def={"tap": {"tap": 1}}, output_dir=str(out),
                     source_path=str(out / "sub-01_task-x_desc-filtered_nirs.snirf"),
                     chromophores=("hbo", "hbr", "hbt"), p_correction=method)
    return {kind: (read_table(out / f"sub-01_task-x_desc-{kind}_nirsmap.tsv"),
                   json.loads((out / f"sub-01_task-x_desc-{kind}_nirsmap.json")
                              .read_text(encoding="utf-8"))["parameters"])
            for kind in ("glm", "contrast")}


@pytest.fixture(scope="module")
def bonferroni(tmp_path_factory):
    return _tables(tmp_path_factory.mktemp("pcorr"), "bonferroni")


@pytest.mark.parametrize("kind, test", [("glm", "Condition"), ("contrast", "contrast")])
def test_each_test_and_chromophore_is_corrected_over_its_long_good_channels(bonferroni, kind,
                                                                            test):
    table, params = bonferroni[kind]
    assert params["p_correction"] == "bonferroni"
    assert "Significant" not in table.columns
    at = list(table.columns).index("p_value")
    assert list(table.columns)[at + 1:at + 3] == ["p_bonferroni", "family"]
    pair = table["ch_name"].str.rsplit(" ", n=1).str[0]
    for (_, chroma), rows in table.groupby([test, "Chroma"]):
        tested = pair[rows.index].isin(LONG_GOOD[chroma])
        assert (rows["family"] == len(LONG_GOOD[chroma])).all()
        expected = np.minimum(rows["p_value"][tested] * len(LONG_GOOD[chroma]), 1.0)
        np.testing.assert_allclose(rows["p_bonferroni"][tested], expected, rtol=1e-12)
        # the short pair and the rejected channels are fitted but never tested
        assert rows["p_bonferroni"][~tested].isna().all()
        assert rows["p_value"][~tested].notna().all()


def test_bh_matches_the_hand_value_on_one_family(tmp_path):
    table, _ = _tables(tmp_path, "fdr_bh")["glm"]
    rows = table[(table["Condition"] == "tap") & (table["Chroma"] == "hbo")
                 & table["ch_name"].str.rsplit(" ", n=1).str[0].isin(LONG_GOOD["hbo"])]
    np.testing.assert_allclose(rows["p_fdr_bh"], _bh(rows["p_value"].to_numpy()), rtol=1e-12)


def test_none_keeps_the_raw_p_and_the_family_only(tmp_path):
    for table, params in _tables(tmp_path, "none").values():
        assert params["p_correction"] == "none"
        assert not [c for c in table.columns if c.startswith("p_") and c != "p_value"]
        assert "Significant" not in table.columns and "family" in table.columns


# ---- the Methods ----

def test_the_glm_sentence_says_whether_and_how_p_was_corrected():
    within = ("within each regressor or contrast and each chromophore, over the long channels "
              "not marked as bad.")
    holm = template_slots("glm", {"p_correction": "holm"})["correction"]
    assert holm == f" P values were corrected by the Holm–Bonferroni method {within}"
    assert template_slots("glm", {"p_correction": "none"})["correction"] == (
        " P values were not corrected for multiple comparisons.")
    # a fit table written before the setting was recorded says nothing rather than guessing
    assert template_slots("glm", {})["correction"] == ""
    # the cohort test keeps its own family
    assert template_slots("hyper_groupnull", {"p_correction": "holm"})["correction"].endswith(
        "within each condition, level and test.")
