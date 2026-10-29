"""The GLM table's df and p values, against statsmodels fitting the same data on the same design."""

from __future__ import annotations

import mne
import numpy as np
import pytest
import statsmodels.api as sm

from nirspipe.io.tables import read_table
from nirspipe.pipeline.glm import run_glm_pipeline

SEPARATIONS_MM = (30.0, 30.0, 30.0)
ONSETS = (5.0, 30.0, 55.0, 80.0)
# HbR falls with the tap regressor's own shape here, so its t is large and negative
DIPPING = "S2_D2 hbr"


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
    data = np.random.default_rng(7).normal(size=(len(names), 1100)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(list(ONSETS), [10.0] * len(ONSETS),
                                        ["tap"] * len(ONSETS)))
    return raw


def _fit(haemo: mne.io.RawArray, out) -> None:
    out.mkdir(parents=True)
    run_glm_pipeline(haemo.copy(), stim_dur=10.0, hrf_model="spm", noise_model="ols",
                     drift_model="polynomial", high_pass=None, drift_order=1, fir_delays=None,
                     contrast_def={"tap": {"tap": 1}, "minus_tap": {"tap": -1}},
                     output_dir=str(out),
                     source_path=str(out / "sub-01_task-x_desc-filtered_nirs.snirf"),
                     chromophores=("hbo", "hbr"))


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("glm_p")
    # the design depends on the events alone, so a first fit hands over the regressor to plant
    blank = root / "blank" / "sub-01" / "nirs"
    _fit(_haemo(), blank)
    tap = read_table(blank / "sub-01_task-x_design.tsv")["tap"].to_numpy()
    haemo = _haemo()
    haemo._data[haemo.ch_names.index(DIPPING)] -= 2e-6 * tap / tap.max()
    out = root / "sub-01" / "nirs"
    _fit(haemo, out)
    design = read_table(out / "sub-01_task-x_design.tsv").drop(columns="censored")
    ols = {ch: sm.OLS(haemo.get_data(picks=[ch])[0], design.to_numpy()).fit()
           for ch in haemo.ch_names}
    tables = {kind: read_table(out / f"sub-01_task-x_desc-{kind}_nirsmap.tsv")
              for kind in ("glm", "contrast")}
    return design, ols, tables


def test_the_glm_table_states_the_residual_df_and_its_two_sided_p(fitted):
    design, ols, tables = fitted
    table = tables["glm"]
    assert len(table) == len(ols) * design.shape[1]
    for _, row in table.iterrows():
        fit = ols[row["ch_name"]]
        at = list(design.columns).index(row["Condition"])
        assert row["df"] == fit.df_resid == len(design) - design.shape[1]
        np.testing.assert_allclose(row["theta"], fit.params[at], rtol=1e-8)
        np.testing.assert_allclose(row["p_value"], fit.pvalues[at], rtol=1e-6, atol=1e-300)


def test_a_planted_dip_reads_significant_in_the_glm_table(fitted):
    table = fitted[2]["glm"]
    row = table[(table["ch_name"] == DIPPING) & (table["Condition"] == "tap")].iloc[0]
    assert row["t"] < -10 and row["p_value"] < 1e-20
