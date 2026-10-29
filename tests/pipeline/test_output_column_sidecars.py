"""Every column of the GLM, contrast, ALFF and channel QC tables is described in its sidecar, and every FC matrix's sidecar says what it holds."""

from __future__ import annotations

import json

import mne
import numpy as np
import pytest

from nirspipe.io.tables import read_table
from nirspipe.pipeline.glm import run_glm_pipeline
from nirspipe.pipeline.post_pipeline import PostConfig, run_post
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep
from nirspipe.qc.common.channel_table import RAW_CHANNEL_METRICS_SUFFIX, save_channel_csv

from tests._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)


def _sidecar(tsv):
    return json.loads(tsv.with_suffix(".json").read_text(encoding="utf-8"))


def _undescribed(tsv) -> list[str]:
    meta = _sidecar(tsv)
    return [c for c in read_table(tsv).columns if not (meta.get(c) or {}).get("Description")]


def _haemo() -> mne.io.RawArray:
    names = [f"S{i}_D{i} {chroma}" for i in (1, 2) for chroma in ("hbo", "hbr")]
    info = mne.create_info(names, sfreq=10.0, ch_types=["hbo", "hbr"] * 2)
    for i, ch in enumerate(info["chs"]):
        src = np.array([0.05 * (i // 2), 0.0, 0.0])
        ch["loc"][3:6], ch["loc"][6:9] = src, src + [0.03, 0.0, 0.0]
        ch["loc"][0:3] = src + [0.015, 0.0, 0.0]
    raw = mne.io.RawArray(np.random.default_rng(3).normal(size=(4, 1100)) * 1e-6, info,
                          verbose="ERROR")
    raw.set_annotations(mne.Annotations([5.0, 30.0, 55.0, 80.0], [10.0] * 4, ["tap"] * 4))
    return raw


@pytest.fixture(scope="module")
def glm_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("glm_cols") / "sub-01" / "nirs"
    out.mkdir(parents=True)
    run_glm_pipeline(_haemo(), stim_dur=10.0, hrf_model="spm", noise_model="ols",
                     drift_model="polynomial", high_pass=None, drift_order=1, fir_delays=None,
                     contrast_def={"tap": {"tap": 1}}, output_dir=str(out),
                     source_path=str(out / "sub-01_task-x_desc-filtered_nirs.snirf"),
                     chromophores=("hbo", "hbr"), p_correction="holm")
    return out


@pytest.mark.parametrize("desc", ["glm", "contrast"])
def test_every_glm_and_contrast_column_is_described(glm_dir, desc):
    tsv = glm_dir / f"sub-01_task-x_desc-{desc}_nirsmap.tsv"
    assert "p_holm" in read_table(tsv).columns
    assert _undescribed(tsv) == []


def test_the_glm_sidecar_keeps_its_provenance_and_gives_mse_a_unit(glm_dir):
    meta = _sidecar(glm_dir / "sub-01_task-x_desc-glm_nirsmap.tsv")
    assert meta["step"] == "glm_fit"
    assert meta["parameters"]["p_correction"] == "holm"
    assert meta["mse"]["Units"] == "M^2"


@pytest.fixture(scope="module")
def rest_dir(tmp_path_factory):
    prep = run_prep(synth_raw("01", "tapping"),
                    PrepConfig(subject="01", dpf=[6.0, 6.0], sci_threshold=0.8,
                               motion_correction="tddr", **_BANDS),
                    output_dir=tmp_path_factory.mktemp("cols_prep"),
                    source_entities={"task": "tapping"})
    haemo = prep.raw_haemo
    pairs = sorted({c.rsplit(" ", 1)[0] for c in haemo.ch_names})
    roi_map = {"A": pairs[: len(pairs) // 2], "B": pairs[len(pairs) // 2:]}
    out = tmp_path_factory.mktemp("cols_rest")
    run_post(haemo.copy(),
             PostConfig(subject="01", high_pass=0.01, low_pass=0.1, drift_model="cosine",
                        drift_high_pass=0.01, roi_map=roi_map, roi_map_name="halves", **_BANDS),
             output_dir=out, mode="rest", source_entities={"task": "tapping"})
    return out


def test_every_alff_column_is_described_and_alff_is_in_molar(rest_dir):
    tables = sorted(rest_dir.rglob("*_stat-alff_nirsmap.tsv"))
    assert len(tables) == 2, "the channel table and the ROI table"
    for tsv in tables:
        assert _undescribed(tsv) == [], tsv.name
        assert _sidecar(tsv)["alff"]["Units"] == "M"


def test_every_fc_matrix_says_what_it_holds(rest_dir):
    matrices = sorted(rest_dir.rglob("*_relmat.tsv"))
    # channel, ROI and seed, Pearson and Fisher z, and the channel matrix before the
    # regression, two chromophores
    assert len(matrices) == 14
    for tsv in matrices:
        description = _sidecar(tsv).get("Description") or ""
        assert description, tsv.name
        square = "agg-seed" not in tsv.name
        if "stat-fisherz" in tsv.name:
            assert ("diagonal" in description) == square, tsv.name


def test_the_channel_qc_table_describes_every_column(tmp_path):
    row = {"name": "S1_D1 760", "sci_win": 0.9, "sci_whole": 0.9, "psp": 0.1, "snr": 20.0,
           "cv": 0.05, "spike": 0.0, "saturated": 0.0, "corr": -0.5, "corr_post": None,
           "short_share_hbo": None, "short_share_hbr": None, "good_frac": 1.0,
           "is_bad": False, "separation": "long"}
    save_channel_csv([row], "sub-01_task-x", tmp_path, 0.8, suffix=RAW_CHANNEL_METRICS_SUFFIX)
    tsv = tmp_path / f"sub-01_task-x{RAW_CHANNEL_METRICS_SUFFIX}"
    assert _undescribed(tsv) == []
