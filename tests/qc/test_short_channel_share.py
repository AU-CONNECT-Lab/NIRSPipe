"""The share of each long channel's variance the fitted short-channel regressors carried.

The closed-form cases build each channel as a part orthogonal to the regressors plus a known
multiple of them, so the share is that multiple's variance over the whole, worked by hand.
The run case checks the record against the regression's own residual file: with only the
short-channel columns and a constant in the design, the share is 1 - var(residual) / var(input).
"""

from __future__ import annotations

import json

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.cli.run import main
from nirspipe.io.derivatives import resolve_bids_uri
from nirspipe.io.snirf import read_snirf
from nirspipe.qc.subject.record_io import read_record
from nirspipe.qc.subject.sqm_record import short_regression_shares
from tests._synth import N_LONG_PAIRS, make_bids_dataset

N = 2000


def _orthonormal(n_cols: int, seed: int = 0) -> np.ndarray:
    """Columns of mean 0 and variance 1, orthogonal to each other."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(np.column_stack([np.ones(N), rng.standard_normal((N, n_cols))]))
    return q[:, 1:] * np.sqrt(N)


def _raw(series: dict[str, np.ndarray]) -> mne.io.RawArray:
    names = list(series)
    info = mne.create_info(names, 10.0, [n.rsplit(" ", 1)[1] for n in names])
    return mne.io.RawArray(np.vstack([series[n] for n in names]), info, verbose=False)


def _betas(thetas: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"Condition": cond, "ch_name": ch, "theta": theta, "bad": False}
                         for ch, by_cond in thetas.items() for cond, theta in by_cond.items()])


def test_the_share_is_the_fitted_short_part_over_the_channel():
    s1, s2, c1, c2 = _orthonormal(4).T
    design = pd.DataFrame({"short_ch_pca01": s1, "short_ch_pca02": s2,
                           "constant": np.ones(N), "censored": np.zeros(N, int)})
    # S1_D1 hbo: 1.0 * s1 + 0.5 * s2 over a unit-variance rest -> 1.25 / 2.25
    # S1_D1 hbr: 0.5 * s2 over a rest of variance 4 -> 0.25 / 4.25
    raw = _raw({"S1_D1 hbo": c1 + s1 + 0.5 * s2 + 3.0, "S1_D1 hbr": 2 * c2 + 0.5 * s2})
    betas = _betas({"S1_D1 hbo": {"short_ch_pca01": 1.0, "short_ch_pca02": 0.5, "constant": 3.0},
                    "S1_D1 hbr": {"short_ch_pca01": 0.0, "short_ch_pca02": 0.5, "constant": 0.0}})
    got = short_regression_shares(design, betas, raw)
    assert got == pytest.approx({"S1_D1 hbo": 1.25 / 2.25, "S1_D1 hbr": 0.25 / 4.25})


def test_frames_the_fit_left_out_are_left_out():
    censored = np.zeros(N, int)
    censored[:100] = 1
    kept = censored == 0
    # orthogonal and of equal variance over the kept frames, so the share there is 1 / 2;
    # the left-out frames hold values that would swamp it
    q, _ = np.linalg.qr(np.column_stack([np.ones(kept.sum()), np.random.default_rng(1)
                                         .standard_normal((kept.sum(), 2))]))
    s1, c1 = np.full(N, 50.0), np.full(N, -80.0)
    s1[kept], c1[kept] = q[:, 1], q[:, 2]
    design = pd.DataFrame({"short_ch_mean": s1, "constant": np.ones(N), "censored": censored})
    raw = _raw({"S1_D1 hbo": c1 + s1})
    got = short_regression_shares(design, _betas({"S1_D1 hbo": {"short_ch_mean": 1.0,
                                                                "constant": 0.0}}), raw)
    assert got == pytest.approx({"S1_D1 hbo": 0.5})


def test_hbt_is_read_off_the_pair_and_a_self_fitted_channel_is_left_out():
    s1, c1, c2 = _orthonormal(3).T
    design = pd.DataFrame({"short_ch_pca01": s1, "constant": np.ones(N),
                           "censored": np.zeros(N, int)})
    # HbO = c1 + s1, HbR = c2 + 0.5 s1, so HbT = c1 + c2 + 1.5 s1 -> 2.25 / 4.25
    raw = _raw({"S1_D1 hbo": c1 + s1, "S1_D1 hbr": c2 + 0.5 * s1, "S9_D9 hbo": s1})
    betas = _betas({"S1_D1 hbt": {"short_ch_pca01": 1.5, "constant": 0.0},
                    "S9_D9 hbo": {"short_ch_pca01": np.nan, "constant": np.nan}})
    assert short_regression_shares(design, betas, raw) == pytest.approx({"S1_D1 hbt": 2.25 / 4.25})


def test_a_design_without_short_channels_gives_nothing():
    design = pd.DataFrame({"constant": np.ones(N), "censored": np.zeros(N, int)})
    raw = _raw({"S1_D1 hbo": _orthonormal(1)[:, 0]})
    assert short_regression_shares(design, _betas({"S1_D1 hbo": {"constant": 0.0}}), raw) == {}


# ---- a denoise run on the synthetic montage ----

SHORT_PAIR = f"S{N_LONG_PAIRS + 1}_D{N_LONG_PAIRS + 1}"
LABEL = "sub-01_task-rest"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    bids = make_bids_dataset(tmp_path_factory.mktemp("share_bids"), subjects=("01",),
                             tasks=("rest",))
    out = tmp_path_factory.mktemp("share_out")
    main([str(bids), str(out), "participant", "--participant-label", "01", "--dpf", "6",
          "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
          "--resp-l-freq", "0.1", "--resp-h-freq", "0.4", "--mode", "denoise",
          "--short-channel", "mean", "--noise-model", "ols", "--skip-bids-validation"])
    return out / "sub-01" / "nirs"


def test_the_record_carries_each_long_channels_share_as_its_residual_says(run):
    record = read_record(run / f"{LABEL}_desc-sqm_qc.json")
    shares = record["per_channel"]["short_regression"]["short_share_per_channel"]
    errts = run / f"{LABEL}_desc-errts_nirs.snirf"
    source = json.loads(errts.with_suffix(".json").read_text(encoding="utf-8"))["Sources"][0]
    y, e = read_snirf(resolve_bids_uri(source, errts)), read_snirf(errts)
    expected = {ch: 1 - e.get_data(picks=[ch])[0].var() / y.get_data(picks=[ch])[0].var()
                for ch in y.ch_names if not ch.startswith(SHORT_PAIR)}
    assert shares == pytest.approx(expected, abs=1e-9)
    bad = set(json.loads((run / f"{LABEL}_design.json").read_text())["bad_channels"])
    for chroma in ("hbo", "hbr"):
        kept = [v for ch, v in shares.items() if ch.endswith(chroma) and ch not in bad]
        assert record["short_regression"][f"short_share_{chroma}"] == pytest.approx(np.mean(kept))


def test_the_channel_table_splits_the_share_by_chromophore(run):
    table = pd.read_csv(run / f"{LABEL}_desc-channel_qc.tsv", sep="\t")
    shares = read_record(run / f"{LABEL}_desc-sqm_qc.json")["per_channel"][
        "short_regression"]["short_share_per_channel"]
    for _, row in table.iterrows():
        pair = row["name"].split(" ")[0]
        if pair == SHORT_PAIR:
            assert np.isnan(row["short_share_hbo"]) and np.isnan(row["short_share_hbr"])
        else:
            assert row["short_share_hbo"] == pytest.approx(shares[f"{pair} hbo"])
            assert row["short_share_hbr"] == pytest.approx(shares[f"{pair} hbr"])


def test_the_report_shows_the_map_the_rows_and_the_column(run):
    html = (run.parent / f"{LABEL}_report.html").read_text(encoding="utf-8")
    assert "Variance carried by the short-channel regressors" in html
    assert "Short-channel share HbO" in html and "Short-channel share HbR" in html
    assert "<th>Short-channel share" in html
