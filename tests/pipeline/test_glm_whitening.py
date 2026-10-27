"""The whitening diagnostic: spectrum against white, lag-1 autocorrelation, and where they land.

Expected values come from theory (white noise and the AR(1) spectral density), from hand-worked
inputs, or from planted tables, never from the module's own helpers.
"""

import mne
import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from nirspipe.io.tables import read_table, write_tsv
from nirspipe.pipeline.glm import run_glm_pipeline
from nirspipe.pipeline.glm_whitening import band_mean, lag1_autocorrelation, residual_spectrum
from nirspipe.qc.figures import design_peak_frequencies, whitening_figure
from nirspipe.qc.subject.sqm_record import _whitening_section
from nirspipe.utils.lineage import stamp
from tests._synth import synth_raw

SFREQ = 10.0
FIT = dict(stim_dur=10.0, hrf_model="spm", drift_model="polynomial",
           high_pass=None, drift_order=1, fir_delays=None)


def _ar1(rho, n, seed):
    return lfilter([1.0], [1.0, -rho], np.random.default_rng(seed).normal(size=n))


# ---- the spectrum ----

def test_white_noise_reads_one_at_every_frequency():
    freqs, ratio = residual_spectrum(np.random.default_rng(0).normal(size=40000), SFREQ)
    assert abs(ratio.mean() - 1.0) < 0.02
    assert abs(band_mean(freqs, ratio, (0.01, 0.1)) - 1.0) < 0.15


def test_an_ar1_residual_follows_its_spectral_density():
    rho = 0.8
    freqs, ratio = residual_spectrum(_ar1(rho, 200000, 1), SFREQ)
    # AR(1) density over its own variance: (1 - rho^2) / |1 - rho e^{-iw}|^2
    theory = (1 - rho ** 2) / (1 - 2 * rho * np.cos(2 * np.pi * freqs / SFREQ) + rho ** 2)
    assert np.median(np.abs(ratio / theory - 1)) < 0.1
    # near zero frequency, (1 + rho) / (1 - rho)
    assert band_mean(freqs, ratio, (0.01, 0.05)) == pytest.approx(9.0, rel=0.15)


def test_the_zero_and_nyquist_bins_are_left_out():
    freqs, _ = residual_spectrum(np.random.default_rng(2).normal(size=5000), SFREQ)
    assert freqs[0] > 0 and freqs[-1] < SFREQ / 2


def test_a_band_with_no_bin_is_nan():
    assert np.isnan(band_mean(np.array([0.5, 1.0]), np.array([1.0, 1.0]), (0.01, 0.1)))


# ---- lag-1 autocorrelation ----

def test_lag1_on_a_hand_worked_series():
    # deviations -1.5 -0.5 0.5 1.5: lagged products 0.75 - 0.25 + 0.75 over squares 5
    assert lag1_autocorrelation([1, 2, 3, 4]) == pytest.approx(0.25)


def test_lag1_recovers_an_ar1_coefficient():
    assert lag1_autocorrelation(_ar1(0.6, 100000, 3)) == pytest.approx(0.6, abs=0.01)


# ---- the GLM step writes it, and the noise model shows in it ----

@pytest.fixture(scope="module")
def haemo():
    raw = synth_raw("01", "tapping", duration=600.0, n_long_pairs=4, bad_pair=None,
                    motion_onset=None)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    haemo = mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)
    # strongly autocorrelated noise, so there is something for whitening to remove
    scale = haemo.get_data().std(axis=1, keepdims=True)
    haemo._data += 5 * scale * np.array([_ar1(0.95, haemo.n_times, 10 + i)
                                         for i in range(len(haemo.ch_names))])
    return stamp(haemo, stage="preproc", step="beer_lambert")


@pytest.fixture(scope="module")
def written(haemo, tmp_path_factory):
    out = {}
    for model in ("ols", "auto"):
        nirs_dir = tmp_path_factory.mktemp(model) / "sub-01" / "nirs"
        nirs_dir.mkdir(parents=True)
        source = str(nirs_dir / "sub-01_task-tapping_desc-preproc_nirs.snirf")
        _, _, dm, _ = run_glm_pipeline(haemo, noise_model=model, output_dir=str(nirs_dir),
                                       source_path=source, **FIT)
        out[model] = (nirs_dir, dm)
    return out


def _channels(written, model):
    return read_table(written[model][0] / "sub-01_task-tapping_desc-whitening_nirsmap.tsv")


def test_both_tables_are_written_with_sidecars(written):
    nirs_dir = written["auto"][0]
    for name in ("sub-01_task-tapping_desc-whitening_nirsmap",
                 "sub-01_task-tapping_stat-psd_desc-whitening_timeseries"):
        assert (nirs_dir / f"{name}.tsv").exists() and (nirs_dir / f"{name}.json").exists()


def test_ols_leaves_the_residual_as_it_is(written):
    # whitening is the identity under ols, so both columns describe one series
    table = _channels(written, "ols").dropna(subset=["acf1_whitened"])
    np.testing.assert_allclose(table["acf1_whitened"], table["acf1_residual"], rtol=1e-9)


def test_an_ar_model_brings_the_low_band_towards_white(written):
    ols = _channels(written, "ols")["lowband_whitened"].median()
    auto = _channels(written, "auto")["lowband_whitened"].median()
    assert ols > 3
    assert abs(auto - 1) < abs(ols - 1) / 3


def test_a_confound_regression_writes_no_diagnostic(haemo, tmp_path):
    nirs_dir = tmp_path / "sub-01" / "nirs"
    nirs_dir.mkdir(parents=True)
    haemo = haemo.copy().set_annotations(None)
    run_glm_pipeline(haemo, noise_model="ols", output_dir=str(nirs_dir),
                     source_path=str(nirs_dir / "sub-01_task-tapping_desc-preproc_nirs.snirf"),
                     **FIT)
    assert not list(nirs_dir.glob("*whitening*"))


# ---- the record and the figure read it back ----

def test_the_record_takes_the_median_of_good_long_channels(tmp_path):
    write_tsv(pd.DataFrame({
        "ch_name": ["S1_D1 hbo", "S2_D2 hbo", "S3_D3 hbo", "S4_D4 hbo", "S5_D5 hbo", "S1_D1 hbr"],
        "Chroma": ["hbo"] * 5 + ["hbr"],
        "long": [True, True, True, True, False, True],
        "bad": [False, False, False, True, False, False],
        "lowband_whitened": [1.0, 2.0, 4.0, 100.0, 50.0, 0.8],
    }), tmp_path / "sub-01_task-x_desc-whitening_nirsmap.tsv")
    stages = {"errts": tmp_path / "sub-01_task-x_desc-errts_nirs.snirf"}
    # the bad channel and the short one are out, so the median of 1, 2 and 4
    assert _whitening_section(stages) == {"whitening_lowband_hbo": 2.0,
                                          "whitening_lowband_hbr": 0.8}


def test_a_run_without_the_table_has_no_section(tmp_path):
    assert _whitening_section({"errts": tmp_path / "sub-01_task-x_desc-errts_nirs.snirf"}) == {}


def test_the_design_peak_is_the_block_rate(written):
    # tapping blocks start every 40 s
    peaks = design_peak_frequencies(written["auto"][1], ["tapping"])
    assert peaks["tapping"] == pytest.approx(1 / 40, abs=0.002)


def test_the_figure_draws_the_chromophores_the_tables_hold():
    spectrum = pd.DataFrame({"freq": np.linspace(0.01, 1.0, 20),
                             **{f"{c}_{kind}": np.ones(20) for c in ("hbo", "hbt")
                                for kind in ("whitened", "residual")}})
    channels = pd.DataFrame({"ch_name": ["S1_D1 hbo", "S1_D1 hbt"], "Chroma": ["hbo", "hbt"],
                             "long": True, "bad": False, "n_samples": 100,
                             "acf1_whitened": [0.1, 0.2]})
    names = {t.name for t in whitening_figure(spectrum, channels, {}, (0.01, 0.1)).data}
    assert {"HbO whitened", "HbT whitened"} <= names and "HbR whitened" not in names


def test_the_shaded_bands_draw_no_corner_dots():
    # a scatter of under 20 points gets markers by default, which read as data
    spectrum = pd.DataFrame({"freq": np.linspace(0.01, 1.0, 20),
                             **{f"hbo_{kind}": np.ones(20) for kind in ("whitened", "residual")}})
    channels = pd.DataFrame({"ch_name": ["S1_D1 hbo", "S2_D1 hbo"], "Chroma": "hbo",
                             "long": True, "bad": False, "n_samples": 100,
                             "acf1_whitened": [0.1, 0.2]})
    bands = [t for t in whitening_figure(spectrum, channels, {}, (0.01, 0.1)).data
             if t.fill == "toself"]
    assert len(bands) == 2 and all(t.mode == "lines" for t in bands)


def test_the_figure_draws_both_panels(written):
    nirs_dir, dm = written["auto"]
    fig = whitening_figure(
        read_table(nirs_dir / "sub-01_task-tapping_stat-psd_desc-whitening_timeseries.tsv"),
        _channels(written, "auto"), design_peak_frequencies(dm, ["tapping"]), (0.01, 0.1))
    names = {t.name for t in fig.data}
    assert {"HbO whitened", "HbR whitened", "HbO before whitening", "white"} <= names
    assert any(t.mode == "markers" for t in fig.data)
