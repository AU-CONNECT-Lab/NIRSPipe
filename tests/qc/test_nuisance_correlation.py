"""The nuisance correlation figure: which series it takes, on which frames, and what it draws.

Each series is built as ``a * global + sqrt(1 - a^2) * e`` from columns orthogonal to each
other, so its correlation with the global signal is ``a`` exactly.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.cli.run import main
from nirspipe.io.auxiliary import ImuTrace
from nirspipe.qc.figures.subject.nuisance_figures import (nuisance_correlation_figure,
                                                          nuisance_series)
from tests._synth import make_bids_dataset

N = 1000


def _orthonormal(n_cols: int, seed: int = 0) -> np.ndarray:
    q, _ = np.linalg.qr(np.column_stack([np.ones(N), np.random.default_rng(seed)
                                         .standard_normal((N, n_cols))]))
    return q[:, 1:] * np.sqrt(N)


def _frame(weights: dict[str, float], seed: int = 0) -> pd.DataFrame:
    """A global HbO and one series per weight, each correlating with it at that weight."""
    cols = _orthonormal(len(weights) + 1, seed).T
    out = {"global HbO": cols[0]}
    for (name, a), e in zip(weights.items(), cols[1:]):
        out[name] = a * cols[0] + np.sqrt(1 - a ** 2) * e
    return pd.DataFrame(out)


def test_the_bars_are_each_series_r_with_the_global_signal_strongest_first():
    fig = nuisance_correlation_figure(_frame({"drift_1": 0.2, "short_ch_hbo_mean": -0.8,
                                              "GVTD": 0.5}))
    bars = [t for t in fig.data if t.type == "bar"]
    assert [b.name for b in bars] == ["global HbO"]
    assert list(bars[0].x) == ["short_ch_hbo_mean", "GVTD", "drift_1"]
    assert np.allclose(bars[0].y, [0.8, 0.5, 0.2])
    heat = next(t for t in fig.data if t.type == "heatmap")
    # the heatmap keeps the frame's order, signed, its diagonal blank
    assert list(heat.x) == ["global HbO", "drift_1", "short_ch_hbo_mean", "GVTD"]
    z = np.array(heat.z, dtype=float)
    assert z[0, 2] == pytest.approx(-0.8) and np.isnan(np.diag(z)).all()


def test_only_the_strongest_series_are_kept():
    weights = {f"drift_{i}": a for i, a in enumerate((0.1, 0.9, 0.3, 0.7, 0.5))}
    fig = nuisance_correlation_figure(_frame(weights), max_series=3)
    heat = next(t for t in fig.data if t.type == "heatmap")
    assert list(heat.x) == ["global HbO", "drift_1", "drift_3"]


def test_no_global_signal_draws_nothing():
    assert nuisance_correlation_figure(_frame({"drift_1": 0.5}).drop(columns="global HbO")) is None


def _haemo() -> mne.io.RawArray:
    names = ["S1_D1 hbo", "S2_D2 hbo", "S1_D1 hbr", "S2_D2 hbr"]
    rng = np.random.default_rng(3)
    raw = mne.io.RawArray(rng.standard_normal((4, N)), mne.create_info(
        names, 10.0, ["hbo", "hbo", "hbr", "hbr"]), verbose=False)
    raw.info["bads"] = ["S2_D2 hbo", "S2_D2 hbr"]
    return raw


def test_the_series_are_the_kept_frames_of_the_design_and_the_global_means():
    haemo = _haemo()
    censored = np.zeros(N, int)
    censored[:50] = 1
    design = pd.DataFrame({"Tapping": np.arange(N, dtype=float), "short_ch_pca01": np.sin(np.arange(N)),
                           "drift_1": np.cos(np.arange(N) / 50), "aux_1": np.arange(N) % 7,
                           "constant": np.ones(N), "censored": censored})
    imu = {"gyro": ImuTrace(np.linspace(0, haemo.times[-1], 50), np.arange(50.0), "deg/s")}
    frame = nuisance_series(haemo, design, imu=imu)
    # the task column and the intercept are not nuisance
    assert list(frame.columns) == ["global HbO", "global HbR", "short_ch_pca01", "drift_1",
                                   "aux_1", "gyro speed"]
    assert len(frame) == N - 50
    # a rejected channel is left out of the global signal
    np.testing.assert_allclose(frame["global HbO"], haemo.get_data(picks=["S1_D1 hbo"])[0][50:])
    # the sensor's straight line stays one on the new samples
    np.testing.assert_allclose(frame["gyro speed"],
                               haemo.times[50:] / haemo.times[-1] * 49, atol=1e-9)


def test_a_constant_series_is_dropped():
    haemo = _haemo()
    design = pd.DataFrame({"drift_1": np.ones(N), "censored": np.zeros(N, int)})
    assert "drift_1" not in nuisance_series(haemo, design).columns


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    bids = make_bids_dataset(tmp_path_factory.mktemp("nuis_bids"), subjects=("01",),
                             tasks=("rest",))
    out = tmp_path_factory.mktemp("nuis_out")
    main([str(bids), str(out), "participant", "--participant-label", "01", "--dpf", "6",
          "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
          "--resp-l-freq", "0.1", "--resp-h-freq", "0.4", "--mode", "denoise",
          "--short-channel", "mean", "--skip-bids-validation"])
    return out / "sub-01"


def test_the_run_report_draws_it_from_the_regression(report):
    html = (report / "sub-01_task-rest_report.html").read_text(encoding="utf-8")
    assert "<h3>Nuisance correlations</h3>" in html
    figure = next((report / "figures").glob("*nuisancecorr*.html")).read_text(encoding="utf-8")
    assert "short_ch_hbo_mean" in figure and "GVTD" in figure
