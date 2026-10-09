"""The GLM's AR coefficients are each channel's own estimate, not snapped to a grid set by the channel count."""

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.pipeline.glm import fit_glm

RHOS = (0.55, 0.72, 0.90)


def _ar1_channels(n=3000, sfreq=10.0, seed=0):
    rng = np.random.default_rng(seed)
    data = np.zeros((len(RHOS), n))
    for c, rho in enumerate(RHOS):
        noise = rng.standard_normal(n)
        for t in range(1, n):
            data[c, t] = rho * data[c, t - 1] + noise[t]
    names = [f"S{i}_D{i} hbo" for i in range(1, len(RHOS) + 1)]
    raw = mne.io.RawArray(data * 1e-6, mne.create_info(names, sfreq, ch_types="hbo"),
                          verbose=False)
    return raw, pd.DataFrame({"constant": np.ones(n)})


def test_an_ar1_fit_keeps_each_channel_s_own_coefficient():
    raw, design = _ar1_channels()
    est = fit_glm(raw, design, noise_model="ar1")
    fitted = [float(est._data[name].model.rho[0]) for name in raw.ch_names]
    # one bin per channel would give 0.33, 0.67, 0.67 here
    assert fitted == pytest.approx(RHOS, abs=0.03)


def test_a_higher_order_fit_still_gives_every_channel_its_own_coefficients():
    raw, design = _ar1_channels()
    est = fit_glm(raw, design, noise_model="ar2")
    first = [float(est._data[name].model.rho[0]) for name in raw.ch_names]
    assert len(set(first)) == len(RHOS)
    assert first == pytest.approx(RHOS, abs=0.05)
