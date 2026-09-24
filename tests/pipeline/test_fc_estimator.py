"""The one estimator every FC product uses, and the seed map's shape.

FC is Pearson, not a shrinkage estimator that pulls correlations toward zero. These pin
that: all three products agree on one estimator, and it carries no bias that varies with the
shape of the recording.
"""

import numpy as np
import pytest

from fnirs_pipe.pipeline.restingstate import (
    compute_fc,
    compute_fc_roi,
    compute_fc_seed,
    fisher_z,
)


@pytest.fixture(scope="module")
def haemo():
    import mne

    from tests._synth import synth_raw

    raw = synth_raw("01", "rest", duration=200.0)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    return mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)


@pytest.fixture(scope="module")
def roi_map(haemo):
    hbo = [c for c in haemo.ch_names if c.endswith("hbo")]
    return {"A": hbo[:2], "B": hbo[2:4]}


# ---- the estimator ----
def test_fc_carries_no_bias_that_tracks_the_recording_shape():
    """A shrinkage estimator fails this: its pull toward zero grows as samples run out.

    Both shapes carry the same true connectivity, so a difference between them is the
    estimator rather than the data, and that difference is what reaches a group comparison.
    """
    import mne

    rng = np.random.default_rng(5)

    def fc_mean(n_ch, n_t, rho, reps=25):
        out = []
        for _ in range(reps):
            g = rng.standard_normal(n_t)
            d = np.array([rho * g + np.sqrt(1 - rho**2) * rng.standard_normal(n_t)
                          for _ in range(n_ch)])
            names = [f"S{i}_D{i} hbo" for i in range(n_ch)]
            info = mne.create_info(names, 10.0, ["hbo"] * n_ch)
            fc = compute_fc(mne.io.RawArray(d, info, verbose="error"), "hbo").to_numpy()
            out.append(fc[~np.eye(n_ch, dtype=bool)].mean())
        return float(np.mean(out))

    true_r = 0.2**2
    long_run, short_run = fc_mean(20, 3000, 0.2), fc_mean(50, 300, 0.2)

    assert long_run == pytest.approx(true_r, abs=0.01)
    assert short_run == pytest.approx(true_r, abs=0.01)


def test_every_fc_product_uses_the_same_estimator(haemo, roi_map):
    hbo = [c for c in haemo.ch_names if c.endswith("hbo")]
    fc = compute_fc(haemo, "hbo")
    seed = compute_fc_seed(haemo, {"one": [hbo[0]]}, "hbo")

    # a one-channel ROI has that channel as its own mean, so its seed row is a row of `fc`
    expected = fc.loc[hbo[0]].drop(hbo[0])
    assert np.allclose(seed.loc["one"].dropna(), expected.to_numpy())


# ---- the seed map ----
def test_the_seed_map_is_roi_by_channel(haemo, roi_map):
    seed = compute_fc_seed(haemo, roi_map, "hbo")
    hbo = [c for c in haemo.ch_names if c.endswith("hbo")]

    assert list(seed.index) == ["A", "B"]
    assert list(seed.columns) == hbo


def test_a_seed_blanks_its_own_channels_by_membership_not_by_value(haemo, roi_map):
    seed = compute_fc_seed(haemo, roi_map, "hbo")

    for roi, chans in roi_map.items():
        assert seed.loc[roi, chans].isna().all()
        assert not seed.loc[roi].drop(chans).isna().any()


def test_the_seed_map_is_not_a_slice_of_either_matrix(haemo, roi_map):
    """Averaging on one side only: the same pair reads differently at each level."""
    seed = compute_fc_seed(haemo, roi_map, "hbo")
    fc_roi = compute_fc_roi(haemo, roi_map, "hbo")

    seed_a_to_b = seed.loc["A", roi_map["B"]].mean()
    assert abs(seed_a_to_b - fc_roi.loc["A", "B"]) > 1e-6


def test_fisher_z_leaves_a_rectangular_frame_alone(haemo, roi_map):
    """`fill_diagonal` on an ROI x channel frame would delete real values at (i, i)."""
    seed = compute_fc_seed(haemo, roi_map, "hbo")
    z = fisher_z(seed)

    assert z.shape == seed.shape
    assert not (z.to_numpy()[np.eye(*z.shape, dtype=bool)] == 0).all()
    assert np.allclose(np.diag(fisher_z(compute_fc(haemo, "hbo")).to_numpy()), 0.0)
