"""mALFF and GLM betas on the optode flat map, the stand-ins for the cortex views of an unplaced run."""

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.qc.figures import activation_flat_figure, alff_flat_figure
from tests._synth import N_LONG_PAIRS, synth_raw


@pytest.fixture(scope="module")
def haemo():
    od = mne.preprocessing.nirs.optical_density(synth_raw("01", "rest", seed=0))
    return mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)


def _scales(fig) -> list[tuple[float, float, set]]:
    """``(cmin, cmax, colours)`` of every trace that carries a colour scale."""
    return [(t.marker.cmin, t.marker.cmax, set(np.round(np.asarray(t.marker.color, float), 6)))
            for t in fig.data if t.marker is not None and t.marker.cmin is not None]


def test_each_chromophore_head_runs_over_its_own_values(haemo):
    # HbO in [0.5, 1.0), HbR in [1.2, 1.7), so a head's range names its chromophore
    rng = np.random.default_rng(1)
    values = {ch: (rng.uniform(0.5, 1.0) if ch.endswith("hbo") else rng.uniform(1.2, 1.7))
              for ch in haemo.ch_names}
    fig = alff_flat_figure(haemo, pd.DataFrame({"channel": list(values),
                                                "malff": list(values.values())}))
    # the map draws long channels only, as the seed maps do; the synthetic long pairs come first
    long_pairs = {f"S{k}_D{k}" for k in range(1, N_LONG_PAIRS + 1)}
    for chroma, lo in (("hbo", 0.5), ("hbr", 1.2)):
        own = [v for ch, v in values.items() if ch.endswith(chroma) and ch.split()[0] in long_pairs]
        heads = [s for s in _scales(fig) if lo <= s[0] < lo + 0.5]
        assert {(cmin, cmax) for cmin, cmax, _ in heads} == {(min(own), max(own))}
        assert set().union(*(c for *_, c in heads)) <= set(np.round(own, 6))


def test_betas_share_one_symmetric_scale_per_chromophore(haemo):
    rng = np.random.default_rng(2)
    channels = list(haemo.ch_names)
    results = {cond: pd.DataFrame({"ch_name": channels, "Coef.": rng.normal(0, 1e-7, len(channels))})
               for cond in ("left", "right")}
    fig = activation_flat_figure(haemo, results)
    expected = set()
    for chroma in ("hbo", "hbr"):
        betas = np.concatenate([df.loc[df["ch_name"].str.endswith(chroma), "Coef."].to_numpy()
                                for df in results.values()])
        v = np.percentile(np.abs(betas), 99.5) * 1e6
        expected.add((round(-v, 9), round(v, 9)))
    assert {(round(cmin, 9), round(cmax, 9)) for cmin, cmax, _ in _scales(fig)} == expected


def test_an_unplaced_montage_is_drawn_as_laid_out_with_no_head(haemo):
    from nirspipe.io.optode_space import Placement, hold_placement
    from nirspipe.qc.figures.common.topomap import _projected_optodes

    raw = hold_placement(haemo.copy(), Placement("none", note="the file has no 3-D optode positions"))
    opt_xy, _, outlines = _projected_optodes(raw)
    assert outlines == {}
    # each optode at its own recorded x and y, untouched by any head projection
    for ch in raw.info["chs"]:
        s = ch["ch_name"].split("_")[0]
        np.testing.assert_allclose(opt_xy[s], ch["loc"][3:5], atol=1e-12)
    fig = alff_flat_figure(raw, pd.DataFrame({"channel": raw.ch_names,
                                              "malff": np.linspace(0.5, 1.5, len(raw.ch_names))}))
    # no outline trace: only the skeleton and the glyphs
    assert all(t.mode != "lines" or t.line.color != "#c9d2da" for t in fig.data)
