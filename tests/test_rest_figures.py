"""The two rest-mode figures whose numbers already existed with nothing drawing them.

`_fcroi.tsv` had been written since ROI FC landed and no panel showed it; ALFF had a bar
chart ordered by channel name, which cannot be read as the spatial claim it is. Both figures
return a base64 PNG, so what is asserted is the geometry that carries the meaning and the
refusals: no frames and no optode positions must give None rather than an exception, because
the report treats None as "skip the panel" and an exception as a broken report.

The seed map's own route is pinned in test_rest_seed_figure.
"""

import base64
import io

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.qc.figures.rest_figures import alff_topo_figure, fc_roi_matrix_figure


def _png_size(b64: str) -> tuple[int, int]:
    """Width and height straight out of the PNG header, without pulling in an image library."""
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    with io.BytesIO(raw[16:24]) as buf:
        return (int.from_bytes(buf.read(4), "big"), int.from_bytes(buf.read(4), "big"))


def _roi_frame(labels):
    rng = np.random.default_rng(0)
    m = rng.uniform(-1, 1, (len(labels), len(labels)))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    return pd.DataFrame(m, index=labels, columns=labels)


# ---- ROI to ROI ----

def test_the_roi_matrix_draws_both_chromophores():
    frame = _roi_frame(["PFC", "TPJ", "M1"])
    one = fc_roi_matrix_figure({"hbo": frame})
    two = fc_roi_matrix_figure({"hbo": frame, "hbr": frame})
    assert _png_size(two)[0] > _png_size(one)[0]      # a second panel, so a wider figure


def test_the_roi_matrix_declines_rather_than_raises():
    assert fc_roi_matrix_figure({}) is None
    assert fc_roi_matrix_figure({"hbo": pd.DataFrame()}) is None


# ---- ALFF on the layout ----

@pytest.fixture(scope="module")
def haemo():
    import mne

    from ._synth import synth_raw

    raw = synth_raw("01", "rest", duration=40.0, motion_onset=None)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    return mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)


@pytest.fixture
def alff_df(haemo):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "channel": haemo.ch_names,
        "alff": rng.uniform(0.1, 1.0, len(haemo.ch_names)),
        "falff": rng.uniform(0.1, 0.9, len(haemo.ch_names)),
    })


def test_the_flat_map_is_drawn_from_the_montage(haemo, alff_df):
    assert alff_topo_figure(haemo, alff_df) is not None


def test_no_optode_positions_means_no_figure(haemo, alff_df):
    """The bar chart still works without coordinates; this panel cannot, and says so."""
    flat = haemo.copy()
    for ch in flat.info["chs"]:
        ch["loc"][:] = 0.0
    assert alff_topo_figure(flat, alff_df) is None


def test_a_rejected_channel_does_not_set_the_colour_scale(haemo, alff_df):
    """One dead channel with a runaway amplitude would otherwise flatten every real difference.

    Drawn twice, once with that channel rejected. The scale is taken from the good channels
    alone, so the two figures differ; without that the outlier would set vmax in both.
    """
    spiked = alff_df.copy()
    spiked.loc[spiked.index[0], "alff"] = 1e6
    normal = alff_topo_figure(haemo, spiked)

    marked = haemo.copy()
    marked.info["bads"] = [str(spiked.iloc[0]["channel"])]
    assert alff_topo_figure(marked, spiked) != normal
