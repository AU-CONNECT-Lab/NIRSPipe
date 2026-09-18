"""Stage carpet: one block per stage, all z-scored by the first stage, cut without rescaling."""

import mne
import numpy as np
import pytest

from fnirs_pipe.qc.figures.subject.carpet_compare import carpet_compare_figure


def _raw(scale: float = 1e-6, n_pairs: int = 4, sfreq: float = 5.0, n: int = 600,
         drop: int = 0, ramp: bool = False) -> mne.io.Raw:
    """HbO/HbR pairs of white noise. ``drop`` removes that many pairs from the end."""
    rng = np.random.default_rng(1)
    pairs = [f"S1_D{i + 1}" for i in range(n_pairs - drop)]
    names = [f"{p} {c}" for p in pairs for c in ("hbo", "hbr")]
    info = mne.create_info(names, sfreq, ["hbo", "hbr"] * len(pairs))
    data = rng.standard_normal((len(names), n)) * scale
    if ramp:  # loud first half, quiet second, for the cut-versus-scale check
        data[:, n // 2:] *= 0.05
    return mne.io.RawArray(data, info, verbose=False)


def _carpets(fig) -> list:
    return [t for t in fig.data if t.type == "heatmap"]


def test_one_block_per_stage():
    fig = carpet_compare_figure(
        [("desc-preproc", _raw()), ("desc-filtered", _raw()), ("desc-errts", _raw())], "hbo")
    assert len(_carpets(fig)) == 3
    titles = [a.text for a in fig.layout.annotations[:3]]
    assert titles[0] == "desc-preproc"
    # every later block names its own SD against the first stage's, beside its label
    assert titles[1].startswith("desc-filtered") and "SD " in titles[1]
    assert titles[2].startswith("desc-errts") and "desc-preproc" in titles[2]


def test_every_block_is_scaled_by_the_first_stage():
    """A stage 100x smaller renders pale, and its title says by how much."""
    fig = carpet_compare_figure(
        [("desc-preproc", _raw(1e-6)), ("desc-errts", _raw(1e-8))], "hbo")
    first, last = (np.asarray(c.z, dtype=float) for c in _carpets(fig))
    # scaled to itself the second block would come back to the first's spread
    assert last.std() < 0.2 * first.std()
    assert "SD 0.01×" in [a.text for a in fig.layout.annotations[:2]][1]


def test_channels_missing_from_a_later_stage_are_dropped_everywhere():
    """A row must mean the same channel in every block, so the set is the intersection."""
    fig = carpet_compare_figure(
        [("desc-preproc", _raw(n_pairs=4)), ("desc-errts", _raw(n_pairs=4, drop=1))], "hbo")
    assert all(np.asarray(c.z).shape[0] == 3 for c in _carpets(fig))


def test_roi_map_in_hbo_names_also_orders_the_hbr_carpet():
    """One map serves both chromophores: it is matched on the S-D base as well as the name."""
    roi_map = {"L": ["S1_D1 hbo", "S1_D2 hbo"], "R": ["S1_D3 hbo", "S1_D4 hbo"]}
    fig = carpet_compare_figure([("desc-preproc", _raw()), ("desc-errts", _raw())],
                                "hbr", roi_map=roi_map)
    assert tuple(fig.layout.yaxis.ticktext) == ("L", "R")


def test_cut_narrows_the_view_without_rescaling_it():
    """The quiet half stays quiet: the z-scoring is the run's, only the columns are cut."""
    stages = [("desc-preproc", _raw(ramp=True)), ("desc-errts", _raw(ramp=True))]
    whole = np.asarray(_carpets(carpet_compare_figure(stages, "hbo"))[0].z, dtype=float)
    quiet = np.asarray(
        _carpets(carpet_compare_figure(stages, "hbo", xlim=(61.0, 119.0)))[0].z, dtype=float)
    assert quiet.shape[1] < whole.shape[1]
    # rescaled to its own SD this would come back to about the whole run's spread
    assert quiet.std() < 0.3 * whole.std()


def test_no_channels_of_that_chromophore_returns_none():
    assert carpet_compare_figure([("desc-preproc", _raw())], "nope") is None
