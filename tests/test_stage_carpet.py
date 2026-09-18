"""Stage carpet: one block per stage, each z-scored to itself, cut without rescaling."""

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


def _tick_labels(fig) -> tuple:
    """Tick text off whichever axis carries it; the carpet is not always the first row."""
    for key in dir(fig.layout):
        if key.startswith("yaxis") and getattr(fig.layout, key).ticktext:
            return tuple(getattr(fig.layout, key).ticktext)
    return ()


def test_one_block_per_stage():
    fig = carpet_compare_figure(
        [("desc-preproc", _raw()), ("desc-filtered", _raw()), ("desc-errts", _raw())],
        "hbo")
    assert len(_carpets(fig)) == 3
    titles = [a.text for a in fig.layout.annotations[:3]]
    assert titles[0] == "desc-preproc"
    # every later block names its own SD against the first stage's, beside its label
    assert titles[1].startswith("desc-filtered") and "SD " in titles[1]
    assert titles[2].startswith("desc-errts") and "desc-preproc" in titles[2]


def test_each_block_is_scaled_to_itself_and_says_by_how_much():
    """A stage 100x smaller still fills its own greyscale; the title carries the ratio."""
    fig = carpet_compare_figure(
        [("desc-preproc", _raw(1e-6)), ("desc-errts", _raw(1e-8))], "hbo")
    first, last = (np.asarray(c.z, dtype=float) for c in _carpets(fig))
    # scaled by the first stage's SD this block would sit near zero everywhere
    assert last.std() > 0.5 * first.std()
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
    assert _tick_labels(fig) == ("L", "R")


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


def test_both_chromophores_stack_in_one_block():
    """HbO and HbR share one image, one colour bar and one seam, like the motion carpet."""
    fig = carpet_compare_figure([("desc-errts", _raw(n_pairs=4))],
                                reference=("desc-preproc", _raw(1e-5, n_pairs=4)))
    carpets = _carpets(fig)
    assert len(carpets) == 1
    assert np.asarray(carpets[0].z).shape[0] == 8      # 4 HbO rows over 4 HbR rows
    title = fig.layout.annotations[0].text
    assert "HBO SD" in title and "HBR SD" in title and "desc-preproc" in title
    # one colour bar per chromophore plus the seam between them
    assert len([sh for sh in fig.layout.shapes if sh.type == "rect"]) == 2
    assert len([sh for sh in fig.layout.shapes if sh.type == "line"]) == 1


def test_the_stage_row_draws_the_reference_and_the_drawn_stage():
    """The before/after comparison lives on the trace, not on a second carpet."""
    fig = carpet_compare_figure([("desc-errts", _raw(1e-8))],
                                reference=("desc-preproc", _raw(1e-6)))
    lines = [t for t in fig.data if t.type == "scatter"]
    assert [t.name for t in lines] == ["desc-preproc", "desc-errts"]
    assert len(_carpets(fig)) == 1
