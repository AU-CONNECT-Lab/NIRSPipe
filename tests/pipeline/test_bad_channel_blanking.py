"""A rejected channel keeps its place in every per-channel product and loses its value.

Blanking rather than dropping is the whole point: every subject's matrix keeps the same
shape and the same channel order, so a group analysis can stack subjects whose rejections
differ. Dropping rows would make position mean something different in every file. It is the
convention `compute_isc` already followed before the resting-state products caught up.

The failure these guard against is silent: before, a rejected channel carried an ordinary
looking correlation into `fc.tsv`, the figures drew it like any other, and a group analysis
reading the table had no way to tell. So each test asserts both halves (the cell is blank,
and the cells around it are not) because a function that blanked everything would pass on
the first half alone.

The input helpers are here too. A rejection is only as good as the naming that produced it,
and `_expand_bad_pairs` is where one wavelength has to become both.
"""

import mne
import numpy as np
import pytest

from fnirs_pipe.pipeline.prep_pipeline import _expand_bad_pairs
from fnirs_pipe.pipeline.restingstate import compute_alff, compute_fc, compute_fc_seed

SFREQ = 10.0
DURATION = 60.0
BAND = dict(high_pass=0.01, low_pass=0.08)


def _haemo(n_pairs: int = 4, seed: int = 0) -> mne.io.Raw:
    """Haemoglobin Raw with n_pairs long S-D pairs, HbO and HbR each."""
    rng = np.random.default_rng(seed)
    names, types = [], []
    for idx in range(n_pairs):
        for chromo in ("hbo", "hbr"):
            names.append(f"S{idx + 1}_D{idx + 1} {chromo}")
            types.append(chromo)
    info = mne.create_info(names, SFREQ, types)
    data = 1e-6 * rng.normal(size=(len(names), int(SFREQ * DURATION)))
    return mne.io.RawArray(data, info, verbose="error")


@pytest.fixture(scope="module")
def haemo():
    return _haemo()


REJECTED = "S2_D2 hbo"
GOOD = "S1_D1 hbo"


def _with_bads(raw, bads):
    out = raw.copy()
    out.info["bads"] = list(bads)
    return out


# ---- the channel correlation matrix ----

def test_a_rejected_channel_blanks_its_row_and_its_column(haemo):
    fc = compute_fc(_with_bads(haemo, [REJECTED]), "hbo")

    assert fc.loc[REJECTED].isna().all()
    assert fc[REJECTED].isna().all()


def test_the_matrix_keeps_its_shape_and_order(haemo):
    """The reason for blanking rather than dropping: position has to mean one thing."""
    clean = compute_fc(haemo, "hbo")
    marked = compute_fc(_with_bads(haemo, [REJECTED]), "hbo")

    assert list(marked.index) == list(clean.index)
    assert list(marked.columns) == list(clean.columns)


def test_the_good_channels_keep_the_correlations_they_had(haemo):
    """Blanking is not recomputation: rejecting one channel must not move another's value."""
    clean = compute_fc(haemo, "hbo")
    marked = compute_fc(_with_bads(haemo, [REJECTED]), "hbo")

    good = [c for c in clean.columns if c != REJECTED]
    np.testing.assert_allclose(marked.loc[good, good].to_numpy(),
                               clean.loc[good, good].to_numpy())


def test_the_other_chromophore_is_untouched_by_an_hbo_rejection(haemo):
    """The two matrices are separate products; a name in `bads` selects one of them."""
    fc_hbr = compute_fc(_with_bads(haemo, [REJECTED]), "hbr")
    assert fc_hbr.notna().all().all()


def test_rejecting_every_channel_leaves_a_frame_rather_than_raising(haemo):
    """A run this bad has to reach the report as an empty panel, not as a traceback."""
    names = [c for c in haemo.ch_names if c.endswith(" hbo")]
    fc = compute_fc(_with_bads(haemo, names), "hbo")

    assert fc.shape == (len(names), len(names))
    assert fc.isna().all().all()


# ---- the seed map ----

ROI_MAP = {"left": ["S1_D1", "S2_D2"], "right": ["S3_D3", "S4_D4"]}


def test_a_rejected_channel_is_blank_in_every_seed_row(haemo):
    """Blank whether or not it was listed in the seed, which are two different reasons.

    `S2_D2` is inside "left", so its cell there was already blank by membership. Its cell in
    "right" is the one this is about: it never entered that average, and before the change it
    carried an ordinary correlation.
    """
    seed = compute_fc_seed(_with_bads(haemo, [REJECTED]), ROI_MAP, "hbo")

    assert REJECTED in seed.columns, "the column stays, only the value goes"
    assert seed[REJECTED].isna().all()


def test_the_seed_keeps_every_column_and_the_rest_of_the_row(haemo):
    seed = compute_fc_seed(_with_bads(haemo, [REJECTED]), ROI_MAP, "hbo")
    hbo = [c for c in haemo.ch_names if c.endswith(" hbo")]

    assert list(seed.columns) == hbo
    # "right" averages S3_D3 and S4_D4, so its own two cells are blank and S1_D1 is not
    assert not np.isnan(seed.loc["right", GOOD])


def test_a_rejected_channel_does_not_enter_the_seed_average(haemo):
    """The companion to the blanking: rejecting a member has to change the seed itself."""
    spiked = haemo.copy()
    spiked._data[spiked.ch_names.index(REJECTED)] += 1e3

    included = compute_fc_seed(spiked, ROI_MAP, "hbo")
    excluded = compute_fc_seed(_with_bads(spiked, [REJECTED]), ROI_MAP, "hbo")

    assert not np.allclose(included.loc["left", GOOD], excluded.loc["left", GOOD])


# ---- ALFF, the fourth product that lists every channel ----

def test_all_four_alff_measures_go_blank_together(haemo):
    """Partial blanking would be worse than none: mALFF without ALFF reads as a real value."""
    frame = compute_alff(_with_bads(haemo, [REJECTED]), **BAND)
    row = frame.set_index("channel").loc[REJECTED]

    assert row["bad"]
    assert row[["alff", "falff", "malff", "zalff"]].isna().all()
    assert frame.set_index("channel").loc[GOOD, ["alff", "falff"]].notna().all()


# ---- naming a bad channel ----

def _od(n_pairs: int = 3) -> mne.io.Raw:
    names = [f"S{i + 1}_D{i + 1} {wl}" for i in range(n_pairs) for wl in ("760", "850")]
    info = mne.create_info(names, SFREQ, "fnirs_cw_amplitude")
    return mne.io.RawArray(np.zeros((len(names), 10)), info, verbose="error")


@pytest.mark.parametrize("label", ["S2_D2", "S2_D2 760", "S2_D2 850"])
def test_naming_a_pair_or_either_wavelength_marks_both(label):
    """Both wavelengths are one measurement, and Beer-Lambert turns them into HbO and HbR.

    Naming only 760 used to mark only 760, so after the conversion the pair's HbO was
    rejected and its HbR was not, and every HbR product kept a channel the operator had
    thrown out.
    """
    assert _expand_bad_pairs(_od(), [label]) == ["S2_D2 760", "S2_D2 850"]


def test_a_label_matching_nothing_yields_nothing():
    assert _expand_bad_pairs(_od(), ["S9_D9"]) == []


def test_bad_channels_can_differ_per_subject(tmp_path):
    """One list applies to everyone; a table gives each subject its own row.

    A cohort where only some caps slipped cannot be described by a single list, and passing
    the union would throw out channels that were fine on most subjects.
    """
    from fnirs_pipe.cli.workflows import _bad_channels_for

    table = tmp_path / "bads.tsv"
    table.write_text("participant_id\tbad_channels\nsub-01\tS1_D1,S2_D3\n02\tS4_D4\n",
                     encoding="utf-8")

    assert _bad_channels_for(str(table), "01") == ["S1_D1", "S2_D3"]
    assert _bad_channels_for(str(table), "sub-02") == ["S4_D4"]
    assert _bad_channels_for(str(table), "03") == []        # unlisted, not an error
    assert _bad_channels_for("S1_D1, S2_D3", "anyone") == ["S1_D1", "S2_D3"]
    assert _bad_channels_for(None, "01") == []


def test_a_table_missing_its_columns_is_refused(tmp_path):
    """Silently reading no rejections out of a malformed table is the dangerous failure."""
    from fnirs_pipe.cli.workflows import _bad_channels_for

    table = tmp_path / "bads.tsv"
    table.write_text("subject\tchannels\nsub-01\tS1_D1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="participant_id"):
        _bad_channels_for(str(table), "01")


# ---- exclude: blanked for a reason that is not rejection ----

def test_an_excluded_channel_is_blanked_like_a_rejected_one(haemo):
    frame = compute_alff(haemo, exclude=[GOOD], **BAND).set_index("channel")

    assert frame.loc[GOOD, ["alff", "falff", "malff", "zalff"]].isna().all()
    # exclude is not rejection, and the group tables read that column
    assert not frame.loc[GOOD, "bad"]


def test_an_excluded_channel_is_out_of_the_standardisation_reference(haemo):
    """The defect this guards: a channel fitted against itself has ALFF ~0, and leaving it
    in the reference drags every other channel's mALFF down."""
    zeroed = haemo.copy()
    data = zeroed.get_data()
    data[zeroed.ch_names.index(GOOD)] = 0.0
    zeroed = mne.io.RawArray(data, zeroed.info, verbose="error")

    kept = compute_alff(zeroed, **BAND).set_index("channel")
    dropped = compute_alff(zeroed, exclude=[GOOD], **BAND).set_index("channel")

    others = [c for c in kept.index if c != GOOD and c.endswith("hbo")]
    assert not np.allclose(kept.loc[others, "malff"], dropped.loc[others, "malff"])
    # with the zero out of the reference the surviving hbo channels average to 1.0
    assert np.isclose(dropped.loc[others, "malff"].mean(), 1.0)


def test_exclude_defaults_to_changing_nothing(haemo):
    a = compute_alff(haemo, **BAND)
    b = compute_alff(haemo, exclude=[], **BAND)
    assert np.allclose(a[["alff", "falff", "malff", "zalff"]],
                       b[["alff", "falff", "malff", "zalff"]])
