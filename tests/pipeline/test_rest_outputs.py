"""What a rest-mode run actually leaves on disk.

Every connectivity TSV is asserted on disk, and `run_post` is given a `roi_map` so that
everything behind `if config.roi_map:` runs. That covers the ROI matrix, the seed map, and
both Fisher z companions.

These read the files back rather than the returned frames, because the write is where the
index label, the NaN convention and the sidecar live, and none of those survive a check made
on the in-memory object.
"""

import json

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.io.naming import derivative_path

from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep

from tests._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)

# every FC product rest mode writes, per chromophore, and whether an ROI map gates it
_UNGATED = ("fc", "fcz")
_ROI_GATED = ("fcroi", "fcroiz", "fcseed", "fcseedz")

# The entities that name each product. The keys are short labels for the tests below.
_PRODUCTS = {
    "fc":      dict(statistic="pearson"),
    "fcz":     dict(statistic="fisherz"),
    "fcroi":   dict(segmentation="custom", aggregation="roi", statistic="pearson"),
    "fcroiz":  dict(segmentation="custom", aggregation="roi", statistic="fisherz"),
    "fcseed":  dict(segmentation="custom", aggregation="seed", statistic="pearson"),
    "fcseedz": dict(segmentation="custom", aggregation="seed", statistic="fisherz"),
}


@pytest.fixture(scope="module")
def haemo(tmp_path_factory):
    config = PrepConfig(subject="01", dpf=[6.0, 6.0], sci_threshold=0.8,
                        motion_correction="tddr", **_BANDS)
    result = run_prep(synth_raw("01", "rest"), config,
                      output_dir=tmp_path_factory.mktemp("rest_out_prep"),
                      source_entities={"task": "rest"})
    return result.raw_haemo


@pytest.fixture(scope="module")
def roi_map(haemo):
    hbo = [c for c in haemo.ch_names if c.endswith("hbo")]
    return {"left": hbo[:2], "right": hbo[2:4]}


def _rest_run(haemo, out_dir, roi_map=None):
    config = PostConfig(subject="01", high_pass=0.01, low_pass=0.1,
                        drift_model="polynomial", drift_order=1, drift_high_pass=0.01,
                        roi_map=roi_map, **_BANDS)
    run_post(haemo.copy(), config, output_dir=out_dir, mode="rest",
             source_entities={"task": "rest"})
    return out_dir


@pytest.fixture(scope="module")
def rest_out(haemo, roi_map, tmp_path_factory):
    return _rest_run(haemo, tmp_path_factory.mktemp("rest_out"), roi_map)


def _expected(out_dir, kind, chromo="hbo"):
    """Built rather than globbed: a `*_stat-alff_*` glob also matches the ROI version."""
    return derivative_path(out_dir, "relmat", ".tsv", subject="01", task="rest",
                           chromophore=chromo, **_PRODUCTS[kind])


def _one(out_dir, suffix, chromo="hbo"):
    path = _expected(out_dir, suffix, chromo)
    assert path.is_file(), f"{suffix} ({chromo}) was not written: {path}"
    return path


def _alff(out_dir, roi=False):
    entities = dict(subject="01", task="rest", statistic="alff")
    if roi:
        entities |= dict(segmentation="custom", aggregation="roi")
    return derivative_path(out_dir, "nirsmap", ".tsv", **entities)


# ---- every product is written, for both chromophores ----
@pytest.mark.parametrize("suffix", _UNGATED + _ROI_GATED)
@pytest.mark.parametrize("chromo", ["hbo", "hbr"])
def test_the_product_reaches_disk(rest_out, suffix, chromo):
    assert _one(rest_out, suffix, chromo).stat().st_size > 0


def test_the_roi_products_are_absent_without_a_roi_map(haemo, tmp_path):
    out = _rest_run(haemo, tmp_path, roi_map=None)

    for suffix in _UNGATED:
        assert _one(out, suffix)
    for suffix in _ROI_GATED:
        assert not _expected(out, suffix).exists()


# ---- shape and labelling survive the write ----
def test_the_channel_matrix_is_square_and_channel_labelled(rest_out, haemo):
    fc = pd.read_csv(_one(rest_out, "fc"), sep="\t", index_col="channel")
    hbo = [c for c in haemo.ch_names if c.endswith("hbo")]
    good = [i for i, c in enumerate(hbo) if c not in set(haemo.info["bads"])]

    assert list(fc.index) == list(fc.columns) == hbo
    # a rejected channel is blanked whole, its own self-correlation included: a diagonal 1
    # sitting in an otherwise empty row would be the one cell claiming the channel is there
    assert np.allclose(np.diag(fc.to_numpy())[good], 1.0)

    rejected = [c for c in hbo if c in set(haemo.info["bads"])]
    assert rejected, "fixture no longer rejects a channel"
    assert fc.loc[rejected].isna().all().all()


def test_the_seed_map_is_roi_by_channel_on_disk(rest_out, haemo, roi_map):
    seed = pd.read_csv(_one(rest_out, "fcseed"), sep="\t", index_col="roi")

    assert list(seed.index) == list(roi_map)
    assert list(seed.columns) == [c for c in haemo.ch_names if c.endswith("hbo")]


def test_a_seed_stays_blank_rather_than_zero_through_the_tsv(rest_out, haemo, roi_map):
    """A zero would read as "no connectivity"; these cells have no meaning at all."""
    seed = pd.read_csv(_one(rest_out, "fcseed"), sep="\t", index_col="roi")
    bads = set(haemo.info["bads"])

    for roi, chans in roi_map.items():
        used = [c for c in chans if c not in bads]
        assert seed.loc[roi, used].isna().all()
        # everything else holds a value, bar the rejected columns, which are blank for their
        # own reason and are covered by the test below
        ordinary = [c for c in seed.columns if c not in used and c not in bads]
        assert seed.loc[roi, ordinary].notna().all()


def test_a_rejected_channel_is_blank_in_every_seed_row(rest_out, haemo, roi_map):
    """A channel preprocessing threw out has no correlation to report, seed member or not.

    The synthetic recording rejects one long pair and the fixture puts it in an ROI, so both
    routes to a blank cell are covered at once: dropped from the seed it was listed in, and
    dropped as a column of every other seed's row.
    """
    seed = pd.read_csv(_one(rest_out, "fcseed"), sep="\t", index_col="roi")
    rejected = [c for c in seed.columns if c in set(haemo.info["bads"])]
    listed = [c for chans in roi_map.values() for c in chans if c in rejected]
    assert rejected, "fixture no longer covers the case it was built for"
    assert listed, "fixture no longer puts a rejected channel inside an ROI"

    assert seed[rejected].isna().all().all()


def test_fisher_z_of_the_seed_map_keeps_every_column(rest_out, roi_map):
    seed = pd.read_csv(_one(rest_out, "fcseed"), sep="\t", index_col="roi")
    z = pd.read_csv(_one(rest_out, "fcseedz"), sep="\t", index_col="roi")

    assert z.shape == seed.shape
    assert np.allclose(z.to_numpy()[seed.notna().to_numpy()],
                       np.arctanh(seed.to_numpy()[seed.notna().to_numpy()]))


# ---- the sidecar records what the map was ----
def test_the_seed_sidecar_names_the_channels_each_seed_was_built_from(rest_out, haemo, roi_map):
    """The resolved membership, not the requested map: the two differ once a channel is rejected."""
    meta = json.loads(_one(rest_out, "fcseed").with_suffix(".json").read_text(encoding="utf-8"))
    bads = set(haemo.info["bads"])
    resolved = {roi: [c for c in chans if c not in bads] for roi, chans in roi_map.items()}

    assert meta["step"] == "fc_seed"
    assert meta["parameters"]["seed_channels"] == resolved
    assert resolved != {k: list(v) for k, v in roi_map.items()}, "fixture stopped covering the difference"


def test_the_roi_sidecar_names_the_channels_each_roi_was_averaged_from(rest_out, haemo, roi_map):
    """An ROI correlation averages its members into a signal that exists nowhere else, so
    without the membership the number cannot be reproduced from the tree."""
    bads = set(haemo.info["bads"])
    resolved = {roi: [c for c in chans if c not in bads] for roi, chans in roi_map.items()}

    for suffix, step in (("fcroi", "fc_roi"), ("fcroiz", "fisher_z")):
        meta = json.loads(_one(rest_out, suffix).with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["step"] == step
        assert meta["parameters"]["roi_channels"] == resolved, suffix


def test_hbo_and_hbr_are_written_separately(rest_out):
    hbo = pd.read_csv(_one(rest_out, "fc", "hbo"), sep="\t", index_col="channel")
    hbr = pd.read_csv(_one(rest_out, "fc", "hbr"), sep="\t", index_col="channel")

    assert all(c.endswith("hbo") for c in hbo.columns)
    assert all(c.endswith("hbr") for c in hbr.columns)


# ---- ROI amplitude ----

def _alff_roi(out_dir):
    path = _alff(out_dir, roi=True)
    assert path.is_file(), f"the ROI amplitude table was not written: {path}"
    return pd.read_csv(path, sep="\t")


def test_the_roi_amplitude_table_is_gated_on_the_roi_map(rest_out, haemo, tmp_path):
    assert not _alff_roi(rest_out).empty
    assert not _alff(_rest_run(haemo, tmp_path, roi_map=None), roi=True).exists()


def test_one_row_per_roi_and_chromophore(rest_out, roi_map):
    df = _alff_roi(rest_out)
    assert set(df["chromophore"]) == {"hbo", "hbr"}
    for chromo in ("hbo", "hbr"):
        assert list(df[df.chromophore == chromo]["roi"]) == list(roi_map)


def test_the_roi_value_is_the_mean_of_its_channels_not_a_measure_of_their_mean(rest_out, roi_map):
    """The whole point of the product: averaging after the measure, not before. A value built
    the other way round would sit well below this one."""
    alff = pd.read_csv(_alff(rest_out), sep="\t").set_index("channel")
    df = _alff_roi(rest_out).set_index(["chromophore", "roi"])

    for roi, chans in roi_map.items():
        expected = alff.loc[chans, "zalff"].mean()
        assert df.loc[("hbo", roi), "zalff"] == pytest.approx(expected)


def test_the_count_is_what_survived_screening_not_what_was_asked_for(rest_out, roi_map):
    """Without it a thinly covered ROI reads like a well covered one. The synthetic montage
    loses a channel out of one of the two ROIs, so the two counts differ here."""
    alff = pd.read_csv(_alff(rest_out), sep="	").set_index("channel")
    df = _alff_roi(rest_out)

    survived = [int(alff.loc[chans, "zalff"].notna().sum()) for chans in roi_map.values()]
    assert survived != [len(v) for v in roi_map.values()]   # the case is actually exercised
    assert df[df.chromophore == "hbo"]["n_channels"].tolist() == survived


def test_the_sidecar_names_the_members_of_both_chromophores(rest_out, roi_map):
    path = _alff(rest_out, roi=True)
    meta = json.loads(path.with_suffix(".json").read_text())

    assert meta["step"] == "alff_roi"
    assert set(meta["parameters"]["roi_channels"]) == {"hbo", "hbr"}
    assert list(meta["parameters"]["roi_channels"]["hbo"]) == list(roi_map)


# ---- the broadband residual says it skipped the bandpass ----
def test_the_broadband_residual_records_no_passband(rest_out):
    def params(desc):
        path = derivative_path(rest_out, "nirs", ".json", subject="01", task="rest", desc=desc)
        return json.loads(path.read_text())["parameters"]

    broad, errts = params("errtsbroad"), params("errts")
    assert {k: broad[k] for k in ("high_pass", "low_pass", "filter_method", "filter_order")} \
        == dict.fromkeys(("high_pass", "low_pass", "filter_method", "filter_order"))
    assert (errts["high_pass"], errts["low_pass"]) == (0.01, 0.1)
    assert broad["drift_model"] == errts["drift_model"] == "polynomial"
