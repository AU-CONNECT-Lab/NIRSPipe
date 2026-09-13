"""What `--mode glm --fc` writes, and what it deliberately does not.

Connectivity on task data is the same computation rest mode runs; the only thing that
differs is which residual it reads. That distinction is the whole point of the flag, so
these tests pin the residual's consequences rather than the correlation itself: the task is
in the design matrix, so the frames come back from a GLM run, and ALFF stays absent because
it needs a broadband residual this mode never produces.

The flag defaulting to off matters as much as the flag working. `--fc` adds twelve files to
every subject directory, and a run that was not asked for them should look exactly as it did
before the flag existed.
"""

import pytest

from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep

from ._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)
# run_post: (result, glm_est, dm, alff_df, fc_df, fc_hbr_df, fc_seed, fc_roi)
_ALFF_DF, _FC_DF, _FC_SEED, _FC_ROI = 3, 4, 6, 7


@pytest.fixture(scope="module")
def haemo(tmp_path_factory):
    config = PrepConfig(subject="01", dpf=[6.0, 6.0], sci_threshold=0.8,
                        motion_correction="tddr", **_BANDS)
    result = run_prep(synth_raw("01", "tapping"), config,
                      output_dir=tmp_path_factory.mktemp("glmfc_prep"),
                      source_entities={"task": "tapping"})
    return result.raw_haemo


@pytest.fixture(scope="module")
def roi_map(haemo):
    pairs = sorted({c.rsplit(" ", 1)[0] for c in haemo.ch_names})
    half = len(pairs) // 2
    return {"ROI_A": [f"{p} hbo" for p in pairs[:half]],
            "ROI_B": [f"{p} hbo" for p in pairs[half:]]}


def _glm(haemo, out_dir, roi_map=None, **overrides):
    config = PostConfig(subject="01", high_pass=0.01, low_pass=0.1,
                        hrf_model="spm", noise_model="ols", drift_model="cosine",
                        drift_high_pass=0.01, drift_order=1, stim_dur=5.0,
                        roi_map=roi_map, **overrides, **_BANDS)
    return run_post(haemo.copy(), config, output_dir=out_dir, mode="glm",
                    source_entities={"task": "tapping"})


def _names(out_dir, pattern):
    return sorted(p.name for p in out_dir.rglob(pattern))


# ---- three runs, not five ----
# The five tests below ask five questions of three configurations, and a GLM run is twenty
# seconds: running one per test made this 98-line file 38% of the suite's wall clock. The
# runs are module-scoped and every test only reads what they produced, so sharing them
# changes no assertion. Each yields (run_post's tuple, the directory it wrote into).

@pytest.fixture(scope="module")
def plain(haemo, tmp_path_factory):
    """No --fc: the run the flag has to leave looking exactly as it did before it existed."""
    out_dir = tmp_path_factory.mktemp("glm_plain")
    return _glm(haemo, out_dir), out_dir


@pytest.fixture(scope="module")
def fc_with_roi(haemo, roi_map, tmp_path_factory):
    """--fc with a ROI map: the full family, channel matrices and ROI products both."""
    out_dir = tmp_path_factory.mktemp("glm_fc_roi")
    return _glm(haemo, out_dir, roi_map=roi_map, fc=True), out_dir


@pytest.fixture(scope="module")
def fc_no_roi(haemo, tmp_path_factory):
    """--fc without a ROI map: channel matrices only, which is what makes the map required."""
    out_dir = tmp_path_factory.mktemp("glm_fc")
    return _glm(haemo, out_dir, fc=True), out_dir


# ---- the flag ----

def test_without_the_flag_nothing_connectivity_is_written(plain):
    out, out_dir = plain
    assert _names(out_dir, "*fc*.tsv") == []
    assert out[_FC_DF] is None and out[_FC_ROI] == {} and out[_FC_SEED] == {}


def test_the_flag_writes_the_same_family_rest_writes(fc_with_roi):
    out, out_dir = fc_with_roi
    written = _names(out_dir, "*fc*.tsv")
    for chromo in ("hbo", "hbr"):
        for suffix in ("fc", "fcz", "fcroi", "fcroiz", "fcseed", "fcseedz"):
            assert f"sub-01_task-tapping_desc-{chromo}_{suffix}.tsv" in written
    assert out[_FC_DF] is not None
    assert sorted(out[_FC_ROI]) == ["hbo", "hbr"]
    assert sorted(out[_FC_SEED]) == ["hbo", "hbr"]


def test_the_roi_frames_are_square_and_labelled_by_roi(fc_with_roi, roi_map):
    fc_roi = fc_with_roi[0][_FC_ROI]
    for frame in fc_roi.values():
        assert list(frame.index) == list(frame.columns) == list(roi_map)


def test_the_roi_products_need_a_roi_map(fc_no_roi):
    out, out_dir = fc_no_roi
    assert _names(out_dir, "*fcroi*.tsv") == []
    assert _names(out_dir, "*fcseed*.tsv") == []
    assert out[_FC_DF] is not None                      # the channel matrices still come


# ---- what the mode cannot produce ----

def test_alff_is_not_written_by_the_glm_mode(fc_no_roi):
    """fALFF's denominator spans the full spectrum, and this mode's residual is bandpassed.

    Writing an ALFF here would put a number in a TSV whose fALFF is ~1 by construction.
    """
    out, out_dir = fc_no_roi
    assert out[_ALFF_DF] is None
    assert _names(out_dir, "*_alff.tsv") == []
