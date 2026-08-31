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
# run_post: (result, glm_est, dm, alff_df, fc_df, fc_hbr_df, gcor_reg, fc_seed, fc_roi)
_ALFF_DF, _FC_DF, _FC_SEED, _FC_ROI = 3, 4, 7, 8


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


# ---- the flag ----

def test_without_the_flag_nothing_connectivity_is_written(haemo, tmp_path):
    out = _glm(haemo, tmp_path)
    assert _names(tmp_path, "*fc*.tsv") == []
    assert out[_FC_DF] is None and out[_FC_ROI] == {} and out[_FC_SEED] == {}


def test_the_flag_writes_the_same_family_rest_writes(haemo, tmp_path, roi_map):
    out = _glm(haemo, tmp_path, roi_map=roi_map, fc=True)
    written = _names(tmp_path, "*fc*.tsv")
    for chromo in ("hbo", "hbr"):
        for suffix in ("fc", "fcz", "fcroi", "fcroiz", "fcseed", "fcseedz"):
            assert f"sub-01_task-tapping_desc-{chromo}_{suffix}.tsv" in written
    assert out[_FC_DF] is not None
    assert sorted(out[_FC_ROI]) == ["hbo", "hbr"]
    assert sorted(out[_FC_SEED]) == ["hbo", "hbr"]


def test_the_roi_frames_are_square_and_labelled_by_roi(haemo, tmp_path, roi_map):
    fc_roi = _glm(haemo, tmp_path, roi_map=roi_map, fc=True)[_FC_ROI]
    for frame in fc_roi.values():
        assert list(frame.index) == list(frame.columns) == list(roi_map)


def test_the_roi_products_need_a_roi_map(haemo, tmp_path):
    out = _glm(haemo, tmp_path, fc=True)
    assert _names(tmp_path, "*fcroi*.tsv") == []
    assert _names(tmp_path, "*fcseed*.tsv") == []
    assert out[_FC_DF] is not None                      # the channel matrices still come


# ---- what the mode cannot produce ----

def test_alff_is_not_written_by_the_glm_mode(haemo, tmp_path):
    """fALFF's denominator spans the full spectrum, and this mode's residual is bandpassed.

    Writing an ALFF here would put a number in a TSV whose fALFF is ~1 by construction.
    """
    out = _glm(haemo, tmp_path, fc=True)
    assert out[_ALFF_DF] is None
    assert _names(tmp_path, "*_alff.tsv") == []
