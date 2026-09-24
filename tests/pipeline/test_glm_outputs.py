"""GLM output naming and the sidecars that go with it.

Every output name is prefixed from the input file, so a second task writes beside the
first rather than over it. Which entities carry over is what the tests below pin down.
"""

import json

import pandas as pd
import pytest

TAB = chr(9)

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.glm import _glm_name, _save_glm_outputs, compute_contrasts

PREPROC = "/out/sub-01/nirs/sub-01_task-tapping_desc-preproc_nirs.snirf"


class _FakeGLM:
    """Stands in for mne_nirs' RegressionResults: _save_glm_outputs only reads to_dataframe."""

    def __init__(self, ch_names):
        self._ch_names = ch_names

    def to_dataframe(self):
        return pd.DataFrame({"ch_name": self._ch_names, "theta": [1.0] * len(self._ch_names)})


class _FakeContrast(_FakeGLM):
    pass


@pytest.fixture
def design_matrix():
    return pd.DataFrame({"tapping": [0.0, 1.0], "constant": [1.0, 1.0]})


# ---- entity prefix ----

@pytest.mark.parametrize("source, expected", [
    (PREPROC, "sub-01_task-tapping_design.tsv"),
    ("/out/sub-01/nirs/sub-01_ses-1_task-a_run-01_desc-resampled_nirs.snirf",
     "sub-01_ses-1_task-a_run-01_design.tsv"),
    ("/bids/sub-01/nirs/sub-01_nirs.snirf", "sub-01_design.tsv"),   # no desc- to drop
])
def test_the_name_carries_the_input_entities(source, expected):
    assert _glm_name(source, "design") == expected


def test_the_name_drops_the_input_desc():
    # desc- describes the GLM's input, not its outputs, so carrying it would name a
    # design matrix after the file it was built from
    assert "desc-resampled" not in _glm_name(PREPROC, "design")


@pytest.mark.parametrize("source", [None, "nirs.snirf"])
def test_a_source_with_no_subject_is_refused(source):
    """A GLM output with no subject has no place in the tree, so it is a failure rather
    than a file called `design.tsv` sitting loose in somebody's nirs folder."""
    with pytest.raises(ValueError, match="no path pattern fits"):
        _glm_name(source, "design")


# ---- output files ----

def test_two_tasks_write_side_by_side(tmp_path, design_matrix):
    glm = _FakeGLM(["S1_D1 hbo", "S1_D1 hbr"])
    for task in ("tapping", "rest"):
        _save_glm_outputs(glm, design_matrix, tmp_path,
                          source_path=PREPROC.replace("tapping", task))

    names = {p.name for p in tmp_path.glob("*.tsv")}
    assert names == {
        "sub-01_task-tapping_design.tsv", "sub-01_task-rest_design.tsv",
        "sub-01_task-tapping_desc-glm_nirsmap.tsv", "sub-01_task-rest_desc-glm_nirsmap.tsv",
    }


def test_an_unknown_source_is_refused_rather_than_written_loose(tmp_path, design_matrix):
    with pytest.raises(ValueError, match="no path pattern fits"):
        _save_glm_outputs(_FakeGLM(["S1_D1 hbo"]), design_matrix, tmp_path)


def test_contrasts_are_written_with_the_same_prefix(tmp_path, design_matrix):
    _save_glm_outputs(_FakeGLM(["S1_D1 hbo"]), design_matrix, tmp_path,
                      contrasts={"tapping-rest": _FakeContrast(["S1_D1 hbo"])},
                      source_path=PREPROC)

    frame = pd.read_csv(tmp_path / "sub-01_task-tapping_desc-contrast_nirsmap.tsv",
                        sep=TAB)
    assert frame["contrast"].unique().tolist() == ["tapping-rest"]


# ---- contrast vectors ----

class _RecordingGLM:
    """Records the vector it was handed; mne-nirs takes one value per design column."""

    def __init__(self):
        self.vectors = []

    def compute_contrast(self, contrast, contrast_type=None):
        self.vectors.append(list(contrast))
        return _FakeContrast(["S1_D1 hbo"])


def test_named_weights_become_a_vector_over_the_design_columns():
    glm = _RecordingGLM()
    design = pd.DataFrame({"left": [], "right": [], "drift_1": [], "constant": []})

    compute_contrasts(glm, {"left_minus_right": {"left": 1.0, "right": -1.0}}, design)

    assert glm.vectors == [[1.0, -1.0, 0.0, 0.0]]


def test_a_condition_the_design_lacks_is_refused():
    design = pd.DataFrame({"left": [], "constant": []})
    with pytest.raises(StageError, match="typo"):
        compute_contrasts(_RecordingGLM(), {"c": {"typo": 1.0}}, design)


# ---- sidecars ----

def test_each_output_gets_a_sidecar_naming_its_step(tmp_path, design_matrix):
    _save_glm_outputs(_FakeGLM(["S1_D1 hbo"]), design_matrix, tmp_path,
                      source_path=PREPROC, noise_model="ar1")

    steps = {}
    for path in tmp_path.glob("*.json"):
        meta = json.loads(path.read_text())
        steps[path.name] = meta["step"]
        assert meta["Sources"] == [PREPROC]
    assert steps == {
        "sub-01_task-tapping_design.json": "design_matrix",
        "sub-01_task-tapping_desc-glm_nirsmap.json": "glm_fit",
    }


def test_the_sidecar_carries_the_parameters_the_fit_used(tmp_path, design_matrix):
    _save_glm_outputs(_FakeGLM(["S1_D1 hbo"]), design_matrix, tmp_path,
                      source_path=PREPROC, noise_model="ar1", drift_model="cosine")

    meta = json.loads((tmp_path / "sub-01_task-tapping_design.json").read_text())
    assert meta["parameters"] == {"noise_model": "ar1", "drift_model": "cosine"}


# ---- bad channels ----

def test_bad_channels_are_flagged_in_the_results_not_dropped(tmp_path, design_matrix):
    # the fit runs on every channel; group analysis reads the frame by position, so a
    # rejected channel has to keep its row
    glm = _FakeGLM(["S1_D1 hbo", "S2_D2 hbo"])
    _save_glm_outputs(glm, design_matrix, tmp_path, source_path=PREPROC, bads=["S2_D2 hbo"])

    frame = pd.read_csv(tmp_path / "sub-01_task-tapping_desc-glm_nirsmap.tsv", sep=TAB)
    assert len(frame) == 2
    assert frame.set_index("ch_name")["bad"].to_dict() == {"S1_D1 hbo": False, "S2_D2 hbo": True}


def test_the_sidecar_lists_the_bad_channels(tmp_path, design_matrix):
    _save_glm_outputs(_FakeGLM(["S1_D1 hbo"]), design_matrix, tmp_path,
                      source_path=PREPROC, bads=["S2_D2 hbo"])

    meta = json.loads((tmp_path / "sub-01_task-tapping_desc-glm_nirsmap.json").read_text())
    assert meta["bad_channels"] == ["S2_D2 hbo"]
