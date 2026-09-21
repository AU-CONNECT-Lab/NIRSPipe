"""GLM output naming and the sidecars that go with it.

The three CSVs used to be written under fixed names, so a second task overwrote the
first subject's results in place and the run still reported success. The names are now
prefixed from the input file, which is a claim about entities the tests below pin down.
"""

import json

import pandas as pd
import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.glm import _entity_prefix, _save_glm_outputs, compute_contrasts

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
    (PREPROC, "sub-01_task-tapping_"),
    ("/out/sub-01/nirs/sub-01_ses-1_task-a_run-01_desc-resampled_nirs.snirf",
     "sub-01_ses-1_task-a_run-01_"),
    ("/bids/sub-01/nirs/sub-01_nirs.snirf", "sub-01_"),          # no desc- to drop
    (None, ""),                                                   # source unknown
    ("nirs.snirf", ""),                                           # suffix only
])
def test_entity_prefix_carries_the_input_entities(source, expected):
    assert _entity_prefix(source) == expected


def test_entity_prefix_drops_desc():
    # desc- describes the GLM's input, not its outputs, so carrying it would name a
    # design matrix after the file it was built from
    assert "desc-" not in _entity_prefix(PREPROC)


# ---- output files ----

def test_two_tasks_write_side_by_side(tmp_path, design_matrix):
    glm = _FakeGLM(["S1_D1 hbo", "S1_D1 hbr"])
    for task in ("tapping", "rest"):
        _save_glm_outputs(glm, design_matrix, tmp_path,
                          source_path=PREPROC.replace("tapping", task))

    names = {p.name for p in tmp_path.glob("*.csv")}
    assert names == {
        "sub-01_task-tapping_design_matrix.csv", "sub-01_task-rest_design_matrix.csv",
        "sub-01_task-tapping_glm_results.csv", "sub-01_task-rest_glm_results.csv",
    }


def test_an_unknown_source_falls_back_to_bare_names(tmp_path, design_matrix):
    _save_glm_outputs(_FakeGLM(["S1_D1 hbo"]), design_matrix, tmp_path)
    assert (tmp_path / "design_matrix.csv").exists()
    assert (tmp_path / "glm_results.csv").exists()


def test_contrasts_are_written_with_the_same_prefix(tmp_path, design_matrix):
    _save_glm_outputs(_FakeGLM(["S1_D1 hbo"]), design_matrix, tmp_path,
                      contrasts={"tapping-rest": _FakeContrast(["S1_D1 hbo"])},
                      source_path=PREPROC)

    frame = pd.read_csv(tmp_path / "sub-01_task-tapping_contrasts.csv")
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
        "sub-01_task-tapping_design_matrix.json": "design_matrix",
        "sub-01_task-tapping_glm_results.json": "glm_fit",
    }


def test_the_sidecar_carries_the_parameters_the_fit_used(tmp_path, design_matrix):
    _save_glm_outputs(_FakeGLM(["S1_D1 hbo"]), design_matrix, tmp_path,
                      source_path=PREPROC, noise_model="ar1", drift_model="cosine")

    meta = json.loads((tmp_path / "sub-01_task-tapping_design_matrix.json").read_text())
    assert meta["parameters"] == {"noise_model": "ar1", "drift_model": "cosine"}


# ---- bad channels ----

def test_bad_channels_are_flagged_in_the_results_not_dropped(tmp_path, design_matrix):
    # the fit runs on every channel; group analysis reads the frame by position, so a
    # rejected channel has to keep its row
    glm = _FakeGLM(["S1_D1 hbo", "S2_D2 hbo"])
    _save_glm_outputs(glm, design_matrix, tmp_path, source_path=PREPROC, bads=["S2_D2 hbo"])

    frame = pd.read_csv(tmp_path / "sub-01_task-tapping_glm_results.csv")
    assert len(frame) == 2
    assert frame.set_index("ch_name")["bad"].to_dict() == {"S1_D1 hbo": False, "S2_D2 hbo": True}


def test_the_sidecar_lists_the_bad_channels(tmp_path, design_matrix):
    _save_glm_outputs(_FakeGLM(["S1_D1 hbo"]), design_matrix, tmp_path,
                      source_path=PREPROC, bads=["S2_D2 hbo"])

    meta = json.loads((tmp_path / "sub-01_task-tapping_glm_results.json").read_text())
    assert meta["bad_channels"] == ["S2_D2 hbo"]
