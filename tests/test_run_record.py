"""The per-subject TOML record must state what actually ran.

Parameters supplied through --config never appear in argv. The record used to be
built from argv and therefore listed them as absent while the pipeline was using
them. It is now built from the resolved config objects, so this drives a real run
with a TOML config plus one CLI override and reads the record back.

test_config_resolution.py covers whether the values resolve correctly; this covers
whether the resolved values reach the record, which is where the bug actually was.
"""

import sys
import tomllib

import pytest

from fnirs_pipe.cli.workflows import run_participant_level

_INVOCATION = ["fnirs-pipe", "/bids", "/out", "participant", "--mode", "glm",
               "--config", "glm.toml", "--noise-model", "ar2"]

# Only in the TOML, never on the command line. drift_order and stim_dur are
# deliberately non-default so a value falling back to a default is visible.
_TOML = """\
hrf_model = "spm"
drift_model = "cosine"
drift_high_pass = 0.01
drift_order = 3
stim_dur = 10.0
high_pass = 0.01
low_pass = 0.5
noise_model = "ols"
"""

_BASE_ARGS = dict(
    analysis_level="participant", session_label=None, task_label=["tapping"],
    bids_filter_file=None, work_dir=None, verbose=False, skip_bids_validation=True,
    ignore=None, n_jobs=1, dry_run=False, no_report=True,
    dpf=[6.0, 6.0], sci_threshold=0.8, motion_correction="tddr",
    cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5,
)


def _run_and_read(bids_dir, out_dir, **overrides):
    config_path = out_dir.parent / "glm.toml"
    config_path.write_text(_TOML)

    args = {**_BASE_ARGS, "bids_dir": bids_dir, "output_dir": out_dir,
            "participant_label": ["01"], **overrides}

    original = sys.argv
    sys.argv = list(_INVOCATION)
    try:
        run_participant_level(args)
    finally:
        sys.argv = original

    # sub-01.toml, not sub-01_<timestamp>.toml: the stamp was dropped in 0.19.0 and a
    # re-run now overwrites the record
    record = next((out_dir / "sub-01" / "logs").glob("sub-01*.toml"))
    return tomllib.loads(record.read_text())


@pytest.fixture(scope="module")
def record(mini_bids, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("record") / "out"
    return _run_and_read(mini_bids, out_dir, mode="glm",
                         config=out_dir.parent / "glm.toml", noise_model="ar2")


@pytest.fixture(scope="module")
def prep_only_record(mini_bids, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("record_prep") / "out"
    return _run_and_read(mini_bids, out_dir, mode=None, config=None)


def test_record_has_the_expected_sections(record):
    assert set(record) == {"environment", "execution", "prep", "post"}


@pytest.mark.parametrize("field, expected", [
    ("hrf_model", "spm"),
    ("drift_model", "cosine"),
    ("drift_order", 3),
    ("stim_dur", 10.0),
    ("high_pass", 0.01),
    ("low_pass", 0.5),
])
def test_toml_supplied_values_reach_the_record(record, field, expected):
    # None of these appear in argv, which is exactly why they used to go missing.
    assert record["post"][field] == expected


def test_a_cli_override_reaches_the_record_and_beats_the_toml(record):
    # The TOML says ols, the command line says ar2, and ar2 is what ran.
    assert record["post"]["noise_model"] == "ar2"


def test_the_mode_is_recorded(record):
    assert record["post"]["mode"] == "glm"


def test_run_command_stays_the_verbatim_invocation(record):
    # [post] is what was used; [execution].run_command is what was typed, and it
    # must not be reconstructed from the resolved values.
    assert record["execution"]["run_command"] == " ".join(_INVOCATION)


def test_prep_section_comes_from_the_resolved_config(record):
    prep = record["prep"]
    assert prep["dpf"] == [6.0, 6.0]
    assert prep["sci_threshold"] == 0.8
    assert prep["motion_correction"] == "tddr"
    assert prep["cardiac_l_freq"] == 0.7


def test_execution_section_records_the_selection(record):
    execution = record["execution"]
    assert execution["participant_label"] == ["01"]
    assert execution["task_label"] == ["tapping"]
    assert execution["analysis_level"] == "participant"


def test_environment_records_the_pipeline_version(record):
    from fnirs_pipe import __version__

    assert record["environment"]["fnirs_pipe_version"] == __version__


def test_a_prep_only_run_has_no_post_section(prep_only_record):
    assert "post" not in prep_only_record
    assert "prep" in prep_only_record
