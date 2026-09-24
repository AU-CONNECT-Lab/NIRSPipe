"""The per-subject TOML record must state what actually ran.

Parameters supplied through --config never appear in argv, so the record is built
from the resolved config objects. This drives a real run with a TOML config plus
one CLI override and reads the record back.

test_config_resolution.py covers whether the values resolve correctly; this covers
whether the resolved values reach the record.
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

    # sub-01.toml, not sub-01_<timestamp>.toml: the record carries no stamp, so a re-run
    # overwrites it
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
    # None of these appear in argv.
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


# ---- the hyperscanning group record ----
#
# The mirror of a subject's record, and it exists for the same reason: the command is the
# only place several of these settings appear, so a run nobody wrote down cannot be redone.

@pytest.fixture
def group_record(tmp_path):
    import tomllib

    from fnirs_pipe.utils.run_record import write_group_run_record

    out = write_group_run_record(
        {"pairs_csv": tmp_path / "pairs.csv", "wtc_fmin": 0.004, "wtc_fmax": 0.2,
         "wtc_channel_cross": False, "task_label": ["chat"], "roi_mapping": None,
         "output_dir": tmp_path, "verbose": False, "func": None,
         "roi_map": {"L": ["S1_D1"]}},
        "D01", "chat", "20260907_120000", tmp_path, tmp_path / "group-D01",
        members=["01", "02"],
    )
    return out, tomllib.loads(out.read_text(encoding="utf-8"))


def test_the_group_record_lands_beside_the_group_reports(group_record):
    out, _ = group_record
    assert out.parent.name == "logs"
    assert out.parent.parent.name == "group-D01"
    assert out.name == "group-D01_task-chat.toml"


def test_the_group_record_names_the_group_and_its_members(group_record):
    _, record = group_record
    assert record["execution"]["group_id"] == "D01"
    assert record["execution"]["task_label"] == "chat"
    assert record["execution"]["participant_label"] == ["01", "02"]


def test_the_group_record_carries_every_resolved_option(group_record):
    _, record = group_record
    assert record["hyper"]["wtc_fmin"] == 0.004
    assert record["hyper"]["wtc_channel_cross"] is False
    # structures are settings for the analysis, not for the record; the ROI file is named
    # by roi_mapping and its contents belong beside the results
    assert "roi_map" not in record["hyper"]
    assert "output_dir" not in record["hyper"]


def test_both_records_report_the_same_environment(group_record, record):
    _, group = group_record
    # free_mem_gb is read when the record is written, so the two disagree by whatever the
    # machine did in between. The claim is that both kinds of record carry the same fields
    # from the same builder, not that memory stood still.
    volatile = {"free_mem_gb"}
    assert group["environment"].keys() == record["environment"].keys()
    assert ({k: v for k, v in group["environment"].items() if k not in volatile}
            == {k: v for k, v in record["environment"].items() if k not in volatile})
