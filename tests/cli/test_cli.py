"""CLI parsing tests — framework-agnostic safety net for the argparse migration.

These assert user-facing behaviour (option names, defaults, nargs, choices,
required args, dispatch), so they survive the Typer -> argparse switch.
"""

import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from fnirs_pipe.cli import run as run_cli


def _parse(argv):
    return run_cli._build_parser().parse_args(argv)


# The four bands carry no default on purpose (population-dependent), so every
# invocation has to supply them.
_BANDS = ["--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
          "--resp-l-freq", "0.2", "--resp-h-freq", "0.5"]

_MIN = ["bids", "out", "participant", "--dpf", "6.0", "--sci-threshold", "0.8", *_BANDS]


def test_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        _parse(["--help"])
    assert e.value.code == 0


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        _parse(["--version"])
    assert e.value.code == 0
    assert "fnirs-pipe" in capsys.readouterr().out


def test_missing_positionals_errors():
    with pytest.raises(SystemExit) as e:
        _parse([])
    assert e.value.code == 2


def test_analysis_level_choice_enforced():
    with pytest.raises(SystemExit):
        _parse(["bids", "out", "bogus"])


def test_participant_label_space_separated():
    # BIDS App convention: space-separated values under one flag.
    args = _parse(_MIN + ["--participant-label", "01", "02", "03"])
    assert args.participant_label == ["01", "02", "03"]


def test_participant_label_repeated_flag():
    # GUI-generated form: flag repeated per value, must accumulate (not overwrite).
    args = _parse(_MIN + ["--participant-label", "01",
                          "--participant-label", "02", "--participant-label", "03"])
    assert args.participant_label == ["01", "02", "03"]


def test_gvtd_censor_is_off_present_or_given_a_channel_set():
    """One flag carries both the switch and the set, so `--gvtd-censor` keeps meaning what
    it always did and `--gvtd-censor all` is the conservative variant. Absent must be None
    rather than a falsy string, since the pipeline gates censoring on the field itself."""
    assert _parse(_MIN).gvtd_censor is None
    assert _parse(_MIN + ["--gvtd-censor"]).gvtd_censor == "long"
    assert _parse(_MIN + ["--gvtd-censor", "all"]).gvtd_censor == "all"
    assert _parse(_MIN + ["--gvtd-censor", "short"]).gvtd_censor == "short"


def test_gvtd_censor_rejects_a_set_it_does_not_have(capsys):
    with pytest.raises(SystemExit):
        _parse(_MIN + ["--gvtd-censor", "medium"])
    assert "invalid choice" in capsys.readouterr().err


def test_dpf_accepts_multiple_values():
    args = _parse(["bids", "out", "participant", "--dpf", "6.0", "6.0",
                   "--sci-threshold", "0.8", *_BANDS])
    assert args.dpf == [6.0, 6.0]


def test_defaults_preserved():
    args = _parse(_MIN)
    assert args.motion_correction == "tddr"
    assert args.drift_order == 1
    assert args.n_jobs == 1
    assert args.combine_runs is False
    assert args.no_report is False
    assert args.participant_label is None


@pytest.mark.parametrize("flag", ["--cardiac-l-freq", "--cardiac-h-freq",
                                  "--resp-l-freq", "--resp-h-freq"])
def test_bands_have_no_default(flag, capsys):
    # Dropping any one of them must be refused rather than filled in: the bands are
    # population-dependent and a wrong band silently corrupts SCI, PSP and band power.
    #
    # Driven through main() rather than the parser: the bands describe preprocessing, so
    # they are required of the participant level rather than of every invocation, and the
    # group level no longer has to name four frequencies it never uses.
    argv = list(_MIN)
    i = argv.index(flag)
    del argv[i:i + 2]

    with pytest.raises(SystemExit):
        run_cli.main(argv)
    assert flag in capsys.readouterr().err


def test_motion_correction_choice_rejected():
    with pytest.raises(SystemExit):
        _parse(_MIN + ["--motion-correction", "bogus"])


def test_combine_runs_negatable():
    assert _parse(_MIN + ["--combine-runs"]).combine_runs is True
    assert _parse(_MIN + ["--no-combine-runs"]).combine_runs is False


def test_dest_names_match_workflow_keys():
    args = _parse(_MIN + ["--sci-threshold", "0.8", "--resample-sfreq", "2.0"])
    d = vars(args)
    for key in ("bids_dir", "output_dir", "analysis_level", "sci_threshold",
                "participant_label", "motion_correction", "resample_sfreq"):
        assert key in d


def test_missing_dpf_exits_one(capsys):
    # --dpf is checked in main() rather than by argparse, so it exits 1, not 2.
    with pytest.raises(SystemExit) as e:
        run_cli.main(["bids", "out", "participant", "--sci-threshold", "0.8", *_BANDS])
    assert e.value.code == 1
    assert "--dpf" in capsys.readouterr().err


def test_dispatch_participant(monkeypatch):
    called = {}
    fake = types.ModuleType("fnirs_pipe.cli.workflows")
    fake.run_participant_level = lambda opts: called.setdefault("participant", opts)
    fake.run_group_level = lambda opts: called.setdefault("group", opts)
    monkeypatch.setitem(sys.modules, "fnirs_pipe.cli.workflows", fake)

    run_cli.main(_MIN + ["--participant-label", "01"])
    assert called["participant"]["participant_label"] == ["01"]


def _fake_workflows(monkeypatch, called):
    fake = types.ModuleType("fnirs_pipe.cli.workflows")
    fake.run_participant_level = lambda opts: called.setdefault("participant", opts)
    fake.run_group_level       = lambda opts: called.setdefault("group", opts)
    monkeypatch.setitem(sys.modules, "fnirs_pipe.cli.workflows", fake)
    return called


def test_dispatch_group(monkeypatch):
    called = _fake_workflows(monkeypatch, {})
    # the prep flags describe preprocessing, so the derivatives-only levels must not ask for them
    run_cli.main(["bids", "out", "group"])
    assert "group" in called


def test_the_dyad_analysis_is_not_a_pipeline_level():
    """It is fnirs-hyper. fnirs-pipe stays on the BIDS App levels, and its three
    positionals stay the app signature every container entry point has to accept."""
    with pytest.raises(SystemExit):
        run_cli.main(["bids", "out", "hyper", "--pairs-csv", "p.csv"])


def test_participant_still_requires_the_bands(capsys):
    with pytest.raises(SystemExit):
        run_cli.main(["bids", "out", "participant", "--dpf", "6.0", "--sci-threshold", "0.8"])
    assert "--cardiac-l-freq" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("fnirs-pipe") is None, reason="console script not installed")
def test_console_script_entry_point():
    r = subprocess.run(["fnirs-pipe", "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "participant" in r.stdout


# ── other commands: parser smoke + dispatch ──────────────────────────────────

from fnirs_pipe.cli import db as db_cli
from fnirs_pipe.cli import gui as gui_cli
from fnirs_pipe.cli import prep as prep_cli
from fnirs_pipe.cli import hyper as hyper_cli
from fnirs_pipe.cli import qc as qc_cli
from fnirs_pipe.cli import rate as rate_cli
from fnirs_pipe.cli import recon as recon_cli


@pytest.mark.parametrize("mod", [db_cli, gui_cli, prep_cli, qc_cli, rate_cli, recon_cli])
def test_help_exits_zero_all(mod):
    with pytest.raises(SystemExit) as e:
        mod._build_parser().parse_args(["--help"])
    assert e.value.code == 0


def test_gui_default_port():
    from fnirs_pipe.interface.app import DEFAULT_PORT

    # None rather than 8050: the launcher has to tell "unset" from "the user asked for 8050",
    # and only the first may be moved when the port is busy
    assert gui_cli._build_parser().parse_args([]).port is None
    assert gui_cli._build_parser().parse_args(["--port", "9000"]).port == 9000
    assert DEFAULT_PORT == 8050


def test_db_merge_subcommand():
    args = db_cli._build_parser().parse_args(["merge", "/out"])
    assert args.func is db_cli.cmd_merge
    assert args.db_path is None


def test_recon_requires_subject_and_task():
    with pytest.raises(SystemExit):
        recon_cli._build_parser().parse_args(["in.snirf", "/bids"])  # no participant/task
    args = recon_cli._build_parser().parse_args(
        ["in.snirf", "/bids", "--participant-label", "01", "--task-label", "tap"])
    assert (args.participant_label, args.task_label, args.overwrite) == ("01", "tap", False)


def test_prep_subcommands_dispatch():
    assert prep_cli._build_parser().parse_args(
        ["crop", "/b", "/d", "--participant-label", "01", "02", "--tmin", "5"]
    ).func is prep_cli.cmd_crop
    assert prep_cli._build_parser().parse_args(
        ["edit-markers", "export", "/b", "/o", "--participant-label", "01"]
    ).func is prep_cli.cmd_markers_export


def test_prep_participant_label_required():
    with pytest.raises(SystemExit):
        prep_cli._build_parser().parse_args(["crop", "/b", "/d", "--tmin", "5"])


def test_qc_subcommands_and_fmin_dest():
    args = qc_cli._build_parser().parse_args(
        ["hyper-raw", "/b", "/o", "--pairs-csv", "p.csv", "--dpf", "6.0",
         "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
         "--fmin", "0.02", "--fmax", "0.2"]
    )
    assert args.func is qc_cli.cmd_hyper_raw
    assert args.coherence_fmin == 0.02
    assert args.coherence_fmax == 0.2


def test_hyper_stage_and_band_flags():
    args = hyper_cli._parsers()["fnirs-hyper"].parse_args(
        ["/deriv", "/o", "group", "--pairs-csv", "p.csv", "--desc", "errts",
         "--wtc-band-fmin", "0.03", "--wtc-band-fmax", "0.10"]
    )
    assert hyper_cli.COMMANDS["fnirs-hyper"] is hyper_cli.cmd_run
    assert (args.desc, args.wtc_band_fmin, args.wtc_band_fmax) == ("errts", 0.03, 0.10)


def test_hyper_reads_preproc_unless_told_otherwise():
    # the band bounds stay None so the report can say it averaged the whole axis
    args = hyper_cli._parsers()["fnirs-hyper"].parse_args(
        ["/deriv", "/o", "group", "--pairs-csv", "p.csv"])
    assert args.desc == "preproc"
    assert (args.wtc_band_fmin, args.wtc_band_fmax) == (None, None)


def test_hyper_reads_derivatives_and_never_raw_bids():
    """Every command here reads a derivatives tree, so none accepts a raw BIDS positional.

    Only the two that open a member's own recording take a source tree at all; giving the
    rest one would be a positional they never read.
    """
    cases = {
        "fnirs-hyper":          ["/deriv", "/o", "group", "--pairs-csv", "p.csv"],
        "fnirs-hyper-pairnull": ["/deriv", "/o", "group", "--pairs-csv", "p.csv"],
        "fnirs-hyper-band":     ["/o", "group", "--wtc-band-fmin", "0.1",
                                 "--wtc-band-fmax", "0.2"],
        "fnirs-hyper-merge":    ["/o", "group"],
    }
    reads_subjects = {"fnirs-hyper", "fnirs-hyper-pairnull"}
    for prog, argv in cases.items():
        args = hyper_cli._parsers()[prog].parse_args(argv)
        assert args.output_dir == Path("/o")
        assert not hasattr(args, "bids_dir")
        assert hasattr(args, "derivatives_dir") is (prog in reads_subjects)


def test_the_band_flags_are_shared_between_run_and_band():
    """One name per parameter: `band` reuses the `run` flags rather than carrying
    --band-fmin / --mask-coi under a second name that has to be kept in step."""
    flags = {f for parser in hyper_cli._parsers().values()
             for a in parser._actions for f in a.option_strings}
    assert not ({"--band-fmin", "--band-fmax", "--mask-coi", "--suffix"} & flags)
    for prog, argv in (("fnirs-hyper", ["/deriv", "/o", "group", "--pairs-csv", "p.csv"]),
                       ("fnirs-hyper-band",
                        ["/o", "group", "--wtc-band-fmin", "0.1", "--wtc-band-fmax", "0.2"])):
        args = hyper_cli._parsers()[prog].parse_args(argv)
        assert hasattr(args, "wtc_band_fmin") and hasattr(args, "wtc_mask_coi")


def test_band_needs_a_band(capsys):
    with pytest.raises(SystemExit):
        hyper_cli.main_band(["/o", "group"])
    assert "--wtc-band-fmin" in capsys.readouterr().err


def test_moved_commands_are_gone_from_qc():
    import argparse as _ap
    sub = [a for a in qc_cli._build_parser()._actions
           if isinstance(a, _ap._SubParsersAction)][0]
    assert set(sub.choices) == {
        "prep-raw", "hyper-raw", "cohort", "cohort-hyper", "provenance"}


def test_rate_subcommands():
    assert rate_cli._build_parser().parse_args(["rate", "/out"]).func is rate_cli.cmd_rate
    assert rate_cli._build_parser().parse_args(
        ["hyper", "/out", "--group-id", "A", "--task-label", "tap", "--pairs-csv", "p.csv"]
    ).func is rate_cli.cmd_hyper


def test_missing_subcommand_errors():
    for mod in (prep_cli, qc_cli, rate_cli, db_cli):
        with pytest.raises(SystemExit):
            mod._build_parser().parse_args([])


def test_qc_provenance_dispatch():
    args = qc_cli._build_parser().parse_args(["provenance", "/out"])
    assert args.func is qc_cli.cmd_provenance
    assert args.output_dir == Path("/out")


def test_qc_provenance_requires_output_dir():
    with pytest.raises(SystemExit):
        qc_cli._build_parser().parse_args(["provenance"])


def test_qc_provenance_accepts_no_options():
    # it rebuilds the graph from sidecars alone; an option here would mean the picture
    # depends on something the run did not record
    with pytest.raises(SystemExit):
        qc_cli._build_parser().parse_args(["provenance", "/out", "--cardiac-l-freq", "0.7"])


@pytest.mark.parametrize("argv", [
    pytest.param(["prep-raw", "/b", "/o", "--participant-label", "01", "--dpf", "6.0"],
                 id="prep-raw"),
    pytest.param(["hyper-raw", "/b", "/o", "--pairs-csv", "p.csv", "--dpf", "6.0"], id="hyper-raw"),
])
def test_cardiac_band_stays_required(argv, capsys):
    # The band is population-dependent and drives SCI, PSP and Cardiac Power. A default
    # would compute all three against the wrong band instead of asking.
    with pytest.raises(SystemExit):
        qc_cli._build_parser().parse_args(argv)
    assert "--cardiac-l-freq" in capsys.readouterr().err
