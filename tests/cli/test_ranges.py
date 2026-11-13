"""A range typed backwards is refused before anything runs. A high-pass above the low-pass
filtered as a band-stop, and a coherence range typed backwards ran every transform, wrote no
coherence table and reported success."""

import sys
import types

import pytest

from nirspipe.cli import _shared
from nirspipe.cli import hyper as hyper_cli
from nirspipe.cli import prep as prep_cli
from nirspipe.cli import qc as qc_cli
from nirspipe.cli import run as run_cli

RUN = ["bids", "out", "participant", "--population", "adult", "--sci-threshold", "0.8"]
HYPER = ["deriv", "out", "group", "--pairs-csv", "no_such_pairs.csv"]


def _refused(main, argv, capsys) -> str:
    with pytest.raises(SystemExit) as exit_:
        main(argv)
    assert exit_.value.code == 1
    return capsys.readouterr().err.strip()


@pytest.fixture
def no_workflows(monkeypatch):
    """The pipeline's workflow module replaced, so reaching it is a test failure."""
    fake = types.ModuleType("nirspipe.cli.workflows")
    fake.run_participant_level = fake.run_group_level = lambda opts: pytest.fail("the run started")
    monkeypatch.setitem(sys.modules, "nirspipe.cli.workflows", fake)


# ---- the rule ----

@pytest.mark.parametrize("low, high", [(2.0, 1.0), (1.0, 1.0), (5, 0.5)])
def test_a_lower_edge_not_below_the_upper_is_refused(low, high, capsys):
    with pytest.raises(SystemExit):
        _shared.refuse_inverted(low, high, "--lo", "--hi")
    assert capsys.readouterr().err.strip() == f"Error: --lo {low:g} has to be below --hi {high:g}."


@pytest.mark.parametrize("low, high", [(0.5, 1.0), (None, 1.0), (2.0, None), (None, None),
                                       (2.0, "none"), ("auto", 1.0)])
def test_an_ordered_or_open_range_passes(low, high):
    _shared.refuse_inverted(low, high, "--lo", "--hi")


# ---- nirspipe and nirspipe-qc ----

def test_a_cardiac_band_typed_backwards_stops_the_pipeline(no_workflows, capsys):
    err = _refused(run_cli.main, ["bids", "out", "participant", "--dpf", "6",
                                  "--sci-threshold", "0.8", "--cardiac-l-freq", "2",
                                  "--cardiac-h-freq", "1", "--resp-l-freq", "0.1",
                                  "--resp-h-freq", "0.5"], capsys)
    assert err == "Error: --cardiac-l-freq 2 has to be below --cardiac-h-freq 1."


def test_an_edge_below_the_one_the_population_fills_stops_it_too(no_workflows, capsys):
    err = _refused(run_cli.main, [*RUN, "--resp-h-freq", "0.05"], capsys)
    assert err == "Error: --resp-l-freq 0.1 has to be below --resp-h-freq 0.05."


def test_an_epoch_window_typed_backwards_stops_the_pipeline(no_workflows, capsys):
    err = _refused(run_cli.main, [*RUN, "--epoch-tmin", "5", "--epoch-tmax", "-5"], capsys)
    assert err == "Error: --epoch-tmin 5 has to be below --epoch-tmax -5."


def test_one_epoch_edge_alone_is_still_refused(no_workflows, capsys):
    err = _refused(run_cli.main, [*RUN, "--epoch-tmin", "-2"], capsys)
    assert err == "Error: --epoch-tmin and --epoch-tmax must be given together."


def test_raw_qc_refuses_the_same_bands(capsys):
    err = _refused(qc_cli.main, [*RUN, "--cardiac-h-freq", "0.5"], capsys)
    assert err == "Error: --cardiac-l-freq 0.7 has to be below --cardiac-h-freq 0.5."


def test_raw_qc_refuses_an_epoch_window_typed_backwards(tmp_path, capsys):
    (tmp_path / "bids").mkdir()
    err = _refused(qc_cli.main, [str(tmp_path / "bids"), str(tmp_path / "qc"), "participant",
                                 "--population", "adult", "--sci-threshold", "0.8",
                                 "--skip-bids-validation", "--epoch-tmin", "3",
                                 "--epoch-tmax", "1"], capsys)
    assert err.endswith("Error: --epoch-tmin 3 has to be below --epoch-tmax 1.")
    assert not (tmp_path / "qc").exists()


def test_raw_qc_group_refuses_a_window_typed_backwards(tmp_path, capsys):
    err = _refused(qc_cli.main, [str(tmp_path / "bids"), str(tmp_path / "qc"), "group",
                                 "--pairs-csv", "p.csv", "--tstart", "60", "--tend", "30"], capsys)
    assert err == "Error: --tstart 60 has to be below --tend 30."


# ---- the passband, checked on the settings as resolved ----

SELECT = ["--population", "adult", "--sci-threshold", "0.8", "--participant-label", "01",
          "--skip-bids-validation", "--dry-run"]


def test_a_high_pass_above_the_low_pass_stops_before_any_subject(mini_bids, tmp_path, capsys):
    out = tmp_path / "out"
    err = _refused(run_cli.main, [str(mini_bids), str(out), "participant", *SELECT,
                                  "--mode", "denoise", "--high-pass", "0.5", "--low-pass", "0.1"],
                   capsys)
    assert err.endswith("Error: --high-pass 0.5 has to be below --low-pass 0.1.")
    assert not list(out.glob("sub-*"))


def test_a_cutoff_crossing_the_modes_own_is_refused_and_says_whose_it_is(mini_bids, tmp_path,
                                                                        capsys):
    out = tmp_path / "out"
    err = _refused(run_cli.main, [str(mini_bids), str(out), "participant", *SELECT,
                                  "--mode", "rest", "--high-pass", "0.1"], capsys)
    assert err.endswith("Error: --high-pass 0.1 has to be below --low-pass 0.08 "
                        "(the rest mode's default).")
    assert not list(out.glob("sub-*"))


def test_a_cutoff_crossing_the_config_files_is_refused_and_says_so(mini_bids, tmp_path, capsys):
    config = tmp_path / "settings.toml"
    config.write_text('[post]\nmode = "denoise"\nhigh_pass = 0.3\n', encoding="utf-8")
    err = _refused(run_cli.main, [str(mini_bids), str(tmp_path / "out"), "participant", *SELECT,
                                  "--config", str(config), "--low-pass", "0.2"], capsys)
    assert err.endswith("Error: --high-pass 0.3 (from --config) has to be below --low-pass 0.2.")


def test_a_low_pass_switched_off_leaves_nothing_to_cross(mini_bids, tmp_path):
    out = tmp_path / "out"
    run_cli.main([str(mini_bids), str(out), "participant", *SELECT, "--mode", "rest",
                  "--high-pass", "0.1", "--low-pass", "none"])
    assert list((out / "sub-01" / "logs").glob("sub-01*.toml"))


# ---- nirspipe-hyper: refused before the pairs table is opened ----

@pytest.mark.parametrize("flags, said", [
    (["--wtc-fmin", "0.2", "--wtc-fmax", "0.02"],
     "Error: --wtc-fmin 0.2 has to be below --wtc-fmax 0.02."),
    (["--band-fmin", "0.15", "--band-fmax", "0.05"],
     "Error: --band-fmin 0.15 has to be below --band-fmax 0.05."),
    (["--wtc-band-fmin", "0.15", "--wtc-band-fmax", "0.05"],
     "Error: --wtc-band-fmin 0.15 has to be below --wtc-band-fmax 0.05."),
    (["--band-fmin", "0.05", "--band-fmax", "0.15", "--isc-fmax", "0.03"],
     "Error: --isc-fmin 0.05 has to be below --isc-fmax 0.03."),
    (["--tstart", "100", "--tend", "100"],
     "Error: --tstart 100 has to be below --tend 100."),
    (["--wtc-fmin", "0.02", "--wtc-fmax", "0.2", "--band-fmin", "0.3", "--band-fmax", "0.5"],
     "Error: the coherence band 0.3 to 0.5 Hz lies outside the computed axis, "
     "--wtc-fmin 0.02 to --wtc-fmax 0.2."),
    (["--wtc-band-fmax", "0.003"],
     "Error: the coherence band 0.004 to 0.003 Hz lies outside the computed axis, "
     "--wtc-fmin 0.004 to --wtc-fmax 0.2."),
])
def test_a_dyad_range_that_cannot_hold_a_value_is_refused(flags, said, tmp_path, capsys):
    err = _refused(hyper_cli.main, ["deriv", str(tmp_path / "out"), "group",
                                    "--pairs-csv", "no_such_pairs.csv", *flags], capsys)
    assert err == said
    assert not (tmp_path / "out").exists()


def test_a_band_inside_the_axis_reaches_the_pairs_table(tmp_path, capsys):
    err = _refused(hyper_cli.main, ["deriv", str(tmp_path / "out"), "group",
                                    "--pairs-csv", str(tmp_path / "no_such_pairs.csv"),
                                    "--wtc-fmin", "0.02", "--wtc-fmax", "0.2",
                                    "--band-fmin", "0.05", "--band-fmax", "0.15",
                                    "--tstart", "0", "--tend", "100"], capsys)
    assert "no_such_pairs.csv" in err and "has to be below" not in err


def test_rebanding_refuses_a_band_typed_backwards(tmp_path, capsys):
    err = _refused(hyper_cli.main_band, [str(tmp_path), "group", "--wtc-band-fmin", "0.2",
                                         "--wtc-band-fmax", "0.05"], capsys)
    assert err == "Error: --wtc-band-fmin 0.2 has to be below --wtc-band-fmax 0.05."


# ---- nirspipe-prep crop ----

def test_a_crop_window_typed_backwards_is_refused(tmp_path, capsys):
    (tmp_path / "bids").mkdir()
    err = _refused(prep_cli.main, ["crop", str(tmp_path / "bids"), str(tmp_path / "out"),
                                   "--participant-label", "01", "--skip-bids-validation",
                                   "--tmin", "60", "--tmax", "30"], capsys)
    assert err == "Error: --tmin 60 has to be below --tmax 30."
    assert not (tmp_path / "out").exists()
