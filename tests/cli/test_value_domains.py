"""A value outside what its flag can mean is refused as it is parsed, on the command line and
in a --config file, where it ran and failed every subject or wrote a result nobody asked for."""

import pytest

from nirspipe.cli import hyper as hyper_cli
from nirspipe.cli import prep as prep_cli
from nirspipe.cli import qc as qc_cli
from nirspipe.cli import rate as rate_cli
from nirspipe.cli import run as run_cli
from nirspipe.interface.commands import parser_for
from nirspipe.interface.forms import fields

RUN = ["bids", "out", "participant", "--population", "adult"]
HYPER = ["deriv", "out", "group", "--pairs-csv", "p.csv"]
CROP = ["crop", "bids", "out", "--participant-label", "01"]

_hyper = lambda prog: lambda: hyper_cli._parsers()[prog]          # noqa: E731

REFUSED = [
    ("a coupling index above 1", run_cli._build_parser, [*RUN, "--sci-threshold", "80"],
     "argument --sci-threshold: must be between -1 and 1, got 80"),
    ("a share above 1", run_cli._build_parser, [*RUN, "--min-good-frac", "5"],
     "argument --min-good-frac: must be between 0 and 1, got 5"),
    ("a window of no length", run_cli._build_parser, [*RUN, "--window-length", "0"],
     "argument --window-length: must be greater than 0, got 0"),
    ("a window of negative length", run_cli._build_parser, [*RUN, "--window-length", "-1"],
     "argument --window-length: must be greater than 0, got -1"),
    ("a band edge at 0 Hz", run_cli._build_parser, [*RUN, "--cardiac-l-freq", "0"],
     "argument --cardiac-l-freq: must be greater than 0, got 0"),
    ("a pathlength factor of 0", run_cli._build_parser, [*RUN, "--dpf", "6", "0"],
     "argument --dpf: must be greater than 0, got 0"),
    ("a negative spectral power", run_cli._build_parser, [*RUN, "--psp-threshold", "-0.1"],
     "argument --psp-threshold: must be 0 or more, got -0.1"),
    ("a drift cutoff of 0 Hz", run_cli._build_parser, [*RUN, "--drift-high-pass", "0"],
     "argument --drift-high-pass: must be greater than 0, got 0"),
    ("a filter of order 0", run_cli._build_parser, [*RUN, "--filter-order", "0"],
     "argument --filter-order: must be 1 or more, got 0"),
    ("a rate that is not a number", run_cli._build_parser, [*RUN, "--resample-sfreq", "nan"],
     "argument --resample-sfreq: must be greater than 0, got nan"),
    ("raw QC's window", qc_cli._build_parser, [*RUN, "--window-length", "0"],
     "argument --window-length: must be greater than 0, got 0"),
    ("raw QC's coupling line", qc_cli._build_parser, [*RUN, "--sci-threshold", "1.5"],
     "argument --sci-threshold: must be between -1 and 1, got 1.5"),
    ("the viewer's second line", rate_cli._build_parser,
     ["raw", "out", "--participant-label", "01", "--sci-threshold", "2"],
     "argument --sci-threshold: must be between -1 and 1, got 2"),
    ("a coherence axis from 0 Hz", _hyper("nirspipe-hyper"), [*HYPER, "--wtc-fmin", "0"],
     "argument --wtc-fmin: must be greater than 0, got 0"),
    ("a coherence above 1", _hyper("nirspipe-hyper"), [*HYPER, "--wtc-arrow-min", "2"],
     "argument --wtc-arrow-min: must be between 0 and 1, got 2"),
    ("a negative padding", _hyper("nirspipe-hyper"), [*HYPER, "--wtc-cond-pad", "-5"],
     "argument --wtc-cond-pad: must be 0 or more, got -5"),
    ("a window of no length, dyad", _hyper("nirspipe-hyper"), [*HYPER, "--wtc-window", "0"],
     "argument --wtc-window: must be greater than 0, got 0"),
    ("a negative seed", _hyper("nirspipe-hyper"), [*HYPER, "--seed", "-1"],
     "argument --seed: must be 0 or more, got -1"),
    ("a negative lag", _hyper("nirspipe-hyper"), [*HYPER, "--isc-max-lag", "-1"],
     "argument --isc-max-lag: must be 0 or more, got -1"),
    ("no resamples", _hyper("nirspipe-hyper-groupnull"),
     ["out", "group", "--task-label", "hold", "--n-resample", "0"],
     "argument --n-resample: must be 1 or more, got 0"),
    ("no draws", _hyper("nirspipe-hyper-pairnull"), [*HYPER, "--wtc-pair-max", "0"],
     "argument --wtc-pair-max: must be 1 or more, got 0"),
    ("a negative margin", prep_cli._build_parser, [*CROP, "--tmin", "0", "--margin", "-1"],
     "argument --margin: must be 0 or more, got -1"),
    ("a band from 0 Hz", prep_cli._build_parser, [*CROP, "--tmin", "0", "--band-fmin", "0"],
     "argument --band-fmin: must be greater than 0, got 0"),
    ("a span of no length", prep_cli._build_parser,
     ["keep-spans", "bids", "out.tsv", "--participant-label", "01", "--conditions", "a",
      "--duration", "0"],
     "argument --duration: must be greater than 0, got 0"),
    ("a negative marker length", prep_cli._build_parser,
     ["edit-markers", "apply", "bids", "out", "--participant-label", "01",
      "--set-duration", "-1"],
     "argument --set-duration: must be 0 or more, got -1"),
]


@pytest.mark.parametrize("build, argv, said", [case[1:] for case in REFUSED],
                         ids=[case[0] for case in REFUSED])
def test_a_value_outside_its_domain_is_refused_by_the_parser(build, argv, said, capsys):
    with pytest.raises(SystemExit) as exit_:
        build().parse_args(argv)
    assert exit_.value.code == 2
    assert capsys.readouterr().err.strip().endswith(said)


def test_the_edges_of_each_domain_are_values():
    args = run_cli._build_parser().parse_args(
        [*RUN, "--sci-threshold", "1", "--min-good-frac", "0", "--psp-threshold", "0",
         "--min-time", "0", "--drift-order", "0", "--stim-dur", "0", "--dpf", "6.5", "age",
         "--drift-high-pass", "auto", "--gvtd-min-epoch", "0"])
    assert (args.sci_threshold, args.min_good_frac, args.psp_threshold, args.min_time,
            args.drift_order, args.dpf, args.drift_high_pass, args.gvtd_min_epoch) == (
        1.0, 0.0, 0.0, 0.0, 0, [6.5, "age"], "auto", 0.0)
    assert run_cli._build_parser().parse_args([*RUN, "--min-good-frac", "1"]).min_good_frac == 1.0
    hyper = hyper_cli._parsers()["nirspipe-hyper"].parse_args(
        [*HYPER, "--wtc-cond-pad", "0", "--isc-phase-null", "0", "--wtc-phase-null", "0",
         "--wtc-whiten", "0", "--wtc-arrow-min", "1", "--seed", "0"])
    assert (hyper.wtc_cond_pad, hyper.isc_phase_null, hyper.wtc_phase_null, hyper.wtc_whiten,
            hyper.wtc_arrow_min, hyper.seed) == (0.0, 0, 0, 0.0, 1.0, 0)


@pytest.mark.parametrize("typed, read", [("auto", "auto"), ("AUTO", "auto"), ("47.1", 47.1)])
def test_a_length_that_may_be_worked_out_takes_seconds_or_auto(typed, read):
    hyper = hyper_cli._parsers()["nirspipe-hyper"].parse_args([*HYPER, "--wtc-cond-pad", typed])
    crop = prep_cli._build_parser().parse_args([*CROP, "--tmin", "0", "--margin", typed])
    assert (hyper.wtc_cond_pad, crop.margin_s) == (read, read)


def test_a_length_that_is_neither_is_refused(capsys):
    with pytest.raises(SystemExit):
        hyper_cli._parsers()["nirspipe-hyper"].parse_args([*HYPER, "--wtc-cond-pad", "soon"])
    assert "argument --wtc-cond-pad: invalid seconds_or_auto value: 'soon'" in capsys.readouterr().err


# ---- the same domains through --config ----

@pytest.mark.parametrize("section, line, said", [
    ("prep", "sci_threshold = 80", "sci_threshold = 80: must be between -1 and 1, got 80"),
    ("prep", "min_good_frac = 5", "min_good_frac = 5: must be between 0 and 1, got 5"),
    ("post", "resample_sfreq = 0", "resample_sfreq = 0: must be greater than 0, got 0"),
])
def test_a_config_file_is_held_to_the_same_domains(section, line, said, tmp_path, capsys):
    config = tmp_path / "settings.toml"
    config.write_text(f"[{section}]\n{line}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_:
        run_cli.read_config(config)
    assert exit_.value.code == 1
    assert capsys.readouterr().err.strip() == f"Error: {config}: {said}"


def test_a_dyad_config_file_is_held_to_them_too(tmp_path, capsys):
    config = tmp_path / "dyad.toml"
    config.write_text("[hyper]\nwtc_fmin = 0\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        hyper_cli.read_hyper_config(config)
    assert "wtc_fmin = 0: must be greater than 0, got 0" in capsys.readouterr().err


# ---- the forms built from the parsers ----

def _kinds(prog, sub=None):
    parser = parser_for(prog)
    if sub:
        from nirspipe.interface.commands import subcommand
        parser = subcommand(parser, sub)
    return {field.name: field.kind for _title, group in fields(parser) for field in group}


def test_a_flag_with_a_domain_is_still_a_number_field():
    run = _kinds("nirspipe")
    assert {run[name] for name in ("--sci-threshold", "--min-good-frac", "--window-length",
                                   "--cardiac-l-freq", "--filter-order", "--nprocs")} == {"number"}
    hyper = _kinds("nirspipe-hyper")
    assert (hyper["--wtc-fmin"], hyper["--seed"]) == ("number", "number")


def test_a_flag_that_also_takes_a_word_stays_a_text_field():
    assert _kinds("nirspipe-hyper")["--wtc-cond-pad"] == "text"
    assert _kinds("nirspipe-prep", "crop")["--margin"] == "text"
    assert _kinds("nirspipe")["--drift-high-pass"] == "text"
