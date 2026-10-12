"""What a page does with what is typed into its form."""

from __future__ import annotations

from nirspipe.interface.callbacks import data_prep_callbacks, hyper_analysis_callbacks


def test_data_preparation_takes_an_output_directory_that_does_not_exist_yet(tmp_path):
    missing = tmp_path / "not-yet"
    _, status = data_prep_callbacks.load_run(
        "sub-01_task-x_nirs.snirf", 0.8, 0.7, 2.0, 6.0, None, None, None, None, [],
        None, None, None, None, None, None, str(missing))
    # past the output check, stopped by the next one
    assert status.children == "Set BIDS Directory first."


def _hyper_values(**opts):
    return [opts.get(key) for key in hyper_analysis_callbacks._KEYS]


def test_every_hyper_field_has_its_key():
    assert len(hyper_analysis_callbacks._KEYS) == len(hyper_analysis_callbacks._STATES)


def test_the_shared_band_reaches_the_command_and_the_overrides_follow_it():
    _, stored = hyper_analysis_callbacks.generate_command(
        1, "run", *_hyper_values(output_dir="/hyper", derivatives_dir="/deriv",
                                 pairs_csv="/pairs.csv", shared_band_fmin=0.02,
                                 shared_band_fmax=0.1, isc_fmin=0.05), "bash")
    argv = stored["argv"]
    assert argv[argv.index("--band-fmin") + 1] == "0.02"
    assert argv[argv.index("--band-fmax") + 1] == "0.1"
    assert argv[argv.index("--isc-fmin") + 1] == "0.05"
    assert "--wtc-band-fmin" not in argv
