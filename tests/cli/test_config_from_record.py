"""A run record reads back through --config and reruns the run it records."""

import tomllib

import numpy as np
import pytest

from nirspipe import __version__
from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_snirf

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

QUIET = ("--participant-label", "01", "--no-report", "--skip-bids-validation")
# every kind of value a record holds: a switched-off mode default, a table, a plain list,
# a band in mm, a flag with an argparse default, a comma list, a list of choices
RUN = (*CLI_ARGS, "--mode", "denoise", "--low-pass", "none", "--drift-model", "polynomial",
       "--short-channel", "mean", "--short-max-dist", "12", "--bad-channels", "S2_D2",
       "--motion-correction", "wavelet", "--fir-delays", "0,1", "--export-mean-filled",
       "--chromophores", "hbt", "hbo")


def _record(out):
    return tomllib.loads((out / "sub-01" / "logs" / "sub-01.toml").read_text(encoding="utf-8"))


def _run(bids, out, *flags):
    run_cli.main([str(bids), str(out), "participant", *flags, *QUIET])
    return _record(out)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("config_from_record")
    bids, _ = make_fingerprint_dataset(root)
    keep = root / "keep.tsv"
    keep.write_text("task\tonset\tduration\ntask-tapping\t0\t300\n", encoding="utf-8")
    return root, bids, keep


@pytest.fixture(scope="module")
def first(dataset):
    root, bids, keep = dataset
    out = root / "first"
    return out, _run(bids, out, *RUN, "--keep-spans", str(keep))


@pytest.fixture(scope="module")
def rerun(dataset, first):
    root, bids, _ = dataset
    out = root / "rerun"
    return out, _run(bids, out, "--config", str(first[0] / "sub-01" / "logs" / "sub-01.toml"))


# ---- the rerun is the same run ----

def test_the_rerun_records_the_same_settings(first, rerun):
    assert rerun[1]["prep"] == first[1]["prep"]
    assert rerun[1]["post"] == first[1]["post"]


def test_the_rerun_writes_the_same_data(first, rerun):
    stages = sorted(p.relative_to(first[0]) for p in first[0].rglob("*_desc-*_nirs.snirf"))
    assert {p.name.split("_desc-")[1] for p in stages} >= {"errts_nirs.snirf", "meanfilled_nirs.snirf"}
    for rel in stages:
        a, b = read_snirf(first[0] / rel), read_snirf(rerun[0] / rel)
        assert np.array_equal(a.get_data(), b.get_data()), rel
        assert a.info["bads"] == b.info["bads"], rel


def test_a_switched_off_mode_default_stays_off(first, rerun):
    # denoise fills low_pass 0.5; a record that left it out would get it back
    assert first[1]["post"]["low_pass"] == rerun[1]["post"]["low_pass"] == "none"


def test_the_record_names_its_inputs_as_typed(first, dataset):
    _, _, keep = dataset
    prep = first[1]["prep"]
    assert prep["keep_spans"] == keep.resolve().as_posix()
    assert prep["bad_channels"] == "S2_D2"
    assert prep["short_max_dist"] == 12.0
    assert first[1]["post"]["fir_delays"] == "0,1"


def test_the_rerun_names_where_its_settings_came_from(rerun):
    assert rerun[1]["execution"]["config"].endswith("first/sub-01/logs/sub-01.toml")
    assert rerun[1]["post_sources"]["mode"] == "config"
    assert rerun[1]["post_sources"]["drift_model"] == "config"


# ---- the command line over the file ----

def test_a_flag_typed_beats_the_file_and_one_left_out_does_not(dataset, first, tmp_path):
    _, bids, _ = dataset
    record = first[0] / "sub-01" / "logs" / "sub-01.toml"
    got = _run(bids, tmp_path / "out", "--config", str(record), "--sci-threshold", "0.6",
               "--dry-run")
    assert got["prep"]["sci_threshold"] == 0.6
    # argparse fills tddr when the flag is absent; the file's wavelet must still win
    assert got["prep"]["motion_correction"] == "wavelet"


def test_a_file_supplies_the_flags_a_participant_run_requires(dataset, tmp_path):
    _, bids, _ = dataset
    config = tmp_path / "settings.toml"
    config.write_text("[prep]\ndpf = [6.0, 6.0]\nsci_threshold = 0.8\ncardiac_l_freq = 0.7\n"
                      "cardiac_h_freq = 1.5\nresp_l_freq = 0.2\nresp_h_freq = 0.5\n",
                      encoding="utf-8")
    got = _run(bids, tmp_path / "out", "--config", str(config), "--dry-run")
    assert got["prep"]["sci_threshold"] == 0.8
    assert "post" not in got


def test_another_version_warns_and_still_runs(dataset, first, tmp_path, capsys):
    _, bids, _ = dataset
    text = (first[0] / "sub-01" / "logs" / "sub-01.toml").read_text(encoding="utf-8")
    old = tmp_path / "old.toml"
    old.write_text(text.replace(f'nirspipe_version = "{__version__}"', 'nirspipe_version = "0.1.0"'),
                   encoding="utf-8")
    got = _run(bids, tmp_path / "out", "--config", str(old), "--dry-run")
    assert "written by nirspipe 0.1.0" in capsys.readouterr().err
    assert got["prep"]["motion_correction"] == "wavelet"


# ---- what the file cannot say ----

@pytest.mark.parametrize("text, names", [
    ('high_pass = 0.01\n', "high_pass sits outside any section"),
    ('[prep]\nsci_treshold = 0.8\n', "no setting 'sci_treshold'"),
    ('[post]\ncardiac_l_freq = 0.7\n', "no setting 'cardiac_l_freq'"),
    ('[prep]\nmotion_correction = "tdr"\n', "expected one of"),
    ('[glm]\nstim_dur = 5\n', "no section [glm]"),
])
def test_a_file_the_flags_cannot_read_is_refused(dataset, tmp_path, capsys, text, names):
    _, bids, _ = dataset
    config = tmp_path / "bad.toml"
    config.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit):
        run_cli.main([str(bids), str(tmp_path / "out"), "participant", *CLI_ARGS,
                      "--config", str(config), *QUIET, "--dry-run"])
    assert names in capsys.readouterr().err
