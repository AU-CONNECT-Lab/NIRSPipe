"""`nirspipe-hyper --config` reruns a dyad from its run record, as `nirspipe --config` does a subject."""

import tomllib

import pytest

from nirspipe.cli import hyper as hyper_cli
from nirspipe.cli.hyper import read_hyper_config

TASK = "hold"


def _record(tmp_path, body: str, version: str = ""):
    path = tmp_path / "group-G01_task-hold.toml"
    env = f'[environment]\nnirspipe_version = "{version}"\n' if version else ""
    path.write_text(env + body, encoding="utf-8")
    return path


# ---- reading ----

def test_a_record_s_own_inputs_and_bands_are_skipped(tmp_path):
    path = _record(tmp_path, '[execution]\ngroup_id = "G01"\n[hyper]\n'
                             'derivatives_dir = "C:/d"\npairs_csv = "C:/p.csv"\n'
                             'check_only = false\nno_report = true\nsep_short_max_mm = 12.0\n'
                             'wtc_fmin = 0.02\nwtc_chroma = ["hbo", "hbt"]\n'
                             'task_label = ["hold"]\n')
    assert read_hyper_config(path)["hyper"] == {"wtc_fmin": 0.02, "wtc_chroma": ["hbo", "hbt"],
                                                "task_label": ["hold"]}


def test_a_value_the_run_derived_from_the_shorthand_is_dropped_and_a_typed_one_kept(tmp_path):
    path = _record(tmp_path, "[hyper]\nband_fmin = 0.05\nband_fmax = 0.1\n"
                             "wtc_band_fmin = 0.05\nwtc_band_fmax = 0.08\n"
                             "isc_fmin = 0.05\nisc_fmax = 0.1\n"
                             "wtc_channel_cross = true\nwtc_phase_null_cross = true\n")
    assert read_hyper_config(path)["hyper"] == {"band_fmin": 0.05, "band_fmax": 0.1,
                                                "wtc_band_fmax": 0.08,
                                                "wtc_channel_cross": True}


@pytest.mark.parametrize("body, said", [
    ("[hyper]\nwtc_fmn = 0.02\n", "[hyper] has no setting 'wtc_fmn'"),
    ("[post]\nmode = \"rest\"\n", "no section [post]"),
    ("wtc_fmin = 0.02\n", "wtc_fmin sits outside any section"),
    ("[hyper]\nwtc_chroma = \"both\"\n", "expected one of"),
])
def test_a_file_that_would_drop_a_setting_is_refused(tmp_path, capsys, body, said):
    with pytest.raises(SystemExit):
        read_hyper_config(_record(tmp_path, body))
    assert said in capsys.readouterr().err


# ---- a rerun ----

@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One dyad analysed, then rerun from its record alone and with a new band over it."""
    from nirspipe.cli import run as run_cli
    from tests._fingerprint import CLI_ARGS
    from tests._synth import make_hyper_dataset

    root = tmp_path_factory.mktemp("hyper_config")
    bids, pairs = make_hyper_dataset(root, tasks=(TASK,))
    deriv = root / "deriv"
    run_cli.main([str(bids), str(deriv), "participant", *CLI_ARGS, "--skip-bids-validation"])
    base = [str(deriv), None, "group", "--pairs-csv", str(pairs)]

    def run(name, *extra):
        out = root / name
        hyper_cli.main([*(str(out) if a is None else a for a in base), *extra])
        return out

    first = run("first", "--wtc-fmin", "0.02", "--band-fmin", "0.03", "--band-fmax", "0.12",
                "--no-channel-cross", "--wtc-chroma", "hbo", "--wtc-phase-null", "2",
                "--isc-phase-null", "3", "--wtc-seed", "0")
    record = next(first.rglob("logs/group-*.toml"))
    # the same run with no seed given, then rerun from its record
    drawn = run("drawn", "--wtc-fmin", "0.02", "--band-fmin", "0.03", "--band-fmax", "0.12",
                "--no-channel-cross", "--wtc-chroma", "hbo", "--wtc-phase-null", "2",
                "--isc-phase-null", "3")
    drawn_record = next(drawn.rglob("logs/group-*.toml"))
    return {"first": first, "record": record,
            "again": run("again", "--config", str(record)),
            "rebanded": run("rebanded", "--config", str(record), "--band-fmin", "0.05"),
            "drawn": drawn, "drawn_again": run("drawn_again", "--config", str(drawn_record))}


def _hyper(out):
    return tomllib.loads(next(out.rglob("logs/group-*.toml")).read_text(encoding="utf-8"))


def test_a_rerun_from_the_record_alone_writes_every_table_bit_identical(runs):
    first = sorted(p.relative_to(runs["first"]) for p in runs["first"].rglob("*.tsv"))
    assert first == sorted(p.relative_to(runs["again"]) for p in runs["again"].rglob("*.tsv"))
    for rel in first:
        assert (runs["first"] / rel).read_bytes() == (runs["again"] / rel).read_bytes(), rel


def test_the_rerun_records_the_same_settings_and_names_the_file(runs):
    first, again = _hyper(runs["first"]), _hyper(runs["again"])
    skip = {"derivatives_dir", "pairs_csv"}
    assert ({k: v for k, v in again["hyper"].items() if k not in skip}
            == {k: v for k, v in first["hyper"].items() if k not in skip})
    assert again["execution"]["config"] == runs["record"].as_posix()
    assert "config" not in again["hyper"]


def test_a_shorthand_typed_over_the_record_reaches_both_measures(runs):
    hyper = _hyper(runs["rebanded"])["hyper"]
    assert hyper["band_fmin"] == hyper["wtc_band_fmin"] == hyper["isc_fmin"] == 0.05
    assert hyper["wtc_band_fmax"] == hyper["isc_fmax"] == 0.12
    assert hyper["wtc_phase_null_cross"] is False


# ---- an unseeded run ----

def test_an_unseeded_run_records_the_seed_it_drew(runs):
    seed = _hyper(runs["drawn"])["hyper"]["wtc_seed"]
    assert isinstance(seed, int) and seed >= 0
    assert _hyper(runs["drawn_again"])["hyper"]["wtc_seed"] == seed


def test_an_unseeded_run_s_rerun_repeats_its_null_draws(runs):
    tables = sorted(p.relative_to(runs["drawn"]) for p in runs["drawn"].rglob("*.tsv"))
    assert any("null-phase" in p.name for p in tables)
    for rel in tables:
        assert (runs["drawn"] / rel).read_bytes() == (runs["drawn_again"] / rel).read_bytes(), rel


def test_an_unseeded_group_null_draws_one_seed_for_every_table_and_prints_it(
        tmp_path, monkeypatch, capsys):
    # the cohort sidecar records the seed it is handed; test_pair_null_group covers repeating
    from nirspipe.pipeline.hyper import pair_null_group
    from nirspipe.qc.hyper import groupnull_report

    seeds = []
    monkeypatch.setattr(pair_null_group, "nulls_on_disk", lambda out, task: ["phase", "repaired"])
    monkeypatch.setattr(pair_null_group, "chromas_on_disk", lambda out, task, null: ["hbo", "hbr"])
    monkeypatch.setattr(pair_null_group, "write_group_null",
                        lambda *a, seed, **k: seeds.append(seed) or [])
    monkeypatch.setattr(groupnull_report, "write_groupnull_report", lambda out, task: None)
    hyper_cli.main_group_null([str(tmp_path), "group", "--task-label", TASK])
    assert len(seeds) == 4 and len(set(seeds)) == 1 and isinstance(seeds[0], int)
    assert f"--seed {seeds[0]}" in capsys.readouterr().err


def test_a_record_from_another_version_warns_and_runs(tmp_path, capsys):
    path = _record(tmp_path, "[hyper]\nwtc_fmin = 0.02\n", version="0.1.0")
    args = hyper_cli._parsers()["nirspipe-hyper"].parse_args(
        ["d", "o", "group", "--pairs-csv", "p.csv", "--config", str(path)])
    hyper_cli._apply_config(args, ["d", "o", "group", "--pairs-csv", "p.csv",
                                   "--config", str(path)])
    assert args.wtc_fmin == 0.02
    assert "written by nirspipe 0.1.0" in capsys.readouterr().err
