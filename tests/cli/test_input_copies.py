"""The files a run's flags name are copied beside its record, and --config reads the copies."""

import hashlib
import tomllib

import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.cli import run as run_cli

from tests._fingerprint import CLI_ARGS, EVENT_DURATION, EVENT_ONSETS, make_fingerprint_dataset

SELECT = ("--participant-label", "01", "--skip-bids-validation", "--no-report")
GLM = ("--mode", "glm", "--noise-model", "ols", "--drift-high-pass", "0.01")


def _record(out, name="sub-01.toml"):
    path = out / "sub-01" / "logs" / name
    return path, tomllib.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("input_copies")
    bids, _ = make_fingerprint_dataset(root)
    contrasts = root / "contrasts.tsv"
    contrasts.write_text("contrast\ttapping\nmain\t1\n", encoding="utf-8")
    events = root / "events"
    events.mkdir()
    rows = "".join(f"{o:g}\t{EVENT_DURATION:g}\ttapping\n" for o in EVENT_ONSETS)
    (events / "sub-01_task-tapping_events.tsv").write_text(
        "onset\tduration\ttrial_type\n" + rows, encoding="utf-8")
    return root, bids, contrasts, events


@pytest.fixture(scope="module")
def first(dataset):
    root, bids, contrasts, events = dataset
    out = root / "first"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, *GLM, *SELECT,
                  "--contrast-file", str(contrasts),
                  "--events-path", str(events / "sub-{sub}_task-{task}_events.tsv")])
    return out


def test_a_file_is_copied_beside_the_record_with_its_hash(dataset, first):
    _, _, contrasts, _ = dataset
    record, got = _record(first)
    entry = got["inputs"]["contrast_file"]
    copy = record.parent / entry["copy"]
    assert entry["copy"] == "inputs/sub-01_contrast-file.tsv"
    assert copy.read_bytes() == contrasts.read_bytes()
    assert entry["sha256"] == _sha(contrasts)
    assert entry["given"] == contrasts.resolve().as_posix()


def test_each_runs_events_table_is_copied_under_the_templates_placeholders(dataset, first):
    _, _, _, events = dataset
    record, got = _record(first)
    entry = got["inputs"]["events_path"]
    assert entry["copy"] == "inputs/sub-01_events-path_sub-{sub}_task-{task}.tsv"
    name = "sub-01_events-path_sub-01_task-tapping.tsv"
    assert entry["sha256"] == {name: _sha(events / "sub-01_task-tapping_events.tsv")}
    assert (record.parent / "inputs" / name).is_file()


def test_a_rerun_reads_the_copies_once_the_originals_are_gone(dataset, first, tmp_path):
    root, bids, contrasts, events = dataset
    record, got = _record(first)
    moved = tmp_path / "moved"
    moved.mkdir()
    contrasts.rename(moved / contrasts.name)
    events.rename(moved / "events")
    try:
        again = tmp_path / "again"
        run_cli.main([str(bids), str(again), "participant", "--config", str(record), *SELECT])
        _, rerun = _record(again)
    finally:
        (moved / contrasts.name).rename(contrasts)
        (moved / "events").rename(events)
    assert rerun["outcome"]["status"] == "success"
    # the first given paths carry over, the copies are taken again
    for key in ("contrast_file", "events_path"):
        assert rerun["inputs"][key]["given"] == got["inputs"][key]["given"]
        assert rerun["inputs"][key]["sha256"] == got["inputs"][key]["sha256"]


def test_a_record_given_as_config_is_not_copied_but_a_settings_file_is(dataset, first, tmp_path):
    _, bids, _, _ = dataset
    record, _ = _record(first)
    from_record = tmp_path / "from_record"
    run_cli.main([str(bids), str(from_record), "participant", "--config", str(record),
                  *SELECT, "--dry-run"])
    assert "config" not in _record(from_record)[1]["inputs"]
    settings = tmp_path / "settings.toml"
    settings.write_text("[prep]\nsci_threshold = 0.8\n", encoding="utf-8")
    from_file = tmp_path / "from_file"
    run_cli.main([str(bids), str(from_file), "participant", *CLI_ARGS, "--config", str(settings),
                  *SELECT, "--dry-run"])
    assert _record(from_file)[1]["inputs"]["config"]["sha256"] == _sha(settings)


def test_raw_qc_copies_its_tables_too(dataset, tmp_path):
    _, bids, _, _ = dataset
    keep = tmp_path / "keep.tsv"
    keep.write_text("task\tonset\tduration\ntapping\t0\t300\n", encoding="utf-8")
    out = tmp_path / "qc"
    qc_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--keep-spans", str(keep),
                 "--participant-label", "01", "--skip-bids-validation"])
    _, got = _record(out, "sub-01_desc-raw.toml")
    assert got["inputs"]["keep_spans"]["sha256"] == _sha(keep)
