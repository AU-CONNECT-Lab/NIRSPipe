"""The run record is written before any recording is read, so it cannot state a value that differs per recording."""

import tomllib

import pytest

from nirspipe.cli import run as run_cli

_ARGS = ("--dpf", "6.0", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
         "--cardiac-h-freq", "1.5", "--resp-l-freq", "0.2", "--resp-h-freq", "0.5",
         "--participant-label", "01", "--task-label", "tapping", "--no-report",
         "--skip-bids-validation", "--dry-run")


def _prep_section(bids_dir, out_dir, *flags) -> dict:
    run_cli.main([str(bids_dir), str(out_dir), "participant", *_ARGS, *flags])
    record = next((out_dir / "sub-01" / "logs").glob("sub-01*.toml"))
    return tomllib.loads(record.read_text(encoding="utf-8"))["prep"]


@pytest.fixture(scope="module")
def tabled_prep(mini_bids, tmp_path_factory):
    root = tmp_path_factory.mktemp("record_tables")
    keep, bad, chans = root / "keep.tsv", root / "bad.tsv", root / "bad_channels.tsv"
    keep.write_text("participant_id\ttask\tonset\tduration\nsub-01\ttapping\t0\t200\n",
                    encoding="utf-8")
    bad.write_text("participant_id\ttask\tonset\tduration\nsub-01\ttapping\t100\t20\n",
                   encoding="utf-8")
    chans.write_text("participant_id\ttask\tbad_channels\nsub-01\ttapping\tS1_D1\n",
                     encoding="utf-8")
    return _prep_section(mini_bids, root / "out",
                         "--keep-spans", str(keep), "--keep-spans-from", "tapping",
                         "--bad-spans", str(bad), "--bad-channels", str(chans))


@pytest.mark.parametrize("field", ["keep_spans", "bad_spans", "channels_tsv_bads", "bad_channels"])
def test_the_record_leaves_out_what_differs_per_recording(tabled_prep, field):
    assert field not in tabled_prep


def test_the_record_never_names_the_keep_spans_marker(tabled_prep):
    assert "keep_spans_from" not in tabled_prep


def test_a_bad_channels_list_applies_to_every_recording_and_stays(mini_bids, tmp_path):
    prep = _prep_section(mini_bids, tmp_path / "out", "--bad-channels", "S1_D1")
    assert prep["bad_channels"] == ["S1_D1"]
