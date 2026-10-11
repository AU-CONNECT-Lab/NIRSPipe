"""The run record is written before any recording is read, so it names each table as typed, never what it resolved to."""

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
    tables = {"keep_spans": keep, "bad_spans": bad, "bad_channels": chans}
    return tables, _prep_section(mini_bids, root / "out",
                                 "--keep-spans", str(keep), "--keep-spans-from", "tapping",
                                 "--bad-spans", str(bad), "--bad-channels", str(chans))


@pytest.mark.parametrize("field", ["keep_spans", "bad_spans", "bad_channels"])
def test_a_table_is_named_by_its_path(tabled_prep, field):
    tables, prep = tabled_prep
    assert prep[field] == tables[field].resolve().as_posix()


def test_what_the_recording_s_own_channels_tsv_marks_is_left_out(tabled_prep):
    assert "channels_tsv_bads" not in tabled_prep[1]


def test_the_keep_spans_marker_is_named_as_typed(tabled_prep):
    assert tabled_prep[1]["keep_spans_from"] == "tapping"


def test_a_bad_channels_list_stays_as_typed(mini_bids, tmp_path):
    prep = _prep_section(mini_bids, tmp_path / "out", "--bad-channels", "S1_D1")
    assert prep["bad_channels"] == "S1_D1"
