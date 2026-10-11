"""Each run's BIDS channels.tsv status column: its bad channels join the rejected set unless ignored."""

import json
import re
import subprocess
import sys

import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.io.bids import channel_status_bads, get_layout
from nirspipe.io.snirf import read_snirf
from nirspipe.qc.boilerplate.vocabulary import _screening_slots

from tests._synth import make_bids_dataset

LISTED = "S1_D1 760"     # one wavelength of a pair the screening keeps
ARGS = ("--dpf", "6", "6", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
        "--cardiac-h-freq", "1.5", "--resp-l-freq", "0.2", "--resp-h-freq", "0.5",
        "--participant-label", "01", "--no-report", "--skip-bids-validation")


def _with_status(bids, rows, name="sub-01/nirs/sub-01_task-tapping_channels.tsv"):
    path = bids / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(["name\ttype\tstatus", *("\t".join(r) for r in rows)]) + "\n",
                    encoding="utf-8")
    return bids


def _snirf(bids):
    (path,) = (bids / "sub-01" / "nirs").glob("*_nirs.snirf")
    return path


# ---- reading the column ----

def test_the_bad_rows_are_read_case_ignored(tmp_path):
    bids = _with_status(make_bids_dataset(tmp_path, subjects=("01",), tasks=("tapping",)),
                        [(LISTED, "NIRSCWAMPLITUDE", "bad"),
                         ("S1_D1 850", "NIRSCWAMPLITUDE", "good"),
                         ("S2_D2 850", "NIRSCWAMPLITUDE", "BAD"),
                         ("S4_D4 760", "NIRSCWAMPLITUDE", "n/a")])
    assert channel_status_bads(get_layout(bids), _snirf(bids)) == [LISTED, "S2_D2 850"]


def test_no_file_or_no_status_column_gives_none(tmp_path):
    bids = make_bids_dataset(tmp_path / "a", subjects=("01",), tasks=("tapping",))
    assert channel_status_bads(get_layout(bids), _snirf(bids)) == []
    path = bids / "sub-01" / "nirs" / "sub-01_task-tapping_channels.tsv"
    path.write_text(f"name\ttype\n{LISTED}\tNIRSCWAMPLITUDE\n", encoding="utf-8")
    assert channel_status_bads(get_layout(bids), _snirf(bids)) == []


def test_a_file_higher_up_is_inherited(tmp_path):
    bids = _with_status(make_bids_dataset(tmp_path, subjects=("01",), tasks=("tapping",)),
                        [(LISTED, "NIRSCWAMPLITUDE", "bad")], name="task-tapping_channels.tsv")
    assert channel_status_bads(get_layout(bids), _snirf(bids)) == [LISTED]


def test_the_methods_name_the_listed_pairs():
    slots = _screening_slots({"channels_tsv_bads": ["S1_D1 760", "S1_D1 850", "S2_D2 760"]})
    assert slots["channels_tsv"] == (" Channels S1_D1 and S2_D2 were also marked as bad, as the "
                                     "recording's BIDS channels.tsv lists them.")
    assert _screening_slots({"channels_tsv_bads": []})["channels_tsv"] == ""


# ---- the whole run ----

def _outputs(root):
    return {p.relative_to(root): p for p in root.rglob("*_nirs.snirf")}


def _sidecar(path):
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One recording run three ways: no channels.tsv, one marking LISTED bad, and that one ignored."""
    root = tmp_path_factory.mktemp("status")
    plain = make_bids_dataset(root / "plain", subjects=("01",), tasks=("tapping",))
    listed = _with_status(make_bids_dataset(root / "listed", subjects=("01",), tasks=("tapping",)),
                          [(LISTED, "NIRSCWAMPLITUDE", "bad"),
                           ("S9_D9 760", "NIRSCWAMPLITUDE", "bad")])
    out = {name: root / f"out_{name}" for name in ("plain", "listed", "ignored")}
    run_cli.main([str(plain), str(out["plain"]), "participant", *ARGS])
    run_cli.main([str(listed), str(out["listed"]), "participant", *ARGS])
    run_cli.main([str(listed), str(out["ignored"]), "participant", *ARGS,
                  "--ignore", "channel-status"])
    return out


def _log(root):
    return (root / "sub-01" / "logs" / "sub-01.log").read_text(encoding="utf-8")


def test_the_listed_pair_is_rejected_whole_and_recorded(runs):
    out = runs
    (sci,) = out["listed"].rglob("*desc-sci_nirs.snirf")
    (plain_sci,) = out["plain"].rglob("*desc-sci_nirs.snirf")
    assert "S1_D1 760" not in read_snirf(plain_sci).info["bads"]
    bads = read_snirf(sci).info["bads"]
    assert {"S1_D1 760", "S1_D1 850"} <= set(bads)
    assert set(bads) - {"S1_D1 760", "S1_D1 850"} == set(read_snirf(plain_sci).info["bads"])
    assert _sidecar(sci)["parameters"]["channels_tsv_bads"] == [LISTED, "S9_D9 760"]


def test_a_listed_name_the_recording_lacks_is_warned_about(runs):
    assert "channels.tsv marks bad channels the recording does not have: ['S9_D9']" in _log(
        runs["listed"])
    assert "channels.tsv" not in _log(runs["ignored"])


def test_ignoring_the_column_comes_out_bit_identical_to_no_column(runs):
    out = runs
    plain, ignored = _outputs(out["plain"]), _outputs(out["ignored"])
    assert plain and sorted(plain) == sorted(ignored)
    for rel in plain:
        a, b = read_snirf(plain[rel]), read_snirf(ignored[rel])
        assert np.array_equal(a.get_data(), b.get_data()), rel
        assert a.info["bads"] == b.info["bads"], rel
        assert a.annotations.description.tolist() == b.annotations.description.tolist()
        pa, pb = _sidecar(plain[rel])["parameters"], _sidecar(ignored[rel])["parameters"]
        if "desc-sci" in rel.name:
            assert pa["channels_tsv_bads"] == pb["channels_tsv_bads"] == []
            assert pb.pop("ignore") == ["channel-status"] and pa.pop("ignore") == []
        assert pa == pb, rel


def test_the_script_rejects_the_same_channels(runs, tmp_path):
    out = runs
    script = (out["listed"] / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script_out = tmp_path / "script"
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({script_out.as_posix()!r})",
                    script, count=1, flags=re.M)
    (tmp_path / "script.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(tmp_path / "script.py")], check=True, cwd=tmp_path)
    for desc in ("sci", "preproc"):
        (cli,) = out["listed"].rglob(f"*task-tapping_desc-{desc}_nirs.snirf")
        (scripted,) = script_out.rglob(f"*task-tapping_desc-{desc}_nirs.snirf")
        assert read_snirf(scripted).info["bads"] == read_snirf(cli).info["bads"], desc
    (sci,) = script_out.rglob("*task-tapping_desc-sci_nirs.snirf")
    assert _sidecar(sci)["parameters"]["channels_tsv_bads"] == [LISTED, "S9_D9 760"]
