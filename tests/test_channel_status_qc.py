"""Raw QC takes each recording's channels.tsv status as the pipeline does, and the dyad page keeps it."""

import json
import logging

import pandas as pd
import pytest

from nirspipe.cli import qc as qc_cli

from tests._raw_qc import raw_qc
from tests._synth import make_bids_dataset, make_hyper_dataset

LISTED = "S1_D1 760"     # one wavelength of a pair the screening keeps
RAW_ARGS = ("--dpf", "6", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
            "--cardiac-h-freq", "1.5", "--resp-l-freq", "0.1", "--resp-h-freq", "0.5", "--skip-bids-validation")


def _with_status(bids, subject, task):
    path = bids / f"sub-{subject}" / "nirs" / f"sub-{subject}_task-{task}_channels.tsv"
    path.write_text(f"name\ttype\tstatus\n{LISTED}\tNIRSCWAMPLITUDE\tbad\n"
                    "S9_D9 760\tNIRSCWAMPLITUDE\tbad\n", encoding="utf-8")
    return bids


def _read_json(path):
    """The file minus its timestamp and the recorded --ignore, which names the skip itself."""
    data = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items()
            if k != "timestamp"}
    if isinstance(data.get("parameters"), dict):
        data["parameters"] = {k: v for k, v in data["parameters"].items() if k != "ignore"}
    return data


def _same_files(a, b, pattern):
    files = sorted(p.relative_to(a) for p in a.rglob(pattern) if "figures" not in p.parts)
    assert files and files == sorted(p.relative_to(b) for p in b.rglob(pattern)
                                     if "figures" not in p.parts)
    for rel in files:
        if rel.suffix == ".json":
            assert _read_json(a / rel) == _read_json(b / rel), rel
        else:
            assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


# ---- participant level ----

@pytest.fixture(scope="module")
def prep_raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("prep_raw_status")
    plain = make_bids_dataset(root / "plain", subjects=("01",), tasks=("tapping",))
    listed = _with_status(make_bids_dataset(root / "listed", subjects=("01",),
                                            tasks=("tapping",)), "01", "tapping")
    outs = {}
    for name, bids, extra in (("plain", plain, ()), ("listed", listed, ()),
                              ("ignored", listed, ("--ignore", "channel-status"))):
        outs[name] = root / f"out_{name}"
        qc_cli.main([str(bids), str(outs[name]), "participant", "--participant-label", "01",
                     *RAW_ARGS, *extra])
    return outs


def _channel_table(out):
    (path,) = (out / "sub-01" / "nirs").glob("*_desc-rawchannel_qc.tsv")
    return pd.read_csv(path, sep="\t").set_index("name")


def test_prep_raw_rejects_the_listed_pair_whole(prep_raw):
    listed, plain = _channel_table(prep_raw["listed"]), _channel_table(prep_raw["plain"])
    pair = ["S1_D1 760", "S1_D1 850"]
    assert not plain.loc[pair, "is_bad"].any()
    assert listed.loc[pair, "is_bad"].all()
    assert (listed.loc[pair, "reason"] == "manual").all()
    others = listed.index.difference(pair)
    assert listed.loc[others, "is_bad"].equals(plain.loc[others, "is_bad"])


def test_prep_raw_names_channels_tsv_under_the_channel_table(prep_raw):
    report = lambda out: (out / "sub-01" / "sub-01_task-tapping_desc-raw_report.html").read_text(  # noqa: E731
        encoding="utf-8")
    assert ("Marked bad by the recording's BIDS channels.tsv, whatever the screening found: "
            "S1_D1.") in report(prep_raw["listed"])
    assert "channels.tsv" not in report(prep_raw["plain"])


def test_prep_raw_warns_about_a_listed_pair_the_recording_lacks(tmp_path, caplog):
    bids = _with_status(make_bids_dataset(tmp_path, subjects=("01",), tasks=("tapping",)),
                        "01", "tapping")
    with caplog.at_level(logging.WARNING, logger="nirspipe.qc.prep_raw_report"):
        qc_cli.main([str(bids), str(tmp_path / "out"), "participant", "--participant-label", "01",
                     *RAW_ARGS])
    assert "channels.tsv marks bad channels the recording does not have: ['S9_D9']" in caplog.text


def test_prep_raw_ignoring_the_column_comes_out_bit_identical_to_no_column(prep_raw):
    for pattern in ("*.tsv", "*.json"):
        _same_files(prep_raw["plain"] / "sub-01" / "nirs", prep_raw["ignored"] / "sub-01" / "nirs",
                    pattern)


# ---- group level ----

@pytest.fixture(scope="module")
def hyper_raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("hyper_raw_status")
    plain, pairs = make_hyper_dataset(root / "plain", tasks=("hold",))
    listed, _ = make_hyper_dataset(root / "listed", tasks=("hold",))
    _with_status(listed, "11", "hold")
    outs = {}
    for name, bids, extra in (("plain", plain, ()), ("listed", listed, ()),
                              ("ignored", listed, ("--ignore", "channel-status"))):
        outs[name] = root / f"out_{name}"
        raw_qc(bids, outs[name], pairs, participant=(*RAW_ARGS, *extra))
    return outs


def _group_table(out, desc):
    (path,) = out.rglob(f"group-G01_task-hold_desc-{desc}_qc.tsv")
    return path, pd.read_csv(path, sep="\t", dtype={"subject_id": str})


def test_hyper_raw_rejects_the_listed_pair_in_that_member_only(hyper_raw):
    _, listed = _group_table(hyper_raw["listed"], "channel")
    _, plain = _group_table(hyper_raw["plain"], "channel")
    key = ["subject_id", "channel"]
    merged = listed.merge(plain, on=key, suffixes=("", "_plain"))
    moved = merged[merged["is_bad"] != merged["is_bad_plain"]]
    assert sorted(map(tuple, moved[key].values)) == [("sub-11", "S1_D1 760"),
                                                     ("sub-11", "S1_D1 850")]
    assert moved["is_bad"].all()


def test_hyper_raw_counts_and_records_the_listed_channels(hyper_raw):
    path, listed = _group_table(hyper_raw["listed"], "subject")
    _, plain = _group_table(hyper_raw["plain"], "subject")
    gained = (listed.set_index("subject_id")["n_bad_channels"]
              - plain.set_index("subject_id")["n_bad_channels"])
    assert gained.to_dict() == {"sub-11": 2, "sub-12": 0}
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["parameters"]["channels_tsv_bads"] == {"sub-11": [LISTED, "S9_D9 760"]}


def test_hyper_raw_names_the_listed_pairs_per_member(hyper_raw):
    report = lambda out: next(out.glob("group-*/*_desc-raw_report.html")).read_text(  # noqa: E731
        encoding="utf-8")
    listed = report(hyper_raw["listed"])
    assert ("sub-11: marked bad by the recording&rsquo;s BIDS channels.tsv, whatever the "
            "screening found: S1_D1.") in listed
    assert "sub-12: marked bad" not in listed
    assert "channels.tsv" not in report(hyper_raw["plain"])
    assert "channels.tsv" not in report(hyper_raw["ignored"])


def test_hyper_raw_ignoring_the_column_comes_out_bit_identical_to_no_column(hyper_raw):
    for pattern in ("*_qc.tsv", "*_qc.json"):
        _same_files(hyper_raw["plain"], hyper_raw["ignored"], pattern)
