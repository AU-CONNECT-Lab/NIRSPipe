"""Raw QC takes the pipeline's reviewer marks and gives the same verdict, and its record says how it screened."""

import json

import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.cli import run as run_cli
from nirspipe.utils import pair_of

from tests._fingerprint import DPF, make_fingerprint_dataset
from tests._synth import make_bids_dataset

# the fingerprint's screening settings, less the respiration band raw QC has no use for
SCREENING = ("--dpf", *(str(v) for v in DPF), "--sci-threshold", "0.75",
             "--psp-threshold", "0.12", "--min-good-frac", "0.85",
             "--cardiac-l-freq", "0.8", "--cardiac-h-freq", "1.9")


def _record(out):
    (path,) = out.rglob("*_desc-sqmraw_qc.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity")
    bids, truth = make_fingerprint_dataset(tmp, blocks=True)
    always_bad = next(p.name for p in truth.pairs if p.bad)
    block_bad, onset, span = truth.block_bad
    marked = next(p.name for p in truth.long_pairs
                  if not p.bad and p.name != block_bad)
    marks = tmp / "marks.tsv"
    marks.write_text(f"bad_channels\tkeep_channels\n{marked}\t{always_bad}\n", encoding="utf-8")
    spans = tmp / "spans.tsv"
    spans.write_text(f"onset\tduration\n{onset - 5:g}\t{span + 10:g}\n", encoding="utf-8")
    flags = ("--bad-channels", str(marks), "--bad-spans", str(spans), "--skip-bids-validation")
    run_cli.main([str(bids), str(tmp / "pipe"), "participant", *SCREENING,
                  "--resp-l-freq", "0.15", "--resp-h-freq", "0.45", "--no-report", *flags])
    qc_cli.main(["prep-raw", str(bids), str(tmp / "qc"), "--participant-label", "01",
                 *SCREENING, *flags])
    (sci,) = (tmp / "pipe").rglob("*_desc-sci_nirs.json")
    return {"pipeline": json.loads(sci.read_text(encoding="utf-8")),
            "record": _record(tmp / "qc"),
            "always_bad": always_bad, "block_bad": block_bad, "marked": marked}


def test_raw_qc_rejects_what_the_pipeline_rejects(both):
    assert sorted(both["record"]["bad_channels"]) == sorted(both["pipeline"]["bad_channels"])


def test_the_marks_and_spans_land_where_the_truth_says(both):
    bad_pairs = {pair_of(c) for c in both["record"]["bad_channels"]}
    assert both["marked"] in bad_pairs                  # --bad-channels
    assert both["always_bad"] not in bad_pairs          # kept on review
    assert both["block_bad"] not in bad_pairs           # its bad block is a --bad-spans span
    assert both["record"]["parameters"]["kept_channels"] == [
        f"{both['always_bad']} 760", f"{both['always_bad']} 850"]


def test_the_record_carries_the_settings_it_screened_with(tmp_path):
    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("tapping",))
    qc_cli.main(["prep-raw", str(bids), str(tmp_path / "qc"), "--participant-label", "01",
                 "--dpf", "6", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
                 "--cardiac-h-freq", "1.5", "--skip-bids-validation"])
    params = _record(tmp_path / "qc")["parameters"]
    assert params["sci_threshold"] == 0.8
    assert params["psp_threshold"] is None
    assert (params["cardiac_l_freq"], params["cardiac_h_freq"]) == (0.7, 1.5)
    # as typed: one factor for every wavelength, the form the pipeline's desc-preproc records
    assert params["dpf"] == [6.0]
    assert params["template"] == "fsaverage"
    # the pipeline's default, so a run that names none corrects as nirspipe would
    assert params["motion_correction"] == "tddr"
