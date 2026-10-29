"""Raw QC takes the pipeline's reviewer marks and gives the same verdict, keeps the run's span lists as the pipeline's record does, and its record says how it screened."""

import json

import numpy as np
import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_raw_snirf
from nirspipe.qc.metrics import gvtd_above_segments
from nirspipe.qc.subject.record_io import read_record
from nirspipe.utils import pair_of

from tests._fingerprint import DPF, make_fingerprint_dataset
from tests._synth import make_bids_dataset

# the fingerprint's screening settings
SCREENING = ("--dpf", *(str(v) for v in DPF), "--sci-threshold", "0.75",
             "--psp-threshold", "0.12", "--min-good-frac", "0.85",
             "--cardiac-l-freq", "0.8", "--cardiac-h-freq", "1.9",
             "--resp-l-freq", "0.15", "--resp-h-freq", "0.45")


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
                  "--no-report", *flags])
    qc_cli.main([str(bids), str(tmp / "qc"), "participant", "--participant-label", "01",
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
    qc_cli.main([str(bids), str(tmp_path / "qc"), "participant", "--participant-label", "01",
                 "--dpf", "6", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
                 "--cardiac-h-freq", "1.5", "--resp-l-freq", "0.2", "--resp-h-freq", "0.6",
                 "--skip-bids-validation"])
    params = _record(tmp_path / "qc")["parameters"]
    assert params["sci_threshold"] == 0.8
    assert params["psp_threshold"] is None
    assert (params["cardiac_l_freq"], params["cardiac_h_freq"]) == (0.7, 1.5)
    assert (params["resp_l_freq"], params["resp_h_freq"]) == (0.2, 0.6)
    # as typed: one factor for every wavelength, the form the pipeline's desc-preproc records
    assert params["dpf"] == [6.0]
    assert params["template"] == "fsaverage"
    # the pipeline's default, so a run that names none corrects as nirspipe would
    assert params["motion_correction"] == "tddr"


# the run's flagged stretches, which a condition counts over its own window
SPAN_KEYS = ("gvtd_above_spans_s", "gvtd_above_spans_short_s", "gvtd_above_spans_all_s",
             "spike_spans_s", "spike_spans_short_s", "spike_spans_all_s")
SPAN_SETTINGS = ("--participant-label", "01", "--dpf", "6", "--sci-threshold", "0.8",
                 "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
                 "--resp-l-freq", "0.1", "--resp-h-freq", "0.4",
                 "--motion-correction", "tddr", "--skip-bids-validation")


@pytest.fixture(scope="module")
def windowed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("spans")
    bids = make_bids_dataset(tmp, subjects=("01",), tasks=("mixed",))
    run_cli.main([str(bids), str(tmp / "pipe"), "participant", *SPAN_SETTINGS, "--no-report"])
    qc_cli.main([str(bids), str(tmp / "qc"), "participant", *SPAN_SETTINGS])
    pipe = read_record(next((tmp / "pipe").rglob("*_desc-sqm_qc.json")))
    raw_qc = read_record(next((tmp / "qc").rglob("*_desc-sqmraw_qc.json")))
    return bids, pipe["windowed"], raw_qc["windowed"]


@pytest.mark.parametrize("key", SPAN_KEYS)
def test_raw_qc_keeps_each_span_list_as_the_pipeline_record_does(windowed, key):
    _bids, pipe, raw_qc = windowed
    assert key in pipe and key in raw_qc
    assert len(raw_qc[key]) == len(pipe[key])
    if pipe[key]:
        np.testing.assert_allclose(np.asarray(raw_qc[key], float),
                                   np.asarray(pipe[key], float), rtol=0, atol=1e-9)


def test_the_recording_moves_its_gvtd_spans_across_the_correction(windowed):
    # without this the parity above would hold for spans taken on either side of the correction
    bids, _pipe, raw_qc = windowed
    raw = read_raw_snirf(next(bids.rglob("*_nirs.snirf")), preload=True, verbose=False)
    before = [list(span) for span in gvtd_above_segments(raw)]
    assert before and before != raw_qc["gvtd_above_spans_s"]
