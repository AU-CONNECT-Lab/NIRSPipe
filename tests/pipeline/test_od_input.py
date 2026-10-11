"""A recording that is already optical density passes through the conversion untouched."""

import json

import mne
import numpy as np

from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_snirf, write_snirf

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset


def test_an_od_input_is_written_as_given_and_named_as_the_next_stage_s_source(tmp_path):
    bids, _ = make_fingerprint_dataset(tmp_path)
    (path,) = bids.rglob("*_nirs.snirf")
    raw = mne.io.read_raw_snirf(path, preload=True, verbose="error")
    write_snirf(mne.preprocessing.nirs.optical_density(raw, verbose="error"), path)
    given = mne.io.read_raw_snirf(path, preload=True, verbose="error").get_data()

    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--participant-label", "01",
                  "--no-report", "--skip-bids-validation"])

    (od,) = out.rglob("*_desc-od_nirs.snirf")
    assert json.loads(od.with_suffix(".json").read_text(encoding="utf-8"))["step"] == "od_passthrough"
    assert np.array_equal(read_snirf(od).get_data(), given)
    (sci,) = out.rglob("*_desc-sci_nirs.json")
    assert json.loads(sci.read_text(encoding="utf-8"))["Sources"] == [
        f"bids::sub-01/nirs/{od.name}"]
