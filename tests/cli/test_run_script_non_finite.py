"""The exported script zeroes non-finite samples and forces their pair bad, as the run does."""

import json
import re
import subprocess
import sys

import mne
import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_snirf, write_snirf

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

ARGS = (*CLI_ARGS, "--mode", "denoise", "--drift-model", "polynomial", "--drift-order", "1",
        "--short-channel", "mean", "--high-pass", "0.01", "--low-pass", "0.1",
        "--participant-label", "01", "--no-report", "--skip-bids-validation")


def _stages(root):
    return {p.name.split("_desc-")[1].split("_")[0]: p for p in root.rglob("*_desc-*_nirs.snirf")}


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    root = tmp_path_factory.mktemp("script_non_finite")
    bids, _ = make_fingerprint_dataset(root)
    (path,) = bids.rglob("*_nirs.snirf")
    raw = mne.io.read_raw_snirf(path, preload=True, verbose="error")
    sfreq = raw.info["sfreq"]
    raw._data[0, int(100 * sfreq):int(120 * sfreq)] = np.nan
    write_snirf(raw, path)

    out = root / "out"
    run_cli.main([str(bids), str(out), "participant", *ARGS])
    script = (out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({(root / 'script').as_posix()!r})",
                    script, count=1, flags=re.M)
    (root / "script.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(root / "script.py")], check=True, cwd=root)
    return out, root / "script"


def test_every_stage_holds_the_same_finite_data(both):
    cli, script = map(_stages, both)
    assert sorted(script) == sorted(cli)
    for stage, path in cli.items():
        ours = read_snirf(script[stage]).get_data()
        assert np.isfinite(ours).all(), stage
        assert np.array_equal(read_snirf(path).get_data(), ours), stage


def test_every_sidecar_matches_the_run_and_names_the_zeroed_samples(both):
    out, script = both
    (od,) = script.rglob("*_desc-od_nirs.json")
    assert json.loads(od.read_text(encoding="utf-8"))["non_finite_samples"]
    for rel in (p.relative_to(script) for p in script.rglob("*.json")):
        cli, ours = (json.loads((root / rel).read_text(encoding="utf-8")) for root in (out, script))
        for side in (cli, ours):
            side.pop("timestamp", None)
        assert ours == cli, rel
