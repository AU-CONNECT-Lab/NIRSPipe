"""The exported script reproduces a run whose separation bands are not the defaults."""

import json
import re
import subprocess
import sys

import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_snirf

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

# the fingerprint's 30 mm pairs count as short under these bands, so the split moves
ARGS = (*CLI_ARGS, "--mode", "denoise", "--drift-model", "polynomial", "--drift-order", "1",
        "--short-channel", "mean", "--short-max-dist", "31", "--long-min-dist", "32",
        "--gvtd-censor", "all", "--high-pass", "0.01", "--low-pass", "0.1",
        "--resample-sfreq", "5", "--export-mean-filled",
        "--participant-label", "01", "--no-report", "--skip-bids-validation")


def _stages(root):
    return {p.name.split("_desc-")[1].split("_")[0]: p for p in root.rglob("*_desc-*_nirs.snirf")}


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    root = tmp_path_factory.mktemp("script_bands")
    bids, _ = make_fingerprint_dataset(root)
    out = root / "out"
    run_cli.main([str(bids), str(out), "participant", *ARGS])
    script = (out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({(root / 'script').as_posix()!r})",
                    script, count=1, flags=re.M)
    (root / "script.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(root / "script.py")], check=True, cwd=root)
    return _stages(out), _stages(root / "script")


def test_the_script_writes_every_stage_the_run_wrote(both):
    cli, script = both
    assert sorted(script) == sorted(cli)


def test_every_stage_holds_the_same_data(both):
    cli, script = both
    for stage, path in cli.items():
        a, b = read_snirf(path), read_snirf(script[stage])
        assert np.array_equal(a.get_data(), b.get_data(), equal_nan=True), stage


def test_every_sidecar_records_the_same_parameters(both):
    cli, script = both
    for stage, path in cli.items():
        params = [json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))["parameters"]
                  for p in (path, script[stage])]
        assert params[1] == params[0], stage
