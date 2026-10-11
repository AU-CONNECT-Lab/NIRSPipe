"""The exported script reproduces a run whose separation bands are not the defaults, in every mode."""

import json
import re
import subprocess
import sys

import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_snirf

from tests._fingerprint import CLI_ARGS, EVENT_DURATION, make_fingerprint_dataset

# the fingerprint's 30 mm pairs count as short under these bands, so the split moves
COMMON = (*CLI_ARGS, "--short-channel", "mean", "--short-max-dist", "31", "--long-min-dist", "32",
          "--participant-label", "01", "--no-report", "--skip-bids-validation")
MODES = {
    "denoise": ("--mode", "denoise", "--drift-model", "polynomial", "--drift-order", "1",
                "--gvtd-censor", "all", "--high-pass", "0.01", "--low-pass", "0.1",
                "--resample-sfreq", "5", "--export-mean-filled"),
    "glm": ("--mode", "glm", "--stim-dur", f"{EVENT_DURATION:g}", "--drift-model", "polynomial",
            "--noise-model", "ar1", "--high-pass", "0.01", "--low-pass", "0.5",
            "--resample-sfreq", "5"),
    "rest": ("--mode", "rest", "--drift-model", "polynomial", "--drift-order", "1",
             "--high-pass", "0.01", "--low-pass", "0.1"),
}


def _stages(root):
    return {p.name.split("_desc-")[1].split("_")[0]: p for p in root.rglob("*_desc-*_nirs.snirf")}


@pytest.fixture(scope="module", params=sorted(MODES))
def both(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"script_bands_{request.param}")
    bids, _ = make_fingerprint_dataset(root)
    out = root / "out"
    run_cli.main([str(bids), str(out), "participant", *COMMON, *MODES[request.param]])
    script = (out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({(root / 'script').as_posix()!r})",
                    script, count=1, flags=re.M)
    (root / "script.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(root / "script.py")], check=True, cwd=root)
    return out, root / "script"


def test_the_script_writes_every_stage_the_run_wrote(both):
    cli, script = map(_stages, both)
    assert sorted(script) == sorted(cli)


def test_every_stage_holds_the_same_data(both):
    cli, script = map(_stages, both)
    for stage, path in cli.items():
        a, b = read_snirf(path), read_snirf(script[stage])
        assert np.array_equal(a.get_data(), b.get_data(), equal_nan=True), stage


def test_every_sidecar_the_script_wrote_matches_the_run(both):
    # the run also writes the quality record, which the script leaves to QC
    out, script = both
    written = [p.relative_to(script) for p in script.rglob("*.json")]
    assert any("errts" in p.name for p in written) and any("design" in p.name for p in written)
    for rel in written:
        cli, ours = (json.loads((root / rel).read_text(encoding="utf-8")) for root in (out, script))
        for side in (cli, ours):
            side.pop("timestamp", None)
        assert ours == cli, rel
