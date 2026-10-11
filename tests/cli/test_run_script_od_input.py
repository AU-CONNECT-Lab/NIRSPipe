"""The exported script handles a recording that is already optical density as the run does."""

import json
import re
import subprocess
import sys

import mne

from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import write_snirf

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset


def test_an_od_input_passes_through_with_the_same_step_and_sources(tmp_path):
    bids, _ = make_fingerprint_dataset(tmp_path)
    (path,) = bids.rglob("*_nirs.snirf")
    raw = mne.io.read_raw_snirf(path, preload=True, verbose="error")
    write_snirf(mne.preprocessing.nirs.optical_density(raw, verbose="error"), path)

    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--participant-label", "01",
                  "--no-report", "--skip-bids-validation"])
    script = (out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({(tmp_path / 'script').as_posix()!r})",
                    script, count=1, flags=re.M)
    (tmp_path / "script.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(tmp_path / "script.py")], check=True, cwd=tmp_path)

    written = sorted(p.relative_to(out) for p in out.rglob("*_desc-*_nirs.json"))
    assert written
    for rel in written:
        cli, ours = (json.loads((root / rel).read_text(encoding="utf-8"))
                     for root in (out, tmp_path / "script"))
        assert (ours["step"], ours["Sources"]) == (cli["step"], cli["Sources"]), rel
    od = json.loads(next(out.rglob("*_desc-od_nirs.json")).read_text(encoding="utf-8"))
    assert od["step"] == "od_passthrough"
