"""The exported script reruns a recording that carries aux channels."""

import re
import subprocess
import sys

from nirspipe.cli import run as run_cli

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset


def test_the_script_writes_the_aux_table_of_a_recording_with_aux(tmp_path):
    bids, _ = make_fingerprint_dataset(tmp_path, aux=True)
    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--participant-label", "01",
                  "--no-report", "--skip-bids-validation"])
    assert list(out.rglob("*_desc-aux_timeseries.json"))
    script = (out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({(tmp_path / 'script').as_posix()!r})",
                    script, count=1, flags=re.M)
    (tmp_path / "script.py").write_text(script, encoding="utf-8")
    done = subprocess.run([sys.executable, str(tmp_path / "script.py")], cwd=tmp_path,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr.strip().splitlines()[-1]
    assert list((tmp_path / "script").rglob("*_desc-aux_timeseries.json"))
