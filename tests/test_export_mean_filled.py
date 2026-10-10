"""--export-mean-filled: the final residual with BAD_ frames at each channel's kept mean, annotations kept."""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone

import mne
import numpy as np

from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.glm_censored import kept_frames
from nirspipe.pipeline.post_pipeline import MEAN_FILLED_DESCRIPTION, mean_filled
from nirspipe.utils.lineage import stage_of

from tests._fingerprint import CLI_ARGS, fingerprint_raw
from tests._synth import _write_dataset_root, _write_subject


def _haemo(spans=()):
    data = np.arange(2 * 100, dtype=float).reshape(2, 100) ** 1.5
    raw = mne.io.RawArray(data, mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], 10.0,
                                                ["hbo", "hbr"]), verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations([s[0] for s in spans], [s[1] for s in spans],
                                        [s[2] for s in spans]))
    return raw


def test_bad_frames_take_each_channels_kept_mean_and_kept_frames_stay():
    raw = _haemo([(2.0, 1.0, "BAD_gvtd"), (6.0, 2.0, "BAD_unselected")])
    out = mean_filled(raw)
    keep = kept_frames(raw)
    data, filled = raw.get_data(), out.get_data()
    np.testing.assert_array_equal(filled[:, keep], data[:, keep])
    for ch in range(2):
        np.testing.assert_allclose(filled[ch, ~keep], data[ch, keep].mean(), rtol=1e-14)
    assert stage_of(out) == "meanfilled" and out.annotations == raw.annotations


def test_without_a_span_the_copy_is_the_data():
    raw = _haemo()
    np.testing.assert_array_equal(mean_filled(raw).get_data(), raw.get_data())


# ---- the whole run ----

def _dataset(root, span=True):
    raw, _ = fingerprint_raw("01", "tapping")
    if span:
        raw.set_annotations(raw.annotations + mne.Annotations(
            [150.0], [6.0], ["BAD_motion"], orig_time=raw.annotations.orig_time))
    bids = root / "bids"
    _write_dataset_root(bids, ["01"])
    _write_subject(bids, "01", "tapping", raw)
    return bids


def _run(bids, out):
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--mode", "denoise",
                  "--high-pass", "0.01", "--low-pass", "0.1", "--short-channel", "mean",
                  "--export-mean-filled", "--no-report", "--skip-bids-validation"])
    return {p.name.split("_desc-")[1].split("_")[0]: p for p in out.rglob("*_nirs.snirf")}


def test_the_run_writes_it_beside_the_residual(tmp_path):
    files = _run(_dataset(tmp_path), tmp_path / "out")
    errts, filled = read_snirf(files["errts"]), read_snirf(files["meanfilled"])
    keep = kept_frames(errts)
    assert not keep.all()
    np.testing.assert_array_equal(filled.get_data()[:, keep], errts.get_data()[:, keep])
    np.testing.assert_allclose(filled.get_data()[:, ~keep],
                               np.broadcast_to(errts.get_data()[:, keep].mean(axis=1,
                                                                             keepdims=True),
                                               filled.get_data()[:, ~keep].shape))
    side = json.loads(files["meanfilled"].with_suffix(".json").read_text(encoding="utf-8"))
    assert side["Description"] == MEAN_FILLED_DESCRIPTION
    assert side["Sources"] and side["Sources"][0].endswith("desc-errts_nirs.snirf")
    assert [s["description"] for s in side["excluded_spans"]] == ["BAD_motion"]
    assert "BAD_motion" in set(filled.annotations.description)


def test_without_a_span_it_is_still_written_and_equals_the_residual(tmp_path):
    files = _run(_dataset(tmp_path, span=False), tmp_path / "out")
    np.testing.assert_array_equal(read_snirf(files["meanfilled"]).get_data(),
                                  read_snirf(files["errts"]).get_data())


def test_the_exported_script_writes_the_same_file(tmp_path):
    bids = _dataset(tmp_path)
    out, script_out = tmp_path / "out", tmp_path / "script"
    files = _run(bids, out)
    script = (out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({script_out.as_posix()!r})",
                    script, count=1, flags=re.M)
    (tmp_path / "script.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(tmp_path / "script.py")], check=True, cwd=tmp_path)
    (script_filled,) = script_out.rglob("*desc-meanfilled_nirs.snirf")
    np.testing.assert_array_equal(read_snirf(script_filled).get_data(),
                                  read_snirf(files["meanfilled"]).get_data())
