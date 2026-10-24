"""The dyad commands under --device cuda write the CPU run's tables and say where they ran."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._dyad_fingerprint import write_roi_map
from tests.cli.test_hyper_nprocs import _outputs, cohort  # noqa: F401  (a fixture)

_DEVICE_KEYS = {"wtc_device", "wtc_device_name", "wtc_cupy_version", "wtc_cuda_version",
                "device_check_pairing", "device_check_cells", "device_check_cells_one_ulp",
                "device_check_cells_near_floor"}


@pytest.fixture(scope="module", autouse=True)
def _card():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA card")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA card")


def _comparable(params: dict) -> dict:
    """The parameters both runs share: the device's own left out, and when each run drew its levels."""
    out = {k: v for k, v in params.items() if k not in _DEVICE_KEYS}
    if "phase_level_files" in out:
        out["phase_level_files"] = {
            chroma: {kind: {k: v for k, v in level.items() if k != "timestamp"}
                     for kind, level in kinds.items()}
            for chroma, kinds in out["phase_level_files"].items()}
    return out


def _same_tables(cpu: Path, gpu: Path) -> None:
    left, right = _outputs(cpu), _outputs(gpu)
    assert set(left) == set(right)
    for name in sorted(left):
        if name.endswith(".tsv"):
            a, b = (pd.read_csv(f[name], sep="\t") for f in (left, right))
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6)
        elif name.endswith(".json"):
            a, b = (json.loads(f[name].read_text(encoding="utf-8")) for f in (left, right))
            pa, pb = a.get("parameters", {}), b.get("parameters", {})
            assert _comparable(pa) == _comparable(pb), name
        elif name.endswith(".npz"):
            x, y = np.load(left[name]), np.load(right[name])
            assert sorted(x.files) == sorted(y.files)
            for key in x.files:
                assert np.allclose(x[key], y[key], rtol=1e-6, atol=1e-6, equal_nan=True), key


def test_nirspipe_hyper_on_the_card_writes_the_cpus_tables(cohort):
    from nirspipe.cli import hyper as hyper_cli

    root, _, pairs, deriv = cohort
    trees = {}
    for device in ("cpu", "cuda"):
        out = root / f"hyper_device_{device}"
        # the triad G02 too: its pairings' real sides share one cache of transforms
        hyper_cli.main([str(deriv), str(out), "group", "--pairs-csv", str(pairs),
                        "--roi-mapping", str(write_roi_map(root)),
                        "--wtc-fmin", "0.02", "--wtc-band-fmin", "0.03",
                        "--wtc-band-fmax", "0.10", "--wtc-phase-null", "3",
                        "--wtc-seed", "7", "--wtc-save-maps", "--no-report",
                        "--device", device])
        trees[device] = out
    _same_tables(trees["cpu"], trees["cuda"])

    record = tomllib.loads(next(trees["cuda"].glob("group-G01/logs/*.toml"))
                           .read_text(encoding="utf-8"))
    assert record["execution"]["device"] == "cuda"
    assert record["execution"]["device_check_cells"] > 0
    assert record["environment"]["gpu_name"]
    assert "device" not in record["hyper"]
    cpu_record = tomllib.loads(next(trees["cpu"].glob("group-G01/logs/*.toml"))
                               .read_text(encoding="utf-8"))
    assert cpu_record["execution"]["device"] == "cpu"
    assert "gpu_name" not in cpu_record["environment"]

    table = next(trees["cuda"].rglob("*_stat-wtc_relmat.json"))
    params = json.loads(table.read_text(encoding="utf-8"))["parameters"]
    assert params["wtc_device"] == "cuda" and params["wtc_device_name"]


def test_the_pair_null_on_the_card_writes_the_cpus_null_and_its_check(cohort):
    from nirspipe.cli import hyper as hyper_cli

    root, _, pairs, deriv = cohort
    trees = {}
    for device in ("cpu", "cuda"):
        out = root / f"pair_device_{device}"
        hyper_cli.main([str(deriv), str(out), "group", "--pairs-csv", str(pairs),
                        "--wtc-fmin", "0.02", "--wtc-band-fmin", "0.03",
                        "--wtc-band-fmax", "0.10", "--no-report"])
        # two processes, so each opens the card in its own worker; by row order the dyad
        # G01 has nobody to stand in for the triad G02's third member, so any member may
        hyper_cli.main_pair_null([str(deriv), str(out), "group", "--pairs-csv", str(pairs),
                                  "--chroma", "hbo", "--device", device, "--nprocs", "2",
                                  "--wtc-pair-pool", "any"])
        trees[device] = out
    _same_tables(trees["cpu"], trees["cuda"])
    nulls = sorted(trees["cuda"].rglob("*_null-pair_stat-wtc_relmat.json"))
    assert {next(part for part in p.parts if part.startswith("group-"))
            for p in nulls} == {"group-G01", "group-G02"}
    for null in nulls:
        params = json.loads(null.read_text(encoding="utf-8"))["parameters"]
        assert params["wtc_device"] == "cuda" and params["device_check_cells"] > 0
