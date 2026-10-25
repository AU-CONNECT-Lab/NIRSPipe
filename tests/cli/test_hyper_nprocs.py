"""The subject and dyad commands under --nprocs write what they write one unit at a time.

Both sides hold BLAS to one thread per unit (--nprocs 1 against --nprocs 2), so the comparison
is exact: anything that differs leaked between subjects or groups, or depends on which thread
or process drew it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nirspipe.io.snirf import read_snirf
from tests._dyad_fingerprint import make_group_dataset, write_roi_map
from tests._fingerprint import CLI_ARGS

# compared byte for byte or array for array; logs and pages carry paths and timestamps
_COMPARED = (".tsv", ".json", ".npz", ".png", ".snirf")


def _flag(name: str) -> list[str]:
    """One flag of CLI_ARGS with its values."""
    i = CLI_ARGS.index(name) + 1
    j = next((k for k in range(i, len(CLI_ARGS)) if CLI_ARGS[k].startswith("--")), len(CLI_ARGS))
    return [name, *CLI_ARGS[i:j]]


_RAW_ARGS = [a for flag in ("--dpf", "--sci-threshold", "--cardiac-l-freq", "--cardiac-h-freq")
             for a in _flag(flag)]


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    from nirspipe.cli import run as run_cli

    root = tmp_path_factory.mktemp("hyper_nprocs")
    bids, pairs, _ = make_group_dataset(root)
    deriv = root / "deriv"
    run_cli.main([str(bids), str(deriv), "participant", *CLI_ARGS, "--by-condition",
                  "--skip-bids-validation"])
    return root, bids, pairs, deriv


def _outputs(tree: Path) -> dict[str, Path]:
    return {p.relative_to(tree).as_posix(): p for p in tree.rglob("*")
            if p.suffix in _COMPARED and "logs" not in p.parts}


def _differences(a: Path, b: Path) -> list[str]:
    left, right = _outputs(a), _outputs(b)
    out = sorted(set(left) ^ set(right))
    for name in sorted(set(left) & set(right)):
        if name.endswith(".npz"):
            x, y = np.load(left[name]), np.load(right[name])
            if sorted(x.files) != sorted(y.files) or any(
                    not np.array_equal(x[k], y[k], equal_nan=True) for k in x.files):
                out.append(name)
        elif name.endswith(".snirf"):
            # HDF5 stamps its own write time, so the recording is compared, not the bytes
            x, y = read_snirf(left[name]), read_snirf(right[name])
            if not np.array_equal(x.get_data(), y.get_data(), equal_nan=True) or (
                    x.info["bads"] != y.info["bads"]):
                out.append(name)
        elif name.endswith(".json"):
            # every sidecar stamps the moment it was written
            x, y = (json.loads(f[name].read_text(encoding="utf-8")) for f in (left, right))
            if {**x, "timestamp": None} != {**y, "timestamp": None}:
                out.append(name)
        elif left[name].read_bytes() != right[name].read_bytes():
            out.append(name)
    return out


def _logs(tree: Path) -> list[Path]:
    return sorted(tree.glob("group-*/logs/*.log"))


def test_nirspipe_writes_the_same_subjects_on_two_threads(cohort):
    from nirspipe.cli import run as run_cli

    root, bids, _, _ = cohort
    trees = {}
    for nprocs in ("1", "2"):
        out = root / f"participant_{nprocs}"
        run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--by-condition",
                      "--skip-bids-validation", "--nprocs", nprocs])
        trees[nprocs] = out

    names = _outputs(trees["1"])
    assert sum(n.endswith(".snirf") for n in names) > 10 and any(n.endswith(".png") for n in names)
    assert _differences(trees["1"], trees["2"]) == []


def test_nirspipe_hyper_writes_the_same_tree_on_two_processes(cohort):
    from nirspipe.cli import hyper as hyper_cli

    root, _, pairs, deriv = cohort
    trees = {}
    for nprocs in ("1", "2"):
        out = root / f"hyper_{nprocs}"
        # the seeded significance contour draws from numpy's global RNG, which a process
        # of its own keeps to one group
        hyper_cli.main([str(deriv), str(out), "group", "--pairs-csv", str(pairs),
                        "--roi-mapping", str(write_roi_map(root)),
                        "--wtc-fmin", "0.02", "--wtc-band-fmin", "0.03",
                        "--wtc-band-fmax", "0.10", "--no-channel-cross",
                        "--wtc-significance", "--wtc-mc-count", "10", "--wtc-seed", "7",
                        "--nprocs", nprocs])
        trees[nprocs] = out

    assert len(_outputs(trees["1"])) > 50
    assert _differences(trees["1"], trees["2"]) == []
    for tree in trees.values():
        logs = _logs(tree)
        assert [p.name for p in logs] == ["group-G01_task-main.log", "group-G02_task-main.log"]
        for log in logs:
            text = log.read_text(encoding="utf-8")
            own, other = ("G01", "G02") if "G01" in log.name else ("G02", "G01")
            assert f"group-{own}" in text and f"group-{other}" not in text


def test_raw_qc_and_the_pair_null_write_the_same_on_two_processes(cohort):
    from nirspipe.cli import hyper as hyper_cli
    from tests._raw_qc import raw_qc

    root, bids, pairs, deriv = cohort
    trees = {}
    for nprocs in ("1", "2"):
        out = root / f"raw_pair_{nprocs}"
        raw_qc(bids, out / "qc", pairs, participant=(*_RAW_ARGS, "--nprocs", nprocs),
               group=("--nprocs", nprocs))
        hyper_cli.main([str(deriv), str(out / "hyper"), "group", "--pairs-csv", str(pairs),
                        "--wtc-fmin", "0.02", "--wtc-band-fmin", "0.03",
                        "--wtc-band-fmax", "0.10", "--no-report", "--nprocs", nprocs])
        # by row order nobody stands in for the triad G02's third member, so any member may
        hyper_cli.main_pair_null([str(deriv), str(out / "hyper"), "group", "--pairs-csv", str(pairs),
                                  "--chroma", "hbo", "--nprocs", nprocs,
                                  "--wtc-pair-pool", "any"])
        trees[nprocs] = out

    names = _outputs(trees["1"])
    assert any("null-pair" in n for n in names) and any("desc-usable_qc" in n for n in names)
    assert _differences(trees["1"], trees["2"]) == []
    for tree in trees.values():
        assert {p.name for p in _logs(tree / "qc") + _logs(tree / "hyper")} >= {
            "group-G01_task-main_desc-raw.log", "group-G02_task-main_desc-raw.log",
            "group-G01_task-main_null-pair.log", "group-G02_task-main_null-pair.log"}
