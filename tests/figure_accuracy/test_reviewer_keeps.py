"""A pair the screening rejected and the reviewer kept is processed as good, and the Methods say so."""

import json

import pytest

from tests._fingerprint import BAD_PAIR
from tests.figure_accuracy._run import Run
from tests.figure_accuracy.conftest import HAND_MARKED, run_capturing

UNCOUPLED = f"S{BAD_PAIR + 1}_D{BAD_PAIR + 1}"


def _run_with(tmp_path, bad: str, keep: str) -> Run:
    tmp_path.mkdir(parents=True, exist_ok=True)
    table = tmp_path / "decisions.tsv"
    table.write_text(f"participant_id\tbad_channels\tkeep_channels\nsub-01\t{bad}\t{keep}\n",
                     encoding="utf-8")
    done = run_capturing(tmp_path, ["--bad-channels", str(table)], ())
    return Run(tmp_path / "out", done["truth"])


def _sidecar(run: Run) -> dict:
    return json.loads(run.stage("sci").with_suffix(".json").read_text(encoding="utf-8"))


def test_the_kept_pair_leaves_the_bad_list_and_the_hand_marked_one_stays(tmp_path):
    baseline = _sidecar(_run_with(tmp_path / "base", HAND_MARKED, ""))
    assert any(c.startswith(UNCOUPLED) for c in baseline["bad_channels"]), \
        "the fingerprint's uncoupled pair has to fail screening for this test to mean anything"

    run = _run_with(tmp_path / "kept", HAND_MARKED, UNCOUPLED)
    sidecar = _sidecar(run)
    assert not any(c.startswith(UNCOUPLED) for c in sidecar["bad_channels"])
    assert any(c.startswith(HAND_MARKED) for c in sidecar["bad_channels"])
    assert {c.split()[0] for c in sidecar["parameters"]["kept_channels"]} == {UNCOUPLED}
    assert f"Channel {UNCOUPLED} was kept after visual review" in run.report.read_text(
        encoding="utf-8")


def test_a_pair_both_marked_bad_and_kept_stops_the_run(tmp_path):
    with pytest.raises(SystemExit):
        _run_with(tmp_path, UNCOUPLED, UNCOUPLED)
    assert not Run(tmp_path / "out", None).stage("sci").exists()
