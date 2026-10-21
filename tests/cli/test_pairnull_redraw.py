"""After the re-paired null, each dyad's pages are drawn again so their condition arrows read it.

A dyad's pages are drawn by `nirspipe-hyper`, before the re-paired null exists, so without a
redraw its condition pages kept the phase null they were first drawn against.
"""

import re
import shutil

import pytest

from nirspipe.cli import hyper as hyper_cli
from nirspipe.cli import run as run_cli

from tests._dyad_fingerprint import TASK, make_group_dataset
from tests._fingerprint import CLI_ARGS

HYPER = ["--wtc-fmin", "0.02", "--wtc-band-fmin", "0.03", "--wtc-band-fmax", "0.10",
         "--wtc-phase-null", "2", "--isc-phase-null", "2", "--wtc-seed", "0"]


def _chords(group_dir) -> "dict[str, str]":
    """Each condition page's ISC chord rule, by page name."""
    out = {}
    for page in group_dir.rglob("*cond-*_report.html"):
        cell = re.search(r"<th>ISC chords</th><td>(.*?)</td>", page.read_text(encoding="utf-8"),
                         re.S)
        out[page.name] = cell.group(1).strip() if cell else ""
    return out


def _bytes(group_dir, pattern):
    return {p.name: p.read_bytes() for p in sorted(group_dir.rglob(pattern))}


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    root = tmp_path_factory.mktemp("pairnull_redraw")
    bids, _, _ = make_group_dataset(root)
    pairs = root / "dyads.csv"
    pairs.write_text("group_id,subject_id,task\n" + "".join(
        f"{g},sub-{s},{TASK}\n" for g, s in (("G01", "01"), ("G01", "02"), ("G02", "03"),
                                             ("G02", "04"))))
    deriv = root / "deriv"
    run_cli.main([str(bids), str(deriv), "participant", *CLI_ARGS, "--mode", "denoise",
                  "--participant-label", "01", "02", "03", "04", "--skip-bids-validation",
                  "--no-report"])
    out = root / "out"
    hyper_cli.main([str(deriv), str(out), "group", "--pairs-csv", str(pairs), *HYPER])
    group = out / "group-G01"
    before = {"chords": _chords(group), "record": _bytes(group, "logs/*.toml"),
              "tables": _bytes(group, "nirs/*_relmat.tsv"),
              "pages": _bytes(group, "*_report.html")}
    kept = root / "kept"
    shutil.copytree(out, kept)
    hyper_cli.main_pair_null([str(deriv), str(out), "group", "--pairs-csv", str(pairs)])
    hyper_cli.main_pair_null([str(deriv), str(kept), "group", "--pairs-csv", str(pairs),
                              "--no-report"])
    # what the redraw stands in for: the whole run again from its record, null drawn anew
    rerun = root / "rerun"
    shutil.copytree(kept, rerun)
    record = rerun / "group-G01" / "logs" / f"group-G01_task-{TASK}.toml"
    hyper_cli.main([str(deriv), str(rerun), "group", "--pairs-csv", str(pairs),
                    "--config", str(record)])
    return {"out": group, "kept": kept / "group-G01", "rerun": rerun / "group-G01",
            "before": before}


def test_the_condition_pages_read_the_re_paired_null_once_it_is_drawn(trees):
    before, after = trees["before"]["chords"], _chords(trees["out"])
    assert before and set(after) == set(before)
    assert not any("re-paired" in rule for rule in before.values())
    assert all("re-paired" in rule for rule in after.values())


def test_the_redraw_keeps_the_run_record(trees):
    assert _bytes(trees["out"], "logs/*.toml") == trees["before"]["record"]


def test_the_redraw_writes_the_tables_a_full_rerun_writes(trees):
    """Reading the phase null off disk instead of drawing it again changes nothing written.

    Against the first run, only the condition rows' phase columns move: they average the
    cells that clear the arrows' level, which the re-paired null now sets.
    """
    redrawn = _bytes(trees["out"], "nirs/*_relmat.tsv")
    assert redrawn == _bytes(trees["rerun"], "nirs/*_relmat.tsv")
    moved = sorted(k for k, v in redrawn.items()
                   if k in trees["before"]["tables"] and v != trees["before"]["tables"][k])
    assert moved == [f"group-G01_task-{TASK}_stat-wtc_relmat.tsv",
                     f"group-G01_task-{TASK}_stat-wtcphase_relmat.tsv"]


def test_the_phase_null_is_read_rather_than_drawn_again(trees):
    log = next(trees["out"].glob("logs/*_redraw.log")).read_text(encoding="utf-8")
    assert "read rather than drawn again" in log
    assert "pages redrawn; run record kept" in log


def test_no_report_leaves_the_pages_as_they_were(trees):
    assert _bytes(trees["kept"], "*_report.html") == trees["before"]["pages"]
    assert not list(trees["kept"].glob("logs/*_redraw.log"))
