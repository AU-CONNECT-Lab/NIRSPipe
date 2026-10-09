"""The two cohort pages share one root ``figures/``, so their panels must never share a name.

`nirspipe group` writes the subjects' page and, where the tree holds groups, the dyads'
page beside it; these pin both pages on one rendered tree.
"""

import json
import re

from nirspipe.io.naming import parse_path
from nirspipe.qc.hyper.group_hyper_writer import build_group_hyper_report
from nirspipe.qc.subject.group_writer import build_group_raw_report
from nirspipe.qc.subject.record_io import write_record


def _tree(out):
    for sub, sci in (("01", 0.9), ("02", 0.7)):
        nirs = out / f"sub-{sub}" / "nirs"
        nirs.mkdir(parents=True)
        write_record(nirs / f"sub-{sub}_task-rest_desc-sqm_qc.json",
                     {"step": "sqm", "raw": {"sci_mean": sci, "gvtd_p95": 1e-3}})
    for group in ("G1", "G2"):
        nirs = out / f"group-{group}" / "nirs"
        nirs.mkdir(parents=True)
        (nirs / f"group-{group}_task-rest_desc-sqm_qc.json").write_text(json.dumps({
            "step": "hyper_sqm", "usable_window_frac": 0.6, "one_member_frac": 0.2,
            "neither_frac": 0.2, "n_long_pairs": 10, "n_windows": 20,
        }))
        (nirs / f"group-{group}_task-rest_desc-usable_qc.tsv").write_text(
            "pair\tcondition\tusable_frac\nS1D1\tgame1\t0.5\nS1D2\tvideo\t0.7\n")


def _linked(page):
    return set(re.findall(r"figures/([^\"'#\s]+)", page.read_text(encoding="utf-8")))


def test_the_two_cohort_pages_link_disjoint_figures_that_exist(tmp_path):
    _tree(tmp_path)
    subjects = _linked(build_group_raw_report(tmp_path))
    groups = _linked(build_group_hyper_report(tmp_path))

    assert subjects and len(groups) == 3
    assert not subjects & groups
    assert all((tmp_path / "figures" / name).exists() for name in subjects | groups)


def test_every_cohort_figure_is_named_by_the_config(tmp_path):
    _tree(tmp_path)
    build_group_raw_report(tmp_path)
    build_group_hyper_report(tmp_path)

    for path in (tmp_path / "figures").iterdir():
        entities = parse_path(f"figures/{path.name}")
        assert entities.get("suffix") == "nirs" and entities.get("desc"), path.name
