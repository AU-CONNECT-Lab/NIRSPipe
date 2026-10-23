"""The cohort page lists the subjects whose last run did not finish, read from their run records."""

from nirspipe.cli.run_record import record_path, write_run_outcome
from nirspipe.qc.subject.group_writer import build_group_raw_report, unfinished_subjects
from tests.figure_accuracy._dyad import table_under
from tests.qc.test_cohort_layout import _subjects

HEADING = "Subjects that did not finish"


def _record(out, sub, *, dry_run=False, sessions=None):
    record = record_path(out / f"sub-{sub}", sub, sessions)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        f'[execution]\nrun_timestamp = "2026-10-19T09:00:0{sub[-1]}"\n'
        f"dry_run = {'true' if dry_run else 'false'}\n", encoding="utf-8")
    return record


def _tree(out):
    _subjects(out)
    write_run_outcome(_record(out, "01"), "failed", 3.0, "ValueError: bad <channel> 'S1'")
    write_run_outcome(_record(out, "02"), "success", 5.0)
    _record(out, "03")                       # killed: the record has no [outcome]
    _record(out, "04", dry_run=True)


def test_each_selection_of_one_subject_is_read_on_its_own(tmp_path):
    # two sessions run in parallel: one finished, one failed
    write_run_outcome(_record(tmp_path, "05", sessions=["1"]), "success", 1.0)
    write_run_outcome(_record(tmp_path, "05", sessions=["2"]), "failed", 1.0, "ValueError: x")

    rows = unfinished_subjects(tmp_path)
    assert [(r["subject"], r["status"]) for r in rows] == [("sub-05_ses-2", "failed")]


def test_failed_and_unfinished_subjects_are_listed_and_the_rest_are_not(tmp_path):
    _tree(tmp_path)
    rows = table_under(build_group_raw_report(tmp_path), HEADING)

    assert rows == [
        ["Subject", "Status", "Started", "Error"],
        ["sub-01", "failed", "2026-10-19T09:00:01", "ValueError: bad <channel> 'S1'"],
        ["sub-03", "unfinished", "2026-10-19T09:00:03", "—"],
    ]


def test_the_error_text_is_escaped_and_the_section_is_in_the_nav(tmp_path):
    _tree(tmp_path)
    html = build_group_raw_report(tmp_path).read_text(encoding="utf-8")

    assert "bad &lt;channel&gt;" in html
    assert 'href="#Unfinished"' in html
    assert html.index('<h2 id="Summary">') < html.index('<section id="Unfinished"')
    assert html.index('<section id="Unfinished"') < html.index('<section id="Overview"')


def test_a_cohort_where_every_subject_finished_has_no_such_section(tmp_path):
    _subjects(tmp_path)
    write_run_outcome(_record(tmp_path, "01"), "success", 1.0)
    html = build_group_raw_report(tmp_path).read_text(encoding="utf-8")

    assert HEADING not in html
    assert 'href="#Unfinished"' not in html
