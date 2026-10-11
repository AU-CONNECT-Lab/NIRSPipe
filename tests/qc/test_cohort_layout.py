"""Where the two cohort pages put their numbers: Summary holds the cohort's context, and the
tables sit under Quality Metrics, as on the subject report."""

from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.hyper.group_hyper_writer import build_group_hyper_report
from nirspipe.qc.subject.group_writer import build_group_raw_report
from nirspipe.qc.subject.record_io import write_record
from tests.figure_accuracy._dyad import html_tables
from tests.qc.test_cohort_figures import _tree

HEADLINE = ["Metric", "Median", "Range"]


def _subjects(out):
    """Two runs carrying a headline metric, so the cohort summary has a row."""
    for sub, psp in (("01", 0.3), ("02", 0.2)):
        nirs = out / f"sub-{sub}" / "nirs"
        nirs.mkdir(parents=True)
        write_record(nirs / f"sub-{sub}_task-rest_desc-sqm_qc.json",
                     {"step": "sqm", "raw": {"psp_mean": psp, "gvtd_p95": 1e-3}})


def _headline_heading(page) -> str:
    return next(head for head, rows in html_tables(page) if rows and rows[0] == HEADLINE)


def _section(html: str, section_id: str) -> str:
    start = html.index(f'<section id="{section_id}"')
    return html[start:html.index("</section>", start)]


def test_the_subject_cohort_summary_sits_under_quality_metrics(tmp_path):
    _subjects(tmp_path)
    page = build_group_raw_report(tmp_path)
    html = page.read_text(encoding="utf-8")

    assert _headline_heading(page) == "Cohort summary"
    assert html.index('<h2 id="Metrics">') < html.index('<section id="CohortSummary"')
    assert html.index('<section id="CohortSummary"') < html.index('<section id="Table"')
    assert 'href="#Metrics"' in html


def test_the_small_cohort_caveat_is_not_repeated_right_under_summary(tmp_path):
    _subjects(tmp_path)
    html = build_group_raw_report(tmp_path).read_text(encoding="utf-8")
    caveat = section_note("group.small_cohort", n=2)

    summary = html[html.index('<h2 id="Summary">'):html.index('<section id="Overview"')]
    assert caveat in summary
    assert caveat not in _section(html, "Overview")
    assert caveat in _section(html, "Distributions")


def test_the_dyad_cohort_summary_sits_under_quality_metrics(tmp_path):
    _tree(tmp_path)
    page = build_group_hyper_report(tmp_path)
    html = page.read_text(encoding="utf-8")

    assert _headline_heading(page) == "Cohort summary"
    assert html.index('<h2 id="Metrics">') < html.index('<section id="Table"')
    assert 'href="#Table"' not in html
