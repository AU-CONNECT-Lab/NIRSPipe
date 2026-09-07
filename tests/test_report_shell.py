"""The page shell every QC report is rendered into.

`_report_base.html.j2` owns the document and `_footer.html.j2` owns the four closing
sections, so a report gets its head, nav, rating bar, errors block, provenance table,
Methods tabs and version table without writing any of them. This file is the guard on that
arrangement, and it exists because both halves of it fail quietly:

- A report template that stops extending the base still renders. It just renders a
  fragment with no `<html>`, which a browser will happily show.
- `footer_vars` decides which of the four sections appear by which keys it returns. Drop a
  key and the section vanishes with no error anywhere.
"""

import re
from pathlib import Path

import pytest
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from fnirs_pipe.qc.report_shell import (
    BASE_CSS,
    FOOTER_CSS,
    TEMPLATE_DIR,
    TOKENS_CSS,
    dashboard_css,
    footer_vars,
    guard,
    note,
    page_vars,
    stylesheet,
)

# Every report template, i.e. every .html.j2 that is not a partial (partials are named
# with a leading underscore and are included rather than rendered).
REPORT_TEMPLATES = sorted(
    p.name for p in TEMPLATE_DIR.glob("*.html.j2") if not p.name.startswith("_")
)

METHODS = {"html": "<p>M</p>", "plain": "M", "markdown": "M", "latex": "M"}


@pytest.fixture
def env():
    # the shell is what is under test, not whether a caller passed every panel variable
    return Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False,
                       undefined=ChainableUndefined)


# ---- the shell is not optional ----

def test_every_report_template_extends_the_shell():
    assert REPORT_TEMPLATES, "no report templates found; the glob or the directory moved"
    for name in REPORT_TEMPLATES:
        first = (TEMPLATE_DIR / name).read_text(encoding="utf-8").lstrip().splitlines()[0]
        assert first == '{% extends "_report_base.html.j2" %}', (
            f"{name} does not extend the shell, so it renders without a document"
        )


def test_no_report_template_opens_its_own_document():
    for name in REPORT_TEMPLATES:
        text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
        for tag in ("<!DOCTYPE", "<html", "<head>", "<body>"):
            assert tag not in text, f"{name} still carries its own {tag}"


def test_no_report_template_writes_the_footer_itself():
    # the four section ids belong to _footer.html.j2 alone; a second copy would render twice
    for name in REPORT_TEMPLATES:
        text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
        for section in ('id="Errors"', 'id="Provenance"', 'id="Methods"', 'id="Versions"'):
            assert section not in text, f"{name} duplicates the footer's {section}"


# ---- footer_vars decides which sections appear ----

def test_an_absent_key_drops_its_section(env):
    html = env.get_template("group_report.html.j2").render(
        **page_vars(title="T", heading="T"),
        **footer_vars(versions={"fnirs-pipe": "0.1"}),
    )
    assert 'id="Versions"' in html
    for absent in ('id="Errors"', 'id="Provenance"', 'id="Methods"'):
        assert absent not in html


def test_an_empty_errors_list_still_prints_the_section(env):
    # "no errors" is a result, unlike a section the report has nothing to say about
    html = env.get_template("group_report.html.j2").render(
        **page_vars(title="T", heading="T"), **footer_vars(errors=[]),
    )
    assert 'id="Errors"' in html
    assert "No errors or warnings." in html


def test_methods_tabs_appear_with_their_script(env):
    html = env.get_template("group_report.html.j2").render(
        **page_vars(title="T", heading="T"), **footer_vars(methods=METHODS),
    )
    for tab in ("tab-rendered", "tab-plain", "tab-md", "tab-latex"):
        assert f'id="{tab}"' in html
    assert "function showTab" in html, "the tabs render with nothing to switch them"


def test_a_tree_with_no_sidecars_gets_no_provenance_section(tmp_path):
    assert "provenance_rows" not in footer_vars(nirs_dir=tmp_path)


def test_a_diagram_alone_is_enough_for_the_section(tmp_path):
    out = footer_vars(nirs_dir=tmp_path, provenance_path="figures/provenance.png")
    assert out["provenance_rows"] == []
    assert out["provenance_path"] == "figures/provenance.png"


# ---- error and note collection ----

def test_guard_records_the_failure_and_lets_the_report_continue():
    errors = []
    with guard("Brain views", errors, "sub-01"):
        raise ValueError("no head coordinates")
    assert errors == ["Brain views: no head coordinates"]


def test_the_same_failure_on_every_channel_collapses_to_one_line():
    errors = ["Channel figure: bad", "Channel figure: bad", "Other: x"]
    out = footer_vars(errors=errors)["errors"]
    assert out == ["Channel figure: bad (x2)", "Other: x"]


def test_notes_are_kept_apart_from_errors():
    notes = []
    note(notes, "sub-01", "no epochs: the run carries no events")
    out = footer_vars(errors=[], notes=notes)
    assert out["errors"] == []
    assert out["notes"] == notes


# ---- stylesheets ----

def test_a_look_replaces_the_other_look_and_not_the_tokens():
    sheet = page_vars(title="T", heading="T", css=stylesheet("subject.css"))["base_css"]
    # the document look, the shared tokens, the footer: all three
    assert "font-size: 14px" in sheet
    assert ":root" in sheet
    assert ".boilerplate-html" in sheet
    # and not the dashboard look. Its ground colour reaching a document report is the
    # visible symptom of the sheets stacking instead of one look replacing the other.
    assert "#f5f7fa" not in sheet


def test_the_tokens_carry_what_both_looks_agree_on():
    for rule in ("h2 {", "table {", "th, td {", "pre {", "img {", ".fig-path {"):
        assert rule in TOKENS_CSS, f"{rule} belongs to both looks and is in neither"


def test_neither_look_keeps_a_copy_of_the_token_rules():
    # the dashboard used to carry .report-footer-scoped copies of every one of these
    for name in ("_base.css", "subject.css"):
        sheet = stylesheet(name)
        for rule in ("th, td {", ".report-footer table {", ".fig-path {"):
            assert rule not in sheet, f"{name} has a second copy of {rule}"


def test_the_raw_viewer_takes_the_dashboard_sheet_rather_than_copying_it():
    # the fourth copy of the dashboard look lived here; the viewer cannot extend the shell
    # (it is one JavaScript-driven document) but it can take the sheet
    text = (TEMPLATE_DIR / "raw_viewer.html").read_text(encoding="utf-8")
    assert "{{ base_css }}" in text
    for rule in ("#qc-nav {", ".card {", ".panel-title {", "box-sizing: border-box"):
        assert rule not in text, f"the viewer still carries its own {rule}"


def test_the_dashboard_sheet_is_the_same_one_page_vars_composes():
    assert dashboard_css() == page_vars(title="T", heading="T")["base_css"]


def test_the_footer_styles_ship_with_the_footer():
    for cls in (".tab-btn", ".boilerplate-html", ".error-list", ".note-list",
                ".section-hint", ".placeholder"):
        assert cls in FOOTER_CSS, f"{cls} is used by the footer and styled nowhere"


def test_page_vars_appends_the_footer_sheet_to_either_look():
    for css in (None, stylesheet("subject.css")):
        sheet = page_vars(title="T", heading="T", css=css)["base_css"]
        assert ".boilerplate-html" in sheet, "the footer would render unstyled"


def test_neither_base_sheet_keeps_its_own_copy_of_the_footer_rules():
    # both sheets carried a verbatim copy until the rules moved to _footer.css; a copy
    # coming back means one look has quietly started overriding the other
    for name in ("_base.css", "subject.css"):
        sheet = stylesheet(name)
        for cls in (".tab-btn {", ".boilerplate-html {", ".error-list {"):
            assert cls not in sheet, f"{name} has a second copy of {cls}"


def test_the_index_shares_the_subject_stylesheet():
    # the index used to carry a near-copy of the subject report's CSS; only the rules that
    # differ belong in its own block, and a block this long means they have re-forked
    text = (TEMPLATE_DIR / "subject_index.html.j2").read_text(encoding="utf-8")
    block = re.search(r"{% block css %}(.*?){% endblock %}", text, re.S)
    assert block, "the index defines no css block; check it still loads subject.css"
    assert len(block.group(1).strip().splitlines()) < 40
