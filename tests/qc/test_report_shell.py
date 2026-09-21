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

import pathlib
import re

import pytest
from jinja2 import ChainableUndefined

from fnirs_pipe.qc.common.report_shell import (
    FOOTER_CSS,
    TEMPLATE_DIR,
    TOKENS_CSS,
    footer_vars,
    guard,
    nav_bar,
    note,
    page_vars,
    stylesheet,
    template_env,
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
    return template_env(undefined=ChainableUndefined)


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

def test_the_default_sheet_is_the_look_the_tokens_and_the_footer():
    # There is one look. A report that names no `css` gets it, which is the point of the
    # default: it used to be a second, dashboard look, and by the time the last report
    # stopped wearing that one the default was a look nothing wore and a new report could
    # reach it by forgetting a keyword.
    sheet = page_vars(title="T", heading="T")["base_css"]
    assert "font-size: 14px" in sheet, "the look is missing"
    assert ":root" in sheet, "the tokens are missing"
    assert ".boilerplate-html" in sheet, "the footer would render unstyled"


def test_the_deleted_dashboard_look_has_not_come_back():
    # `#f5f7fa` was its ground colour and `#qc-nav` its sticky bar. Either one appearing in
    # a composed sheet means the sheet is back, whether as a file or pasted into the look.
    sheet = page_vars(title="T", heading="T")["base_css"]
    for rule in ("#f5f7fa", "#qc-nav {", ".card {", ".panel-title {"):
        assert rule not in sheet, f"the dashboard look is back: {rule}"
    assert not (TEMPLATE_DIR / "_base.css").exists(), "_base.css is back"


def test_a_named_css_replaces_the_look_and_not_the_tokens():
    # the seam a second look would come through, exercised with a stand-in
    sheet = page_vars(title="T", heading="T", css="body { font-size: 99px; }")["base_css"]
    assert "font-size: 99px" in sheet
    assert "font-size: 14px" not in sheet, "the two looks stacked instead of replacing"
    assert ":root" in sheet and ".boilerplate-html" in sheet


def test_the_tokens_carry_what_the_look_and_the_footer_agree_on():
    for rule in ("h2 {", "table {", "th, td {", "pre {", "img {", ".fig-path {"):
        assert rule in TOKENS_CSS, f"{rule} is shared and is in neither sheet"


def test_the_look_keeps_no_copy_of_the_token_rules():
    sheet = stylesheet("document.css")
    for rule in ("th, td {", ".report-footer table {", ".fig-path {"):
        assert rule not in sheet, f"document.css has a second copy of {rule}"


def test_the_raw_viewer_takes_the_shells_sheet_rather_than_copying_it():
    # a fourth copy of the look lived here; the viewer cannot extend the shell (it is one
    # JavaScript-driven document) but it can take the sheet page_vars composes
    text = (TEMPLATE_DIR / "raw_viewer.html").read_text(encoding="utf-8")
    assert "{{ base_css }}" in text
    for rule in ("#qc-nav {", ".card {", ".panel-title {", "box-sizing: border-box"):
        assert rule not in text, f"the viewer still carries its own {rule}"


def test_the_footer_styles_ship_with_the_footer():
    for cls in (".tab-btn", ".boilerplate-html", ".error-list", ".note-list",
                ".section-hint", ".placeholder"):
        assert cls in FOOTER_CSS, f"{cls} is used by the footer and styled nowhere"


def test_the_look_keeps_no_copy_of_the_footer_rules():
    # it carried a verbatim copy until the rules moved to _footer.css; a copy coming back
    # means the look has quietly started overriding the footer
    sheet = stylesheet("document.css")
    for cls in (".tab-btn {", ".boilerplate-html {", ".error-list {"):
        assert cls not in sheet, f"document.css has a second copy of {cls}"


def test_no_report_writer_names_a_stylesheet():
    # The arrangement the deletion rests on: `stylesheet` is how a caller names a look, and
    # nothing outside the shell calls it, so the one look reaches every page through
    # page_vars and there is no keyword to forget. Not `extra_css`, which is per-figure CSS
    # injected into an iframe and is a different thing.
    import fnirs_pipe.qc as qc_pkg
    qc = pathlib.Path(qc_pkg.__file__).resolve().parent
    for path in sorted(qc.rglob("*.py")):
        if path.name == "report_shell.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "stylesheet(" not in text, (
            f"{path.name} names its own stylesheet; there is one look and it is the default")


def test_the_index_shares_the_one_stylesheet():
    # the index used to carry a near-copy of the subject report's CSS; only the rules that
    # differ belong in its own block, and a block this long means they have re-forked
    text = (TEMPLATE_DIR / "subject_index.html.j2").read_text(encoding="utf-8")
    block = re.search(r"{% block css %}(.*?){% endblock %}", text, re.S)
    assert block, "the index defines no css block; check it still loads document.css"
    assert len(block.group(1).strip().splitlines()) < 40


# ---- the top bar is one builder ----

def test_every_report_carries_a_top_bar():
    # the bar is what a reader moves through a report with, and a template that stops
    # setting nav_modules still renders, just without one
    for name in REPORT_TEMPLATES + ["raw_viewer.html"]:
        text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
        assert "nav_modules" in text, f"{name} renders with no top bar"
        assert "nav_bar(" in text, f"{name} spells its own bar instead of calling nav_bar"


def test_nav_bar_wraps_the_sections_in_the_shared_skeleton():
    bar = nav_bar([("Motion", "Motion"), None, ("Metrics", "Metrics")],
                  siblings=[{"label": "video", "href": "v.html"}],
                  index="sub-01_qc.html")
    assert [e.get("name", "|") for e in bar] == [
        "Summary", "|", "Motion", "|", "Metrics", "|", "Provenance", "Methods",
        "|", "video", "|", "← All runs"]
    assert bar[0]["href"] == "#Final"
    assert [e["ratable"] for e in bar if "name" in e].count(True) == 2


def test_a_key_reaches_only_the_ids_a_verdict_is_filed_under():
    # the anchor is the section's, the id the run's: a subject's runs share the headings
    # and must not share the verdict
    bar = nav_bar([("Motion", "Motion")], key="__sub-01_task-chat")
    motion = next(e for e in bar if e.get("name") == "Motion")
    assert motion["id"] == "Motion__sub-01_task-chat"
    assert motion["href"] == "#Motion"
    assert all(e["id"] == "Provenance" for e in bar if e.get("name") == "Provenance")


def test_an_unrated_bar_has_nothing_to_rate_with():
    assert not any(e.get("ratable") for e in nav_bar([("Runs", "Runs")], rated=False))
    # and the key cannot reach an id that no longer carries a pill
    assert nav_bar([("Runs", "Runs")], rated=False, key="__x")[2]["id"] == "Runs"


def test_a_summary_page_renders_no_rating_layer(env):
    html = env.get_template("group_report.html.j2").render(
        **page_vars(title="T", heading="T"), **footer_vars(versions={"fnirs-pipe": "0.1"}),
    )
    assert 'id="qc-container"' in html, "the cohort report lost its bar"
    for rating_only in ('class="qc-module"', "saveEndpoint", 'id="rating-static-banner"'):
        assert rating_only not in html, f"an unrated page still ships {rating_only}"


def test_the_two_index_pages_share_one_stylesheet():
    for name in ("subject_index.html.j2", "hyper_index.html.j2"):
        text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
        assert '{% include "_index.css" %}' in text, f"{name} spells the index look again"
