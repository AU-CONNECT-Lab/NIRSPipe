"""Every report's navigation names the heading each entry jumps to, or that heading's declared short form."""

import html as html_lib
import re
from pathlib import Path

from nirspipe.qc.common.report_shell import NAV_SHORT

_LINK = re.compile(r'<a class="qc-nav-cell" href="#([^"]+)"([^>]*)>(.*?)</a>', re.S)
_HEADING = re.compile(r"<(h[23])\b[^>]*>(.*?)</\1>", re.S)
_HINT = re.compile(r'<span class="hint".*?</span>', re.S)
_TITLE = re.compile(r'\stitle="([^"]*)"')


def _text(fragment: str) -> str:
    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", _HINT.sub("", fragment))).split())


def mismatches(page: Path) -> list:
    """``[(label, heading)]`` for each entry naming neither its heading nor its short form with the heading on hover."""
    html = page.read_text(encoding="utf-8")
    out = []
    for target, attrs, label in _LINK.findall(html):
        if label == "FINAL":
            continue                 # the run's verdict cell, not a section
        found = re.search(rf'\sid="{re.escape(target)}"', html)
        if not found:
            out.append((_text(label), None))
            continue
        # the element carrying the id: its own text when it is a heading, else its first h2 or h3
        start = html.rfind("<", 0, found.start())
        heading = _text(_HEADING.search(html, start).group(2))
        title = _TITLE.search(attrs)
        short = (_text(label) == NAV_SHORT.get(heading)
                 and title is not None and html_lib.unescape(title.group(1)) == heading)
        if _text(label) != heading and not short:
            out.append((_text(label), heading))
    return out


def test_every_page_of_the_group_tree_names_its_sections_as_its_headings(groups):
    pages = sorted(groups.root.rglob("*_report.html"))
    assert len(pages) > 10
    assert {p.name: m for p in pages if (m := mismatches(p))} == {}


def test_the_raw_viewer_names_its_sections_as_its_headings(raw_viewer_run):
    pages = sorted((raw_viewer_run.out / "sub-01").glob("*_desc-raw_report.html"))
    assert pages
    assert {p.name: m for p in pages if (m := mismatches(p))} == {}
