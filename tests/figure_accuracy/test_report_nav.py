"""Every report's navigation reads the same as the heading each entry jumps to."""

import html as html_lib
import re
from pathlib import Path

_LINK = re.compile(r'<a class="qc-nav-cell" href="#([^"]+)"[^>]*>(.*?)</a>', re.S)
_HEADING = re.compile(r"<(h[23])\b[^>]*>(.*?)</\1>", re.S)
_HINT = re.compile(r'<span class="hint".*?</span>', re.S)


def _text(fragment: str) -> str:
    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", _HINT.sub("", fragment))).split())


def mismatches(page: Path) -> list:
    """``[(label, heading)]`` for every in-page entry whose target's heading reads otherwise.

    The target is the element carrying the id; its heading is its own text when it is one,
    and otherwise the first h2 or h3 inside it.
    """
    html = page.read_text(encoding="utf-8")
    out = []
    for target, label in _LINK.findall(html):
        if label == "FINAL":
            continue                 # the run's verdict cell, not a section
        found =re.search(rf'\sid="{re.escape(target)}"', html)
        if not found:
            out.append((_text(label), None))
            continue
        start = html.rfind("<", 0, found.start())
        heading = _HEADING.search(html, start)
        if _text(heading.group(2)) != _text(label):
            out.append((_text(label), _text(heading.group(2))))
    return out


def test_every_page_of_the_group_tree_names_its_sections_as_its_headings(groups):
    pages = sorted(groups.root.rglob("*_report.html"))
    assert len(pages) > 10
    assert {p.name: m for p in pages if (m := mismatches(p))} == {}


def test_the_raw_viewer_names_its_sections_as_its_headings(raw_viewer_run):
    pages = sorted((raw_viewer_run.out / "sub-01").glob("*_desc-raw_report.html"))
    assert pages
    assert {p.name: m for p in pages if (m := mismatches(p))} == {}
