"""A report section closes its tables and itself before the next one opens; a section left
open inside a table is lifted out by the browser and drawn above that table."""

import re
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[2] / "nirspipe" / "qc" / "templates"
PAGES = sorted(p.name for p in TEMPLATES.iterdir() if p.suffix in {".html", ".j2"})

_TAG = re.compile(r"<(/?)(section|table)\b")


@pytest.mark.parametrize("page", PAGES)
def test_no_section_opens_inside_another_or_a_table(page):
    source = (TEMPLATES / page).read_text(encoding="utf-8")
    open_tags = []
    for match in _TAG.finditer(source):
        line = source.count("\n", 0, match.start()) + 1
        closing, tag = match.group(1) == "/", match.group(2)
        if closing:
            assert open_tags and open_tags[-1] == tag, f"line {line}: </{tag}> closes {open_tags}"
            open_tags.pop()
        else:
            assert not (tag == "section" and open_tags), f"line {line}: <section> inside {open_tags}"
            open_tags.append(tag)
    assert not open_tags
