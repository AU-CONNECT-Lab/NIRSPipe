"""Every element a report script looks up by a fixed id has to exist on the page."""

import re
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[2] / "fnirs_pipe" / "qc" / "templates"
PAGES = sorted(p.name for p in TEMPLATES.iterdir() if p.suffix in {".html", ".j2"})

_LOOKUP = re.compile(r"""getElementById\(\s*["']([\w\-]+)["']\s*\)""")
_DECLARED = re.compile(r"""\bid=["']([\w\-]+)["']|\.id\s*=\s*["']([\w\-]+)""")


@pytest.mark.parametrize("page", PAGES)
def test_every_looked_up_id_is_declared(page):
    # a lookup that finds nothing is skipped by its `if (el)` guard, so the text never shows
    source = (TEMPLATES / page).read_text(encoding="utf-8")
    declared = {a or b for a, b in _DECLARED.findall(source)}
    assert set(_LOOKUP.findall(source)) <= declared
