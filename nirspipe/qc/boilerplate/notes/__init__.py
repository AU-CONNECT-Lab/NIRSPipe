"""Section prose for the QC reports, so a paragraph is written once and read anywhere.

The per-metric hover text is next door in :mod:`~nirspipe.qc.boilerplate.vocabulary`;
this is the layer above it, the paragraphs that introduce a section or say how to read the
panel under it, and the caveats a run attaches to them from Python (``caveats``). One
module per report family, merged into one table here.

**Every paragraph a report prints lives here.** A template keeps the ``{% if %}`` that picks
which paragraph to say, never the sentence itself: a paragraph that differs between two runs
is split into one key per variant, and a clause that comes and goes is a key of its own
(``hyper_post.selector_order``) or a slot filled by one (``hyper_raw.onset_residuals``). Headings,
table headers and the short hints beside a ``<summary>`` are labels and stay in the template.

The strings carry HTML (``<code>``, ``<b>``, ``&nbsp;``) and reach the page unescaped,
which is what ``render()`` does with every other variable; a report environment that turned
autoescaping on would have to wrap these.
"""

from __future__ import annotations

from nirspipe.qc.boilerplate.notes.caveats import NOTES as _CAVEATS
from nirspipe.qc.boilerplate.notes.hyper import NOTES as _HYPER
from nirspipe.qc.boilerplate.notes.raw import NOTES as _RAW
from nirspipe.qc.boilerplate.notes.subject import NOTES as _SUBJECT


def _merge(*tables: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for table in tables:
        clash = merged.keys() & table.keys()
        if clash:
            raise ValueError(f"section note keys defined twice: {sorted(clash)}")
        merged.update(table)
    return merged


SECTION_NOTES = _merge(_SUBJECT, _RAW, _HYPER, _CAVEATS)


def section_note(key: str, **values: object) -> str:
    """One section's paragraph, with any value slots filled.

    ::

      section_note("trial_qc", window="-5 to 25 s")

    Returns '' for a key nothing is written for, the way
    :func:`~nirspipe.qc.boilerplate.vocabulary.metric_summary` does: a renamed key leaves
    a missing paragraph rather than stopping the render half way down a report.
    """
    text = SECTION_NOTES.get(key, "")
    return text.format(**values) if (text and values) else text
