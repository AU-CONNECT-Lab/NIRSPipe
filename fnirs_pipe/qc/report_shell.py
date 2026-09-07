"""The page shell every QC report is rendered into, and the variables that fill it.

``templates/_report_base.html.j2`` owns the document: head, nav bar, rating bar, the four
closing sections and the tab script. A report template extends it and fills
``{% block content %}``. This module builds what that shell reads, so what a report header
shows and what its footer carries is decided once instead of once per builder.

Three groups of helpers:

- :func:`guard` / :func:`note` collect what a section failed at and what it skipped on
  purpose, both of which the footer prints.
- :func:`page_vars` fills the head and the nav bar.
- :func:`footer_vars` fills the footer: errors, the provenance table, the Methods prose and
  the version table.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.report_shell")

TEMPLATE_DIR = Path(__file__).parent / "templates"


def stylesheet(name: str) -> str:
    """A stylesheet from ``templates/``, read once per process."""
    return _SHEETS.setdefault(
        name, (TEMPLATE_DIR / name).read_text(encoding="utf-8"))


_SHEETS: dict[str, str] = {}
BASE_CSS = stylesheet("_base.css")


# ---- Error handling ----

@contextmanager
def guard(label: str, errors: list, scope: str):
    """Run a report section, and on failure record it instead of losing the whole report.

    ``scope`` is what the run is called in the log: ``sub-01``, ``group-D01_task-chat``.
    """
    try:
        yield
    except Exception as exc:
        errors.append(f"{label}: {exc}")
        logger.exception("%s | %s failed", scope, label)


def note(notes: list, scope: str, message: str) -> None:
    """Record something the report leaves out on purpose.

    Kept apart from the errors list: a section skipped because the run does not carry what
    it needs is a property of the data, not a failure, and it reads as one in the report and
    in the run log. The caller decides what is worth a note; nothing here is fatal.
    """
    notes.append(message)
    logger.info("%s | %s", scope, message)


# ---- Rendering ----

def render(template_name: str, **variables: Any) -> str:
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False)
    return env.get_template(template_name).render(**variables)


# ---- Head and nav ----

def page_vars(
    *,
    title: str,
    heading: str,
    nav_meta: "list[tuple[str, str]] | None" = None,
    nav_note: str = "",
    css: str | None = None,
) -> dict:
    """Variables the shell's head and nav bar read.

    page_vars(title="fnirs-pipe Hyper Raw Report - D01 / chat",
              heading="fnirs-pipe Hyper Raw Report",
              nav_meta=[("group", "D01"), ("task", "chat")],
              nav_note="SCI thr: 0.80")

    ``nav_meta`` prints as ``label: <b>value</b>`` chips in run order; ``nav_note`` is the
    right-aligned line, which is where the parameters a reader needs to judge the numbers
    belong. ``css`` replaces the base stylesheet, for a report with its own look.
    """
    return {
        "page_title":   title,
        "page_heading": heading,
        "nav_meta":     list(nav_meta or []),
        "nav_note":     nav_note,
        "base_css":     BASE_CSS if css is None else css,
        "run_date":     date.today().isoformat(),
    }


# ---- Footer ----

def provenance_rows(
    nirs_dir: Path,
    mode: str | None = None,
    *,
    scope: str = "",
    errors: list | None = None,
    label: str | None = None,
) -> list[dict]:
    """One row per output: what it is, what made it, and what that step did.

    The diagram carries the topology; this carries the narrative, which does not fit in a
    node box. Both come from the same sidecars, so this works on a subject's ``nirs/`` and
    on a group's alike.
    """
    rows: list[dict] = []
    with guard("Provenance table", errors if errors is not None else [], scope):
        from fnirs_pipe.qc.boilerplate.generate import step_sentence
        from fnirs_pipe.qc.provenance import scan

        seen: set[tuple] = set()
        for node in sorted(scan(nirs_dir, label=label).values(),
                           key=lambda n: (n.depth, n.label)):
            if not node.step:
                continue
            row = (node.label, node.step,
                   step_sentence(node.step, node.params, mode),
                   node.detail.replace("\n", " "))
            # only bites when no label was given and the scan spans several tasks, which
            # repeat every step with the same settings
            if row in seen:
                continue
            seen.add(row)
            rows.append(dict(zip(("name", "step", "what", "settings"), row)))
    return rows


def _collapse(messages: list[str]) -> list[str]:
    """One line per distinct message, in first-seen order, counted when it repeats.

    ["Channel S1_D1: no data", "Channel S1_D2: no data"] stays two lines; the same
    message twice becomes one line ending in " (x2)". A per-channel section that fails
    the same way on every channel used to print forty identical lines.
    """
    counts: dict[str, int] = {}
    for message in messages:
        counts[message] = counts.get(message, 0) + 1
    return [m if c == 1 else f"{m} (x{c})" for m, c in counts.items()]


def footer_vars(
    *,
    scope: str = "",
    errors: list | None = None,
    notes: list | None = None,
    nirs_dir: Path | None = None,
    mode: str | None = None,
    label: str | None = None,
    provenance_path: str | None = None,
    methods: dict | None = None,
    versions: dict[str, str] | None = None,
) -> dict:
    """Variables the shell's four closing sections read.

    Only the keys a report has something to say about are returned, and the shell skips a
    section whose key is absent. Passing ``errors`` gets the errors block even when the
    list is empty, because "no errors" is itself worth printing; passing ``nirs_dir`` gets
    the provenance table, ``methods`` the Methods tabs, ``versions`` the version table.
    """
    out: dict[str, Any] = {}
    if errors is not None:
        out["errors"] = _collapse(errors)
        out["notes"] = _collapse(notes) if notes is not None else []
    if nirs_dir is not None:
        rows = provenance_rows(nirs_dir, mode, scope=scope, errors=errors, label=label)
        # a tree with no sidecars has no provenance to show, and a heading over an empty
        # table plus a "diagram not rendered" placeholder says less than nothing
        if rows or provenance_path is not None:
            out["provenance_rows"] = rows
            out["provenance_path"] = provenance_path
    elif provenance_path is not None:
        out["provenance_path"] = provenance_path
    if methods is not None:
        out["methods"] = methods
    if versions is not None:
        out["versions"] = versions
    return out
