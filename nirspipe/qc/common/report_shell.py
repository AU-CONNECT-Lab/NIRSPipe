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

import statistics
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.report_shell")

# qc/templates/, a directory up: the templates are shared by both report paths, so they
# stay at the package root rather than following this module into common/
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


def stylesheet(name: str) -> str:
    """A stylesheet from ``templates/``, read once per process."""
    return _SHEETS.setdefault(
        name, (TEMPLATE_DIR / name).read_text(encoding="utf-8"))


_SHEETS: dict[str, str] = {}

# Three sheets, composed in this order by page_vars: what every report agrees on, then
# the look, then the footer. Tokens first so the look can override them, the footer last
# so it can override either.
#
# `css` overrides the look rather than choosing between two.
TOKENS_CSS = stylesheet("_tokens.css")
LOOK_CSS = stylesheet("document.css")
FOOTER_CSS = stylesheet("_footer.css")


# ---- Error handling ----

@contextmanager
def guard(label: str, errors: list, scope: str):
    """Run a report section, and on failure record it instead of losing the whole report.

    ``scope`` is what the run is called in the log: ``sub-01``, ``group-G01_task-main``.
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


# ---- Index pages ----

# How far from its neighbours a row has to sit before its cell is marked. One constant, so a
# flag means the same thing on the subject index and on the dyad index.
OUTLIER_Z = 3.5


def outlier_flags(values: "list[float | None]", z: float = OUTLIER_Z) -> list[bool]:
    """Which rows of an index table sit apart from the others on one metric.

    ::

      [0.96, 0.95, 0.96, 0.40, 0.97] -> [False, False, False, True, False]

    Scaled by the median absolute deviation, so the row being looked for cannot widen the
    scale that is meant to catch it. Under four rows there is nothing to compare against,
    which is why a two-condition design gets no flags at all.

    A column whose rows agree exactly has a zero MAD, which would divide by nothing. The mean
    deviation takes over there: it is zero only when every row agrees, and otherwise still
    marks the single row standing away from a set that agrees with itself.
    """
    present = [v for v in values if v is not None]
    if len(present) < 4:
        return [False] * len(values)

    median = statistics.median(present)
    deviations = [abs(v - median) for v in present]
    scale = statistics.median(deviations) * 1.4826
    if scale == 0:
        scale = sum(deviations) / len(deviations)
    if scale == 0:
        return [False] * len(values)
    return [v is not None and abs(v - median) / scale >= z for v in values]


# ---- Rendering ----

def template_env(**options: Any):
    """The environment the reports render in, the globals every template may call included.

    Globals rather than variables each builder passes, so a template can use a paragraph or
    a nav bar that another already has without being wired for it. ``section_note`` is not
    to be confused with :func:`note` above, which collects what a section skipped.
    """
    from jinja2 import Environment, FileSystemLoader

    from fnirs_pipe.qc.boilerplate.notes import section_note

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False,
                      **options)
    env.globals["section_note"] = section_note
    env.globals["nav_bar"] = nav_bar
    return env


def render(template_name: str, **variables: Any) -> str:
    return template_env().get_template(template_name).render(**variables)


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

    page_vars(title="fnirs-pipe Hyper Raw Report - G01 / main",
              heading="fnirs-pipe Hyper Raw Report",
              nav_meta=[("group", "G01"), ("task", "main")],
              nav_note="SCI thr: 0.80")

    ``nav_meta`` prints as ``label: <b>value</b>`` chips in run order; ``nav_note`` is the
    right-aligned line, which is where the parameters a reader needs to judge the numbers
    belong.

    ``css`` replaces the look, for a report that needs one of its own; the shared tokens come
    before it and the footer's own sheet after, either way, because neither belongs to a
    look. Every report in the package leaves it alone, there being one look.
    """
    return {
        "page_title":   title,
        "page_heading": heading,
        "nav_meta":     list(nav_meta or []),
        "nav_note":     nav_note,
        "base_css":     "\n".join(
            (TOKENS_CSS, LOOK_CSS if css is None else css, FOOTER_CSS)),
        "run_date":     date.today().isoformat(),
    }


# The sections every report closes with, in the order the footer prints them. An index page
# carries none of them and passes its own.
CLOSING_SECTIONS = (("Provenance", "Provenance"), ("Methods", "Methods"))


def nav_bar(
    sections,
    *,
    key: str = "",
    rated: bool = True,
    group: "int | None" = None,
    closing=CLOSING_SECTIONS,
    summary: str = "#Final",
    siblings=(),
    index: str = "",
    index_label: str = "← All runs",
) -> list[dict]:
    """The entries of a report's top bar, from the sections that report drew.

    ::

      nav_bar([("Motion", "Motion"), ("GLM", "GLM")], key="__sub-01_task-main",
              index="sub-01_desc-index_report.html")
      -> Summary | Motion  GLM | Provenance  Methods | <- All runs

    A section is ``(id, name)``, anchored at ``#id``, or ``(id, name, href)`` where the
    anchor is spelled differently from the id a verdict is filed under; ``None`` is a
    divider between two runs of them. ``key`` is appended
    to every ratable id, so a subject's runs do not file one verdict between them, and
    ``rated=False`` drops the pills, which is what a page summarising other pages wants.
    ``group`` stamps one run's entries on a viewer holding several.

    ``siblings`` are the pages read alongside this one, as ``{label, href, current}``, and
    ``index`` the page above it. ``summary=""`` drops the leading entry, for an index page
    whose first section is its summary. The skeleton lives here rather than in each template.
    """
    bar: list[dict] = []
    if summary:
        bar.append({"id": "Summary", "name": "Summary", "href": summary,
                    "ratable": False})

    def extend(entries: list[dict]) -> None:
        if entries:
            if bar:
                bar.append({"sep": True})
            bar.extend(entries)

    extend([{"sep": True} if section is None else
            {"id": section[0] + (key if rated else ""), "name": section[1],
             "href": section[2] if len(section) > 2 else "#" + section[0],
             "ratable": rated}
            for section in sections])
    extend([{"id": sid, "name": name, "href": "#" + sid, "ratable": False}
            for sid, name in closing])
    extend([{"id": f"page{i}", "name": link["label"],
             "href": False if link.get("current") else link["href"], "ratable": False}
            for i, link in enumerate(siblings, start=1)])
    if index:
        extend([{"id": "Index", "name": index_label, "href": index, "ratable": False}])

    if group is not None:
        for entry in bar:
            entry["group"] = group
    return bar


# ---- Footer ----

def provenance_rows(
    nirs_dir: Path,
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
        from fnirs_pipe.qc.common.provenance import scan

        seen: set[tuple] = set()
        for node in sorted(scan(nirs_dir, label=label).values(),
                           key=lambda n: (n.depth, n.label)):
            if not node.step:
                continue
            row = (node.label, node.step,
                   step_sentence(node.step, node.params),
                   node.detail.replace("\n", " "))
            # only bites when no label was given and the scan spans several tasks, which
            # repeat every step with the same settings
            if row in seen:
                continue
            seen.add(row)
            rows.append(dict(zip(("name", "step", "what", "settings"), row)))
    return rows


def collapse_messages(messages: list[str]) -> list[str]:
    """One line per distinct message, in first-seen order, counted when it repeats.

    ["Channel S1_D1: no data", "Channel S1_D2: no data"] stays two lines; the same
    message twice becomes one line ending in " (x2)", so a per-channel section that fails
    the same way on every channel prints one line rather than one per channel.

    Public because the raw viewer prints its own lists: it renders in JavaScript and can
    never share the footer's markup, but the strings it renders come from here.
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
        out["errors"] = collapse_messages(errors)
        out["notes"] = collapse_messages(notes) if notes is not None else []
    if nirs_dir is not None:
        rows = provenance_rows(nirs_dir, scope=scope, errors=errors, label=label)
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
