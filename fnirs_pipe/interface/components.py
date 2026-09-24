"""Layout kit shared by the interface's pages.

A parameter block is one grid that fits as many columns as the content area allows, so
blocks line up with each other and reflow on their own.

    params(field("DPF", dbc.Input(id="an-dpf")),
           band("Cardiac band (Hz)", lo_input, "-", hi_input))
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


# ---- Containers ----

def card(title, *children, subtitle=None, className="mb-3", body_class=None, body_style=None):
    head = [title]
    if subtitle is not None:
        head.append(html.Small(subtitle, className="ms-2 fw-normal"))
    return dbc.Card(
        [dbc.CardHeader(head),
         dbc.CardBody(list(children), className=body_class, style=body_style)],
        className=className,
    )


def section(title, *children, key=None, open=True):
    """A titled group inside a card, so related parameters share one card instead of four.

    With a `key` the group folds away, and its header carries a summary of what is set
    inside it. Collapsing a group of parameters only helps if the values stay readable;
    the summary is what a page fills in through `{key}-summary`.
    """
    if key is None:
        return html.Div([html.Div(title, className="fp-section"), *children])

    return html.Div([
        html.Div(
            [html.Span(title, className="fp-section-name"),
             html.Span(id=f"{key}-summary", className="fp-section-summary"),
             html.Span("▾", className="fp-section-caret")],
            id={"type": "fp-section-toggle", "key": key},
            className="fp-section fp-section-head" + ("" if open else " fp-section-closed"),
            n_clicks=0,
        ),
        dbc.Collapse(list(children),
                     id={"type": "fp-section-body", "key": key},
                     is_open=open),
    ])


def split(main, aside):
    """Page body as a form column plus a panel that stays put while the form scrolls."""
    return html.Div(
        [html.Div(main, className="fp-split-main"),
         html.Div(aside, className="fp-split-aside")],
        className="fp-split",
    )


# ---- Parameter blocks ----

def params(*items, className=""):
    return html.Div(list(items), className=("fp-params " + className).strip())


# columns out of twelve, by what the control holds rather than by where the row falls
NUMBER, CHOICE, RANGE, PATH, FULL = 2, 3, 4, 6, 12


def _width_for(control) -> int:
    """A control's width follows what it holds; a path or a long label passes its own."""
    if isinstance(control, dbc.InputGroup):
        return RANGE
    if isinstance(control, dbc.Input) and getattr(control, "type", None) == "number":
        return NUMBER
    return CHOICE


def field(label, control, span=None, hint=None, id=None):
    children = [dbc.Label(label) if label else html.Span()]
    children.append(control)
    # the hint row exists either way, so a row of fields keeps one baseline
    children.append(html.Small(hint, className="fp-hint") if hint else html.Span())
    width = span if span is not None else _width_for(control)
    # an id here, rather than on a wrapper, keeps the field a direct child of the grid
    kwargs = {"id": id} if id else {}
    return html.Div(children, className=f"fp-field fp-w-{width}", **kwargs)


def band(label, *parts, span=RANGE, hint=None):
    """A field whose control is several inputs joined into one, e.g. a frequency range."""
    parts = [dbc.InputGroupText(p) if isinstance(p, str) else p for p in parts]
    return field(label, dbc.InputGroup(list(parts)), span=span, hint=hint)


def action_field(*children, span=CHOICE):
    """Buttons that belong on a form row: they take a field's slot so they share its baseline."""
    return html.Div(
        [html.Span(), html.Div(list(children), className="fp-actions"), html.Span()],
        className=f"fp-field fp-w-{span}",
    )


def switches(control, hint=None, columns=False):
    """Checkboxes, on a row of their own: they are not fields and do not line up as ones."""
    children = [html.Div(control, className="fp-switches-cols" if columns else None)]
    if hint:
        children.append(html.Small(hint, className="fp-hint d-block mt-1"))
    return html.Div(children, className="fp-switches")


def actions(*children, className=""):
    """A row of buttons that keeps its own baseline instead of borrowing a field's."""
    return html.Div(list(children), className=("fp-actions " + className).strip())
