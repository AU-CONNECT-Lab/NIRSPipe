"""Layout kit shared by the interface's pages.

Each page used to carry its own ``_card`` and lay parameters out with ``dbc.Row`` plus a
hardcoded ``width=``, which put a different column count on every card and stretched the
inputs on a wide screen. Here a parameter block is one grid that fits as many columns as
the content area allows, so blocks line up with each other and reflow on their own.

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


def section(title, *children):
    """A titled group inside a card, so related parameters share one card instead of four."""
    return html.Div([html.Div(title, className="fp-section"), *children])


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


def field(label, control, span=1, hint=None):
    children = [dbc.Label(label)] if label else []
    children.append(control)
    if hint:
        children.append(html.Small(hint, className="fp-hint"))
    cls = "fp-field" if span == 1 else f"fp-field fp-span-{span}"
    return html.Div(children, className=cls)


def band(label, *parts, span=2, hint=None):
    """A field whose control is several inputs joined into one, e.g. a frequency range."""
    parts = [dbc.InputGroupText(p) if isinstance(p, str) else p for p in parts]
    return field(label, dbc.InputGroup(list(parts)), span=span, hint=hint)


def switches(control, span=1, hint=None):
    """A checklist that sits in a parameter grid without a label above it."""
    children = [control]
    if hint:
        children.append(html.Small(hint, className="fp-hint"))
    cls = "fp-field fp-field-switches" + ("" if span == 1 else f" fp-span-{span}")
    return html.Div(children, className=cls)


def actions(*children, className=""):
    """A row of buttons that keeps its own baseline instead of borrowing a field's."""
    return html.Div(list(children), className=("fp-actions " + className).strip())
