"""One toggle for every collapsible parameter group, on every page."""

from __future__ import annotations

from dash import MATCH, Input, Output, State, callback


@callback(
    Output({"type": "fp-section-body", "key": MATCH}, "is_open"),
    Output({"type": "fp-section-toggle", "key": MATCH}, "className"),
    Input({"type": "fp-section-toggle", "key": MATCH}, "n_clicks"),
    State({"type": "fp-section-body", "key": MATCH}, "is_open"),
    prevent_initial_call=True,
)
def toggle_section(n_clicks, is_open):
    opening = not is_open
    base = "fp-section fp-section-head"
    return opening, base if opening else f"{base} fp-section-closed"


def summary(*parts) -> str:
    """The header line for a collapsed group: what is set, in the order it is asked for."""
    return " · ".join(str(p) for p in parts if p not in (None, "", False))


def value(label: str, value, unit: str = "") -> str | None:
    if value in (None, ""):
        return None
    return f"{label} {value}{unit}"


def rng(label: str, lo, hi, unit: str = "") -> str | None:
    if lo in (None, "") and hi in (None, ""):
        return None
    return f"{label} {lo if lo not in (None, '') else '?'}–{hi if hi not in (None, '') else '?'}{unit}"
