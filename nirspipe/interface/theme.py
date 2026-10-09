"""Visual tokens shared by the interface's Dash layout and its plotly figures.

The CSS half lives in `assets/interface.css`; this module carries the part that has to
reach plotly. Styling is applied to the figure dicts on their way to the browser rather
than through `plotly.io.templates.default`, so report figures written during a GUI session
still look exactly like the ones the CLI writes.
"""

from __future__ import annotations

FONT_FAMILY = (
    'Segoe UI, -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif'
)

TEXT_COLOR = "#1c2530"
MUTED_COLOR = "#6c7684"
ACCENT = "#3a6ea5"

SIDEBAR_BG = "#172029"

_FONT = {"family": FONT_FAMILY, "size": 11, "color": TEXT_COLOR}
_HOVERLABEL = {"font": {"family": FONT_FAMILY, "size": 11}, "bordercolor": "#d8dee6"}


def style_figure(fig):
    """Give a figure the page's typography, leaving every per-figure setting intact.

    e.g. `{"data": [...], "layout": {"height": 400}}` comes back with `layout.font` and
    `layout.hoverlabel` filled in and `height` untouched. Anything already set on the
    figure wins, so a figure that picked its own font size keeps it.
    """
    if isinstance(fig, dict):
        layout = dict(fig.get("layout") or {})
        layout["font"] = {**_FONT, **(layout.get("font") or {})}
        layout["hoverlabel"] = {**_HOVERLABEL, **(layout.get("hoverlabel") or {})}
        layout.setdefault("paper_bgcolor", "white")
        return {**fig, "layout": layout}
    if hasattr(fig, "update_layout"):
        fig.update_layout(font_family=FONT_FAMILY, hoverlabel_font_family=FONT_FAMILY)
    return fig
