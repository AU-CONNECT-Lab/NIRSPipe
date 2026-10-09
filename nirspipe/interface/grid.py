"""Defaults shared by the interface's AG Grid tables.

AG Grid is styled through `--ag-*` custom properties in `assets/interface.css`, so what is
here is the tables' behaviour, plus the delete column.
"""

from __future__ import annotations

# no sorting or filtering
COL_DEF = {"sortable": False, "filter": False, "resizable": True}

AUTO_HEIGHT = {"domLayout": "autoHeight"}

# The glyph is drawn by CSS rather than carried in the row data, so the dicts the callbacks
# read hold only the data columns.
DEL_COL = {
    "headerName": "", "field": "_del", "editable": False,
    "width": 34, "minWidth": 34, "maxWidth": 34,
    "resizable": False, "cellClass": "fp-grid-del", "cellDataType": False,
}


def rows_minus_clicked(cell, rows):
    """Rows without the one whose cross was clicked, or None if the click was elsewhere.

    e.g. `{"colId": "_del", "rowIndex": 1}` over three rows gives back rows 0 and 2; a click
    on any other column gives back None, which the callers pass on as `no_update`.
    """
    if not cell or cell.get("colId") != DEL_COL["field"]:
        return None
    rows = rows or []
    index = cell.get("rowIndex")
    if index is None or not 0 <= index < len(rows):
        return None
    return [row for i, row in enumerate(rows) if i != index]
