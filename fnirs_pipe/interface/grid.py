"""Defaults shared by the interface's AG Grid tables.

These replaced `dash_table.DataTable`, which was styled through component props. AG Grid is
styled through `--ag-*` custom properties in `assets/interface.css` instead, so what is left
here is the behaviour those tables had, plus the delete column `row_deletable` used to draw.
"""

from __future__ import annotations

# off so the grids behave as the DataTables did; neither offered sorting or filtering
COL_DEF = {"sortable": False, "filter": False, "resizable": True}

AUTO_HEIGHT = {"domLayout": "autoHeight"}

# The glyph is drawn by CSS rather than carried in the row data, so the dicts the callbacks
# read stay exactly the columns they were.
DEL_COL = {
    "headerName": "", "field": "_del", "editable": False,
    "width": 34, "minWidth": 34, "maxWidth": 34,
    "resizable": False, "cellClass": "fp-grid-del",
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
