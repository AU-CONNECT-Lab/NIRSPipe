"""Read the figures a report wrote back out of its HTML, as plain data and numpy arrays."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np

_DECODER = json.JSONDecoder()
_NEW_PLOT = re.compile(r"Plotly\.newPlot\(\s*")


def _decode(node):
    # plotly writes numeric arrays as {"dtype", "bdata", "shape"}, base64 of the raw buffer
    if isinstance(node, dict):
        if "bdata" in node and "dtype" in node:
            values = np.frombuffer(base64.b64decode(node["bdata"]), dtype=np.dtype(node["dtype"]))
            shape = node.get("shape")
            if isinstance(shape, str):
                shape = [int(s) for s in shape.split(",")]
            return values.reshape(shape) if shape else values
        return {k: _decode(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_decode(v) for v in node]
    return node


def _next_argument(text: str, i: int) -> int:
    i = text.index(",", i) + 1
    while text[i].isspace():
        i += 1
    return i


def plotly_figures(path: Path) -> list[dict]:
    """Every ``Plotly.newPlot`` call in the file, as ``{"data": [...], "layout": {...}}``."""
    text = Path(path).read_text(encoding="utf-8")
    figures = []
    for match in _NEW_PLOT.finditer(text):
        _, i = _DECODER.raw_decode(text, match.end())
        data, i = _DECODER.raw_decode(text, _next_argument(text, i))
        layout, _ = _DECODER.raw_decode(text, _next_argument(text, i))
        figures.append({"data": _decode(data), "layout": _decode(layout)})
    return figures


def one_figure(path: Path) -> dict:
    figures = plotly_figures(path)
    assert len(figures) == 1, f"{Path(path).name}: {len(figures)} plots, expected one"
    return figures[0]
