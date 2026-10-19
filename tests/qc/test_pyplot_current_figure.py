"""Figures are drawn on threads under --n-jobs, so no package code may act on pyplot's shared current figure."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "nirspipe"

# these name their figure, or make a new one; everything else on pyplot reaches for gcf()
ALLOWED = {"subplots", "figure", "close"}


def _pyplot_aliases(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.asname or a.name for a in node.names if a.name == "matplotlib.pyplot"}
        elif isinstance(node, ast.ImportFrom) and node.module == "matplotlib":
            names |= {a.asname or a.name for a in node.names if a.name == "pyplot"}
    return names


def test_no_current_figure_calls():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = _pyplot_aliases(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in aliases and node.attr not in ALLOWED):
                offenders.append(f"{path.relative_to(PACKAGE.parent)}:{node.lineno} "
                                 f"{node.value.id}.{node.attr}")
    assert not offenders, "\n".join(offenders)
