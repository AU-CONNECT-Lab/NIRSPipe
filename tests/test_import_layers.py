"""The three blocks import one way: data preparation and the App never reach each other or the front end."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "nirspipe"

# package inits that hold shared modules; matched exactly, not as prefixes
SHARED_PACKAGES = ("nirspipe", "nirspipe.cli", "nirspipe.pipeline", "nirspipe.pipeline.hyper")
SHARED = ("nirspipe.exceptions", "nirspipe.io", "nirspipe.utils", "nirspipe.cli._shared",
          "nirspipe.pipeline.hyper.alignment")
DATA_PREPARATION = ("nirspipe.pipeline.crop", "nirspipe.pipeline.edit_markers",
                    "nirspipe.pipeline.keep_spans", "nirspipe.cli.prep", "nirspipe.cli.snirf2bids")
FRONT_END = ("nirspipe.interface", "nirspipe.qc.rating", "nirspipe.cli.gui", "nirspipe.cli.rate")
ALLOWED = {
    "shared": {"shared"},
    "data preparation": {"shared", "data preparation"},
    "app": {"shared", "app"},
    "front end": {"shared", "data preparation", "app", "front end"},
}


def _under(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _block(module: str) -> str:
    if module in SHARED_PACKAGES or _under(module, SHARED):
        return "shared"
    if _under(module, DATA_PREPARATION):
        return "data preparation"
    if _under(module, FRONT_END):
        return "front end"
    return "app"


def _modules() -> dict[str, Path]:
    out = {}
    for path in PACKAGE.rglob("*.py"):
        parts = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        out[".".join(parts)] = path
    return out


def _imports(module: str, path: Path, modules: dict[str, Path]) -> list[tuple[int, str]]:
    """(line, nirspipe module) for every import in the file, function-level ones included."""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package
                for _ in range(node.level - 1):
                    anchor = anchor.rpartition(".")[0]
                base = f"{anchor}.{base}" if base else anchor
            names = [f"{base}.{alias.name}" if f"{base}.{alias.name}" in modules else base
                     for alias in node.names]
        else:
            continue
        for name in names:
            while name and name not in modules:
                name = name.rpartition(".")[0]
            if name and name != module:
                found.append((node.lineno, name))
    return found


def test_no_import_crosses_a_block_the_wrong_way():
    modules = _modules()
    wrong = [f"{module}:{line} ({_block(module)}) imports {target} ({_block(target)})"
             for module, path in sorted(modules.items())
             for line, target in _imports(module, path, modules)
             if _block(target) not in ALLOWED[_block(module)]]
    assert not wrong, "\n".join(wrong)


@pytest.mark.parametrize("name", SHARED_PACKAGES + SHARED + DATA_PREPARATION + FRONT_END)
def test_every_module_a_block_names_still_exists(name):
    assert name in _modules()
