"""The `package-data` globs against what is actually in the package directory.

Nothing currently depends on this list. setuptools defaults `include-package-data` to true
under a pyproject build, and a build from this git working tree carries every tracked file
inside the package whether or not a glob names it: verified on 2026-09-12 by building an
sdist and a wheel with `qc/templates/*.css` removed from the list, and finding the three
sheets in both. An untracked file matching no glob is left out of the wheel, which is the
other half of the same check.

So the list is what holds when that implicit behaviour does not: a build from an exported
tree with no VCS, a different backend, `include-package-data` turned off. It was already
incomplete when this was written (the three sheets), which is the failure shape worth
pinning: a glob list nobody reads drifts from the directory it describes, and no build
says so.
"""

import glob
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "fnirs_pipe"


def _patterns() -> list[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return config["tool"]["setuptools"]["package-data"]["fnirs_pipe"]


def _declared() -> set[str]:
    """The globs expanded the way setuptools expands them: `*` does not cross a directory.

    ::

      ["qc/templates/*.css"]  ->  {"qc/templates/_tokens.css", ...}
    """
    return {Path(hit).as_posix()
            for pattern in _patterns()
            for hit in glob.glob(pattern, root_dir=PACKAGE)}


def test_every_non_python_file_in_the_package_is_declared():
    on_disk = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*")
               if p.is_file() and p.suffix != ".py" and "__pycache__" not in p.parts}
    missing = sorted(on_disk - _declared())
    assert not missing, f"named by no package-data glob: {missing}"


def test_no_pattern_has_stopped_matching_anything():
    # the other direction: a file moves and the glob left behind matches nothing, which the
    # test above cannot see because the file is covered by whichever glob it moved under
    dead = [p for p in _patterns() if not glob.glob(p, root_dir=PACKAGE)]
    assert not dead, f"package-data patterns matching no file: {dead}"
