"""Sphinx configuration.

Builds the whole site: narrative pages plus an API reference generated from the
source. Replaces the Jupyter Book 2 setup that myst.yml used to drive.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

from nirspipe import __version__ as release  # noqa: E402  needs the path insert above

project = "NIRSPipe"
author = "Qiuyu"

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
    "sphinxarg.ext",
    "metric_docs",
    # footcite inside autodoc2 docstrings and eval-rst blocks resolves only with this
    "nested_footnotes",
]

# The usage page's argparse directives import nirspipe.cli (factories in _ext/cli_parsers.py),
# and the QC page's metric entries import nirspipe.qc (_ext/metric_docs.py), so a docs build
# needs the package installed with its runtime dependencies.

# NOT sphinx.ext.viewcode: it resolves [source] links by importing every module, the GUI and
# report writers included. Once the repo is public, sphinx.ext.linkcode gives the same links
# by URL, without importing anything.

# autodoc2 parses the source statically and never executes module-level code. Switching back
# to sphinx.ext.autodoc + napoleon is a conf.py-only change: the docstrings stay numpydoc
# either way.
autodoc2_packages = [
    {"path": "../nirspipe/pipeline", "module": "nirspipe.pipeline"},
    {"path": "../nirspipe/io", "module": "nirspipe.io"},
    {"path": "../nirspipe/qc", "module": "nirspipe.qc"},
]
autodoc2_render_plugin = "rst"  # keeps the existing `.. math::` blocks working
autodoc2_hidden_objects = ["private", "dunder", "inherited"]
autodoc2_output_dir = "api/_generated"
autodoc2_index_template = None

# autodoc2 hands docstrings straight to docutils, so napoleon's autodoc hook never fires
# and numpydoc sections would be read as section titles and dropped. _ext/numpydoc_rst_parser
# runs napoleon's converter first. Without this, every Parameters/Returns block is lost.
autodoc2_docstring_parser_regexes = [(r".*", "numpydoc_rst_parser")]

# Docstrings cite with :footcite: / .. footbibliography:: (see qc/metrics/gvtd.py).
bibtex_bibfiles = ["../nirspipe/qc/boilerplate/references.bib"]
bibtex_reference_style = "author_year"

root_doc = "index"
# no dash conversion: the generated flag help names options in plain text, and --mode must stay --mode
smartquotes_action = "qe"
exclude_patterns = [
    "_build",
    "README.md",
]

# ---- Markdown ----
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 4

# without this a ```mermaid fence is treated as source code with an unknown lexer and
# rendered as a grey block. Keeping the plain fence (rather than a {mermaid} directive)
# means the same file also renders as a diagram on GitHub.
myst_fence_as_directive = ["mermaid"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "mne": ("https://mne.tools/stable", None),
    "nilearn": ("https://nilearn.github.io/stable", None),
}

# ---- HTML ----
html_theme = "sphinx_book_theme"
html_title = "NIRSPipe"
html_theme_options = {
    "use_download_button": False,
    "repository_url": "https://github.com/AU-CONNECT-Lab/NIRSPipe",
    "use_repository_button": True,
    "show_navbar_depth": 2,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
