"""docutils parser that pre-converts numpydoc sections to RST via napoleon.

autodoc2 renders docstrings through a docutils parser instead of going through
sphinx.ext.autodoc, so napoleon's ``autodoc-process-docstring`` hook never fires and
numpydoc sections ("Parameters" + dashes) reach docutils as section titles, which it
rejects and drops. Running napoleon's converter first turns them into ordinary RST
field lists, which the normal RST parser then renders as a parameter table.

Registered via ``autodoc2_docstring_parser_regexes``; docutils resolves the name to
this module and uses the ``Parser`` class below.
"""

from docutils.parsers.rst import Parser as _RstParser
from sphinx.ext.napoleon import Config as _NapoleonConfig
from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring

_CONFIG = _NapoleonConfig(
    napoleon_use_param=True,
    napoleon_use_rtype=True,
    napoleon_use_ivar=True,
    napoleon_preprocess_types=True,
)


class Parser(_RstParser):
    def parse(self, inputstring, document):
        text = inputstring if isinstance(inputstring, str) else "\n".join(inputstring)
        converted = str(GoogleDocstring(str(NumpyDocstring(text, _CONFIG)), _CONFIG))
        super().parse(converted, document)
