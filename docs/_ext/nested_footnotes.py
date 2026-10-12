"""Make footcite resolve inside autodoc2 docstrings and MyST eval-rst blocks."""

import copy

from autodoc2.sphinx.docstring import DocstringRenderer
from docutils import nodes
from docutils.transforms import Transform

_BIBTEX_STATE = "bibtex_foot"  # prefix of sphinxcontrib-bibtex's per-page footnote state
_BIBTEX_HOLDER = "bibtex_foot_bibliography"


class AdoptNestedFootnotes(Transform):
    # autodoc2 docstrings and MyST eval-rst blocks are parsed into their own document, so the
    # footnotes and references footcite creates there are never registered with the page
    default_priority = 610  # before docutils numbers and resolves footnotes at 620

    def apply(self):
        document = self.document
        known = {id(f) for f in document.autofootnotes}
        for footnote in document.findall(nodes.footnote):
            if id(footnote) in known or not footnote.get("auto"):
                continue
            self._drop_foreign_ids(footnote)
            document.note_autofootnote(footnote)
            if footnote["names"] and footnote["names"][0] not in document.nameids:
                document.note_explicit_target(footnote, footnote)
        registered = {id(r) for refs in document.footnote_refs.values() for r in refs}
        for ref in document.findall(nodes.footnote_reference):
            if id(ref) not in registered and ref.get("refname"):
                self._drop_foreign_ids(ref)
                document.note_footnote_ref(ref)

    def _drop_foreign_ids(self, node):
        # ids the nested document handed out ("id1") collide with the page's own
        node["ids"] = [i for i in node["ids"] if self.document.ids.get(i, node) is node]


class SummaryKeepsFootnotes(DocstringRenderer):
    # a summary-table cell parses the whole docstring and keeps one paragraph; the footnotes
    # footcite made in the discarded rest would otherwise be gone before the docstring renders
    def run(self):
        if "summary" not in self.options:
            return super().run()
        data = self.env.temp_data
        saved = {k: copy.copy(v) for k, v in data.items()
                 if k.startswith(_BIBTEX_STATE) and k != _BIBTEX_HOLDER}
        holder = data.get(_BIBTEX_HOLDER)
        children = list(holder.children) if holder is not None else None
        try:
            return super().run()
        finally:
            for key in [k for k in data if k.startswith(_BIBTEX_STATE)]:
                del data[key]
            data.update(saved)
            if holder is not None:
                holder.children = children
                data[_BIBTEX_HOLDER] = holder


def setup(app):
    app.add_transform(AdoptNestedFootnotes)
    app.add_directive("autodoc2-docstring", SummaryKeepsFootnotes, override=True)
    return {"parallel_read_safe": True}
