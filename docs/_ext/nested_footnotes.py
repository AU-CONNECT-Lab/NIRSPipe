"""Register footnotes a nested parse left in a document of its own, so footcite resolves on the page."""

from docutils import nodes
from docutils.transforms import Transform


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


def setup(app):
    app.add_transform(AdoptNestedFootnotes)
    return {"parallel_read_safe": True}
