"""The scalp-share notes on the channel maps sit over the column they describe."""
import numpy as np

from nirspipe.qc.figures.common.topomap import _assemble, _glyph_points

CONDS = ["one", "two", "three", "four", "five"]
OPT = {"S1": (0.0, 0.0), "D1": (1.0, 0.0), "S2": (0.0, 1.0), "D2": (1.0, 1.0)}


def _figure():
    scoped = {"long": [("S1", "D1")], "short": [("S2", "D2")]}
    geometry = {scope: (_glyph_points(pairs, OPT, scope == "short"), pairs)
                for scope, pairs in scoped.items()}

    def row_values(chromo, scope, cond, t):
        (_, _, _, per_pair), _ = geometry[scope]
        return np.array([1.0] * sum(per_pair))

    rows = [("hbo", "long"), ("hbo", "short")]
    verdicts = {("hbo", c): 1.6 for c in CONDS}
    return _assemble(rows, CONDS, np.array([0.0]), 0, geometry, {}, OPT,
                     row_values, {"hbo": 1.0}, ["hbo"], verdicts, CONDS)


def test_scalp_note_is_centred_on_its_own_column():
    fig = _figure()
    titles = {a.text: a.x for a in fig.layout.annotations if a.text in CONDS}
    notes = [a for a in fig.layout.annotations if "scalp" in a.text]

    assert len(notes) == len(CONDS)
    # an auto anchor resolves to left or right off the middle third, which walks the note
    # out of its column
    assert all(a.xanchor == "center" for a in notes)
    assert sorted(a.x for a in notes) == sorted(titles.values())
