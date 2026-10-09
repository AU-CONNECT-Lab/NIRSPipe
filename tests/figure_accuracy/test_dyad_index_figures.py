"""The dyad index (fnirs-hyper-index): one row per pairing and window, each from its own tables."""

import re
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from tests.figure_accuracy._dyad import wtc_tables

GROUPS = ("G01", "G02")


def _index(groups, group):
    return groups.gdir(group) / f"group-{group}_desc-index_report.html"


def _rows(groups, group):
    """(first link, cell texts) per row of the Windows table."""
    html = _index(groups, group).read_text(encoding="utf-8")
    body = html.split("<tbody", 1)[1].split("</tbody>", 1)[0]
    out = []
    for chunk in body.split("<tr")[1:]:
        href = re.search(r'href="([^"]+)"', chunk)
        cells = [" ".join(re.sub(r"<[^>]+>", " ", c).split())
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", chunk, re.S)]
        out.append((href.group(1) if href else None, cells))
    return out


def _pairings(groups, group):
    return list(combinations([m.sid for m in groups.truth.group(group)], 2))


def _expected(groups, group):
    """{(pairing, window): (page, hbo wtc, hbr wtc, hbo isc, hbr isc)} off the tables."""
    triad = len(groups.truth.group(group)) > 2
    table = wtc_tables(groups, group)
    homologous = table[table.label == table.label2]
    out = {}
    for a, b in _pairings(groups, group):
        slug = f"_pair-{a.replace('-', '')}x{b.replace('-', '')}" if triad else ""
        for window in ("", "ca", "cb"):
            wtc = {c: homologous[(homologous.sub1 == a) & (homologous.sub2 == b)
                                 & (homologous.chromophore == c)
                                 & (homologous.condition == window)].coherence.mean()
                   for c in ("hbo", "hbr")}
            isc = {}
            for c in ("hbo", "hbr"):
                m = pd.read_csv(groups.gdir(group) / "nirs" /
                                f"group-{group}_task-main{slug}_chromo-{c}"
                                f"{'_cond-' + window if window else ''}_stat-isc_relmat.tsv",
                                sep="\t", index_col=0)
                isc[c] = np.nanmean(np.diag(m.loc[m.columns, m.columns].to_numpy(float)))
            page = f"group-{group}_task-main{slug}{'_cond-' + window if window else ''}_report.html"
            out[((a, b), window)] = (page, wtc["hbo"], wtc["hbr"], isc["hbo"], isc["hbr"])
    return out


@pytest.mark.parametrize("group", GROUPS)
def test_each_index_row_is_its_own_pairing_and_window_and_links_its_page(groups, group):
    expected = _expected(groups, group)
    rows = _rows(groups, group)
    assert len(rows) == len(expected)
    triad = len(groups.truth.group(group)) > 2
    for href, cells in rows:
        window = "" if cells[0].startswith("whole run") else cells[0]
        pair = (tuple(cells[1].split(" × ")) if triad
                else tuple(m.sid for m in groups.truth.group(group)))
        page, hbo, hbr, isc_hbo, isc_hbr = expected[(pair, window)]
        assert href == page, cells
        assert (groups.gdir(group) / href).exists(), href
        k = 3 if triad else 2                   # the WTC HbO column, past Window, Pair, Task, Span
        values = [float(v.rstrip("%")) for v in cells[k + 1: k + 7]]
        assert values[0] == pytest.approx(hbo, abs=5e-4), cells
        assert values[1] == pytest.approx(hbr, abs=5e-4), cells
        assert values[4] == pytest.approx(isc_hbo, abs=5e-4), cells
        assert values[5] == pytest.approx(isc_hbr, abs=5e-4), cells


@pytest.mark.parametrize("group", GROUPS)
def test_every_link_on_the_index_resolves(groups, group):
    html = _index(groups, group).read_text(encoding="utf-8")
    for href in re.findall(r'href="([^"#:]+)"', html):
        assert (groups.gdir(group) / href).exists(), href


def test_the_index_shows_the_block_coupling_in_its_block(groups):
    """K1 is homologous and lives in ca only, so ca's homologous HbO mean stands above cb's."""
    rows = {cells[0]: float(cells[3]) for _, cells in _rows(groups, "G01")}
    assert rows["ca"] > rows["cb"] + 0.05
