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


def _head(groups, group) -> list[str]:
    html = _index(groups, group).read_text(encoding="utf-8")
    head = html.split("<thead", 1)[1].split("</thead>", 1)[0]
    return [" ".join(re.sub(r"<[^>]+>", " ", c).split())
            for c in re.findall(r"<th[^>]*>(.*?)</th>", head, re.S)]


def _z_mean(values) -> float:
    values = np.asarray(values, float)
    return float(np.tanh(np.arctanh(values[np.isfinite(values)]).mean()))


def _expected(groups, group):
    """{(pairing, window): (page, {column: value})} off the tables, same-channel and crossed."""
    triad = len(groups.truth.group(group)) > 2
    table = wtc_tables(groups, group)
    out = {}
    for a, b in _pairings(groups, group):
        slug = f"_pair-{a.replace('-', '')}x{b.replace('-', '')}" if triad else ""
        for window in ("", "ca", "cb"):
            values = {}
            for c, name in (("hbo", "HbO"), ("hbr", "HbR")):
                rows = table[(table.sub1 == a) & (table.sub2 == b) & (table.chromophore == c)
                             & (table.condition == window)]
                values[f"WTC {name}"] = rows[rows.label == rows.label2].coherence.mean()
                values[f"WTC {name} crossed"] = rows.coherence.mean()
                m = pd.read_csv(groups.gdir(group) / "nirs" /
                                f"group-{group}_task-main{slug}_chromo-{c}"
                                f"{'_cond-' + window if window else ''}_stat-isc_relmat.tsv",
                                sep="	", index_col=0)
                m = m.loc[m.columns, m.columns].to_numpy(float)
                values[f"ISC {name}"] = _z_mean(np.diag(m))
                values[f"ISC {name} crossed"] = _z_mean(m.ravel())
            page = f"group-{group}_task-main{slug}{'_cond-' + window if window else ''}_report.html"
            out[((a, b), window)] = (page, values)
    return out


@pytest.mark.parametrize("group", GROUPS)
def test_each_index_row_is_its_own_pairing_and_window_and_links_its_page(groups, group):
    expected = _expected(groups, group)
    rows = _rows(groups, group)
    head = _head(groups, group)
    assert len(rows) == len(expected)
    triad = len(groups.truth.group(group)) > 2
    for href, cells in rows:
        window = "" if cells[0].startswith("whole run") else cells[0]
        pair = (tuple(cells[1].split(" × ")) if triad
                else tuple(m.sid for m in groups.truth.group(group)))
        page, values = expected[(pair, window)]
        assert href == page, cells
        assert (groups.gdir(group) / href).exists(), href
        for column, want in values.items():
            got = float(cells[head.index(column)])
            assert got == pytest.approx(want, abs=5e-4), (column, cells)


def test_the_crossed_columns_differ_from_the_same_channel_ones(groups):
    """The fixture is crossed, so a crossed column copying its same-channel twin is a defect."""
    head = _head(groups, "G01")
    for _, cells in _rows(groups, "G01"):
        for column in ("WTC HbO", "ISC HbO", "ISC HbR"):
            assert cells[head.index(column)] != cells[head.index(f"{column} crossed")], cells


@pytest.mark.parametrize("group", GROUPS)
def test_every_link_on_the_index_resolves(groups, group):
    html = _index(groups, group).read_text(encoding="utf-8")
    for href in re.findall(r'href="([^"#:]+)"', html):
        assert (groups.gdir(group) / href).exists(), href


def test_the_index_shows_the_block_coupling_in_its_block(groups):
    """K1 is homologous and lives in ca only, so ca's homologous HbO mean stands above cb's."""
    rows = {cells[0]: float(cells[3]) for _, cells in _rows(groups, "G01")}
    assert rows["ca"] > rows["cb"] + 0.05
