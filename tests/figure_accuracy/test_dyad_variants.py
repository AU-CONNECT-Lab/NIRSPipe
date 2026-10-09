"""G01 under the flags the main run leaves at their defaults, and the cohort page with a gap."""

import html as html_lib
import re

import numpy as np
import pytest

from tests.figure_accuracy._dyad import entities, js_var, table_under
from tests.figure_accuracy._payload import one_figure, plotly_figures


def _maps(g, whole_run=False):
    paths = sorted((g.gdir("G01") / "figures").glob("*_desc-wtcmap_nirs.png"))
    return [p for p in paths if not (whole_run and entities(p).get("cond"))]


def _summary_row(html: str, name: str) -> str:
    found = re.search(rf"<th>{name}</th>\s*<td>(.*?)</td>", html, re.S)
    assert found, name
    return " ".join(re.sub(r"<[^>]+>", " ", found.group(1)).split())


# ---- --wtc-cond-transform ----

@pytest.mark.xfail(strict=True, reason="A: a condition transformed on its own keeps the cut's clock, not the aligned one")
def test_a_condition_transformed_on_its_own_is_drawn_on_the_aligned_clock(dyad_variants):
    g = dyad_variants["condtransform"]
    seen = 0
    for path in _maps(g):
        cond = entities(path).get("cond")
        if not cond:
            continue
        times = np.asarray(g.map_inputs(path)[0][2], float)
        t0, t1 = g.truth.block(cond)
        assert times.min() >= t0 - 1 and times.max() <= t1 + 1, (path.name, times.min(), times.max())
        seen += 1
    assert seen


# ---- the phase-scrambled null, crossed and not ----

def test_a_crossed_phase_null_reaches_every_whole_run_channel_map(dyad_variants):
    g = dyad_variants["phasenull"]
    for path in _maps(g, whole_run=True):
        assert g.map_inputs(path)[0][0].get("sig") is not None, path.name


@pytest.mark.xfail(strict=True, reason="J: an uncrossed null's levels are keyed by label, a crossed map by label pair")
def test_an_uncrossed_phase_null_reaches_the_same_channel_maps(dyad_variants):
    """The homologous null is the null for the homologous cells of a crossed real table."""
    g = dyad_variants["phasenull_homologous"]
    same = [p for p in _maps(g, whole_run=True) if "x" not in entities(p)["chan"]]
    assert same
    for path in same:
        assert g.map_inputs(path)[0][0].get("sig") is not None, path.name


@pytest.mark.xfail(strict=True, reason="I: the ROI maps drop the level and draw at --wtc-arrow-min, under a note saying 'as above'")
def test_the_roi_maps_say_what_their_arrows_cleared(dyad_variants):
    """The ROI maps carry no level of their own, so a page whose channel maps used the null
    cannot describe the ROI arrows as the channel ones."""
    html = dyad_variants["phasenull"].page("G01").read_text(encoding="utf-8")
    assert "null" in _summary_row(html, "Arrow threshold")
    roi_note = re.search(r'id="WTCRoi".*?<p class="step-desc">(.*?)</p>', html, re.S).group(1)
    assert "as above" not in roi_note, roi_note


# ---- --no-channel-cross ----

def test_an_uncrossed_run_keeps_the_isc_to_the_same_channel_pairs(dyad_variants):
    import pandas as pd
    g = dyad_variants["uncrossed"]
    for chroma in ("hbo", "hbr"):
        m = pd.read_csv(g.gdir("G01") / "nirs" / f"group-G01_task-main_chromo-{chroma}_stat-isc_relmat.tsv",
                        sep="\t", index_col=0)
        m = m.loc[m.columns, m.columns].to_numpy(float)
        assert np.isnan(m[~np.eye(len(m), dtype=bool)]).all(), chroma
        assert np.isfinite(np.diag(m)).sum() >= len(m) - 2, chroma
    rows = table_under(g.page("G01"), "Channel pairs")
    assert all(r[0] == r[1] for r in rows[2:] if len(r) > 2)


def test_an_uncrossed_run_prints_dashes_in_the_index_s_crossed_columns(dyad_variants):
    g = dyad_variants["uncrossed"]
    html = (g.gdir("G01") / "group-G01_desc-index_report.html").read_text(encoding="utf-8")
    head = [" ".join(re.sub(r"<[^>]+>", " ", c).split())
            for c in re.findall(r"<th[^>]*>(.*?)</th>", html.split("</thead>")[0], re.S)]
    body = html.split("<tbody", 1)[1].split("</tbody>", 1)[0]
    for chunk in body.split("<tr")[1:]:
        cells = [html_lib.unescape(" ".join(re.sub(r"<[^>]+>", " ", c).split()))
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", chunk, re.S)]
        for column in ("WTC HbO crossed", "ISC HbO crossed"):
            assert cells[head.index(column)] == "—", (column, cells)
        float(cells[head.index("WTC HbO")])


# ---- fnirs-qc hyper-raw --tstart / --normalize ----

@pytest.mark.xfail(strict=True, reason="S10: the record's alignment block keeps the offsets from before the --tstart cut")
def test_the_record_s_offsets_are_the_page_s_under_tstart(dyad_variants):
    g = dyad_variants["tstart"]
    html = g.page("G01", raw=True).read_text(encoding="utf-8")
    page = {r["subject_id"]: r["offset_s"] for r in js_var(html, "_ALIGN")}
    for m in g.truth.group("G01"):
        assert page[m.sid] == pytest.approx(m.offset + 20, abs=0.1), m.sid
    recorded = g.record("G01")["alignment"]["align_offset_s"]
    for sid, offset in page.items():
        assert recorded[sid] == pytest.approx(offset, abs=1e-3), sid


@pytest.mark.xfail(strict=True, reason="S11: --normalize z-scores the traces and the axis still reads umol/L")
def test_a_normalized_detail_trace_is_not_labelled_in_concentration(dyad_variants):
    g = dyad_variants["normalize"]
    path = next(iter(sorted((g.gdir("G01") / "figures").glob("*_desc-rawdetail_nirs.html"))))
    titles = [str((ax.get("title") or {}).get("text"))
              for fig in plotly_figures(path) for key, ax in fig["layout"].items()
              if key.startswith("yaxis") and isinstance(ax, dict)]
    assert not any("mol" in t for t in titles), titles


# ---- fnirs-qc cohort-hyper with a block one dyad lacks ----

@pytest.mark.xfail(strict=True, reason="cohort-hyper draws a block a dyad lacks as a 0% arc")
def test_a_block_a_dyad_lacks_draws_no_arc(dyad_variants):
    g = dyad_variants["cohortgap"]
    fig = one_figure(g.hyper / "figures" / "desc-groupsconditiondials_nirs.html")
    arcs = [t["customdata"][0] for t in fig["data"]
            if t.get("type") == "barpolar" and t.get("customdata")]
    assert any(a[1] == "cb" for a in arcs)           # the other dyad still draws its cb
    assert not [a for a in arcs if a[0].startswith("group-G01") and a[1] == "cb"], arcs
