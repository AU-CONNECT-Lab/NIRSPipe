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


def test_an_uncrossed_phase_null_reaches_the_same_channel_maps(dyad_variants):
    """The homologous null is the null for the homologous cells of a crossed real table."""
    g = dyad_variants["phasenull_homologous"]
    same = [p for p in _maps(g, whole_run=True) if "x" not in entities(p)["chan"]]
    assert same
    for path in same:
        assert g.map_inputs(path)[0][0].get("sig") is not None, path.name


def _roi_maps(g):
    return [p for p in sorted((g.gdir("G01") / "figures").glob("*_agg-roi_desc-wtcmap_nirs.html"))
            if not entities(p).get("cond")]


def _levels(g, chroma, roi=False):
    from fnirs_pipe.pipeline.hyper.wtc_store import load_null_levels
    seg = "_seg-roidyad_agg-roi" if roi else ""
    path = (g.gdir("G01") / "nirs"
            / f"group-G01_task-main_chromo-{chroma}{seg}_null-phase_stat-wtc_desc-level_relmat.npz")
    return load_null_levels(path)[("sub-01", "sub-02")] if path.exists() else None


def _roi_key(path):
    label = entities(path)["label"]
    return tuple(label.split("x")) if "x" in label else (label, label)


def _drawn_level(path):
    """The per-frequency level a live map was drawn against, off its own contour of wtc/level."""
    fig = one_figure(path)
    heat = np.asarray(fig["data"][0]["z"], float)
    ratio = [t for t in fig["data"] if t["type"] == "contour"]
    if not ratio:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = heat / np.asarray(ratio[0]["z"], float)
    implied[heat < 0.2] = np.nan
    return np.nanmedian(implied, axis=1)


def test_the_roi_maps_carry_their_own_level_when_a_null_ran(dyad_variants):
    g = dyad_variants["phasenull"]
    maps = _roi_maps(g)
    assert maps
    for path in maps:
        chroma = entities(path)["chromo"]
        want = _levels(g, chroma, roi=True)[_roi_key(path)]
        got = _drawn_level(path)
        assert got is not None, path.name
        ok = np.isfinite(got) & np.isfinite(want)
        assert ok.sum() > 10, path.name
        np.testing.assert_allclose(got[ok], want[ok], rtol=0.02, err_msg=path.name)
        assert "phase-scrambled null" in path.read_text(encoding="utf-8"), path.name
    html = g.page("G01").read_text(encoding="utf-8")
    note = re.search(r'id="WTCRoi".*?<p class="step-desc">(.*?)</p>', html, re.S).group(1)
    assert "its own null" in note and "as above, and arrows where the averaged map" in note


def test_an_roi_level_is_the_averaged_maps_own_not_its_channels(dyad_variants):
    """Averaging surrogate maps narrows their spread, so an ROI's own level sits below the
    mean of its channel pairings' levels; the channels' average would not."""
    from tests._dyad_fingerprint import ROI_MAP
    g = dyad_variants["phasenull"]
    seen = 0
    for chroma in ("hbo", "hbr"):
        roi, chan = _levels(g, chroma, roi=True), _levels(g, chroma)
        for (r1, r2), level in roi.items():
            members = [chan[(a, b)] for a in ROI_MAP[r1] for b in ROI_MAP[r2] if (a, b) in chan]
            if len(members) < 2:
                continue
            mean = np.nanmean(members, axis=0)
            ok = np.isfinite(level) & np.isfinite(mean)
            assert np.nanmedian(mean[ok] - level[ok]) > 0.02, (chroma, r1, r2)
            seen += 1
    assert seen


def test_an_uncrossed_null_gives_a_crossed_run_s_roi_maps_no_level(dyad_variants):
    """A crossed (R, R) map averages every pairing inside R, an uncrossed null's R only the
    same-channel ones: no level is drawn for the maps, and the note says what they cleared."""
    g = dyad_variants["phasenull_homologous"]
    assert _levels(g, "hbo", roi=True) is None
    for path in _roi_maps(g):
        assert _drawn_level(path) is None, path.name
    html = g.page("G01").read_text(encoding="utf-8")
    note = re.search(r'id="WTCRoi".*?<p class="step-desc">(.*?)</p>', html, re.S).group(1)
    assert "reaches 0.5, no null having been drawn" in note, note


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

def test_the_record_s_offsets_are_the_page_s_under_tstart(dyad_variants):
    g = dyad_variants["tstart"]
    html = g.page("G01", raw=True).read_text(encoding="utf-8")
    page = {r["subject_id"]: r["offset_s"] for r in js_var(html, "_ALIGN")}
    for m in g.truth.group("G01"):
        assert page[m.sid] == pytest.approx(m.offset + 20, abs=0.1), m.sid
    recorded = g.record("G01")["alignment"]["align_offset_s"]
    for sid, offset in page.items():
        assert recorded[sid] == pytest.approx(offset, abs=1e-3), sid


def test_a_normalized_detail_trace_is_not_labelled_in_concentration(dyad_variants):
    g = dyad_variants["normalize"]
    path = next(iter(sorted((g.gdir("G01") / "figures").glob("*_desc-rawdetail_nirs.html"))))
    titles = [str((ax.get("title") or {}).get("text"))
              for fig in plotly_figures(path) for key, ax in fig["layout"].items()
              if key.startswith("yaxis") and isinstance(ax, dict)]
    assert not any("mol" in t for t in titles), titles


# ---- fnirs-qc cohort-hyper with a block one dyad lacks ----

def test_a_block_a_dyad_lacks_draws_no_arc(dyad_variants):
    g = dyad_variants["cohortgap"]
    fig = one_figure(g.hyper / "figures" / "desc-groupsconditiondials_nirs.html")
    arcs = [t["customdata"][0] for t in fig["data"]
            if t.get("type") == "barpolar" and t.get("customdata")]
    assert any(a[1] == "cb" for a in arcs)           # the other dyad still draws its cb
    assert not [a for a in arcs if a[0].startswith("group-G01") and a[1] == "cb"], arcs
