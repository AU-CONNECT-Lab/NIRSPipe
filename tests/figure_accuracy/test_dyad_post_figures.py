"""The dyad post page (fnirs-hyper) and its condition pages: each figure against its table and the planted truth."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from tests.figure_accuracy._dyad import (BAND, band_mean, entities, js_var, table_under,
                                         wtc_tables)
from tests.figure_accuracy._payload import one_figure

GROUPS = ("G01", "G02")
SCOPES = ("", "ca", "cb")             # the whole run, then each block


def _pairings(groups, group):
    return list(combinations([m.sid for m in groups.truth.group(group)], 2))


def _slug(groups, group, pair) -> str:
    """The pairing's file entity, empty while a group holds one pairing."""
    if len(groups.truth.group(group)) == 2:
        return ""
    return f"pair-{pair[0].replace('-', '')}x{pair[1].replace('-', '')}_"


def _page_pair(groups, group, pair):
    return pair if len(groups.truth.group(group)) > 2 else None


def _cond(scope):
    return f"cond-{scope}_" if scope else ""


def _cells(rows, a, b, chroma, scope, label2=True):
    sel = rows[(rows.sub1 == a) & (rows.sub2 == b) & (rows.condition.fillna("") == scope)]
    if "chromophore" in sel:
        sel = sel[sel.chromophore == chroma]
    return {(r.label, r.label2 if label2 else r.label): r.coherence for r in sel.itertuples()}


def _planted(groups, a, b, chroma, scope):
    """The cells a pairing was coupled on in one scope, as (sub1 label, sub2 label)."""
    return {(c.a_pair, c.b_pair) for c in groups.truth.couplings_between(a, b, chroma)
            if c.block in (None, scope or None) and (c.block is None or scope == c.block)}


def _rejected(groups, sid):
    return groups.truth.member(sid).rejected


# ---- the channel coherence maps (PNG): the builder's inputs ----

def _maps(groups, group):
    return sorted(groups.figure(group, "x").parent.glob("*_desc-wtcmap_nirs.png"))


def _labels(site: str) -> tuple[str, str]:
    a, _, b = site.partition(" × ")
    return a, b or a


@pytest.mark.parametrize("group", GROUPS)
def test_every_channel_map_is_the_band_mean_its_table_records(groups, group):
    table = wtc_tables(groups, group)
    pairs = _pairings(groups, group)
    paths = _maps(groups, group)
    assert paths
    for path in paths:
        args, _ = groups.map_inputs(path)
        data, freqs, times, pair_label, _, _, site = args
        ent = entities(path)
        a, b = next(p for p in pairs if not ent.get("pair")
                    or ent["pair"] == f"{p[0].replace('-', '')}x{p[1].replace('-', '')}")
        assert pair_label == f"{a} × {b}", path.name
        l1, l2 = _labels(site)
        assert ent["chan"] == (l1.replace("_", "") if l1 == l2
                               else f"{l1.replace('_', '')}x{l2.replace('_', '')}"), path.name
        recorded = _cells(table, a, b, ent["chromo"], ent.get("cond", ""))[(l1, l2)]
        drawn = band_mean(data["wtc"], data["coi"], np.asarray(freqs))
        # the map arrives in float32, the table was averaged in float64
        assert drawn == pytest.approx(recorded, rel=1e-6), path.name


@pytest.mark.parametrize("group", GROUPS)
def test_a_condition_map_covers_its_block_on_the_aligned_clock(groups, group):
    for path in _maps(groups, group):
        args, _ = groups.map_inputs(path)
        times = np.asarray(args[2], float)
        cond = entities(path).get("cond")
        t0, t1 = groups.truth.block(cond) if cond else (0.0, 400.0)
        assert times.min() >= t0 - 1 and times.max() <= t1 + 1, path.name
        assert times.max() - times.min() > 0.9 * (t1 - t0), path.name


@pytest.mark.parametrize("group", GROUPS)
def test_every_map_marks_the_blocks_where_they_are_on_the_aligned_clock(groups, group):
    for path in _maps(groups, group)[:4]:
        markers = {m["description"]: (m["onset"], m["onset"] + m["duration"])
                   for m in groups.map_inputs(path)[0][4]}
        for name in ("ca", "cb"):
            assert markers[name] == pytest.approx(groups.truth.block(name), abs=0.1), path.name


@pytest.mark.parametrize("group", GROUPS)
def test_each_coupling_s_map_shows_its_lag_as_the_phase_at_every_scale(groups, group):
    by_name = {p.name: p for p in _maps(groups, group)}
    seen = 0
    for a, b in _pairings(groups, group):
        for c in groups.truth.couplings_between(a, b):
            if c.block is not None:
                continue
            chan = f"{c.a_pair.replace('_', '')}x{c.b_pair.replace('_', '')}"
            name = (f"group-{group}_task-main_{_slug(groups, group, (a, b))}"
                    f"chromo-{c.chroma}_chan-{chan}_desc-wtcmap_nirs.png")
            data, freqs = groups.map_inputs(by_name.get(name) or by_name[name])[0][:2]
            freqs = np.asarray(freqs)
            rows = (freqs > BAND[0]) & (freqs < BAND[1])
            strong = np.asarray(data["wtc"])[rows] > 0.7
            want = np.array([c.phase_deg(f) for f in freqs[rows]])[:, None]
            err = (np.degrees(np.asarray(data["phase"])[rows]) - want + 180) % 360 - 180
            assert strong.mean() > 0.3, name
            assert np.median(np.abs(err[strong])) < 20, name
            seen += 1
    assert seen


# ---- the ROI coherence maps (Plotly): one file per region pairing, condition views inside ----

def _whole_run_maps(groups, group, pair, chroma):
    """Every whole-run channel map of one pairing and chromophore, keyed by its label pair."""
    out = {}
    prefix = f"group-{group}_task-main_{_slug(groups, group, pair)}chromo-{chroma}_chan-"
    for path in _maps(groups, group):
        if path.name.startswith(prefix) and "_cond-" not in path.name:
            args, _ = groups.map_inputs(path)
            out[_labels(args[6])] = args
    return out


@pytest.mark.parametrize("group", GROUPS)
def test_each_roi_map_is_the_mean_of_its_channel_pairs_maps(groups, group):
    from tests._dyad_fingerprint import ROI_MAP
    for a, b in _pairings(groups, group):
        for chroma in ("hbo", "hbr"):
            channel = _whole_run_maps(groups, group, (a, b), chroma)
            for r1, r2 in ((r1, r2) for r1 in ROI_MAP for r2 in ROI_MAP):
                label = r1 if r1 == r2 else f"{r1}x{r2}"
                path = groups.figure(group, f"{_slug(groups, group, (a, b))}chromo-{chroma}_"
                                            f"label-{label}_agg-roi_desc-wtcmap_nirs.html")
                members = [channel[(l1, l2)] for l1 in ROI_MAP[r1] for l2 in ROI_MAP[r2]
                           if (l1, l2) in channel]
                if not members:
                    assert not path.exists(), path.name
                    continue
                heat = one_figure(path)["data"][0]
                want = np.nanmean([np.asarray(m[0]["wtc"], float) for m in members], axis=0)
                np.testing.assert_allclose(np.asarray(heat["z"], float), want, atol=6e-4,
                                           err_msg=path.name)
                np.testing.assert_allclose(np.asarray(heat["x"], float),
                                           np.asarray(members[0][2], float), err_msg=path.name)


@pytest.mark.parametrize("group", GROUPS)
def test_each_roi_map_opens_every_condition_on_its_block(groups, group):
    for a, b in _pairings(groups, group):
        path = groups.figure(group, f"{_slug(groups, group, (a, b))}chromo-hbo_label-L_agg-roi_"
                                    "desc-wtcmap_nirs.html")
        views = js_var(path.read_text(encoding="utf-8"), "__COND_VIEWS__")
        assert set(views) == {"ca", "cb"}
        for name, view in views.items():
            assert view["x"] == pytest.approx(list(groups.truth.block(name)), abs=0.1), name


# ---- the channel and ROI matrices ----

def _matrix(path):
    fig = one_figure(path)
    heat = [t for t in fig["data"] if t["type"] == "heatmap"]
    out = {}
    for panel, chroma in zip(heat, ("hbo", "hbr")):
        z = np.asarray(panel["z"], float)
        out[chroma] = {(y, x): z[i, j] for i, y in enumerate(panel["y"])
                       for j, x in enumerate(panel["x"])}
    return fig, out


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("scope", SCOPES)
def test_the_channel_matrix_is_the_band_mean_table(groups, group, scope):
    table = wtc_tables(groups, group)
    for a, b in _pairings(groups, group):
        path = groups.figure(group, f"{_slug(groups, group, (a, b))}agg-chan_{_cond(scope)}"
                                    "desc-wtcmatrix_relmat.html")
        fig, panels = _matrix(path)
        assert fig["layout"]["yaxis"]["title"]["text"].startswith(a), path.name
        assert fig["layout"]["xaxis"]["title"]["text"].startswith(b), path.name
        for chroma, cells in panels.items():
            recorded = _cells(table, a, b, chroma, scope)
            for (l1, l2), drawn in cells.items():
                if l1 == _rejected(groups, a) or l2 == _rejected(groups, b):
                    assert np.isnan(drawn), (path.name, chroma, l1, l2)
                else:
                    assert drawn == pytest.approx(recorded[(l1, l2)], abs=5e-4), \
                        (path.name, chroma, l1, l2)


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("scope", SCOPES)
def test_the_channel_matrix_puts_each_coupling_on_its_own_cell(groups, group, scope):
    for a, b in _pairings(groups, group):
        path = groups.figure(group, f"{_slug(groups, group, (a, b))}agg-chan_{_cond(scope)}"
                                    "desc-wtcmatrix_relmat.html")
        _, panels = _matrix(path)
        for chroma, cells in panels.items():
            hot = _planted(groups, a, b, chroma, scope)
            finite = {k: v for k, v in cells.items() if np.isfinite(v)}
            top = sorted(finite, key=finite.get, reverse=True)[:len(hot) or 1]
            if hot:
                assert set(top) == hot, (path.name, chroma, top)
            else:
                assert max(finite.values()) < 0.6, (path.name, chroma)


def _roi_cells(groups, group, a, b, chroma, scope):
    whole = groups.table(group, "seg-roidyad_agg-roi_stat-wtc_relmat.tsv").assign(condition="")
    cond = groups.table(group, "seg-roidyad_agg-roi_cond-all_stat-wtc_relmat.tsv")
    return _cells(pd.concat([whole, cond]), a, b, chroma, scope)


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("scope", SCOPES)
def test_the_roi_matrix_is_the_roi_table(groups, group, scope):
    for a, b in _pairings(groups, group):
        path = groups.figure(group, f"{_slug(groups, group, (a, b))}agg-roi_{_cond(scope)}"
                                    "desc-wtcmatrix_relmat.html")
        _, panels = _matrix(path)
        for chroma, cells in panels.items():
            recorded = _roi_cells(groups, group, a, b, chroma, scope)
            for key, drawn in cells.items():
                want = recorded.get(key, np.nan)
                if np.isnan(want):
                    assert np.isnan(drawn), (path.name, chroma, key)
                else:
                    assert drawn == pytest.approx(want, abs=5e-4), (path.name, chroma, key)


def test_the_roi_matrix_puts_each_whole_run_coupling_in_its_regions(groups):
    from tests._dyad_fingerprint import ROI_MAP
    region = {ch: roi for roi, chans in ROI_MAP.items() for ch in chans}
    _, panels = _matrix(groups.figure("G01", "agg-roi_desc-wtcmatrix_relmat.html"))
    for c in groups.truth.couplings_between("sub-01", "sub-02"):
        if c.block is None:
            cells = {k: v for k, v in panels[c.chroma].items() if np.isfinite(v)}
            assert max(cells, key=cells.get) == (region[c.a_pair], region[c.b_pair]), c


# ---- correlation ----

def _isc_matrix(groups, group, pair, chroma, scope):
    path = (groups.gdir(group) / "nirs" / f"group-{group}_task-main_"
            f"{_slug(groups, group, pair)}chromo-{chroma}_{_cond(scope)}stat-isc_relmat.tsv")
    return pd.read_csv(path, sep="\t", index_col=0)


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("scope", SCOPES)
def test_the_isc_panel_is_the_isc_table(groups, group, scope):
    for a, b in _pairings(groups, group):
        for chroma in ("hbo", "hbr"):
            path = groups.figure(group, f"{_slug(groups, group, (a, b))}chromo-{chroma}_"
                                        f"{_cond(scope)}desc-iscpanel_nirs.html")
            fig = one_figure(path)
            heat = next(t for t in fig["data"] if t["type"] == "heatmap")
            table = _isc_matrix(groups, group, (a, b), chroma, scope)
            z = np.asarray(heat["z"], float)
            for i, l1 in enumerate(heat["y"]):
                for j, l2 in enumerate(heat["x"]):
                    want = table.loc[l1, l2]
                    if np.isnan(want):
                        assert np.isnan(z[i, j]), (path.name, l1, l2)
                    else:
                        assert z[i, j] == pytest.approx(want, abs=5e-4), (path.name, l1, l2)


@pytest.mark.parametrize("group", GROUPS)
def test_the_isc_panel_blanks_each_member_s_own_rejection_on_its_own_axis(groups, group):
    for a, b in _pairings(groups, group):
        fig = one_figure(groups.figure(
            group, f"{_slug(groups, group, (a, b))}chromo-hbo_desc-iscpanel_nirs.html"))
        heat = next(t for t in fig["data"] if t["type"] == "heatmap")
        z = np.asarray(heat["z"], float)
        rows, cols = list(heat["y"]), list(heat["x"])
        blank_rows = {rows[i] for i in range(len(rows)) if np.isnan(z[i]).all()}
        blank_cols = {cols[j] for j in range(len(cols)) if np.isnan(z[:, j]).all()}
        assert blank_rows == {_rejected(groups, a)}, (a, b)
        assert blank_cols == {_rejected(groups, b)}, (a, b)


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("scope", SCOPES)
def test_the_roi_isc_matrix_is_the_roi_isc_table(groups, group, scope):
    for a, b in _pairings(groups, group):
        path = groups.figure(group, f"{_slug(groups, group, (a, b))}agg-roi_{_cond(scope)}"
                                    "desc-iscmatrix_relmat.html")
        _, panels = _matrix(path)
        for chroma, cells in panels.items():
            table = pd.read_csv(
                groups.gdir(group) / "nirs" / f"group-{group}_task-main_"
                f"{_slug(groups, group, (a, b))}chromo-{chroma}_seg-roidyad_agg-roi_"
                f"{_cond(scope)}stat-isc_relmat.tsv", sep="\t", index_col=0)
            for (l1, l2), drawn in cells.items():
                want = table.loc[l1, l2]
                if np.isnan(want):
                    assert np.isnan(drawn), (path.name, chroma, l1, l2)
                else:
                    assert drawn == pytest.approx(want, abs=5e-4), (path.name, chroma, l1, l2)


# ---- the tables on the page ----

@pytest.mark.parametrize("group", GROUPS)
def test_the_numbers_table_prints_each_pairing_s_own_band_means(groups, group):
    table = wtc_tables(groups, group)
    for a, b in _pairings(groups, group):
        page = groups.page(group, pair=_page_pair(groups, group, (a, b)))
        rows = table_under(page, "Channel pairs")
        body = [r for r in rows[2:] if len(r) > 2]
        assert len(body) == 16
        for r in body:
            l1, l2 = r[0], r[1]
            for k, scope in enumerate(SCOPES):
                for c, chroma in enumerate(("hbo", "hbr")):
                    want = _cells(table, a, b, chroma, scope)[(l1, l2)]
                    assert float(r[2 + 4 * k + c]) == pytest.approx(want, abs=5e-4), \
                        (page.name, l1, l2, scope, chroma)


@pytest.mark.xfail(strict=True, reason="the homologous table's ISC is the crossed ROI diagonal, "
                                        "every pairing inside the region")
@pytest.mark.parametrize("scope", SCOPES)
def test_the_homologous_table_s_isc_averages_only_same_channel_pairs(groups, scope):
    from tests._dyad_fingerprint import ROI_MAP
    rows = table_under(groups.page("G01", cond=scope or None), "ROI homologous pairs")
    head = rows[1]
    for r in (r for r in rows[2:] if len(r) > 2):
        assert r[0] == r[1], r
        for c, chroma in enumerate(("hbo", "hbr")):
            isc = pd.read_csv(groups.gdir("G01") / "nirs" / f"group-G01_task-main_chromo-{chroma}_"
                              f"{_cond(scope)}stat-isc_relmat.tsv", sep="\t", index_col=0)
            same = [isc.loc[ch, ch] for ch in ROI_MAP[r[0]] if np.isfinite(isc.loc[ch, ch])]
            want = np.tanh(np.mean(np.arctanh(same)))
            col = head.index(f"{'HbO' if chroma == 'hbo' else 'HbR'} ISC")
            assert float(r[2 + col]) == pytest.approx(want, abs=5e-4), (scope, r[0], chroma)


@pytest.mark.parametrize("group", GROUPS)
def test_the_post_page_carries_each_member_s_planted_offset(groups, group):
    for a, b in _pairings(groups, group):
        html = groups.page(group, pair=_page_pair(groups, group, (a, b))).read_text(
            encoding="utf-8")
        rows = {r["subject_id"]: r["offset_s"] for r in js_var(html, "_ALIGN")}
        for sid in (a, b):
            assert rows[sid] == pytest.approx(groups.truth.member(sid).offset, abs=0.1)


@pytest.mark.parametrize("group", GROUPS)
def test_each_member_s_channel_selector_marks_only_its_own_rejection(groups, group):
    for a, b in _pairings(groups, group):
        html = groups.page(group, pair=_page_pair(groups, group, (a, b))).read_text(
            encoding="utf-8")
        bad = js_var(html, "_BAD_PAIRS")
        per = bad if isinstance(bad, dict) else {a: bad, b: bad}
        for sid in (a, b):
            assert set(per[sid]) == {_rejected(groups, sid)}, (a, b, sid)


def _quality(page, heading):
    rows = table_under(page, heading)
    head = rows[0]
    return {r[0]: dict(zip(head[1:], r[1:])) for r in rows[1:]}


@pytest.mark.parametrize("scope", SCOPES)
def test_the_quality_table_prints_each_member_s_own_record(groups, scope):
    page = groups.page("G01", cond=scope or None)
    table = _quality(page, "Long channels")
    for m in groups.truth.group("G01"):
        rec = groups.member_record(m.sid)
        section = (rec["by_condition"][scope]["od_by_set"]["long"] if scope
                   else rec["raw_long"])
        printed = table[m.sid]
        sci = next(v for k, v in printed.items() if k.startswith("Mean SCI ("))
        assert float(sci) == pytest.approx(section["sci_win_mean"], abs=5e-4), (m.sid, scope)


@pytest.mark.parametrize("scope", ["ca", "cb"])
def test_a_condition_table_names_the_grid_its_windowed_values_were_cut_from(groups, scope):
    window = groups.member_record("sub-01")["windowed"]["qc_window_s"]
    head = list(_quality(groups.page("G01", cond=scope), "Long channels").values())[0]
    for name in ("SCI", "PSP", "CV", "SNR"):
        label = next(k for k in head if k.startswith(f"Mean {name} ("))
        assert f"({window:g} s)" in label, label


def test_the_condition_quality_table_shows_each_member_s_movement_in_its_own_block(groups):
    from tests._dyad_fingerprint import OWN_SPIKE, SHARED_SPIKE
    for scope in ("ca", "cb"):
        t0, t1 = groups.truth.block(scope)
        table = _quality(groups.page("G01", cond=scope), "Long channels")
        for m in groups.truth.group("G01"):
            moved = any(t0 <= t < t1 for t in (OWN_SPIKE[m.subject], SHARED_SPIKE))
            spikes = float(table[m.sid]["Spike % frames"].rstrip("%"))
            assert (spikes > 0) == moved, (scope, m.sid, spikes)


def test_a_pairing_page_lists_only_its_own_two_members(groups):
    for a, b in _pairings(groups, "G02"):
        page = groups.page("G02", pair=(a, b))
        assert set(_quality(page, "Long channels")) == {a, b}, (a, b)
        html = page.read_text(encoding="utf-8")
        assert {r["subject_id"] for r in js_var(html, "_ALIGN")} == {a, b}, (a, b)
