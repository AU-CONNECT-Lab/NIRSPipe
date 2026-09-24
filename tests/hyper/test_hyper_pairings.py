"""A group of more than two members: one report per pairing, and no pairing's numbers on
another pairing's page.

Every inter-brain figure is of two brains, so three members are three pairings. The
band-mean table carries every pairing, so a page keying its matrices and its numbers table by
channel pair alone would collapse the three onto one cell, whichever row was read last
winning, under axes naming the first two members.

The decisive test is :func:`test_a_pairing_page_carries_its_own_pairing_s_coherence`. Two of
the three members share a rhythm and the third carries independent noise, so a page showing
the wrong pairing's numbers cannot come out looking right: it would report the noise as
coherence or the rhythm as noise.

A dyad has exactly one pairing and carries no pairing suffix, which
:func:`test_a_dyad_carries_no_pairing_suffix` holds to.
"""

import re

import mne
import numpy as np
import pytest

from fnirs_pipe.io.naming import report_name

from fnirs_pipe.qc.common.figure_io import pair_slug

SFREQ = 5.0
DURATION = 300.0
LABELS = ["S1_D1", "S2_D2"]
BAND = (0.03, 0.10)
SHARED_FREQ = 0.05          # inside BAND, so the band mean sees it


def _raw(seed: int, hbo_signal: np.ndarray) -> mne.io.Raw:
    """A haemoglobin Raw carrying `hbo_signal` on every HbO channel."""
    rng = np.random.default_rng(seed)
    names, types, data = [], [], []
    for label in LABELS:
        for chroma in ("hbo", "hbr"):
            names.append(f"{label} {chroma}")
            types.append("hbo" if chroma == "hbo" else "hbr")
            base = hbo_signal if chroma == "hbo" else rng.standard_normal(hbo_signal.size)
            data.append(base * 1e-6 + rng.standard_normal(hbo_signal.size) * 1e-8)
    info = mne.create_info(names, SFREQ, types)
    raw = mne.io.RawArray(np.asarray(data), info, verbose=False)
    # the montage the separation bands are read off; both members wear the same one
    montage = {}
    for i, label in enumerate(LABELS):
        src, det = label.split("_")
        montage[src] = np.array([0.0, i * 0.03, 0.0])
        montage[det] = np.array([0.03, i * 0.03, 0.0])
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=montage, coord_frame="head"),
                    on_missing="ignore", verbose=False)
    return raw


@pytest.fixture
def triad():
    """Three members where only the first two share a rhythm.

    So pairing 1x2 has to come out high and the two pairings involving the third low: a page
    that mixed them could not satisfy both.
    """
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    shared = np.sin(2 * np.pi * SHARED_FREQ * t)
    rng = np.random.default_rng(99)
    return {
        "sub-01": _raw(1, shared),
        "sub-02": _raw(2, shared),
        "sub-03": _raw(3, rng.standard_normal(t.size)),
    }


def _run(members, where, **kwargs):
    """Build the post report over `members` and return the directory it wrote into."""
    from fnirs_pipe.pipeline.hyper import GroupEntry
    from fnirs_pipe.qc.hyper.hyper_report import build_hyper_post_report

    ids = list(members)
    path = build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", sid, "tap") for sid in ids],
        aligned_raws=members, offsets={sid: 0.0 for sid in ids}, output_dir=where,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1],
        wtc_chroma=("hbo",), **kwargs,
    )
    return path.parent


def _coherence_cells(page, scope="Whole run"):
    """``{(site a, site b): value}`` off one scope's columns of a page's channel table.

    The table carries every scope side by side, a group of columns each, so a cell is found
    by counting: two label columns, then the scope's group, then the column inside it.
    """
    html = page.read_text(encoding="utf-8", errors="replace")
    i = html.find('class="qm-sub">Channel pairs (')
    if i < 0:
        return {}
    table = html[i:html.find("</table>", i)]
    head = table[:table.find("</thead>")]
    # a th spanning anything is a label or a scope banner; the plain ones are the columns.
    # The lookahead is what keeps `<thead>` from matching as a th with attributes
    heads = re.findall(r"<th(?![a-z])([^>]*)>(.*?)</th>", head, re.S)
    scopes = [re.sub(r"<[^>]+>.*", "", h, flags=re.S).strip()
              for attrs, h in heads if "colspan" in attrs]
    columns = [re.sub(r"<[^>]+>", "", c).strip()
               for attrs, c in heads if "span" not in attrs]
    width = len(columns) // len(scopes) if scopes else 0
    if scope not in scopes or "HbO WTC" not in columns[:width]:
        return {}
    column = 2 + scopes.index(scope) * width + columns.index("HbO WTC")

    out = {}
    for row in re.findall(r"<tr>(.*?)</tr>", table, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if len(cells) > column and re.fullmatch(r"S\d+_D\d+", cells[0] or ""):
            try:
                out[(cells[0], cells[1])] = float(cells[column])
            except ValueError:
                pass
    return out


# ---- the file naming ----

def test_the_slug_is_empty_while_a_group_holds_one_pairing():
    """A dyad's files would otherwise all be renamed to carry a name that distinguishes
    nothing from nothing."""
    assert pair_slug(("sub-01", "sub-02"), 1) == ""
    assert pair_slug(None, 3) == ""
    assert pair_slug(("sub-01", "sub-02"), 3) == "_sub01xsub02"


def test_a_triad_writes_one_page_per_pairing(triad, tmp_path):
    folder = _run(triad, tmp_path)
    pages = sorted(p.name for p in folder.glob("*_pair-*_report.html"))
    assert len(pages) == 3, pages
    # each names the two members it is of, and no two land on one file
    assert len(set(pages)) == 3
    for a, b in (("01", "02"), ("01", "03"), ("02", "03")):
        assert any(f"sub{a}xsub{b}" in name for name in pages), (a, b, pages)


def test_a_dyad_carries_no_pairing_suffix(triad, tmp_path):
    """The pairing dimension must cost a dyad nothing: one pairing, no suffix."""
    dyad = {sid: triad[sid] for sid in ("sub-01", "sub-02")}
    folder = _run(dyad, tmp_path)
    assert (folder / report_name("group-G1_task-tap")).exists()
    assert not list(folder.glob("*_pair-*"))


# ---- the numbers, which is what a collapse gets wrong ----

def test_a_pairing_page_carries_its_own_pairings_coherence(triad, tmp_path):
    """Only sub-01 and sub-02 share a rhythm, so their page has to read high and the other
    two low. A page drawing whichever pairing was written last could not do both."""
    folder = _run(triad, tmp_path, wtc_channel_cross=True)

    def mean_of(slug):
        cells = _coherence_cells(
            folder / report_name("group-G1_task-tap", pairing=slug))
        assert cells, slug
        return sum(cells.values()) / len(cells)

    shared = mean_of("sub01xsub02")
    assert shared > 0.6
    for slug in ("sub01xsub03", "sub02xsub03"):
        assert mean_of(slug) < shared


def test_no_two_pairings_print_the_same_numbers(triad, tmp_path):
    """A collapse shows up as two pages agreeing cell for cell."""
    folder = _run(triad, tmp_path, wtc_channel_cross=True)
    seen = {slug: _coherence_cells(
                folder / report_name("group-G1_task-tap", pairing=slug))
            for slug in ("sub01xsub02", "sub01xsub03", "sub02xsub03")}
    slugs = list(seen)
    for i, a in enumerate(slugs):
        for b in slugs[i + 1:]:
            assert seen[a] != seen[b], (a, b)


def test_an_uncrossed_run_still_prints_the_coherence_it_measured(triad, tmp_path):
    """Without crossing there is no `label2` column, and the table reads those rows as the
    diagonal they are rather than dropping the whole metric."""
    dyad = {sid: triad[sid] for sid in ("sub-01", "sub-02")}
    folder = _run(dyad, tmp_path)
    cells = _coherence_cells(folder / report_name("group-G1_task-tap"))
    assert cells, "an uncrossed run printed no coherence at all"
    assert all(a == b for a, b in cells), "an uncrossed run has nothing off the diagonal"
    assert {a for a, _ in cells} == set(LABELS)
