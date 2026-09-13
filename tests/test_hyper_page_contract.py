"""What the dyad page promises its own JavaScript, on a built page rather than in the source.

The coherence panels are a pair of selectors over a nested table of figure URLs embedded in
the HTML (`_PER_CH[chromophore][site1][site2]`, and `_PER_ROI` for regions). Everything that
can go wrong with that is silent in a way no other test here catches: the page renders, the
selectors populate, and a viewer sees *a* coherence map. It is the wrong one, or the same one
twice, or a broken image on a pairing nobody clicked during review.

These started as a scratchpad smoke script and are here because it caught three real defects
that the report's other tests did not:

- both chromophores pointing at one dict, so HbR showed HbO's figures
- a condition page linking the whole-run figures, the window slug never reaching the filename
- a rejected channel blanking its *row* as well as its column, losing every pairing the
  surviving member could still have been read against

Each of the three has a test below saying so. The build is module-scoped: one report serves
every assertion, since the failures are all in what it wrote rather than in how it was asked.
"""

import json
from pathlib import Path

import mne
import numpy as np
import pytest

from fnirs_pipe.pipeline.hyperscanning import GroupEntry

SFREQ, DURATION = 5.0, 400.0
LABELS = ["S1_D1", "S2_D2", "S3_D3"]
CONDITIONS = ("rest", "talk")
# rejected in sub-02 only, so the two members disagree and the blanking rule has a side
REJECTED = "S2_D2"


def _raw(seed: int, shared: np.ndarray, bads: "tuple[str, ...]" = ()) -> mne.io.Raw:
    """A three-pair montage carrying `shared` in HbO and independent noise in HbR."""
    rng = np.random.default_rng(seed)
    names = [f"{label} {c}" for label in LABELS for c in ("hbo", "hbr")]
    types = [c for _ in LABELS for c in ("hbo", "hbr")]
    info = mne.create_info(names, SFREQ, types)
    for i, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        loc[3:6] = [i * 0.05, 0.0, 0.0]                # source
        loc[6:9] = [i * 0.05 + 0.03, 0.0, 0.0]         # detector, 30 mm away
        loc[:3] = (loc[3:6] + loc[6:9]) / 2
        ch["loc"] = loc
    data = np.empty((len(names), len(shared)))
    for i, ch_type in enumerate(types):
        noise = rng.standard_normal(len(shared))
        data[i] = 1e-6 * (shared + 0.2 * noise if ch_type == "hbo" else noise)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.info["bads"] = list(bads)
    raw.set_annotations(mne.Annotations([20.0, 210.0], [180.0, 180.0], list(CONDITIONS)))
    return raw


@pytest.fixture(scope="module")
def dyad():
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    shared = np.sin(2 * np.pi * 0.05 * t)
    bads = tuple(f"{REJECTED} {c}" for c in ("hbo", "hbr"))
    pair = {"sub-01": _raw(1, shared), "sub-02": _raw(2, shared, bads=bads)}
    # a nonzero first_time on both members: the condition boundaries used to be drawn at the
    # raw onset, which is late by exactly this much
    return {sid: raw.copy().crop(tmin=20.0) for sid, raw in pair.items()}


@pytest.fixture(scope="module")
def pages(dyad, tmp_path_factory) -> "list[Path]":
    """The run's page and one per condition, crossed and over both chromophores."""
    from fnirs_pipe.qc.hyper.hyper_report import build_hyper_post_report

    out = tmp_path_factory.mktemp("hyper_pages")
    path = build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", "sub-01", "tap"), GroupEntry("G1", "sub-02", "tap")],
        aligned_raws=dyad, offsets={"sub-01": 0.0, "sub-02": 0.0}, output_dir=out,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=0.03, wtc_band_fmax=0.10,
        wtc_chroma=("hbo", "hbr"), wtc_channel_cross=True, wtc_by_condition=True,
        roi_map={"L": ["S1_D1", "S2_D2"], "R": ["S3_D3"]}, wtc_roi_min_channels=1,
    )
    return [p for p in sorted(path.parent.glob("*.html")) if "index" not in p.name]


def _js(html: str, name: str):
    """One `var NAME = {...};` literal out of the page, as Python.

    ::

      "var _PER_CH = {\\"hbo\\": {...}};"  ->  {"hbo": {...}}
    """
    start = html.index(f"var {name}")
    line = html[start:html.index("\n", start)]
    return json.loads(line.split("=", 1)[1].rsplit(";", 1)[0].strip())


def _tables(page: Path) -> tuple:
    html = page.read_text(encoding="utf-8")
    return (_js(html, "_PER_CH"), _js(html, "_PER_ROI"),
            _js(html, "_CH_PAIRS"), _js(html, "_ROI_LABELS"))


def _window_of(page: Path) -> str:
    """The condition a page is for, "" for the run's own page."""
    if "desc-" not in page.name:
        return ""
    slug = page.name.split("desc-")[1].split("_")[0]
    return "" if slug == "hyperpost" else slug


def _urls(per_ch, per_roi, axis, rois) -> set:
    return ({u for c in per_ch for a in axis for b in axis if (u := per_ch[c][a][b]["wtc"])}
            | {u for c in per_roi for a in rois for b in rois
               if (u := per_roi[c][a][b]["wtc"])})


# ---- the shape the selectors index into ----

def test_a_page_is_written_for_the_run_and_for_each_condition(pages):
    assert {_window_of(p) for p in pages} == {"", *CONDITIONS}


def test_every_pairing_of_the_axis_is_present_at_both_levels(pages):
    """Both selectors range over the whole axis, so a missing key is an undefined lookup
    rather than an empty panel."""
    for page in pages:
        per_ch, per_roi, axis, rois = _tables(page)
        for chromophore in ("hbo", "hbr"):
            assert sorted(per_ch[chromophore]) == sorted(axis), page.name
            for site in axis:
                assert sorted(per_ch[chromophore][site]) == sorted(axis), (page.name, site)
            for roi in rois:
                assert sorted(per_roi[chromophore][roi]) == sorted(rois), (page.name, roi)


def test_the_two_chromophores_are_two_tables(pages):
    """The shared-mutable-default defect: one dict filled twice, so the switch moved the
    label and not the figure."""
    for page in pages:
        per_ch, _, _, _ = _tables(page)
        assert per_ch["hbo"] != per_ch["hbr"], page.name


# ---- the figures the URLs point at ----

def _file_of(url: str) -> str:
    """The path a URL names, without the fragment that selects a view inside it."""
    return url.split("#", 1)[0]


def _window_in(url: str) -> str:
    """Which window a URL is of, however this panel says so.

    A channel map is one file per window and carries the slug in its name; an ROI map is one
    file holding every window and carries it as the fragment. Both are the same claim.
    """
    if "#" in url:
        return url.split("#", 1)[1]
    tail = Path(url).stem.split("_")[-1]
    return tail if tail in CONDITIONS else ""


def test_every_url_on_the_page_exists_on_disk(pages):
    for page in pages:
        for url in _urls(*_tables(page)):
            assert (page.parent / _file_of(url)).exists(), (page.name, url)


def test_a_condition_page_links_its_own_window_and_no_other(pages):
    """The defect: the window slug never reached the filename, so all six pages linked the
    whole-run figures and every condition looked identical to the run."""
    for page in pages:
        expected = _window_of(page)
        for url in _urls(*_tables(page)):
            assert _window_in(url) == expected, (
                page.name, url, f"expected {expected or '(run)'}")


def test_the_run_page_and_a_condition_page_do_not_share_a_figure(pages):
    run = next(p for p in pages if not _window_of(p))
    talk = next(p for p in pages if _window_of(p) == "talk")
    assert not _urls(*_tables(run)) & _urls(*_tables(talk))


# ---- a rejection is one member's, and costs one direction ----

def test_a_rejected_channel_blanks_the_column_it_is_read_as(pages):
    for page in pages:
        per_ch, _, axis, _ = _tables(page)
        assert all(per_ch["hbo"][site][REJECTED]["wtc"] is None for site in axis), page.name


def test_the_rejected_channel_keeps_the_row_the_other_member_can_still_be_read_against(pages):
    """Blanking both directions was the third defect. sub-01's S2_D2 is fine; only sub-02's
    was rejected, so that row is real data and dropping it loses three pairings."""
    for page in pages:
        per_ch, _, _, _ = _tables(page)
        assert per_ch["hbo"][REJECTED]["S1_D1"]["wtc"], page.name


def test_the_pairings_that_touch_no_rejection_are_untouched(pages):
    """The other half of every blanking assertion: something that blanked the whole table
    would pass the two above on its own."""
    for page in pages:
        per_ch, _, _, _ = _tables(page)
        assert per_ch["hbo"]["S1_D1"]["S3_D3"]["wtc"], page.name


# ---- the controls the tables exist for ----

def test_the_page_carries_a_second_selector_per_panel(pages):
    """A crossed run needs to name both ends of a pairing. One selector can only reach the
    homologous diagonal."""
    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert 'id="ch-select-post-2"' in html, page.name
        assert 'id="roi-select-post-2"' in html, page.name


def test_the_roi_thumbnail_grid_is_gone(pages):
    """Replaced by the selector pair. It rendered every ROI pairing at a size nothing could
    be read at, and it is the reason the page was measured in megabytes."""
    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert "_ROI_GRID" not in html and "wtc-roi-grid-img" not in html, page.name


# ---- how many cycles of the slowest analysed frequency a window holds ----
#
# Not a cone and not contamination: the count stays the same however clean the edges are.
# A coherence at `--wtc-band-fmin` is a claim about a phase relationship, and a window
# holding one cycle of that frequency has seen the relationship once. Lowering the band
# without lengthening the blocks is the way into that, and nothing used to say so.

# the count is always printed; this sentence only appears when it is below it
COUNT = "cycles of the slowest"
CAVEAT = "Four to six cycles is the usual minimum"


@pytest.mark.parametrize("window, band_fmin, expected", [
    ((3580.0, 3880.0), 0.06, 18.0),      # d01's conversation as it is run
    ((3580.0, 3880.0), 0.01, 3.0),       # the same block with the band lowered
    ((0.0, 900.0), 0.06, 54.0),
    ((0.0, 60.0), 0.02, 1.2),
])
def test_the_count_is_the_window_in_units_of_the_slowest_period(window, band_fmin, expected):
    from fnirs_pipe.qc.hyper.hyper_report import _band_cycles

    assert _band_cycles(window, band_fmin) == pytest.approx(expected, abs=0.05)


def test_a_run_with_no_window_has_no_count():
    """The whole-run page describes the recording, which has no block to be short."""
    from fnirs_pipe.qc.hyper.hyper_report import _band_cycles

    assert _band_cycles(None, 0.06) is None


def test_a_long_enough_condition_prints_the_count_without_the_caveat(pages):
    """The fixture's blocks are 180 s against a 0.03 Hz floor, so 5.4 cycles: above the
    minimum, and the reader still gets the number."""
    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert CAVEAT not in html, page.name
        if _window_of(page):
            assert COUNT in html, page.name


def test_a_condition_too_short_for_the_band_says_so(dyad, tmp_path_factory):
    """The same blocks against a 0.01 Hz floor: 1.8 cycles, and the page has to say it."""
    from fnirs_pipe.qc.hyper.hyper_report import build_hyper_post_report

    out = tmp_path_factory.mktemp("hyper_short")
    path = build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", "sub-01", "tap"), GroupEntry("G1", "sub-02", "tap")],
        aligned_raws=dyad, offsets={"sub-01": 0.0, "sub-02": 0.0}, output_dir=out,
        wtc_fmin=0.008, wtc_fmax=0.2, wtc_band_fmin=0.01, wtc_band_fmax=0.10,
        wtc_chroma=("hbo",), wtc_by_condition=True,
    )
    condition_pages = [p for p in path.parent.glob("*.html")
                       if "index" not in p.name and _window_of(p)]
    assert condition_pages, "no condition page was written"
    for page in condition_pages:
        html = page.read_text(encoding="utf-8")
        assert CAVEAT in html, page.name
        assert "1.8 cycles" in html, page.name

    # and the run's own page, which describes no block, carries neither
    run = next(p for p in path.parent.glob("*.html")
               if "index" not in p.name and not _window_of(p))
    run_html = run.read_text(encoding="utf-8")
    assert COUNT not in run_html and CAVEAT not in run_html
