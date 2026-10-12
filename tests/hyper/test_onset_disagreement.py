"""A dyad whose members marked a block at different moments says so on its analysis pages."""

import re

import mne
import numpy as np
import pytest

from nirspipe.io.naming import parse_path
from nirspipe.pipeline.hyper import GroupEntry
from nirspipe.qc.hyper.hyper_report import member_condition_entries

SFREQ, DURATION, CROP = 5.0, 400.0, 20.0
LABELS = ["S1_D1", "S2_D2", "S3_D3"]
# sub-02 marked "talk" a second after sub-01, past the 2-sample tolerance of 0.4 s
ONSETS = {"sub-01": (20.0, 210.0), "sub-02": (20.0, 211.0)}


def _raw(seed: int, shared: np.ndarray, onsets) -> mne.io.Raw:
    rng = np.random.default_rng(seed)
    names = [f"{label} {c}" for label in LABELS for c in ("hbo", "hbr")]
    types = [c for _ in LABELS for c in ("hbo", "hbr")]
    info = mne.create_info(names, SFREQ, types)
    for i, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        loc[3:6] = [i * 0.05, 0.0, 0.0]
        loc[6:9] = [i * 0.05 + 0.03, 0.0, 0.0]
        loc[:3] = (loc[3:6] + loc[6:9]) / 2
        ch["loc"] = loc
    data = 1e-6 * (shared + 0.2 * rng.standard_normal((len(names), len(shared))))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(list(onsets), [180.0, 180.0], ["rest", "talk"]))
    return raw


def _entry(window):
    return {"window_s": list(window),
            "od_by_set": {"all": {"channel_retention_rate": 1.0, "sci_win_mean": 0.9}}}


# each member's record on its own clock, which starts CROP seconds before the aligned one
SQM = {sid: {"by_condition": {"rest": _entry((rest, rest + 180.0)),
                              "talk": _entry((talk, talk + 180.0))}}
       for sid, (rest, talk) in ONSETS.items()}
OFFSETS = {sid: CROP for sid in ONSETS}


@pytest.fixture(scope="module")
def pages(tmp_path_factory) -> dict:
    from nirspipe.qc.hyper.hyper_report import build_hyper_post_report

    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    shared = np.sin(2 * np.pi * 0.05 * t)
    raws = {sid: _raw(i, shared, onsets).crop(tmin=CROP)
            for i, (sid, onsets) in enumerate(ONSETS.items())}
    out = tmp_path_factory.mktemp("onset_pages")
    path = build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", sid, "tap") for sid in ONSETS],
        aligned_raws=raws, offsets=OFFSETS, output_dir=out, subject_sqm=SQM,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=0.03, wtc_band_fmax=0.10,
        wtc_chroma=("hbo",), wtc_by_condition=True,
    )
    return {parse_path(p.name).get("condition") or "": p.read_text(encoding="utf-8")
            for p in path.parent.glob("*.html") if "index" not in p.name}


def _onset_table(html: str) -> list:
    body = re.search(r'id="onset-residual-table">.*?<tbody>(.*?)</tbody>', html, re.S).group(1)
    return [(c, m, float(d), w) for c, m, d, w in
            re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td></tr>",
                       body)]


def _member_section(html: str) -> str:
    return re.search(r'<section id="MemberMetrics".*?</section>', html, re.S).group(0)


def test_a_refused_record_window_is_kept_with_its_bounds_on_the_aligned_clock():
    refused: dict = {}
    entries = member_condition_entries(SQM, list(ONSETS), [("rest", 0.0, 180.0),
                                                           ("talk", 190.0, 370.0)],
                                       OFFSETS, SFREQ, refused=refused)
    assert set(entries["sub-02"]) == {"rest"} and set(entries["sub-01"]) == {"rest", "talk"}
    assert refused == {"talk": {"sub-02": (pytest.approx(191.0), pytest.approx(371.0))}}


def test_every_page_notes_the_block_whose_markers_disagree(pages):
    assert set(pages) == {"", "rest", "talk"}
    for html in pages.values():
        assert "talk sub-02 +1.000 s" in html
        assert "rest sub-02" not in html


def test_the_condition_page_says_why_the_member_column_is_blank(pages):
    talk = _member_section(pages["talk"])
    assert "sub-02&rsquo;s record puts this condition at 191.000" in talk
    assert "190.000" in talk
    assert "record puts this condition" not in _member_section(pages["rest"])


def test_the_alignment_check_lists_a_condition_page_s_own_block_and_the_run_page_every_one(pages):
    assert _onset_table(pages["talk"]) == [("talk", "sub-02", pytest.approx(1.0), "no")]
    assert _onset_table(pages[""]) == [("rest", "sub-02", pytest.approx(0.0), "yes"),
                                       ("talk", "sub-02", pytest.approx(1.0), "no")]
