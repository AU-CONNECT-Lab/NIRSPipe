"""A run given its own separation bands is drawn and labelled with those, not the defaults."""

import json
import re

import pytest

from fnirs_pipe.qc.common.channel_table import _neither_range_title
from tests._fingerprint import LONG_DISTANCE, SHORT_DISTANCE
from tests.figure_accuracy._payload import _decode, one_figure
from tests.figure_accuracy._run import Run
from tests.figure_accuracy.conftest import run_capturing

# the short pairs fall below the long range and above this, so they belong to neither
SHORT_MAX_MM = 5.0
BANDS = (SHORT_MAX_MM / 1e3, 0.015, None)
SEP_ARGS = ["--by-condition", "--short-max-dist", f"{SHORT_MAX_MM:g}"]

assert SHORT_MAX_MM / 1e3 < SHORT_DISTANCE < BANDS[1] < LONG_DISTANCE


@pytest.fixture(scope="module")
def raw_sep(tmp_path_factory) -> Run:
    root = tmp_path_factory.mktemp("fingerprint_raw_sep")
    done = run_capturing(root, ["--participant-label", "01", *SEP_ARGS], (), task="main",
                         blocks=True, raw_viewer=True)
    return Run(root / "out", done["truth"], task="main")


@pytest.fixture(scope="module")
def prep_sep(tmp_path_factory) -> Run:
    root = tmp_path_factory.mktemp("fingerprint_prep_sep")
    done = run_capturing(root, SEP_ARGS, (), task="main", blocks=True)
    return Run(root / "out", done["truth"], task="main")


def _raw_pages(run):
    return sorted((run.out / "sub-01").glob("*desc-raw_report.html"))


def _raw_blocks(page):
    html = page.read_text(encoding="utf-8")
    data = json.loads(re.search(r"var _STATIC_DATA = (\[.*?\]);\n", html, re.S).group(1))[0]
    return _decode(data)["channels"]["blocks"]


def _pipeline_pages(run):
    return sorted(p for p in (run.out / "sub-01").glob("sub-01_task-main*_report.html")
                  if "desc-" not in p.name)


def test_the_short_pairs_fall_in_neither_range(raw_sep):
    short = {p.name for p in raw_sep.truth.pairs if p.short}
    for page in _raw_pages(raw_sep):
        neither = [rows for title, rows in _raw_blocks(page) if title.startswith("Neither range")]
        assert len(neither) == 1 and {row["name"] for row in neither[0]} == short, page.name


def test_the_mean_psd_groups_channels_by_the_run_s_bands(raw_sep):
    n_long = 2 * len(raw_sep.truth.long_pairs)
    n_short = 2 * sum(p.short for p in raw_sep.truth.pairs)
    psd = one_figure(raw_sep.figure("rawpsd"))
    legends = [t["name"] for t in psd["data"] if t.get("name")]
    assert legends == [f"Long channels (n={n_long})", f"{_neither_range_title(BANDS)} (n={n_short})"]


@pytest.mark.parametrize("which", ["raw viewer", "pipeline"])
def test_every_channel_table_names_the_run_s_gap(raw_sep, prep_sep, which):
    wanted = _neither_range_title(BANDS)
    if which == "raw viewer":
        pages = _raw_pages(raw_sep)
        titles = {page.name: [t for t, _ in _raw_blocks(page) if t.startswith("Neither")]
                  for page in pages}
    else:
        pages = _pipeline_pages(prep_sep)
        titles = {page.name: re.findall(r'class="ch-group">(Neither range \([^)]*\))',
                                        page.read_text(encoding="utf-8"))
                  for page in pages}
    assert len(pages) == 3, sorted(titles)
    assert all(found == [wanted] for found in titles.values()), titles
