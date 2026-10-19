"""Reports load Plotly.js from the CDN first and fall back to one copy at the tree's root."""

from __future__ import annotations

from pathlib import Path

import plotly

from nirspipe.io.derivatives import write_bidsignore
from nirspipe.qc.common.figure_io import (
    PLOTLY_CDN_URL, PLOTLY_LOCAL_NAME, _save_figure_html, plotly_script_tags,
)

_BUNDLED = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


def test_the_cdn_comes_first_and_the_fallback_climbs_to_the_tree_root(tmp_path):
    page = tmp_path / "sub-01" / "figures" / "x.html"
    page.parent.mkdir(parents=True)
    tags = plotly_script_tags(page)
    assert tags.index(PLOTLY_CDN_URL) < tags.index("window.Plotly||")
    assert f'src="../../{PLOTLY_LOCAL_NAME}"' in tags


def test_the_copy_is_written_once_at_the_root_and_is_plotly_s_own(tmp_path):
    for page in (tmp_path / "sub-01" / "figures" / "a.html",
                 tmp_path / "sub-02" / "ses-1" / "figures" / "b.html",
                 tmp_path / "group-1" / "c.html"):
        page.parent.mkdir(parents=True)
        plotly_script_tags(page)
    copies = list(tmp_path.rglob("plotly-*.min.js"))
    assert copies == [tmp_path / PLOTLY_LOCAL_NAME]
    assert copies[0].read_bytes().replace(b"\r\n", b"\n") == _BUNDLED.read_bytes().replace(b"\r\n", b"\n")
    assert not list(tmp_path.rglob("*.tmp"))


def test_the_cdn_version_is_the_one_plotly_ships():
    head = _BUNDLED.read_text(encoding="utf-8")[:200]
    version = PLOTLY_LOCAL_NAME.removeprefix("plotly-").removesuffix(".min.js")
    assert f"plotly.js v{version}" in head
    assert PLOTLY_CDN_URL.endswith(f"/plotly-{version}.min.js")


def test_a_written_figure_carries_both_tags(tmp_path):
    import plotly.graph_objects as go
    page = tmp_path / "sub-01" / "figures" / "x.html"
    _save_figure_html(go.Figure(go.Scatter(x=[0, 1], y=[1, 2])), page)
    html = page.read_text(encoding="utf-8")
    assert PLOTLY_CDN_URL in html and f"../../{PLOTLY_LOCAL_NAME}" in html


def test_the_copy_is_out_of_the_validator_s_way(tmp_path):
    write_bidsignore(tmp_path)
    patterns = (tmp_path / ".bidsignore").read_text(encoding="utf-8").split()
    assert "plotly-*.min.js" in patterns


def test_the_app_serves_the_fallback_wherever_a_page_climbs_to():
    from nirspipe.interface.server import WebApp
    client = WebApp().flask.test_client()
    for url in (f"/view/{PLOTLY_LOCAL_NAME}", f"/{PLOTLY_LOCAL_NAME}", f"/report/{PLOTLY_LOCAL_NAME}"):
        reply = client.get(url)
        assert reply.status_code == 200, url
        assert reply.mimetype == "text/javascript"
        assert reply.data[:200].replace(b"\r\n", b"\n") == _BUNDLED.read_bytes()[:200].replace(b"\r\n", b"\n")


def test_outside_a_dataset_the_copy_sits_beside_the_report_not_among_the_figures(tmp_path):
    page = tmp_path / "figures" / "x.html"
    page.parent.mkdir()
    assert f'src="../{PLOTLY_LOCAL_NAME}"' in plotly_script_tags(page)
    assert (tmp_path / PLOTLY_LOCAL_NAME).is_file()
    assert not (page.parent / PLOTLY_LOCAL_NAME).exists()


def test_parallel_subjects_writing_the_copy_at_once_all_succeed(tmp_path, monkeypatch):
    import threading

    from nirspipe.qc.common import figure_io

    real = figure_io.get_plotlyjs
    for attempt in range(5):
        # every thread is held until all have passed the exists-check, where the race was
        barrier = threading.Barrier(4)

        def held_plotlyjs():
            try:
                barrier.wait(timeout=1)
            except threading.BrokenBarrierError:
                pass
            return real()

        monkeypatch.setattr(figure_io, "get_plotlyjs", held_plotlyjs)
        target = tmp_path / str(attempt) / PLOTLY_LOCAL_NAME
        errors = []

        def write():
            try:
                figure_io._write_local_plotly(target)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert target.read_bytes().replace(b"\r\n", b"\n") == _BUNDLED.read_bytes().replace(b"\r\n", b"\n")
        assert not list(target.parent.glob("*.tmp"))
