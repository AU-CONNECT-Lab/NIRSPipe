"""Serving a generated QC report back into the GUI without opening the filesystem.

The route exists because `srcDoc` cannot host the group reports: they are iframe shells whose
panels are sibling files, and a srcdoc document has no base URL for a relative `src`. Once
files are served, the interesting question stops being "does the report show" and becomes
"what else can be asked for", which is most of what is tested here.
"""

import pytest
from flask import Flask

from nirspipe.interface.report_serve import _ROOTS, attach, register, report_url


@pytest.fixture
def report_tree(tmp_path):
    """A group report and the sibling panel it embeds, plus a file outside the tree."""
    reports = tmp_path / "derivatives"
    reports.mkdir()
    (reports / "desc-subjects_report.html").write_text(
        "<html><iframe src='panels/heatmap.html'></iframe></html>")
    (reports / "panels").mkdir()
    (reports / "panels" / "heatmap.html").write_text("<b>heatmap</b>")
    (tmp_path / "secret.txt").write_text("not for the browser")
    return reports


@pytest.fixture
def client(report_tree):
    app = Flask(__name__)
    attach(app)
    return app.test_client()


def test_the_report_and_its_panels_both_load(client, report_tree):
    """The panel is the reason this is a route and not a srcDoc."""
    url = report_url(report_tree / "desc-subjects_report.html")
    assert client.get(url).status_code == 200

    base = url.rsplit("/", 1)[0]
    assert client.get(f"{base}/panels/heatmap.html").status_code == 200


def test_a_directory_registers_once(report_tree):
    """Re-running a report must not leak a token per run, or the iframe URL would churn."""
    first = register(report_tree)
    assert register(report_tree) == first
    assert sum(1 for root in _ROOTS.values() if root == report_tree.resolve()) == 1


def test_an_unknown_token_is_not_served(client):
    assert client.get("/report/ffffffffffff/anything.html").status_code == 404


@pytest.mark.parametrize("escape", [
    "../secret.txt",
    "..%2fsecret.txt",
    "panels/../../secret.txt",
])
def test_a_token_cannot_be_walked_out_of(client, report_tree, escape):
    """The token names one directory; it must not become a handle on the filesystem."""
    base = report_url(report_tree / "desc-subjects_report.html").rsplit("/", 1)[0]
    assert client.get(f"{base}/{escape}").status_code in (400, 403, 404)


def test_a_missing_file_inside_a_known_root_is_a_404(client, report_tree):
    base = report_url(report_tree / "desc-subjects_report.html").rsplit("/", 1)[0]
    assert client.get(f"{base}/never_written.html").status_code == 404


def test_the_url_points_at_the_file_by_name(report_tree):
    url = report_url(report_tree / "desc-subjects_report.html")
    assert url.startswith("/report/")
    assert url.endswith("/desc-subjects_report.html")
