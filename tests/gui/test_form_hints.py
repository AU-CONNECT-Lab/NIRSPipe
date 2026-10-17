"""What the generated forms carry for the page's helpers: field kinds for the folded summaries, and Detect's subjects."""

from nirspipe.interface.server import WebApp
from tests._synth import make_bids_dataset


def test_detect_lists_the_datasets_subjects(tmp_path):
    bids = make_bids_dataset(tmp_path, subjects=("02", "01"), tasks=("rest",))
    found = WebApp().flask.test_client().get("/subjects", query_string={"bids_dir": str(bids)})
    assert found.get_json() == {"subjects": ["01", "02"]}


def test_detect_says_why_it_found_nothing(tmp_path):
    client = WebApp().flask.test_client()

    assert client.get("/subjects").get_json() == {"error": "Fill the BIDS folder first."}
    missing = tmp_path / "nowhere"
    assert client.get("/subjects", query_string={"bids_dir": str(missing)}).get_json() == {
        "error": f"No folder {missing}."}
    assert client.get("/subjects", query_string={"bids_dir": str(tmp_path)}).get_json() == {
        "error": f"No subjects in {tmp_path}."}


def test_every_field_names_its_kind_and_the_form_names_detect():
    html = WebApp().flask.test_client().get("/raw-qc/prep-raw/").get_data(as_text=True)

    assert 'data-subjects="/subjects"' in html
    assert html.count('class="field ') == html.count("data-kind=")
    for kind in ("switch", "boolean", "choice", "number", "text", "list"):
        assert f'data-kind="{kind}"' in html
