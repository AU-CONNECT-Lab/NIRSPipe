"""What a page does with what is typed into its form."""

from __future__ import annotations

import os

import dash
import pytest

from nirspipe.interface.callbacks import data_prep_callbacks, hyper_analysis_callbacks


@pytest.fixture(scope="module")
def page_layouts():
    import nirspipe.interface as interface_pkg
    pages = os.path.join(os.path.dirname(interface_pkg.__file__), "pages")
    dash.Dash(__name__, use_pages=True, pages_folder=pages, suppress_callback_exceptions=True)
    layouts = [page["layout"] for page in dash.page_registry.values()]
    return [layout() if callable(layout) else layout for layout in layouts]


def _number_steps(component, found):
    if getattr(component, "type", None) == "number":
        found[str(component.id)] = getattr(component, "step", None)
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and not isinstance(child, (str, int, float)):
            _number_steps(child, found)
    return found


def test_no_number_field_drops_a_value_off_its_step_grid(page_layouts):
    # the browser holds a value off min + n*step invalid and dbc then sends None, so the flag
    # vanishes from the command while the field still shows it
    steps = {}
    for layout in page_layouts + [data_prep_callbacks._make_marker_row(0, {})]:
        _number_steps(layout, steps)
    assert {"an-short-max-dist", "hy-wtc-phasenull", "bp-crop-tmin"} <= steps.keys()
    assert {cid: step for cid, step in steps.items() if step not in ("any", 1)} == {}


def test_data_preparation_takes_an_output_directory_that_does_not_exist_yet(tmp_path):
    missing = tmp_path / "not-yet"
    _, status = data_prep_callbacks.load_run(
        "sub-01_task-x_nirs.snirf", 0.8, 0.7, 2.0, 6.0, None, None, None, None, [],
        None, None, None, None, None, None, str(missing))
    # past the output check, stopped by the next one
    assert status.children == "Set BIDS Directory first."


def _hyper_values(**opts):
    return [opts.get(key) for key in hyper_analysis_callbacks._KEYS]


def test_every_hyper_field_has_its_key():
    assert len(hyper_analysis_callbacks._KEYS) == len(hyper_analysis_callbacks._STATES)


def test_the_shared_band_reaches_the_command_and_the_overrides_follow_it():
    _, stored = hyper_analysis_callbacks.generate_command(
        1, "run", *_hyper_values(output_dir="/hyper", derivatives_dir="/deriv",
                                 pairs_csv="/pairs.csv", shared_band_fmin=0.02,
                                 shared_band_fmax=0.1, isc_fmin=0.05), "bash")
    argv = stored["argv"]
    assert argv[argv.index("--band-fmin") + 1] == "0.02"
    assert argv[argv.index("--band-fmax") + 1] == "0.1"
    assert argv[argv.index("--isc-fmin") + 1] == "0.05"
    assert "--wtc-band-fmin" not in argv
