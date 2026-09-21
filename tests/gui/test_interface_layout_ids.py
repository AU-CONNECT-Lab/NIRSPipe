"""Every id a callback binds to must exist in a page layout, or the callback silently dies."""

from __future__ import annotations

import os

import dash
import pytest

# ids the app shell owns, and ids the callbacks build at runtime rather than ship in a layout
_SHELL_IDS = {
    "app-bids-dir", "app-output-dir", "dp-run-store",
    "app-sidebar", "app-sidebar-nav", "app-sidebar-title", "app-sidebar-toggle",
    "app-page-content",
}
_RUNTIME_IDS = {"an-subjects-checklist", "dp-subject-radio"}


def _walk(component, found):
    cid = getattr(component, "id", None)
    if isinstance(cid, str):
        found.add(cid)
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            _walk(child, found)
    else:
        _walk(children, found)


@pytest.fixture(scope="module")
def registered():
    pages = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "fnirs_pipe", "interface", "pages")
    dash.Dash(__name__, use_pages=True, pages_folder=pages, suppress_callback_exceptions=True)

    import fnirs_pipe.interface.callbacks._sections               # noqa: F401
    import fnirs_pipe.interface.callbacks.analysis_callbacks      # noqa: F401
    import fnirs_pipe.interface.callbacks.batch_prep_callbacks    # noqa: F401
    import fnirs_pipe.interface.callbacks.data_prep_callbacks     # noqa: F401
    import fnirs_pipe.interface.callbacks.hyper_align_callbacks    # noqa: F401
    import fnirs_pipe.interface.callbacks.hyper_analysis_callbacks  # noqa: F401
    import fnirs_pipe.interface.callbacks.qc_callbacks            # noqa: F401
    import fnirs_pipe.interface.callbacks.recon_callbacks         # noqa: F401

    layout_ids = set()
    for page in dash.page_registry.values():
        layout = page["layout"]
        _walk(layout() if callable(layout) else layout, layout_ids)
    return layout_ids


def _callback_ids():
    from dash._callback import GLOBAL_CALLBACK_MAP

    ids = set()
    for key, spec in GLOBAL_CALLBACK_MAP.items():
        for dep in list(spec.get("inputs", [])) + list(spec.get("state", [])):
            dep_id = dep.get("id") if isinstance(dep, dict) else getattr(dep, "component_id", None)
            if isinstance(dep_id, str):
                ids.add(dep_id)
        for target in (key.split("...") if "..." in key else [key]):
            target = target.strip(".")
            if "." in target:
                ids.add(target.rsplit(".", 1)[0])
    # pattern-matching ids serialise as JSON, and no layout ships them
    return {i for i in ids if not i.startswith("{")}


def test_every_callback_id_is_in_a_layout(registered):
    missing = sorted(_callback_ids() - registered - _SHELL_IDS - _RUNTIME_IDS)
    assert not missing, f"callbacks bind ids no layout renders: {missing}"


def test_pages_all_register(registered):
    assert len(dash.page_registry) == 7
