"""A per-condition report page must not carry a figure measured over the whole run.

This is the failure the pages are most exposed to and the one no other check sees. `ruff`
passes, the CLI guards pass, the run exits zero, and the page still shows the run's carpet
under numbers that describe one condition, with nothing on it saying so. It happened once
already: `denoise_carpet_path` reached the template as a loose keyword rather than inside a
section dict, so the blanking pass never saw it and every condition page kept it.

Two halves are pinned here. `_blanked` empties whatever it is handed, and `_figure_leaks`
catches what it was not handed, by looking at the assembled values rather than at a list of
names somebody has to remember to extend.
"""

from fnirs_pipe.qc.report import _blanked, _figure_leaks


# ---- blanking a section keeps its type ----

def test_paths_blank_to_empty_and_collections_to_empty():
    # every panel is gated on its path and "" is falsy, so the section vanishes; the height
    # has to stay a number because the markup adds to it
    out = _blanked(({"a_path": "figures/a.html", "a_h": 420,
                     "pairs": [{"pair": "S1_D1"}], "lookup": {"x": 1}},))
    assert out == {"a_path": "", "a_h": 0, "pairs": [], "lookup": {}}


def test_every_group_handed_in_is_emptied():
    out = _blanked(({"one": "figures/one.png"}, {"two": "figures/two.html", "two_h": 12}))
    assert out == {"one": "", "two": "", "two_h": 0}


def test_a_blanked_string_is_empty_and_never_the_word_none():
    # Jinja prints None as "None", so an unguarded {{ }} would put that in the page;
    # trial_qc_window is a sentence printed inside its section
    assert _blanked(({"window": "each event's own duration"},))["window"] == ""


def test_a_flag_blanks_to_none_not_to_zero():
    # sqm_split is a bool the template branches on; 0 would read as False either way, but
    # None keeps a bool from silently becoming an int in the payload
    assert _blanked(({"split": True},))["split"] is None


# ---- and the guard for what blanking never saw ----

def test_the_conditions_own_figures_are_not_leaks():
    # every panel a condition page rebuilds is written as <panel>_<slug>, so the check is a
    # suffix rather than a list of panel names nobody would remember to extend
    page = {"channel_summary_path": "figures/x/channel_summary_game1.html",
            "sci_psp_panel_path":   "figures/x/sci_psp_panel_game1.html",
            "carpet_gvtd_path":     "figures/x/carpet_gvtd_game1.html",
            "bad_segment_zoom_path": "figures/x/bad_segment_zoom_game1.png"}
    assert _figure_leaks(page, "game1") == []


def test_a_longer_label_starting_with_this_one_is_still_a_leak():
    # "game1" must not accept "game10"'s figures, nor the other way round
    assert _figure_leaks({"a": "figures/x/carpet_gvtd_game10.html"}, "game1")
    assert _figure_leaks({"a": "figures/x/carpet_gvtd_game1.html"}, "game10")


def test_the_provenance_graph_is_not_a_leak():
    # it describes the run's file lineage, which is the same for every condition
    assert _figure_leaks({"provenance_path": "figures/x/provenance.png"}, "game1") == []


def test_a_run_wide_figure_is_reported():
    # the case that actually happened
    leaks = _figure_leaks({"denoise_carpet_path": "figures/x/denoise_carpet.png"}, "game1")
    assert leaks == ["denoise_carpet_path=denoise_carpet.png"]


def test_another_conditions_figure_is_a_leak():
    # a copied payload pointing at the neighbour's is worse than a run-wide figure, since
    # the page would look per-condition and be the wrong condition
    assert _figure_leaks(
        {"channel_summary_path": "figures/x/channel_summary_video.html"}, "game1")
    assert _figure_leaks({"carpet_gvtd_path": "figures/x/carpet_gvtd_video.html"}, "game1")


def test_values_that_are_not_figure_paths_are_ignored():
    page = {"subject": "01", "sqm": {"sci_mean": 0.9}, "n_bad": 2,
            "index_href": "sub-01_task-full_qc.html", "nothing": None}
    assert _figure_leaks(page, "game1") == []


def test_several_leaks_are_all_reported():
    leaks = _figure_leaks({"a": "figures/x/carpet_gvtd.html",
                           "b": "figures/x/psd_panel.html"}, "game1")
    assert len(leaks) == 2
