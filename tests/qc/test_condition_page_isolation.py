"""A per-condition report page must not carry a figure measured over the whole run.

This is the failure the pages are most exposed to and the one no other check sees. `ruff`
passes, the CLI guards pass, the run exits zero, and the page still shows the run's carpet
under numbers that describe one condition, with nothing on it saying so.

Two halves are pinned here. `_blanked` empties whatever it is handed, and `_figure_leaks`
catches what it was not handed, by looking at the assembled values rather than at a list of
names somebody has to remember to extend.
"""

from nirspipe.qc.common.figure_io import figure_namer
from nirspipe.qc.subject.report import (
    _blanked, _carpet_views, _condition_carpet, _figure_leaks,
    _segments_in_window,
)

RUN = figure_namer("sub-01_task-main")


def fig(desc, condition=None, **entities):
    """A figure URL as a page carries it, named by the writers' own namer.

    Spelled by hand, the test would agree with itself while neither half agreed with what
    lands on disk.
    """
    return "figures/" + figure_namer("sub-01_task-main", condition)(desc, **entities)


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
    # every panel a condition page rebuilds carries the condition in its own cond- entity,
    # so the check reads that entity rather than a list of panel names nobody would
    # remember to extend
    page = {"channel_summary_path": fig("chsummary", "game1", suffix="qc"),
            "sci_psp_panel_path":   fig("scipsp", "game1"),
            "carpet_gvtd_path":     fig("carpet", "game1"),
            "bad_segment_zoom_path": fig("badsegmentzoom", "game1", extension=".png")}
    assert _figure_leaks(page, "game1") == []


def test_a_longer_label_starting_with_this_one_is_still_a_leak():
    # "game1" must not accept "game10"'s figures, nor the other way round
    assert _figure_leaks({"a": fig("carpet", "game10")}, "game1")
    assert _figure_leaks({"a": fig("carpet", "game1")}, "game10")


def test_the_provenance_graph_is_not_a_leak():
    # it describes the run's file lineage, which is the same for every condition
    assert _figure_leaks({"provenance_path": fig("provenance", extension=".png")},
                         "game1") == []


def test_the_glm_design_matrix_is_not_a_leak():
    # one model over the whole recording with every condition drawn as a column of it, so
    # it reads as the model rather than as this condition
    assert _figure_leaks(
        {"glm_design_path": fig("timeseries", suffix="design", extension=".png"),
         "glm_design_heatmap_path": fig("heatmap", suffix="design", extension=".png")},
        "game1") == []


def test_the_glm_activation_still_has_to_be_this_conditions():
    # unlike the design matrix, an activation map is one condition's and nothing on it says
    # which, so the run's own must not survive
    assert _figure_leaks(
        {"glm_activation_path": fig("glmactivation", "video", suffix="nirsmap",
                                    extension=".png")}, "game1")


def test_a_run_wide_figure_is_reported():
    # a loose keyword outside any section dict, which the blanking pass never sees
    name = RUN("carpetstage")
    leaks = _figure_leaks({"denoise_carpet_path": f"figures/{name}"}, "game1")
    assert leaks == [f"denoise_carpet_path={name}"]


def test_another_conditions_figure_is_a_leak():
    # a copied payload pointing at the neighbour's is worse than a run-wide figure, since
    # the page would look per-condition and be the wrong condition
    assert _figure_leaks(
        {"channel_summary_path": fig("chsummary", "video", suffix="qc")}, "game1")
    assert _figure_leaks({"carpet_gvtd_path": fig("carpet", "video")}, "game1")


def test_values_that_are_not_figure_paths_are_ignored():
    page = {"subject": "01", "sqm": {"sci_mean": 0.9}, "n_bad": 2,
            "index_href": "sub-01_task-main_report.html", "nothing": None}
    assert _figure_leaks(page, "game1") == []


def test_several_leaks_are_all_reported():
    leaks = _figure_leaks({"a": fig("carpet"), "b": fig("psd")}, "game1")
    assert len(leaks) == 2


# ---- a run-wide file addressed at one condition ----

def test_a_fragment_naming_this_condition_is_not_a_leak():
    # the per-channel motion figures are one file per channel carrying every condition's
    # window, because the traces are identical across conditions and only the axes move
    motion = fig("motion", channel="S1D1")
    page = {"motion_detail_pairs": [
        {"pair": "S1D1", "path": f"{motion}#game1", "h": 400}]}
    assert _figure_leaks(page, "game1") == []


def test_the_same_file_with_no_fragment_is_still_a_leak():
    motion = RUN("motion", channel="S1D1")
    page = {"motion_detail_pairs": [
        {"pair": "S1D1", "path": f"figures/{motion}", "h": 400}]}
    assert _figure_leaks(page, "game1") == [f"motion_detail_pairs={motion}"]


def test_a_fragment_naming_another_condition_is_a_leak():
    motion = RUN("motion", channel="S1D1")
    page = {"motion_detail_pairs": [
        {"pair": "S1D1", "path": f"figures/{motion}#video", "h": 400}]}
    assert _figure_leaks(page, "game1")


def test_the_per_channel_panels_are_checked_at_all():
    # they arrive as a list of dicts rather than a path, and they are the panels there are
    # the most files of
    detail = RUN("detail", channel="S1D1")
    page = {"ch_detail_pairs": [{"pair": "S1D1", "path": f"figures/{detail}"}]}
    assert _figure_leaks(page, "game1") == [f"ch_detail_pairs={detail}"]


def test_the_carpet_reaches_a_condition_page_as_a_fragment_too():
    carpet = f"figures/{RUN('carpet')}"
    page = _condition_carpet({"carpet_gvtd_path": carpet,
                              "carpet_gvtd_h": 600}, "game1")
    assert page["carpet_gvtd_path"] == f"{carpet}#game1"
    assert page["carpet_gvtd_h"] == 600
    assert _figure_leaks(page, "game1") == []


def test_a_run_with_no_carpet_hands_the_condition_page_none():
    # the build is guarded, so a failed carpet must not become the string "None#game1"
    assert _condition_carpet({}, "game1")["carpet_gvtd_path"] is None


def test_a_carpet_view_names_the_condition_and_its_window():
    # what a view carries beyond the window is pinned in test_condition_views, against a
    # figure the real builder made; here it is only the slug and the span
    from nirspipe.qc.figures.common.motion_panel import carpet_gvtd_figure  # noqa: F401
    views = _carpet_views(_FakeFig(), [("game 1", 10.0, 20.0), ("video", 30.0, 40.0)])
    assert set(views) == {"game1", "video"}
    assert views["game1"]["x"] == [10.0, 20.0]


def test_a_run_with_no_conditions_bakes_no_view_table():
    assert _carpet_views(_FakeFig(), []) is None


# ---- the bad-segment zoom belongs to the condition it is printed under ----

class _FakeFig:
    """Enough of a figure for the view table: no GVTD rows, so only the window survives."""

    class _Layout:
        annotations = ()
        shapes = ()

    data = ()
    layout = _Layout()

    def add_annotation(self, **_kw):
        raise AssertionError("the view table must not write to the figure")

    def update_yaxes(self, **_kw):
        raise AssertionError("the view table must not write to the figure")


SEGMENTS = {"BAD_gvtd": [(10.0, 5.0), (150.0, 40.0), (900.0, 30.0)],
            "BAD_manual": [(95.0, 20.0)]}


def test_the_zoom_shows_only_what_happened_during_this_condition():
    # the recording's segments on a quiet condition's page would show what another
    # condition was censored for
    assert _segments_in_window(SEGMENTS, (100.0, 300.0)) == [(150.0, 40.0), (95.0, 20.0)]


def test_a_segment_across_the_boundary_counts():
    # 95-115 s straddles the start: the condition sat through it whichever side it began on
    assert (95.0, 20.0) in _segments_in_window(SEGMENTS, (100.0, 300.0))


def test_a_segment_ending_exactly_at_the_window_start_does_not():
    assert _segments_in_window({"BAD": [(80.0, 20.0)]}, (100.0, 300.0)) == []


def test_the_run_keeps_every_segment():
    assert len(_segments_in_window(SEGMENTS, None)) == 4


def test_a_condition_with_nothing_flagged_gets_no_zoom():
    # the template gates the panel on the path, so an empty list means no panel rather than
    # the run's figure under this condition's heading
    assert _segments_in_window(SEGMENTS, (400.0, 800.0)) == []


# ---- a page says which page it is ----

def test_the_shell_reads_page_heading_and_page_title():
    # the keys a page must set to be titled at all
    from nirspipe.qc.common.report_shell import page_vars
    keys = page_vars(title="t", heading="h")
    assert keys["page_heading"] == "h" and keys["page_title"] == "t"
    assert "heading" not in keys


def test_a_condition_page_sets_the_keys_the_shell_reads():
    # nothing reads `heading`, so a page setting it would carry the run's own title and the
    # Scope row would be the only thing telling them apart
    import inspect

    from nirspipe.qc.subject.report import _write_condition_reports

    source = inspect.getsource(_write_condition_reports)
    assert '"page_heading": f"{report_vars[' in source
    assert '"page_title": f"{report_vars[' in source
    assert '"heading":' not in source
