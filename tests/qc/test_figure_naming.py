"""Every figure this package writes is named by one namer, and three things depend on it.

A subject's figures share one ``figures/``, so the run is part of every name, which is
only safe if three collisions cannot happen: two runs of a subject, the raw viewer against
the pipeline's own report, and a condition page against the run's. Each is pinned below on
the namer rather than on a rendered tree, because a collision is a file silently
overwritten and a rendered tree shows nothing.

`_figure_leaks` is the fourth reader of these names and lives in
`tests/qc/test_condition_page_isolation.py`, which builds its cases from this same namer.
"""

from fnirs_pipe.io.naming import parse_path
from fnirs_pipe.qc.common.figure_io import figure_namer

# Every panel the subject report writes whose name comes from the namer alone, as
# (desc, kwargs). The list is here so a panel added without a thought for collisions turns
# one of the tests below red. The GLM activation is not among them: it names its own
# condition, see `test_the_glm_activation_is_the_one_file_a_condition_page_shares`.
PANELS = [
    ("scipsp", {}),
    ("carpet", {}),
    ("carpetstage", {}),
    ("detail", {"channel": "S1D1"}),
    ("psddetail", {"channel": "S1D1"}),
    ("motion", {"channel": "S1D1760"}),
    ("psd", {}),
    ("trigger", {}),
    ("epochmean", {}),
    ("hbohbrcorr", {}),
    ("denoisestages", {}),
    ("badsegmentzoom", {"extension": ".png"}),
    ("brainviews", {"extension": ".png"}),
    ("restpanel", {}),
    ("trialimage", {"channel": "S1D1hbo"}),
    ("trialimage", {"segmentation": "aal", "label": "PFC"}),
    ("trialqc", {"suffix": "qc"}),
    ("chsummary", {"suffix": "qc"}),
    ("evokedtopo", {"suffix": "nirsmap"}),
    ("topo", {"suffix": "nirsmap", "statistic": "alff"}),
    ("timeseries", {"suffix": "design", "extension": ".png"}),
    ("heatmap", {"suffix": "design", "extension": ".png"}),
    ("matrix", {"suffix": "relmat", "segmentation": "aal", "aggregation": "roi",
                "statistic": "pearson"}),
    ("matrix", {"suffix": "relmat", "segmentation": "aal", "aggregation": "seed",
                "statistic": "pearson"}),
    ("provenance", {"extension": ".png"}),
]


def _names(namer) -> list[str]:
    return [namer(desc, **kwargs) for desc, kwargs in PANELS]


# ---- the three collisions the flat directory is exposed to ----

def test_two_runs_of_one_subject_share_no_filename():
    rest = _names(figure_namer("sub-01_task-rest"))
    tapping = _names(figure_namer("sub-01_task-tapping"))
    assert not set(rest) & set(tapping)


def test_the_raw_viewer_and_the_report_share_no_filename():
    """Both draw the same panels of one run, at different stages, into one figures/."""
    report = _names(figure_namer("sub-01_task-rest"))
    viewer = _names(figure_namer("sub-01_task-rest", prefix="raw"))
    assert not set(report) & set(viewer)


def test_a_condition_page_shares_no_filename_with_the_run():
    run = _names(figure_namer("sub-01_task-rest"))
    game1 = _names(figure_namer("sub-01_task-rest", "game1"))
    game10 = _names(figure_namer("sub-01_task-rest", "game10"))
    assert not set(run) & set(game1)
    assert not set(game1) & set(game10)


def test_the_glm_activation_is_the_one_file_a_condition_page_shares():
    """One model is fitted over the whole recording and the run draws every condition of it
    in one pass on a shared colour scale, so a condition page points at the file that pass
    already wrote rather than drawing its own."""
    from_run = figure_namer("sub-01_task-rest")(
        "glmactivation", suffix="nirsmap", extension=".png", condition="game1")
    on_page = figure_namer("sub-01_task-rest", "game1")(
        "glmactivation", suffix="nirsmap", extension=".png", condition="game1")
    assert from_run == on_page


def test_no_two_panels_of_one_run_collide():
    """Two rows of PANELS that differ only in an entity nobody put in the name."""
    names = _names(figure_namer("sub-01_task-rest"))
    assert len(set(names)) == len(names)


# ---- what the readers take back out ----

def test_the_condition_is_readable_off_the_name():
    """`_figure_leaks` gates a per-condition page on this entity and nothing else."""
    name = figure_namer("sub-01_task-rest", "game1")("carpet")
    assert parse_path(name)["condition"] == "game1"
    assert "condition" not in parse_path(figure_namer("sub-01_task-rest")("carpet"))


def test_an_entity_a_panel_passes_itself_wins_over_the_bound_condition():
    """The GLM activation set is written from the run's namer, one file per condition."""
    name = figure_namer("sub-01_task-rest")("glmactivation", suffix="nirsmap",
                                            extension=".png", condition="video")
    assert parse_path(name)["condition"] == "video"


def test_the_namer_says_what_it_is_bound_to():
    namer = figure_namer("sub-01_task-rest", "game1")
    assert namer.label == "sub-01_task-rest" and namer.condition == "game1"
    assert figure_namer("sub-01_task-rest").condition is None


# ---- the dyad side, whose axis entity is not always a channel ----

def test_a_crossed_map_names_both_sites_in_one_entity():
    namer = figure_namer("group-G1_task-rest")
    crossed = namer("wtcmap", extension=".png", chromophore="hbo",
                    channel="S1D1xS1D2")
    assert parse_path(crossed)["channel"] == "S1D1xS1D2"
    # the ROI panel puts its sites in label- instead, so the two cannot be confused
    roi = namer("wtcmap", chromophore="hbo", label="PFCxTPJ", aggregation="roi")
    assert parse_path(roi)["label"] == "PFCxTPJ"
    assert parse_path(roi)["aggregation"] == "roi"


def test_the_two_cross_matrices_differ_by_their_aggregation():
    namer = figure_namer("group-G1_task-rest")
    chan = namer("wtcmatrix", suffix="relmat", aggregation="chan")
    roi = namer("wtcmatrix", suffix="relmat", aggregation="roi")
    assert chan != roi
    assert parse_path(chan)["suffix"] == "relmat"


def test_a_pairing_separates_one_dyad_of_a_group_from_the_next():
    namer = figure_namer("group-G1_task-rest")
    first = namer("iscpanel", chromophore="hbo", pairing="subaxsubb")
    second = namer("iscpanel", chromophore="hbo", pairing="subaxsubc")
    assert first != second
    assert parse_path(first)["pairing"] == "subaxsubb"
