"""The names in the scheme, spelled out, and the guarantee that they parse back.

Two things are held here. The table below is the scheme itself: change a pattern in
`fnirs_pipe/data/fnirs_pipe_bids_config.json` and the rows that name a different file turn
red, so the config cannot drift from what the package meant to write. And every row has to
survive a build-then-parse round trip, which is the property that lets a reader recover what
a file is from its name alone. Without it the entities are decoration and the tables have to
be opened to be told apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fnirs_pipe.io.naming import derivative_path, parse_path

OUT = Path("/out")

# (note, entities, the name it must build). Entities carry suffix and extension separately
# because derivative_path takes them positionally.
CASES = [
    ("a pipeline stage",
     dict(subject="01", task="rest", desc="preproc"), "nirs", ".snirf",
     "sub-01/nirs/sub-01_task-rest_desc-preproc_nirs.snirf"),
    ("a session and a run both present",
     dict(subject="01", session="a", task="rest", run=2, desc="errts"), "nirs", ".snirf",
     "sub-01/ses-a/nirs/sub-01_ses-a_task-rest_run-2_desc-errts_nirs.snirf"),
    ("channel-by-channel correlation",
     dict(subject="01", task="rest", chromophore="hbo", statistic="pearson"),
     "relmat", ".tsv",
     "sub-01/nirs/sub-01_task-rest_chromo-hbo_stat-pearson_relmat.tsv"),
    ("the same thing averaged into regions, Fisher z",
     dict(subject="01", task="rest", chromophore="hbo", segmentation="custom",
          aggregation="roi", statistic="fisherz"), "relmat", ".tsv",
     "sub-01/nirs/sub-01_task-rest_chromo-hbo_seg-custom_agg-roi_stat-fisherz_relmat.tsv"),
    ("one value per channel, so the chromophore is a column not an entity",
     dict(subject="01", task="rest", statistic="alff"), "nirsmap", ".tsv",
     "sub-01/nirs/sub-01_task-rest_stat-alff_nirsmap.tsv"),
    ("a quality record, which is numbers rather than a page",
     dict(subject="01", task="rest", desc="preproc"), "qc", ".json",
     "sub-01/nirs/sub-01_task-rest_desc-preproc_qc.json"),
    ("the auxiliary regressors",
     dict(subject="01", task="rest", desc="aux"), "timeseries", ".tsv",
     "sub-01/nirs/sub-01_task-rest_desc-aux_timeseries.tsv"),
    ("a quality record's channel by window matrix",
     dict(subject="01", task="rest", statistic="sci", desc="sqm"), "timeseries", ".tsv",
     "sub-01/nirs/sub-01_task-rest_stat-sci_desc-sqm_timeseries.tsv"),
    ("a quality record's per-channel values",
     dict(subject="01", task="rest", desc="sqm"), "nirsmap", ".tsv",
     "sub-01/nirs/sub-01_task-rest_desc-sqm_nirsmap.tsv"),
    ("four orthogonal dimensions, each its own entity",
     dict(group="G1", task="rest", segmentation="custom", aggregation="homologous",
          condition="all", nulldist="pair", statistic="wtc"), "relmat", ".tsv",
     "group-G1/nirs/group-G1_task-rest_seg-custom_agg-homologous_cond-all_null-pair"
     "_stat-wtc_relmat.tsv"),
    ("one pairing of a group with more than two members, one condition",
     dict(group="G1", task="rest", pairing="01x02", chromophore="hbo", condition="game1",
          statistic="isc"), "relmat", ".tsv",
     "group-G1/nirs/group-G1_task-rest_pair-01x02_chromo-hbo_cond-game1_stat-isc_relmat.tsv"),
    ("the saved transform, where one file cannot hold both chromophores",
     dict(group="G1", task="rest", chromophore="hbo", statistic="wtc"), "relmat", ".npz",
     "group-G1/nirs/group-G1_task-rest_chromo-hbo_stat-wtc_relmat.npz"),
    ("re-averaged over another band, beside the original rather than over it",
     dict(group="G1", task="rest", chromophore="hbo", band="0p05to0p2", statistic="wtc"),
     "relmat", ".tsv",
     "group-G1/nirs/group-G1_task-rest_chromo-hbo_band-0p05to0p2_stat-wtc_relmat.tsv"),
    ("a figure of one channel's signal",
     dict(subject="01", task="rest", channel="S1D1", desc="detail", datatype="figures"),
     "nirs", ".html",
     "sub-01/figures/sub-01_task-rest_chan-S1D1_desc-detail_nirs.html"),
    ("a figure of a crossed channel pair between two brains",
     dict(group="G1", task="rest", pairing="01x02", chromophore="hbo",
          channel="S1D1xS1D2", statistic="wtc", datatype="figures"), "relmat", ".png",
     "group-G1/figures/group-G1_task-rest_pair-01x02_chromo-hbo_chan-S1D1xS1D2"
     "_stat-wtc_relmat.png"),
    ("a figure of one ROI's trials, whose sites are labels rather than channels",
     dict(subject="01", task="rest", segmentation="custom", label="PFC",
          desc="trialimage", datatype="figures"), "nirs", ".html",
     "sub-01/figures/sub-01_task-rest_seg-custom_label-PFC_desc-trialimage_nirs.html"),
    ("the provenance graph's mermaid source, which is a figure written twice",
     dict(subject="01", task="rest", desc="provenance", datatype="figures"),
     "nirs", ".mmd",
     "sub-01/figures/sub-01_task-rest_desc-provenance_nirs.mmd"),
    ("one condition's copy of a panel, which is what keeps it off the run's file",
     dict(subject="01", task="rest", condition="game1", desc="carpet",
          datatype="figures"), "nirs", ".html",
     "sub-01/figures/sub-01_task-rest_cond-game1_desc-carpet_nirs.html"),
    ("a run's report",
     dict(subject="01", task="rest"), "report", ".html",
     "sub-01/sub-01_task-rest_report.html"),
    ("one condition's page of that report",
     dict(subject="01", task="rest", condition="game1"), "report", ".html",
     "sub-01/sub-01_task-rest_cond-game1_report.html"),
    # no analysis unit and no task: a merged table spans every dyad and every task, and
    # carries both as columns. Being at the root with no sub-/group- is what marks it.
    ("the re-paired null of the homologous ROI means, per condition",
     dict(group="G1", task="rest", segmentation="custom", aggregation="homologous",
          condition="all", nulldist="pair", statistic="wtc"), "relmat", ".tsv",
     "group-G1/nirs/group-G1_task-rest_seg-custom_agg-homologous_cond-all_null-pair"
     "_stat-wtc_relmat.tsv"),
    ("the same null at full detail, which is a desc- and not a third null",
     dict(group="G1", task="rest", condition="all", nulldist="pair", statistic="wtc",
          desc="draws"), "relmat", ".tsv",
     "group-G1/nirs/group-G1_task-rest_cond-all_null-pair_stat-wtc_desc-draws_relmat.tsv"),
    ("a re-averaged band, the one output whose band is in its name",
     dict(group="G1", task="rest", chromophore="hbo", band="0p05to0p2", statistic="wtc"),
     "relmat", ".tsv",
     "group-G1/nirs/group-G1_task-rest_chromo-hbo_band-0p05to0p2_stat-wtc_relmat.tsv"),
    ("the dyad's cross-subject channel quality, which may not be _channels.tsv",
     dict(group="G1", task="rest", desc="channel"), "qc", ".tsv",
     "group-G1/nirs/group-G1_task-rest_desc-channel_qc.tsv"),
    ("one page's human ratings",
     dict(subject="01", task="rest", desc="rawrating"), "qc", ".json",
     "sub-01/nirs/sub-01_task-rest_desc-rawrating_qc.json"),
    # the chromophore leads because pybids' own task- pattern needs a separator before it,
    # so a root-level name cannot start with task-. Every entity this package declares is
    # anchored to the start as well as to an underscore, which is why they can.
    ("one cohort verdict, at the root and therefore across dyads",
     dict(chromophore="hbo", task="rest", condition="all", nulldist="pair",
          statistic="wtc", desc="bycell"), "relmat", ".tsv",
     "chromo-hbo_task-rest_cond-all_null-pair_stat-wtc_desc-bycell_relmat.tsv"),
    ("a cross-dyad summary",
     dict(statistic="wtc"), "relmat", ".tsv",
     "stat-wtc_relmat.tsv"),
    ("a cross-dyad summary of the re-paired null on homologous regions",
     dict(segmentation="custom", aggregation="homologous", condition="all",
          nulldist="pair", statistic="wtc"), "relmat", ".tsv",
     "seg-custom_agg-homologous_cond-all_null-pair_stat-wtc_relmat.tsv"),
    ("the cohort quality table",
     dict(desc="subjects"), "qc", ".tsv",
     "desc-subjects_qc.tsv"),
]

IDS = [case[0] for case in CASES]


@pytest.mark.parametrize("note,entities,suffix,extension,expected", CASES, ids=IDS)
def test_the_name_is_what_the_scheme_says(note, entities, suffix, extension, expected):
    built = derivative_path(OUT, suffix, extension, **entities)
    assert built.relative_to(OUT).as_posix() == expected


@pytest.mark.parametrize("note,entities,suffix,extension,expected", CASES, ids=IDS)
def test_every_entity_survives_the_round_trip(note, entities, suffix, extension, expected):
    read = parse_path(expected)
    lost = {key: (value, read.get(key)) for key, value in entities.items()
            if key != "datatype" and str(read.get(key)) != str(value)}
    assert not lost, f"{expected} does not give back {lost}"
    assert read.get("suffix") == suffix
    assert read.get("extension") == extension


@pytest.mark.parametrize("note,entities,suffix,extension,expected", CASES, ids=IDS)
def test_no_name_carries_a_stray_separator(note, entities, suffix, extension, expected):
    """An optional segment written with the underscore outside leaves one behind when the
    entity is absent."""
    assert "__" not in expected and "/_" not in expected and not expected.startswith("_")


def test_an_entity_left_as_none_is_simply_absent():
    """Callers pass session=None rather than branching, so None must not reach the name."""
    with_none = derivative_path(OUT, "nirs", ".snirf", subject="01", task="rest",
                                session=None, run=None, desc="preproc")
    assert with_none.relative_to(OUT).as_posix() == \
        "sub-01/nirs/sub-01_task-rest_desc-preproc_nirs.snirf"


def test_a_suffix_outside_the_scheme_is_refused():
    """Silently returning a name no pattern covers would let hand-built strings drift."""
    with pytest.raises(ValueError, match="no path pattern fits"):
        derivative_path(OUT, "fc", ".tsv", subject="01", task="rest")


def test_pybids_can_index_and_query_a_tree_written_this_way(tmp_path):
    """The point of the scheme: somebody else's BIDS tooling can read the dyad results.

    Every entity queried here is found by a query, not by globbing the whole tree and
    parsing names by hand.
    """
    import json

    from bids import BIDSLayout

    from fnirs_pipe.io.naming import layout_config

    (tmp_path / "dataset_description.json").write_text(json.dumps({
        "Name": "fnirs-hyper output", "BIDSVersion": "1.9.0",
        "DatasetType": "derivative", "GeneratedBy": [{"Name": "fnirs-hyper"}],
    }), encoding="utf-8")

    written = [
        dict(group="G1", task="rest", statistic="wtc"),
        dict(group="G1", task="rest", chromophore="hbo", statistic="isc"),
        dict(group="G1", task="rest", segmentation="custom", aggregation="homologous",
             nulldist="pair", statistic="wtc"),
        dict(group="G2", task="rest", statistic="wtc"),
    ]
    for entities in written:
        path = derivative_path(tmp_path, "relmat", ".tsv", **entities)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")

    layout = BIDSLayout(tmp_path, validate=False, is_derivative=True,
                        config=layout_config())

    assert len(layout.get(suffix="relmat")) == len(written)
    assert len(layout.get(suffix="relmat", group="G1")) == 3
    assert len(layout.get(suffix="relmat", statistic="wtc", nulldist="pair")) == 1
    assert len(layout.get(suffix="relmat", chromophore="hbo")) == 1
    # no null entity means the real observation, so it must not come back with the draws
    assert len(layout.get(suffix="relmat", statistic="wtc")) == 3


def test_the_config_ships_with_the_package():
    from fnirs_pipe.io.naming import _CONFIG_FILE

    assert _CONFIG_FILE.is_file(), f"{_CONFIG_FILE} is missing from the installed package"
