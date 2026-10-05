"""Provenance graph reconstruction from sidecars on disk.

Everything goes through the public entry points (scan / to_mermaid /
write_provenance) rather than the private helpers, so a refactor of the naming
or layout internals does not turn these red while behaviour is unchanged.
"""

import json

import pytest

from fnirs_pipe.qc.common.provenance import scan, to_mermaid
from fnirs_pipe.qc.common.figure_io import figure_namer
from fnirs_pipe.qc.figures.common.provenance_figure import write_provenance


def _sidecar(directory, name, step=None, sources=(), data_file=True, **extra):
    """Write one sidecar, and by default the file it describes.

    Omitting step produces a non-provenance JSON, like SQM. `data_file=False` leaves the
    sidecar without its output, which is the state a mode switch leaves behind and which
    `scan` reports as missing rather than as an ordinary node.
    """
    payload = {"Sources": list(sources), **extra}
    if step is not None:
        payload["step"] = step
    (directory / f"{name}.json").write_text(json.dumps(payload))
    if data_file:
        (directory / f"{name}.snirf").write_bytes(b"")


# ---- what counts as a node ----

def test_json_without_step_is_not_a_node(tmp_path):
    # step is what marks a JSON as provenance; the SQM group records and any other
    # JSON sharing the folder carry none and stay out of the graph.
    _sidecar(tmp_path, "sub-01_desc-sqm_nirs", sci_mean=0.84)
    assert scan(tmp_path) == {}


def test_unreadable_json_is_skipped(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    _sidecar(tmp_path, "sub-01_desc-od_nirs", step="od_conversion")
    assert set(scan(tmp_path)) == {"sub-01_desc-od_nirs"}


def test_source_outside_the_directory_becomes_a_root(tmp_path):
    _sidecar(tmp_path, "sub-01_desc-od_nirs", step="od_conversion",
             sources=["/bids/sub-01/nirs/sub-01_task-tapping_nirs.snirf"])
    nodes = scan(tmp_path)

    root = nodes["sub-01_task-tapping_nirs"]
    assert root.step is None
    assert root.sources == []
    assert root.domain == "input"


def test_empty_directory_yields_no_nodes(tmp_path):
    assert scan(tmp_path) == {}


# ---- labels ----

@pytest.mark.parametrize("key, expected", [
    ("sub-01_task-tapping_desc-preproc_nirs", "preproc"),   # generic suffix, desc wins
    ("sub-01_task-tapping_chromo-hbo_seg-custom_agg-roi_stat-fisherz_relmat",
     "relmat fisherz hbo custom roi"),                      # the measure, then the slices
    ("sub-01_task-tapping_stat-alff_nirsmap", "nirsmap alff"),
    ("sub-01_task-tapping_desc-glm_nirsmap", "nirsmap glm"),
    ("group-G1_task-rest_chromo-hbo_cond-game1_stat-isc_relmat", "relmat isc hbo (game1)"),
    ("sub-01_task-tapping_design", "design"),               # nothing past who it belongs to
    ("design_matrix", "design_matrix"),                     # no BIDS entities at all
])
def test_label_shortens_the_filename(tmp_path, key, expected):
    _sidecar(tmp_path, key, step="a_step")
    assert scan(tmp_path)[key].label == expected


def test_a_collapsed_box_counts_the_conditions_it_holds(tmp_path):
    """Two chromophores times the whole run and two conditions, drawn as one box.

    The condition count reads the `cond-` entity; a member without one is the whole run.
    """
    from fnirs_pipe.qc.common.provenance import simplify

    root = ["/bids/group-G1_task-rest_desc-errts_nirs.snirf"]
    for chromo in ("hbo", "hbr"):
        for cond in ("", "_cond-game1", "_cond-video"):
            _sidecar(tmp_path, f"group-G1_task-rest_chromo-{chromo}{cond}_stat-isc_relmat",
                     step="isc", sources=root)

    (box,) = [n for n in simplify(scan(tmp_path)).values() if n.step == "isc"]
    assert box.label == "relmat isc"
    assert box.state == "hbo, hbr  ·  whole run + 2 conditions"


# ---- domains ----

@pytest.mark.parametrize("key, expected", [
    ("sub-01_desc-od_nirs", "od"),
    ("sub-01_desc-preproc_nirs", "haemo"),
    ("sub-01_alff", "derivative"),
])
def test_domain_follows_the_stage(tmp_path, key, expected):
    _sidecar(tmp_path, key, step="a_step", sources=["/bids/in.snirf"])
    assert scan(tmp_path)[key].domain == expected


# ---- topology ----

def test_depth_is_the_longest_path_not_the_shortest(tmp_path):
    # Diamond with one long arm: d must sit right of c, not beside it. A shortest-path
    # implementation puts d at depth 2 and draws an arrow pointing backwards.
    _sidecar(tmp_path, "a", step="s", sources=["/x/input.snirf"])
    _sidecar(tmp_path, "b", step="s", sources=["/x/input.snirf"])
    _sidecar(tmp_path, "c", step="s", sources=["/x/b.snirf"])
    _sidecar(tmp_path, "d", step="s", sources=["/x/a.snirf", "/x/c.snirf"])

    nodes = scan(tmp_path)
    assert nodes["input"].depth == 0
    assert nodes["a"].depth == 1
    assert nodes["c"].depth == 2
    assert nodes["d"].depth == 3


def test_a_cycle_terminates(tmp_path):
    # Nothing produces cycles today, but the graph is rebuilt from files anyone can
    # edit, and a hang here would stall a run rather than fail it.
    _sidecar(tmp_path, "x", step="s", sources=["/x/y.snirf"])
    _sidecar(tmp_path, "y", step="s", sources=["/x/x.snirf"])

    nodes = scan(tmp_path)
    assert all(isinstance(n.depth, int) for n in nodes.values())


def test_chain_depth_increases_by_one_per_step(tmp_path):
    _sidecar(tmp_path, "sub-01_desc-od_nirs", step="od_conversion", sources=["/bids/in.snirf"])
    _sidecar(tmp_path, "sub-01_desc-preproc_nirs", step="beer_lambert",
             sources=["/out/sub-01_desc-od_nirs.snirf"])

    nodes = scan(tmp_path)
    assert nodes["sub-01_desc-preproc_nirs"].depth == nodes["sub-01_desc-od_nirs"].depth + 1


def test_parameters_are_carried_onto_the_node(tmp_path):
    _sidecar(tmp_path, "sub-01_desc-preproc_nirs", step="beer_lambert",
             parameters={"dpf": [6.0, 6.0]})
    assert scan(tmp_path)["sub-01_desc-preproc_nirs"].params == {"dpf": [6.0, 6.0]}


# ---- mermaid ----

def test_mermaid_ids_drop_characters_mermaid_rejects(tmp_path):
    _sidecar(tmp_path, "sub-01_desc-preproc_nirs", step="beer_lambert",
             sources=["/bids/sub-01_task-tapping_nirs.snirf"])
    text = to_mermaid(scan(tmp_path))

    ids = [line.strip().split("[")[0] for line in text.splitlines() if "[" in line]
    assert ids and all("-" not in i and "." not in i for i in ids)


def test_mermaid_labels_each_edge_with_its_step(tmp_path):
    _sidecar(tmp_path, "sub-01_desc-preproc_nirs", step="beer_lambert",
             sources=["/bids/sub-01_task-tapping_nirs.snirf"])
    assert "-- beer_lambert -->" in to_mermaid(scan(tmp_path))


# ---- settings on the edge ----
#
# Without these the diagram only restates the mode name: it says a bandpass ran, not
# which band. Each step names the few keys that describe it, out of the whole config
# the sidecar carries.

@pytest.mark.parametrize("step, params, expected", [
    ("bandpass", {"high_pass": 0.01, "low_pass": 0.5}, "0.01-0.5 Hz"),
    ("bandpass", {"low_pass": 0.5}, "<0.5 Hz"),
    ("bandpass", {"high_pass": 0.01}, ">0.01 Hz"),
    ("resample", {"sfreq": 2.0}, "2 Hz"),
    ("beer_lambert", {"dpf": [6.0, 6.0]}, "dpf 6, 6"),
    ("sci_pruning", {"sci_threshold": 0.75}, "thr 0.75"),
    ("motion_correction", {"motion_correction": "tddr"}, "tddr"),
    ("glm_residuals", {"hrf_model": "glover", "noise_model": "ar1"}, "glover / ar1"),
    ("bandpass", {}, ""),                        # the keys this step reads are absent
    ("od_conversion", {"dpf": [6.0]}, ""),       # step has no settings worth showing
])
def test_the_edge_carries_the_settings_the_step_used(tmp_path, step, params, expected):
    _sidecar(tmp_path, "out", step=step, sources=["/bids/in.snirf"], parameters=params)
    assert scan(tmp_path)["out"].detail == expected


def test_the_settings_reach_the_mermaid_edge(tmp_path):
    _sidecar(tmp_path, "sub-01_desc-filtered_nirs", step="bandpass",
             sources=["/bids/in.snirf"], parameters={"high_pass": 0.01, "low_pass": 0.5})
    assert "-- bandpass 0.01-0.5 Hz -->" in to_mermaid(scan(tmp_path))


# ---- SQM checkpoints ----
#
# Both checkpoints are named after their step, so the node's data is what tells a raw
# checkpoint from a final one.

_RAW_METRICS = ["sci_mean", "channel_retention_rate", "ch_dist_mean", "psp_mean",
                "cp_mean", "cv_mean_760", "snr_mean", "mean_amp_mean",
                "spike_count", "gvtd_p95"]


def test_an_sqm_node_says_what_it_holds(tmp_path):
    _sidecar(tmp_path, "sub-01_sqm_raw", step="sqm_raw", sources=["/out/in.snirf"],
             data={"n_metrics": len(_RAW_METRICS), "metrics": _RAW_METRICS})

    node = scan(tmp_path)["sub-01_sqm_raw"]
    assert node.detail == "quantitative QC metrics"
    assert node.state == "10 metrics"


def test_the_two_checkpoints_read_differently(tmp_path):
    # the point of the whole field: post replaces the prep checkpoint with a thinner one
    _sidecar(tmp_path, "sub-01_sqm_raw", step="sqm_raw", sources=["/out/in.snirf"],
             data={"n_metrics": len(_RAW_METRICS), "metrics": _RAW_METRICS})
    _sidecar(tmp_path, "sub-01_sqm", step="sqm", sources=["/out/in.snirf"],
             data={"n_metrics": 4, "metrics": ["hbo_hbr_corr_mean", "gcor_hbo",
                                               "gcor_hbr", "pct_data_retained"]})

    nodes = scan(tmp_path)
    assert nodes["sub-01_sqm"].state == "4 metrics"
    assert nodes["sub-01_sqm"].state != nodes["sub-01_sqm_raw"].state


def test_no_node_text_carries_a_newline(tmp_path):
    # the figure shrinks the font to the longest line, so a wrapped line would render every
    # node's text at a size chosen by the worst one
    _sidecar(tmp_path, "sub-01_sqm_raw", step="sqm_raw", sources=["/out/in.snirf"],
             data={"n_metrics": len(_RAW_METRICS), "metrics": _RAW_METRICS})

    for node in scan(tmp_path).values():
        assert "\n" not in node.detail
        assert "\n" not in node.state


def test_no_edge_label_carries_a_newline(tmp_path):
    # a mermaid edge label is one line, so a wrapped detail on the arrow would break it
    _sidecar(tmp_path, "sub-01_desc-od_nirs", step="od_conversion", sources=["/out/in.snirf"])
    _sidecar(tmp_path, "sub-01_desc-filtered_nirs", step="bandpass",
             sources=["sub-01_desc-od_nirs.snirf"],
             parameters={"high_pass": 0.01, "low_pass": 0.5})

    for line in to_mermaid(scan(tmp_path)).splitlines():
        if "-- " in line and "-->" in line:
            assert "\n" not in line.split("-- ")[1].split("-->")[0]


def test_the_node_keeps_the_data_it_was_given(tmp_path):
    # state is compressed to a count and detail says nothing of the names, so the raw dict
    # is the only place they survive
    _sidecar(tmp_path, "sub-01_sqm_raw", step="sqm_raw", sources=["/out/in.snirf"],
             data={"n_metrics": len(_RAW_METRICS), "metrics": _RAW_METRICS})

    assert scan(tmp_path)["sub-01_sqm_raw"].data["metrics"] == _RAW_METRICS


def test_a_root_says_it_is_the_recording(tmp_path):
    # the BIDS input has no sidecar of its own, so the box would carry a bare name
    _sidecar(tmp_path, "sub-01_desc-od_nirs", step="od_conversion",
             sources=["/bids/sub-01_task-tapping_nirs.snirf"])
    assert scan(tmp_path)["sub-01_task-tapping_nirs"].detail == "raw data"


# ---- a checkpoint is a node, not a step ----

def _chain_plus_record(directory):
    """The od -> preproc chain, plus an SQM record naming both of them as sources."""
    _sidecar(directory, "sub-01_desc-od_nirs", step="od_conversion", sources=["/bids/in.snirf"])
    _sidecar(directory, "sub-01_desc-preproc_nirs", step="beer_lambert",
             sources=["sub-01_desc-od_nirs.snirf"])
    _sidecar(directory, "sub-01_desc-sqm_nirs", step="sqm", data_file=False,
             sources=["/bids/in.snirf", "sub-01_desc-od_nirs.snirf",
                      "sub-01_desc-preproc_nirs.snirf"],
             data={"sections": ["raw", "preproc"], "metrics": ["raw_sci_mean"],
                   "n_metrics": 121})


def test_a_checkpoint_draws_no_edges(tmp_path):
    # it measures every stage, so its arrows would cross the whole diagram and bury the
    # chain they are drawn over
    _chain_plus_record(tmp_path)

    mermaid = to_mermaid(scan(tmp_path))
    assert "sub_01_desc_sqm_nirs[" in mermaid
    assert "--> sub_01_desc_sqm_nirs" not in mermaid
    assert "sub_01_desc_od_nirs -- beer_lambert --> sub_01_desc_preproc_nirs" in mermaid


def test_a_checkpoint_still_sits_after_what_it_measured(tmp_path):
    # the edges are dropped at render time only: the sources stay, so the depth does and
    # the node is not mistaken for a root
    _chain_plus_record(tmp_path)

    nodes = scan(tmp_path)
    assert nodes["sub-01_desc-sqm_nirs"].depth > nodes["sub-01_desc-preproc_nirs"].depth
    assert nodes["sub-01_desc-sqm_nirs"].domain != "input"
    assert len(nodes["sub-01_desc-sqm_nirs"].sources) == 3


def test_only_a_checkpoint_loses_its_edges(tmp_path):
    _chain_plus_record(tmp_path)
    assert all(n.checkpoint is (n.step == "sqm") for n in scan(tmp_path).values())


def test_the_table_names_the_record_rather_than_the_metrics(tmp_path):
    # 121 metric names in one cell is a paragraph nobody reads; they stay in the record
    from fnirs_pipe.qc.common.report_shell import provenance_rows

    _chain_plus_record(tmp_path)
    rows = provenance_rows(tmp_path, scope="sub-01")

    row = next(r for r in rows if r["step"] == "sqm")
    assert "metrics" not in row
    assert row["settings"] == "quantitative QC metrics"


# ---- data shape on the node ----

@pytest.mark.parametrize("data, expected", [
    ({"n_channels": 20, "n_bad": 4, "sfreq": 2.0, "duration_s": 120.5}, "16/20 ch · 2 Hz · 120.5 s"),
    ({"n_channels": 20, "n_bad": 0, "sfreq": 10.0, "duration_s": 60.0}, "20 ch · 10 Hz · 60 s"),
    ({"n_channels": 20}, "20 ch"),
    ({}, ""),                                    # a sidecar with no data field
])
def test_the_node_carries_the_shape_of_its_data(tmp_path, data, expected):
    _sidecar(tmp_path, "out", step="bandpass", sources=["/bids/in.snirf"], data=data)
    assert scan(tmp_path)["out"].state == expected


def test_the_shape_reaches_the_mermaid_label(tmp_path):
    _sidecar(tmp_path, "sub-01_desc-preproc_nirs", step="beer_lambert",
             sources=["/bids/in.snirf"],
             data={"n_channels": 8, "n_bad": 2, "sfreq": 10.0, "duration_s": 60.0})
    assert "preproc<br/><small>6/8 ch · 10 Hz · 60 s</small>" in to_mermaid(scan(tmp_path))


def test_a_node_without_data_keeps_a_plain_label(tmp_path):
    _sidecar(tmp_path, "sub-01_desc-preproc_nirs", step="beer_lambert", sources=["/bids/in.snirf"])
    assert "<small>" not in to_mermaid(scan(tmp_path))


# ---- write_provenance ----

def test_write_provenance_writes_nothing_without_sidecars(tmp_path):
    out = tmp_path / "out"
    assert write_provenance(tmp_path, out, figure_namer("sub-01")) == []


def test_cmd_provenance_writes_where_the_report_looks_for_it(tmp_path, capsys):
    # a subject's report embeds the diagram by relative path, so re-rendering has to land
    # on the name the namer gives or an existing report keeps showing the old diagram
    from fnirs_pipe.cli.qc import cmd_provenance

    nirs = tmp_path / "sub-01" / "nirs"
    nirs.mkdir(parents=True)
    _sidecar(nirs, "sub-01_task-tapping_desc-od_nirs", step="od_conversion",
             sources=["/bids/in.snirf"])

    cmd_provenance(tmp_path)

    figures = tmp_path / "sub-01" / "figures"
    run = figure_namer("sub-01_task-tapping")
    assert (figures / run("provenance", extension=".png")).exists()
    assert (figures / run("provenance", extension=".mmd")).exists()


def test_a_group_gets_the_graph_its_dyad_report_links(tmp_path):
    """One graph per task, under the name the dyad report embeds.

    The report links `group-G01_task-rest_desc-provenance`, so the name carries the task.
    """
    from fnirs_pipe.cli.qc import cmd_provenance

    nirs = tmp_path / "group-G01" / "nirs"
    nirs.mkdir(parents=True)
    _sidecar(nirs, "group-G01_task-rest_stat-wtc_relmat", step="hyper_wtc")

    cmd_provenance(tmp_path)

    name = figure_namer("group-G01_task-rest")("provenance", extension=".png")
    assert (tmp_path / "group-G01" / "figures" / name).exists()


def test_a_session_tree_is_found_and_drawn_beside_the_reports(tmp_path):
    # a subject's reports sit in sub-<id>/ whatever its sessions, and so do their figures
    from fnirs_pipe.cli.qc import cmd_provenance

    nirs = tmp_path / "sub-01" / "ses-a" / "nirs"
    nirs.mkdir(parents=True)
    _sidecar(nirs, "sub-01_ses-a_task-tapping_desc-od_nirs", step="od_conversion",
             sources=["/bids/in.snirf"])

    cmd_provenance(tmp_path)

    name = figure_namer("sub-01_ses-a_task-tapping")("provenance", extension=".png")
    assert (tmp_path / "sub-01" / "figures" / name).exists()


def test_write_provenance_writes_png_and_mermaid(tmp_path):
    nirs = tmp_path / "nirs"
    nirs.mkdir()
    _sidecar(nirs, "sub-01_desc-od_nirs", step="od_conversion", sources=["/bids/in.snirf"])
    out = tmp_path / "logs"

    written = write_provenance(nirs, out, figure_namer("sub-01"), title="sub-01")

    assert {p.suffix for p in written} == {".png", ".mmd"}
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


# ---- a sidecar that outlived its file ----

def test_a_sidecar_whose_file_is_gone_is_marked_missing(tmp_path):
    """Switching a tree from rest to denoise leaves the old mode's sidecars behind.

    Nothing removes them, so the graph would otherwise claim steps the run did not perform.
    """
    _sidecar(tmp_path, "sub-01_desc-errts_nirs", step="glm_residuals")
    _sidecar(tmp_path, "sub-01_desc-errtsbroad_nirs", step="glm_residuals", data_file=False)

    nodes = scan(tmp_path)
    assert nodes["sub-01_desc-errts_nirs"].missing is False
    assert nodes["sub-01_desc-errtsbroad_nirs"].missing is True


def test_a_missing_node_is_drawn_rather_than_dropped(tmp_path):
    """It stays in the graph: a step that was run and whose output is gone is worth seeing."""
    _sidecar(tmp_path, "sub-01_desc-od_nirs", step="od_conversion", sources=["/bids/in.snirf"])
    _sidecar(tmp_path, "sub-01_alff", step="alff", sources=["sub-01_desc-od_nirs.snirf"],
             data_file=False)

    mermaid = to_mermaid(scan(tmp_path))
    assert "file missing" in mermaid
    assert ":::missing" in mermaid
    assert "sub_01_desc_od_nirs -- alff --> sub_01_alff" in mermaid


def test_the_sqm_record_is_its_own_file_and_never_missing(tmp_path):
    """It holds the numbers itself, so it has no companion by design."""
    _sidecar(tmp_path, "sub-01_desc-sqm_nirs", step="sqm", data_file=False,
             data={"sections": ["raw"], "metrics": ["raw_sci_mean"], "n_metrics": 1})
    assert scan(tmp_path)["sub-01_desc-sqm_nirs"].missing is False


def test_a_companion_with_any_extension_counts(tmp_path):
    """A table, a figure and a compressed table are all the file the sidecar describes."""
    for name, suffix in (("sub-01_glm_results", ".csv"),
                         ("sub-01_hyper-wtc", ".tsv"),
                         ("sub-01_desc-aux_timeseries", ".tsv.gz")):
        _sidecar(tmp_path, name, step="glm_fit", data_file=False)
        (tmp_path / f"{name}{suffix}").write_bytes(b"")
    assert not any(n.missing for n in scan(tmp_path).values())
