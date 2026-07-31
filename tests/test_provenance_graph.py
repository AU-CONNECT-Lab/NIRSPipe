"""Provenance graph reconstruction from sidecars on disk.

Everything goes through the public entry points (scan / to_mermaid /
write_provenance) rather than the private helpers, so a refactor of the naming
or layout internals does not turn these red while behaviour is unchanged.
"""

import json

import pytest

from fnirs_pipe.qc.provenance import scan, to_mermaid, write_provenance


def _sidecar(directory, name, step=None, sources=(), **extra):
    """Write one sidecar. Omitting step produces a non-provenance JSON, like SQM."""
    payload = {"Sources": list(sources), **extra}
    if step is not None:
        payload["step"] = step
    (directory / f"{name}.json").write_text(json.dumps(payload))


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
    ("sub-01_task-tapping_desc-hbo_fc", "fc (hbo)"),        # both informative
    ("sub-01_task-tapping_alff", "alff"),                   # no desc
    ("design_matrix", "design_matrix"),                     # no BIDS entities at all
])
def test_label_shortens_the_filename(tmp_path, key, expected):
    _sidecar(tmp_path, key, step="a_step")
    assert scan(tmp_path)[key].label == expected


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
# Both checkpoints are named after their step, so before this they read "sqm" and nothing
# else, and the graph could not tell a 27-metric raw checkpoint from a 4-metric final one.

_RAW_METRICS = ["sci_mean", "channel_retention_rate", "ch_dist_mean", "psp_mean",
                "cp_mean", "cv_mean_760", "snr_mean", "mean_amp_mean",
                "spike_count", "gvtd_p95"]


def test_an_sqm_node_names_the_metric_families(tmp_path):
    _sidecar(tmp_path, "sub-01_sqm_raw", step="sqm_raw", sources=["/out/in.snirf"],
             data={"n_metrics": len(_RAW_METRICS), "metrics": _RAW_METRICS})

    node = scan(tmp_path)["sub-01_sqm_raw"]
    assert node.detail.replace("\n", " ") == (
        "sci retention distance psp cp cv snr amplitude spike gvtd")
    assert node.state == "10 metrics"


def test_the_two_checkpoints_read_differently(tmp_path):
    # the point of the whole field: post replaces the prep checkpoint with a thinner one
    _sidecar(tmp_path, "sub-01_sqm_raw", step="sqm_raw", sources=["/out/in.snirf"],
             data={"n_metrics": len(_RAW_METRICS), "metrics": _RAW_METRICS})
    _sidecar(tmp_path, "sub-01_sqm", step="sqm", sources=["/out/in.snirf"],
             data={"n_metrics": 4, "metrics": ["hbo_hbr_corr_mean", "gcor_hbo",
                                               "gcor_hbr", "pct_data_retained"]})

    nodes = scan(tmp_path)
    assert nodes["sub-01_sqm"].detail == "hbo/hbr gcor retention"
    assert nodes["sub-01_sqm"].detail != nodes["sub-01_sqm_raw"].detail


def test_a_long_family_list_is_wrapped_not_run_on(tmp_path):
    # the figure shrinks the font to the longest line, so an unwrapped list would render
    # every node's text at a size chosen by the worst one
    _sidecar(tmp_path, "sub-01_sqm_raw", step="sqm_raw", sources=["/out/in.snirf"],
             data={"n_metrics": len(_RAW_METRICS), "metrics": _RAW_METRICS})

    lines = scan(tmp_path)["sub-01_sqm_raw"].detail.split("\n")
    assert len(lines) > 1
    assert all(len(line) <= 24 for line in lines)


def test_the_wrapped_detail_stays_on_one_mermaid_line(tmp_path):
    _sidecar(tmp_path, "sub-01_sqm_raw", step="sqm_raw", sources=["/out/in.snirf"],
             data={"n_metrics": len(_RAW_METRICS), "metrics": _RAW_METRICS})

    text = to_mermaid(scan(tmp_path))
    assert "\n" not in text.split("-- sqm_raw ")[1].split("-->")[0]


def test_the_node_keeps_the_data_it_was_given(tmp_path):
    # the report's table shows the full metric names, which only the raw dict has: state
    # is already compressed to a count and detail to the families
    _sidecar(tmp_path, "sub-01_sqm_raw", step="sqm_raw", sources=["/out/in.snirf"],
             data={"n_metrics": len(_RAW_METRICS), "metrics": _RAW_METRICS})

    assert scan(tmp_path)["sub-01_sqm_raw"].data["metrics"] == _RAW_METRICS


def test_an_sqm_sidecar_without_metrics_says_nothing(tmp_path):
    # sidecars written before the field existed
    _sidecar(tmp_path, "sub-01_sqm", step="sqm", sources=["/out/in.snirf"])
    assert scan(tmp_path)["sub-01_sqm"].detail == ""


# ---- data shape on the node ----

@pytest.mark.parametrize("data, expected", [
    ({"n_channels": 56, "n_bad": 16, "sfreq": 2.0, "duration_s": 595.2}, "40/56 ch · 2 Hz · 595.2 s"),
    ({"n_channels": 56, "n_bad": 0, "sfreq": 10.0, "duration_s": 60.0}, "56 ch · 10 Hz · 60 s"),
    ({"n_channels": 56}, "56 ch"),
    ({}, ""),                                    # sidecar written before the field existed
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
    assert write_provenance(tmp_path, out, stem="sub-01_provenance") == []


def test_cmd_provenance_writes_where_the_report_looks_for_it(tmp_path, capsys):
    # the QC report embeds figures/provenance.png by relative path, so re-rendering has
    # to land on that exact name or an existing report keeps showing the old diagram
    from fnirs_pipe.cli.qc import cmd_provenance

    nirs = tmp_path / "sub-01" / "nirs"
    nirs.mkdir(parents=True)
    _sidecar(nirs, "sub-01_desc-od_nirs", step="od_conversion", sources=["/bids/in.snirf"])

    cmd_provenance(tmp_path)

    figures = tmp_path / "sub-01" / "figures"
    assert (figures / "provenance.png").exists()
    assert (figures / "provenance.mmd").exists()


def test_write_provenance_writes_png_and_mermaid(tmp_path):
    nirs = tmp_path / "nirs"
    nirs.mkdir()
    _sidecar(nirs, "sub-01_desc-od_nirs", step="od_conversion", sources=["/bids/in.snirf"])
    out = tmp_path / "logs"

    written = write_provenance(nirs, out, stem="sub-01_provenance", title="sub-01")

    assert {p.suffix for p in written} == {".png", ".mmd"}
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
