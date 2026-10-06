"""Output contract: every derivative names what produced it and what it came from.

Runs the real pipeline over a synthetic BIDS dataset and checks the sidecars it
leaves behind. This fails whenever an output is added without wiring its
provenance, which is silent at runtime: the file is written, the run succeeds,
and only the graph quietly loses a node.

The prep and post passes are deliberately separate, mirroring the CLI: post
re-reads the desc-preproc file from disk and takes that path as its source, so
the chain joins up across the two Recorder instances.
"""

import json

import pytest

from fnirs_pipe.io.derivatives import LINK_RAW, write_dataset_description
from fnirs_pipe.io.snirf import read_snirf
from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep
from fnirs_pipe.qc.common.provenance import scan

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)

_POST = {
    "denoise": dict(high_pass=0.01, low_pass=0.5),
    # stim_dur, not the annotation durations: the GLM takes onsets from the
    # annotations but requires the duration to be stated explicitly.
    "glm": dict(high_pass=0.01, low_pass=0.5, hrf_model="spm", noise_model="ar1",
                drift_model="cosine", drift_high_pass=0.01, drift_order=1, stim_dur=10.0),
    "rest": dict(high_pass=0.01, low_pass=0.1,
                 drift_model="cosine", drift_high_pass=0.01, drift_order=1),
}

# Stages each mode must leave on disk, beyond the four prep steps.
_EXTRA_STAGES = {
    "denoise": {"filtered"},
    "glm": {"filtered", "errts"},
    "rest": {"filtered", "errts", "errtsbroad"},
}


def _run(bids_dir, out_dir, mode, subject="01", task="tapping"):
    source = bids_dir / f"sub-{subject}" / "nirs" / f"sub-{subject}_task-{task}_nirs.snirf"
    entities = {"task": task}
    # what the command writes first, so the raw input has a BIDS URI to be named by
    write_dataset_description(out_dir, source=bids_dir, link=LINK_RAW)

    prep_config = PrepConfig(subject=subject, dpf=[6.0, 6.0], sci_threshold=0.8,
                             motion_correction="tddr", **_BANDS)
    run_prep(read_snirf(source), prep_config, output_dir=out_dir,
             source_entities=entities, source_path=source)

    preproc = next((out_dir / f"sub-{subject}" / "nirs").glob("*desc-preproc_nirs.snirf"))
    post_config = PostConfig(subject=subject, **_BANDS, **_POST[mode])
    run_post(read_snirf(preproc), post_config, output_dir=out_dir, mode=mode,
             source_entities=entities, source_path=preproc)

    return out_dir / f"sub-{subject}" / "nirs"


@pytest.fixture(scope="session", params=["denoise", "glm", "rest"])
def derivatives(request, mini_bids, tmp_path_factory):
    """(mode, nirs_dir) for one completed run per postprocessing mode."""
    mode = request.param
    out_dir = tmp_path_factory.mktemp(f"deriv_{mode}")
    return mode, _run(mini_bids, out_dir, mode)


def _provenance_sidecars(nirs_dir):
    """Every sidecar that claims to describe provenance, as (name, contents)."""
    out = []
    for path in sorted(nirs_dir.glob("*.json")):
        meta = json.loads(path.read_text())
        if "step" in meta:            # SQM and other sidecars are not provenance records
            out.append((path.name, meta))
    return out


def test_the_run_produced_provenance_sidecars(derivatives):
    _, nirs_dir = derivatives
    assert _provenance_sidecars(nirs_dir), f"no provenance sidecars under {nirs_dir}"


def test_every_output_names_the_step_that_produced_it(derivatives):
    _, nirs_dir = derivatives
    missing = [name for name, meta in _provenance_sidecars(nirs_dir) if not meta.get("step")]
    assert not missing, f"sidecars with an empty step: {missing}"


def test_every_output_names_its_sources(derivatives):
    _, nirs_dir = derivatives
    missing = [name for name, meta in _provenance_sidecars(nirs_dir) if not meta.get("Sources")]
    assert not missing, f"sidecars with no Sources: {missing}"


def test_the_graph_has_exactly_one_root(derivatives):
    # More than one root means a chain was broken and an output was credited to
    # something outside the run instead of to the file it actually came from.
    _, nirs_dir = derivatives
    roots = sorted(k for k, node in scan(nirs_dir).items() if not node.sources)
    assert len(roots) == 1, f"expected a single input, got roots: {roots}"


def test_the_root_is_the_bids_input(derivatives):
    _, nirs_dir = derivatives
    root = next(k for k, node in scan(nirs_dir).items() if not node.sources)
    assert "desc-" not in root, f"root should be the unprocessed input, got {root!r}"


def test_no_node_is_orphaned(derivatives):
    # An orphan is a node whose declared source is not in the graph, so following
    # Sources from it dead-ends instead of reaching the input.
    _, nirs_dir = derivatives
    nodes = scan(nirs_dir)
    dangling = {key: [s for s in node.sources if s not in nodes]
                for key, node in nodes.items()}
    assert not any(dangling.values()), f"sources not in the graph: { {k: v for k, v in dangling.items() if v} }"


def test_every_node_reaches_the_input(derivatives):
    _, nirs_dir = derivatives
    nodes = scan(nirs_dir)
    root = next(k for k, node in nodes.items() if not node.sources)

    def reaches(key, seen=frozenset()):
        if key == root:
            return True
        if key in seen:
            return False
        return any(reaches(s, seen | {key}) for s in nodes[key].sources)

    stranded = sorted(k for k in nodes if not reaches(k))
    assert not stranded, f"nodes that do not trace back to the input: {stranded}"


def test_the_mode_wrote_the_stages_it_should_have(derivatives):
    mode, nirs_dir = derivatives
    written = {p.name.split("desc-")[1].split("_")[0]
               for p in nirs_dir.glob("*desc-*_nirs.snirf")}
    expected = {"od", "sci", "motcorrected", "preproc"} | _EXTRA_STAGES[mode]
    assert expected <= written, f"{mode}: missing {sorted(expected - written)}"


def test_prep_outputs_form_a_chain_not_a_fan(derivatives):
    # Each prep step must name the previous step's file, not the BIDS input. A fan
    # means the stage-to-file mapping was lost and everything fell back to the root.
    _, nirs_dir = derivatives
    nodes = scan(nirs_dir)
    by_label = {node.label: node for node in nodes.values()}

    for child, parent in [("sci", "od"), ("motcorrected", "sci"), ("preproc", "motcorrected")]:
        assert child in by_label, f"missing {child}"
        parents = {nodes[s].label for s in by_label[child].sources}
        assert parents == {parent}, f"{child} came from {parents}, expected {{{parent!r}}}"
