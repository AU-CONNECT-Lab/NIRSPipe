"""A regression names the tables it read besides the haemoglobin, and a contrast its weights."""

import json

import pytest

from nirspipe.cli import run as run_cli

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

GLM = ("--mode", "glm", "--drift-high-pass", "0.008", "--noise-model", "ols",
       "--participant-label", "01", "--no-report", "--skip-bids-validation")
WEIGHTS = {"ca_minus_cb": {"ca": 1, "cb": -1}}


def _sidecars(out, pattern):
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in out.rglob(pattern)}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("glm_input_sources")
    bids, _ = make_fingerprint_dataset(root, blocks=True, aux=True)
    contrast = root / "contrast.toml"
    contrast.write_text('ca_minus_cb = { ca = 1, cb = -1 }\n')
    return root, bids, contrast


@pytest.fixture(scope="module")
def inside(dataset):
    """The raw dataset's own events table, aux regressors and a contrast file."""
    root, bids, contrast = dataset
    events = next(bids.rglob("*_events.tsv"))
    out = root / "inside"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, *GLM, "--events-path",
                  str(events), "--aux-regressors", "--contrast-file", str(contrast)])
    return out


@pytest.fixture(scope="module")
def outside(dataset):
    """An events table that sits in no dataset the output links to."""
    root, bids, _ = dataset
    events = root / "my_events.tsv"
    events.write_text("onset\tduration\ttrial_type\n40\t5\ttrial\n100\t5\ttrial\n160\t5\ttrial\n")
    out = root / "outside"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, *GLM, "--events-path", str(events)])
    return out, events


@pytest.mark.parametrize("pattern", ["*_design.json", "*_desc-glm_nirsmap.json",
                                     "*_desc-contrast_nirsmap.json", "*_desc-errts_nirs.json"])
def test_every_glm_output_names_the_events_and_aux_tables(inside, pattern):
    (side,) = _sidecars(inside, pattern).values()
    sources = side["Sources"]
    assert sources[0].endswith("_desc-preproc_nirs.snirf")
    assert any(s.startswith("bids:raw:") and s.endswith("_events.tsv") for s in sources)
    assert any(s.startswith("bids::") and s.endswith("_desc-aux_timeseries.tsv.gz") for s in sources)


def test_the_contrast_sidecar_carries_the_weights(inside):
    (side,) = _sidecars(inside, "*_desc-contrast_nirsmap.json").values()
    assert side["parameters"]["contrast_weights"] == WEIGHTS


def test_a_table_outside_every_dataset_is_named_by_its_path(outside):
    out, events = outside
    for side in (*_sidecars(out, "*_design.json").values(), *_sidecars(out, "*_desc-errts_nirs.json").values()):
        assert side["external_sources"] == [events.resolve().as_posix()]
        assert not any(s.endswith("my_events.tsv") for s in side["Sources"])


def test_a_regression_without_tables_names_only_its_input(dataset, tmp_path):
    _, bids, _ = dataset
    out = tmp_path / "denoise"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--mode", "denoise",
                  "--short-channel", "mean", "--participant-label", "01", "--no-report",
                  "--skip-bids-validation"])
    (side,) = _sidecars(out, "*_desc-errts_nirs.json").values()
    assert len(side["Sources"]) == 1 and "external_sources" not in side
