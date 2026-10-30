"""Raw QC measures through the pipeline's own code, keeps its stage files apart, and leaves a shared tree's analysis alone."""

import hashlib

import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.cli import run as run_cli
from nirspipe.io.derivatives import read_json
from nirspipe.qc.subject.record_io import read_record
from nirspipe.qc.subject.sqm_record import SECTIONS

from tests._fingerprint import DPF, make_fingerprint_dataset

SCREENING = ("--dpf", *(str(v) for v in DPF), "--sci-threshold", "0.75",
             "--psp-threshold", "0.12", "--min-good-frac", "0.85",
             "--cardiac-l-freq", "0.8", "--cardiac-h-freq", "1.9",
             "--resp-l-freq", "0.15", "--resp-h-freq", "0.45", "--skip-bids-validation")
# reads the rejected channels when it corrects, which raw QC once did not pass it
MOTION = ("--motion-correction", "tpca")


def _digests(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*_desc-*") if p.is_file() and "_desc-sqmraw" not in p.name
            and "_desc-raw" not in p.name and "logs" not in p.parts and "figures" not in p.parts}


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stages")
    bids, truth = make_fingerprint_dataset(tmp, blocks=True)
    marked = next(p.name for p in truth.long_pairs if not p.bad)
    marks = tmp / "marks.tsv"
    marks.write_text(f"bad_channels\n{marked}\n", encoding="utf-8")
    flags = (*SCREENING, *MOTION, "--bad-channels", str(marks))
    run_cli.main([str(bids), str(tmp / "pipe"), "participant", *flags, "--no-report"])
    qc_cli.main([str(bids), str(tmp / "qc"), "participant", "--participant-label", "01", *flags])
    shared = tmp / "shared"
    run_cli.main([str(bids), str(shared), "participant", *flags, "--no-report"])
    before = _digests(shared)
    qc_cli.main([str(bids), str(shared), "participant", "--participant-label", "01", *flags])
    return {"pipe": tmp / "pipe", "qc": tmp / "qc", "shared": shared, "before": before}


def _record(tree, desc):
    (path,) = tree.rglob(f"*_desc-{desc}_qc.json")
    return read_record(path)


def test_the_stage_files_sit_in_their_own_dataset(trees):
    stages = trees["qc"] / "stages"
    assert read_json(stages / "dataset_description.json")["GeneratedBy"][0]["Name"] == "nirspipe-qc"
    assert sorted(p.name.split("_desc-")[1].split("_")[0] for p in stages.rglob("*.snirf")) == [
        "motcorrected", "od", "preproc", "sci"]


def test_nothing_an_analysis_reads_sits_beside_the_record(trees):
    assert not list((trees["qc"] / "sub-01").rglob("*.snirf"))


def test_the_tree_tells_a_validator_to_skip_them(trees):
    assert "stages" in (trees["qc"] / ".bidsignore").read_text(encoding="utf-8").split()


def test_the_record_names_the_recording_alone(trees):
    sources = _record(trees["qc"], "sqmraw")["Sources"]
    assert len(sources) == 1 and sources[0].startswith("bids:raw:")
    assert "_desc-" not in sources[0]


def test_raw_qc_measures_what_the_pipeline_measures(trees):
    raw, pipeline = _record(trees["qc"], "sqmraw"), _record(trees["pipe"], "sqm")
    for section in ("raw", "raw_long", "raw_short", "motion", "motion_long", "motion_post",
                    "preproc", "preproc_long", "rawhaemo", "rawhaemo_long", "excluded"):
        assert section in SECTIONS or section.startswith("rawhaemo")
        for key, value in pipeline[section].items():
            if isinstance(value, float):
                assert raw[section][key] == pytest.approx(value, abs=1e-9), (section, key)
            else:
                assert raw[section][key] == value, (section, key)


def test_each_condition_is_measured_as_the_pipeline_measures_it(trees):
    raw, pipeline = _record(trees["qc"], "sqmraw"), _record(trees["pipe"], "sqm")
    assert raw["by_condition"].keys() == pipeline["by_condition"].keys()
    for label, entry in pipeline["by_condition"].items():
        assert raw["by_condition"][label]["bad_channels"] == entry["bad_channels"]
        for name, row in entry["haemo_by_set"].items():
            assert raw["by_condition"][label]["haemo_by_set"][name] == pytest.approx(
                row, abs=1e-9, nan_ok=True)


def test_a_shared_tree_keeps_the_analysis_files_byte_for_byte(trees):
    after = _digests(trees["shared"])
    assert {k: after.get(k) for k in trees["before"] if "stages/" not in k} == trees["before"]


def test_a_shared_tree_holds_both_records(trees):
    assert _record(trees["shared"], "sqm") and _record(trees["shared"], "sqmraw")
