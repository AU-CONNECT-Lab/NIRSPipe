"""desc-conditions_qc.tsv: one row per run and condition, under the cohort table's column names."""

import json

import pytest

from nirspipe.cli.run import main
from nirspipe.io.tables import read_table
from nirspipe.qc.subject.record_io import read_record
from tests._raw_qc import SCREENING, raw_qc
from tests._synth import make_bids_dataset

ID_COLUMNS = ("bids_name", "condition", "onset", "duration", "kept_s", "counted_windows")


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    root = tmp_path_factory.mktemp("conditions")
    bids = make_bids_dataset(root, subjects=("01", "02"), tasks=("mixed",))
    pipeline = root / "derivatives"
    main([str(bids), str(pipeline), "participant", *SCREENING, "--mode", "denoise",
          "--by-condition", "--no-report", "--skip-bids-validation"])
    main([str(bids), str(pipeline), "group", "--skip-bids-validation"])
    raw = root / "qc"
    raw_qc(bids, raw)
    return {"pipeline": (pipeline, "_desc-sqm_qc.json"), "raw": (raw, "_desc-sqmraw_qc.json")}


def _records(tree, suffix):
    return {p.name.removesuffix(suffix): read_record(p)
            for p in sorted(tree.glob(f"sub-*/**/nirs/*{suffix}"))}


@pytest.mark.parametrize("kind", ["pipeline", "raw"])
def test_one_row_per_run_and_condition_under_the_cohort_names(trees, kind):
    tree, suffix = trees[kind]
    table = read_table(tree / "desc-conditions_qc.tsv")
    records = _records(tree, suffix)

    expected = {(name, c) for name, rec in records.items() for c in rec["by_condition"]}
    assert len(expected) == 4
    assert set(zip(table["bids_name"], table["condition"])) == expected
    assert len(table) == len(expected)
    cohort = read_table(tree / "desc-subjects_qc.tsv").columns
    assert set(table.columns) - set(ID_COLUMNS) <= set(cohort)


@pytest.mark.parametrize("kind", ["pipeline", "raw"])
def test_each_value_is_its_conditions_under_the_section_its_run_value_sits_in(trees, kind):
    tree, suffix = trees[kind]
    table = read_table(tree / "desc-conditions_qc.tsv").set_index(["bids_name", "condition"])
    for name, rec in _records(tree, suffix).items():
        for condition, entry in rec["by_condition"].items():
            row = table.loc[(name, condition)]
            assert row["onset"] == pytest.approx(entry["window_s"][0])
            assert row["raw_long_sci_win_mean"] == pytest.approx(
                entry["od_by_set"]["long"]["sci_win_mean"])
            assert row["motion_post_gvtd_filt_mean"] == pytest.approx(
                entry["motion_by_set"]["all"]["gvtd_filt_mean"])
            haemo = entry["haemo_by_set"]["all"]
            assert row["preproc_hbo_hbr_corr_mean"] == pytest.approx(haemo["hbo_hbr_corr_mean"])
            assert row["rawhaemo_hbo_hbr_corr_mean"] == pytest.approx(
                haemo["hbo_hbr_corr_mean_uncorrected"])
            if kind == "pipeline":
                # the regression's after side is the run's errts stage
                assert row["errts_gcor_hbo"] == pytest.approx(haemo["gcor_hbo_postreg"])


@pytest.mark.parametrize("kind", ["pipeline", "raw"])
def test_every_column_is_described_and_none_says_rejected(trees, kind):
    tree, _ = trees[kind]
    sidecar = json.loads((tree / "desc-conditions_qc.json").read_text(encoding="utf-8"))
    for column in ID_COLUMNS + ("raw_channel_retention_rate", "raw_long_sci_win_mean"):
        assert sidecar[column]["Description"], column
    assert "passing in this condition" in sidecar["raw_channel_retention_rate"]["Description"]
    assert not [c for c, e in sidecar.items() if "reject" in e["Description"].lower()]
