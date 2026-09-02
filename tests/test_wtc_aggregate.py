"""Merging the per-dyad WTC tables, and the two merges it refuses to perform.

The concatenation itself is not where the risk is. The risk is that a coherence value carries
no record of the band it was averaged over once it is a row in a combined table, so mixing
bands produces a file that is wrong in a way nothing downstream can detect. Most of these
tests are about that refusal rather than about the merge.
"""

import json

import pandas as pd
import pytest

from fnirs_pipe.qc.wtc_aggregate import aggregate_wtc, write_aggregate_wtc


def _table(root, group_id, task, kind="wtc", band=(0.01, 0.1), mask_coi=True,
           crossed=False, sidecar=True):
    """One dyad's band-mean table, laid out the way hyper-post lays it out."""
    directory = root / f"group-{group_id}" / "nirs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"group-{group_id}_task-{task}_hyper-{kind}.tsv"

    data = {
        "sub1": [f"{group_id}a", f"{group_id}a"],
        "sub2": [f"{group_id}b", f"{group_id}b"],
        "label": ["S1_D1 hbo", "S1_D2 hbo"],
        "coherence": [0.5, 0.6],
        "n_valid_frac": [0.9, 0.8],
    }
    if crossed:
        data["label2"] = ["S2_D1 hbo", "S2_D2 hbo"]
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)

    if sidecar:
        path.with_suffix(".json").write_text(json.dumps({"parameters": {
            "band_fmin": band[0], "band_fmax": band[1], "mask_coi": mask_coi}}))
    return path


# ---- the merge ----

def test_the_filename_entities_become_columns(tmp_path):
    """Nothing about a row should depend on which file it came from."""
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "rest")
    _table(tmp_path, "01", "tapping")

    merged = aggregate_wtc(tmp_path)
    assert list(merged.columns[:2]) == ["group_id", "task"]
    assert set(merged["group_id"]) == {"01", "02"}
    assert set(merged["task"]) == {"rest", "tapping"}
    assert len(merged) == 6


def test_the_order_does_not_depend_on_the_filesystem(tmp_path_factory):
    """Two trees with the same tables written in a different order merge identically."""
    first = tmp_path_factory.mktemp("first")
    for group_id, task in [("01", "rest"), ("02", "rest"), ("02", "tapping")]:
        _table(first, group_id, task)

    second = tmp_path_factory.mktemp("second")
    for group_id, task in [("02", "tapping"), ("01", "rest"), ("02", "rest")]:
        _table(second, group_id, task)

    pd.testing.assert_frame_equal(aggregate_wtc(first), aggregate_wtc(second))


def test_nothing_to_merge_is_not_an_error(tmp_path):
    """A study that never ran WTC should not look like a failure."""
    assert aggregate_wtc(tmp_path).empty
    assert write_aggregate_wtc(tmp_path) is None


def test_unrelated_tables_are_left_alone(tmp_path):
    _table(tmp_path, "01", "rest")
    (tmp_path / "group-01" / "nirs" / "something_else.tsv").write_text("a\tb\n1\t2\n")
    assert len(aggregate_wtc(tmp_path)) == 2


def test_a_second_run_does_not_swallow_its_own_output(tmp_path):
    """The merged table sits beside its inputs; re-running must not fold it back in."""
    _table(tmp_path, "01", "rest")
    write_aggregate_wtc(tmp_path)
    assert len(aggregate_wtc(tmp_path)) == 2


# ---- what it refuses ----

@pytest.mark.parametrize("odd_one_out, key", [
    ({"band": (0.01, 0.2)}, "band_fmax"),
    ({"band": (0.02, 0.1)}, "band_fmin"),
    ({"mask_coi": False},   "mask_coi"),
])
def test_a_disagreement_about_the_band_is_a_stop(tmp_path, odd_one_out, key):
    """Warning and merging anyway would put the mistake somewhere nothing can find it."""
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "rest", **odd_one_out)

    with pytest.raises(ValueError, match=key):
        aggregate_wtc(tmp_path)


def test_the_refusal_names_which_file_disagreed(tmp_path):
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "rest", band=(0.01, 0.2))

    with pytest.raises(ValueError) as excinfo:
        aggregate_wtc(tmp_path)
    message = str(excinfo.value)
    assert "group-01_task-rest_hyper-wtc.tsv" in message
    assert "group-02_task-rest_hyper-wtc.tsv" in message


def test_a_table_with_no_sidecar_carries_no_opinion(tmp_path):
    """It cannot be checked, so it cannot object; the merge proceeds rather than failing."""
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "rest", sidecar=False)
    assert len(aggregate_wtc(tmp_path)) == 4


def test_crossed_and_homologous_roi_tables_do_not_mix(tmp_path):
    """An empty label2 reads as missing data rather than as a different analysis."""
    _table(tmp_path, "01", "rest", kind="wtc-roichan", crossed=True)
    _table(tmp_path, "02", "rest", kind="wtc-roichan", crossed=False)

    with pytest.raises(ValueError, match="label2"):
        aggregate_wtc(tmp_path, kind="wtc-roichan")


def test_all_crossed_roi_tables_merge_and_keep_label2(tmp_path):
    _table(tmp_path, "01", "rest", kind="wtc-roichan", crossed=True)
    _table(tmp_path, "02", "rest", kind="wtc-roichan", crossed=True)

    merged = aggregate_wtc(tmp_path, kind="wtc-roichan")
    assert "label2" in merged.columns
    assert merged["label2"].notna().all()


def test_an_unknown_kind_is_refused(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        aggregate_wtc(tmp_path, kind="coherence")


# ---- what reaches disk ----

def test_the_two_kinds_land_in_separate_files(tmp_path):
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "01", "rest", kind="wtc-roichan")

    assert write_aggregate_wtc(tmp_path, "wtc").name == "group_hyper_wtc.tsv"
    assert write_aggregate_wtc(tmp_path, "wtc-roichan").name == "group_hyper_wtc_roichan.tsv"


def test_the_sidecar_says_what_went_in(tmp_path):
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "rest")
    _table(tmp_path, "02", "tapping")

    out = write_aggregate_wtc(tmp_path)
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["step"] == "group_hyper_wtc"
    assert len(sidecar["Sources"]) == 3
    assert sidecar["parameters"]["n_tables"] == 3
    assert sidecar["parameters"]["n_dyads"] == 2
    assert sidecar["parameters"]["tasks"] == ["rest", "tapping"]
    # the band is uniform by then, so recording it makes the merged file self-describing
    assert sidecar["parameters"]["band_fmin"] == 0.01
    assert sidecar["parameters"]["band_fmax"] == 0.1


def test_the_written_table_round_trips(tmp_path):
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "tapping")

    out = write_aggregate_wtc(tmp_path)
    from_disk = pd.read_csv(out, sep="\t", dtype={"group_id": str})
    pd.testing.assert_frame_equal(from_disk, aggregate_wtc(tmp_path))
