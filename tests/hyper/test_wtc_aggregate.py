"""Merging the per-dyad WTC tables, and the two merges it refuses to perform.

The concatenation itself is not where the risk is. The risk is that a coherence value carries
no record of the band it was averaged over once it is a row in a combined table, so mixing
bands produces a file that is wrong in a way nothing downstream can detect. Most of these
tests are about that refusal rather than about the merge.

What counts as one kind is the entity set a file carries besides its group and its task, so
the discovery half is tested through `merge_kinds` rather than by naming kinds here: a table
added to the pipeline joins a merge without anything in this module being told about it.
"""

import json

import pandas as pd
import pytest

from fnirs_pipe.io.derivatives import group_output_path
from fnirs_pipe.pipeline.hyper.wtc_aggregate import (
    aggregate_wtc, merge_kinds, write_aggregate_wtc, write_all_aggregates,
)

WTC = {"statistic": "wtc"}
ROI = {"segmentation": "custom", "aggregation": "roi", "statistic": "wtc"}


def _table(root, group_id, task, entities=None, band=(0.01, 0.1), mask_coi=True,
           crossed=False, sidecar=True, chroma=None):
    """One dyad's band-mean table, laid out the way hyper-post lays it out."""
    path = group_output_path(root, group_id, {"task": task, **(entities or WTC)},
                             "relmat", ".tsv")

    data = {
        "sub1": [f"{group_id}a", f"{group_id}a"],
        "sub2": [f"{group_id}b", f"{group_id}b"],
        "label": ["S1_D1 hbo", "S1_D2 hbo"],
        "coherence": [0.5, 0.6],
        "n_valid_frac": [0.9, 0.8],
    }
    if crossed:
        data["label2"] = ["S2_D1 hbo", "S2_D2 hbo"]
    df = pd.DataFrame(data)
    if chroma:
        df = pd.concat([df.assign(chromophore=c) for c in chroma], ignore_index=True)
        df.insert(0, "chromophore", df.pop("chromophore"))
    df.to_csv(path, sep="\t", index=False)

    if sidecar:
        path.with_suffix(".json").write_text(json.dumps({"parameters": {
            "band_fmin": band[0], "band_fmax": band[1], "mask_coi": mask_coi}}))
    return path


def _sources(root, entities=None):
    """The files one kind's merge would take, as the command discovers them."""
    key = tuple(sorted((k, str(v)) for k, v in (entities or WTC).items()))
    return merge_kinds(root).get(key, [])


def _merge(root, entities=None):
    return aggregate_wtc(root, _sources(root, entities))


# ---- the merge ----

def test_the_filename_entities_become_columns(tmp_path):
    """Nothing about a row should depend on which file it came from."""
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "rest")
    _table(tmp_path, "01", "tapping")

    merged = _merge(tmp_path)
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

    pd.testing.assert_frame_equal(_merge(first), _merge(second))


def test_nothing_to_merge_is_not_an_error(tmp_path):
    """A study that never ran WTC should not look like a failure."""
    assert merge_kinds(tmp_path) == {}
    assert write_all_aggregates(tmp_path) == []


def test_unrelated_tables_are_left_alone(tmp_path):
    _table(tmp_path, "01", "rest")
    (tmp_path / "group-01" / "nirs" / "something_else.tsv").write_text("a\tb\n1\t2\n")
    assert len(_merge(tmp_path)) == 2


def test_a_second_run_does_not_swallow_its_own_output(tmp_path):
    """The merged table sits beside its inputs; re-running must not fold it back in.

    It is at the root and carries no group- entity, which is exactly what marks it as the
    merge rather than one of its inputs.
    """
    _table(tmp_path, "01", "rest")
    write_all_aggregates(tmp_path)
    assert len(_merge(tmp_path)) == 2


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
        _merge(tmp_path)


def test_the_refusal_names_which_file_disagreed(tmp_path):
    first = _table(tmp_path, "01", "rest")
    second = _table(tmp_path, "02", "rest", band=(0.01, 0.2))

    with pytest.raises(ValueError) as excinfo:
        _merge(tmp_path)
    message = str(excinfo.value)
    assert first.name in message and second.name in message


def test_a_table_with_no_sidecar_carries_no_opinion(tmp_path):
    """It cannot be checked, so it cannot object; the merge proceeds rather than failing."""
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "rest", sidecar=False)
    assert len(_merge(tmp_path)) == 4


def test_crossed_and_homologous_roi_tables_do_not_mix(tmp_path):
    """An empty label2 reads as missing data rather than as a different analysis."""
    _table(tmp_path, "01", "rest", ROI, crossed=True)
    _table(tmp_path, "02", "rest", ROI, crossed=False)

    with pytest.raises(ValueError, match="label2"):
        _merge(tmp_path, ROI)


def test_all_crossed_roi_tables_merge_and_keep_label2(tmp_path):
    _table(tmp_path, "01", "rest", ROI, crossed=True)
    _table(tmp_path, "02", "rest", ROI, crossed=True)

    merged = _merge(tmp_path, ROI)
    assert "label2" in merged.columns
    assert merged["label2"].notna().all()


# ---- which files are one kind ----

def test_two_tables_differing_only_in_group_and_task_are_one_kind(tmp_path):
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "tapping")
    assert len(merge_kinds(tmp_path)) == 1


def test_an_entity_beyond_group_and_task_makes_a_second_kind(tmp_path):
    """The whole-run table and the ROI one describe different things and never merge."""
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "01", "rest", ROI)
    assert len(merge_kinds(tmp_path)) == 2


def test_the_draws_are_not_merged(tmp_path):
    """They are the same null at full detail, so merging them would double every row."""
    _table(tmp_path, "01", "rest", {"nulldist": "pair", "statistic": "wtc"})
    _table(tmp_path, "01", "rest",
           {"nulldist": "pair", "statistic": "wtc", "desc": "draws"})
    assert len(merge_kinds(tmp_path)) == 1


# ---- what reaches disk ----

def test_the_two_kinds_land_in_separate_files(tmp_path):
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "01", "rest", ROI)

    written = {p.name for p in write_all_aggregates(tmp_path)}
    assert written == {"stat-wtc_relmat.tsv",
                       "seg-custom_agg-roi_stat-wtc_relmat.tsv"}


def test_the_merged_name_is_the_inputs_name_without_group_and_task(tmp_path):
    """The rule a merge follows: drop what varied, keep what agreed. Being at the root with
    no analysis unit in the name is what marks a table as cross-dyad."""
    _table(tmp_path, "01", "rest", ROI)
    _table(tmp_path, "02", "tapping", ROI)

    out, = write_all_aggregates(tmp_path)
    assert out.parent == tmp_path
    assert out.name == "seg-custom_agg-roi_stat-wtc_relmat.tsv"


def test_the_sidecar_says_what_went_in(tmp_path):
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "rest")
    _table(tmp_path, "02", "tapping")

    out = write_aggregate_wtc(tmp_path, _sources(tmp_path))
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["step"] == "hyper_merge"
    assert len(sidecar["Sources"]) == 3
    assert sidecar["parameters"]["n_tables"] == 3
    assert sidecar["parameters"]["n_dyads"] == 2
    assert sidecar["parameters"]["tasks"] == ["rest", "tapping"]
    # what the merged file is of, so its own name does not have to be parsed to find out
    assert sidecar["parameters"]["entities"] == {"statistic": "wtc"}
    # the band is uniform by then, so recording it makes the merged file self-describing
    assert sidecar["parameters"]["band_fmin"] == 0.01
    assert sidecar["parameters"]["band_fmax"] == 0.1


def test_the_written_table_round_trips(tmp_path):
    _table(tmp_path, "01", "rest")
    _table(tmp_path, "02", "tapping")

    out = write_aggregate_wtc(tmp_path, _sources(tmp_path))
    from_disk = pd.read_csv(out, sep="\t", dtype={"group_id": str})
    pd.testing.assert_frame_equal(from_disk, _merge(tmp_path))


# ---- chromophores ----
# A chromophore is a row label rather than a column-wide parameter, so mixing does not
# corrupt anything the way a mixed band does. It is warned about, not refused.

def test_the_chromophore_is_a_sort_key(tmp_path):
    """It is the coarsest grouping in the table, so the merged file groups by it."""
    _table(tmp_path, "01", "rest", chroma=("hbr", "hbo"))
    merged = _merge(tmp_path)
    assert "chromophore" in merged.columns
    assert list(merged["chromophore"]) == ["hbo", "hbo", "hbr", "hbr"]


def test_tables_carrying_different_chromophores_merge_with_a_warning(tmp_path, caplog):
    """One dyad run on both and another on HbO alone is a real study state, and the rows
    stay separable, so refusing would be wrong. What a group model will not notice on its
    own is the unequal dyad count per chromophore."""
    import logging

    _table(tmp_path, "01", "rest", chroma=("hbo", "hbr"))
    _table(tmp_path, "02", "rest", chroma=("hbo",))
    with caplog.at_level(logging.WARNING):
        merged = _merge(tmp_path)
    assert merged["chromophore"].value_counts().to_dict() == {"hbo": 4, "hbr": 2}
    assert "same chromophores" in caplog.text


def test_a_table_predating_the_column_merges_with_a_warning(tmp_path, caplog):
    import logging

    _table(tmp_path, "01", "rest", chroma=("hbo", "hbr"))
    _table(tmp_path, "02", "rest")
    with caplog.at_level(logging.WARNING):
        _merge(tmp_path)
    assert "no chromophore column" in caplog.text


def test_uniform_chromophores_warn_about_nothing(tmp_path, caplog):
    import logging

    _table(tmp_path, "01", "rest", chroma=("hbo", "hbr"))
    _table(tmp_path, "02", "rest", chroma=("hbo", "hbr"))
    with caplog.at_level(logging.WARNING):
        _merge(tmp_path)
    assert "same chromophores" not in caplog.text
