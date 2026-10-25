"""The run report's per-channel table names each pair's region, as the run's channel table records it.

Each case writes the channel table by hand, so the expected region is the one written there
and not one the package computed. Views that have no channel table (raw QC, the dyad raw
report) and the per-channel QC TSV never carry the column.
"""

from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.boilerplate.vocabulary import MISSING_VALUE
from nirspipe.qc.common.channel_table import CSV_FIELDS, channel_columns, format_rows
from nirspipe.qc.hyper.hyper_report import _CH_COLUMNS
from nirspipe.qc.subject.prep_raw_report import _CH_COLUMN_VARS, _COND_CH_COLUMN_VARS
from nirspipe.qc.subject.record_io import write_record
from nirspipe.qc.subject.report import _section_sqm

_LABEL = "sub-01_task-main"
_SCI = {f"S{i}_D{i} {wl}": 0.95 for i in (1, 2) for wl in (760, 850)}
_TABLE = ("name\tx\ty\tz\taparc\tbrodmann\n"
          "S1_D1 hbo\t-40\t20\t50\tprecentral-lh\tBrodmann.4-lh\n"
          "S1_D1 hbr\t-40\t20\t50\tprecentral-lh\tBrodmann.4-lh\n"
          "S2_D2 hbo\tn/a\tn/a\tn/a\tn/a\tn/a\n"
          "S2_D2 hbr\tn/a\tn/a\tn/a\tn/a\tn/a\n")


def _sqm_section(tmp_path, with_table: bool) -> dict:
    nirs = tmp_path / "nirs"
    nirs.mkdir(parents=True)
    write_record(nirs / f"{_LABEL}_desc-sqm_qc.json",
                 {"raw": {"sci_mean": 0.95}, "per_channel": {}})
    if with_table:
        (nirs / f"{_LABEL}_space-fsaverage_channels.tsv").write_text(_TABLE, encoding="utf-8")
    errors: list = []
    out = _section_sqm(_SCI, [], "01", errors, nirs, sqm_label=_LABEL, sci_threshold=0.8)
    assert not errors, errors
    return out


def _keys(columns) -> list[str]:
    return [key for key, _ in columns]


def test_each_pair_prints_the_region_its_channel_table_names(tmp_path):
    out = _sqm_section(tmp_path, with_table=True)
    assert _keys(out["channel_columns"])[:2] == ["name", "region"]
    by_pair = {cell["name"]: cell["region"] for cell in out["channel_cells"]}
    # an unplaced pair has no region on disk, so it prints the table's dash
    assert by_pair == {"S1_D1": "precentral-lh", "S2_D2": MISSING_VALUE}
    assert out["channel_regions"] == {"S1_D1": "precentral-lh"}


def test_a_run_without_a_channel_table_prints_no_region_column(tmp_path):
    out = _sqm_section(tmp_path, with_table=False)
    assert "region" not in _keys(out["channel_columns"])
    assert not out["channel_regions"]


def test_rows_named_by_wavelength_find_their_pairs_region():
    rows = [{"name": "S1_D1 760", "is_bad": False, "separation": "long"},
            {"name": "S3_D3 850", "is_bad": False, "separation": "long"}]
    cells = format_rows(rows, 0.8, regions={"S1_D1": "superiorfrontal-rh"})
    assert [c["region"] for c in cells] == ["superiorfrontal-rh", MISSING_VALUE]


def test_views_without_a_channel_table_and_the_qc_tsv_carry_no_region():
    assert "region" not in CSV_FIELDS
    for columns in (_CH_COLUMN_VARS["ch_columns"], _COND_CH_COLUMN_VARS["ch_columns"], _CH_COLUMNS):
        assert "region" not in _keys(columns)
    assert "region" in _keys(channel_columns())


def test_the_table_note_says_the_region_is_the_nearest_one_not_the_light_path():
    note = section_note("metrics.per_channel_region")
    assert "nearest" in note and "not where" in note
