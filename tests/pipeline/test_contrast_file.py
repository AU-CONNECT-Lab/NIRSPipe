"""`--contrast-file` as a TSV or CSV table, one row per contrast, beside the TOML it always read."""

import pandas as pd
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.pipeline.glm import read_contrasts

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

TABLE = ("contrast\tTapping_Left\tTapping_Right\tControl\n"
         "tapping_vs_control\t0.5\t0.5\t-1\n"
         "left_minus_right\t1\t-1\t\n")
EXPECTED = {"tapping_vs_control": {"Tapping_Left": 0.5, "Tapping_Right": 0.5, "Control": -1.0},
            "left_minus_right": {"Tapping_Left": 1.0, "Tapping_Right": -1.0}}


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---- reading ----

def test_a_tsv_gives_each_row_its_weights_and_leaves_empty_cells_out(tmp_path):
    assert read_contrasts(_write(tmp_path, "c.tsv", TABLE)) == EXPECTED


def test_a_csv_reads_the_same(tmp_path):
    assert read_contrasts(_write(tmp_path, "c.csv", TABLE.replace("\t", ","))) == EXPECTED


def test_the_toml_it_always_read_still_reads(tmp_path):
    toml = ("[tapping_vs_control]\nTapping_Left = 0.5\nTapping_Right = 0.5\nControl = -1\n"
            "[left_minus_right]\nTapping_Left = 1\nTapping_Right = -1\n")
    assert read_contrasts(_write(tmp_path, "c.toml", toml)) == EXPECTED


@pytest.mark.parametrize("text, message", [
    ("name\tTapping_Left\nlr\t1\n", "needs a contrast column"),
    ("contrast\tTapping_Left\nlr\tone\n", "line 2: Tapping_Left 'one' is not a number"),
    ("contrast\tTapping_Left\nlr\t1\nlr\t2\n", "line 3: contrast 'lr' is named twice"),
    ("contrast\tTapping_Left\nlr\t\n", "line 2: contrast 'lr' weights no condition"),
    ("contrast\tTapping_Left\n\t1\n", "line 2: no contrast name"),
])
def test_a_table_it_cannot_read_is_refused_naming_the_line(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_contrasts(_write(tmp_path, "c.tsv", text))


# ---- through the command line ----

GLM = ("--mode", "glm", "--drift-high-pass", "0.008", "--noise-model", "ols",
       "--participant-label", "01", "--no-report", "--skip-bids-validation")


@pytest.fixture(scope="module")
def bids(tmp_path_factory):
    return make_fingerprint_dataset(tmp_path_factory.mktemp("contrast_file"), blocks=True)[0]


def _glm(bids, out, contrast):
    events = next(bids.rglob("*_events.tsv"))
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, *GLM,
                  "--events-path", str(events), "--contrast-file", str(contrast)])
    return pd.read_csv(next(out.rglob("*_desc-contrast_nirsmap.tsv")), sep="\t")


def test_a_tsv_run_writes_the_contrasts_a_toml_run_does(bids, tmp_path):
    table = _glm(bids, tmp_path / "tsv",
                 _write(tmp_path, "c.tsv", "contrast\tca\tcb\nca_minus_cb\t1\t-1\n"))
    toml = _glm(bids, tmp_path / "toml", _write(tmp_path, "c.toml", "[ca_minus_cb]\nca = 1\ncb = -1\n"))
    pd.testing.assert_frame_equal(table, toml)
    assert set(table["contrast"]) == {"ca_minus_cb"}


def test_a_bad_table_stops_the_run_before_any_subject(bids, tmp_path):
    with pytest.raises(SystemExit, match="'one' is not a number"):
        _glm(bids, tmp_path / "out", _write(tmp_path, "c.tsv", "contrast\tca\nx\tone\n"))
    assert not (tmp_path / "out" / "sub-01" / "nirs").exists()
