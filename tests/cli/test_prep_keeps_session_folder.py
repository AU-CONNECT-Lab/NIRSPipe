"""A session-labelled input keeps its ses- folder through nirspipe-prep, named or not on the command line."""

import pytest

from nirspipe.cli import prep as prep_cli
from nirspipe.pipeline.hyper import parse_group_csv
from tests._synth import make_hyper_dataset


@pytest.fixture(scope="module")
def bids(tmp_path_factory):
    bids_dir, _pairs = make_hyper_dataset(tmp_path_factory.mktemp("ses"), tasks=("hold",),
                                          sessions=("1",))
    return bids_dir


@pytest.mark.parametrize("command, options, tree", [
    (["edit-markers", "apply"], ["--set-duration", "5"], "marker_edited"),
    (["crop"], ["--tmin", "0", "--tmax", "20"], "cropped"),
])
def test_the_output_sits_under_the_session_its_filename_names(bids, tmp_path, command,
                                                               options, tree):
    out = tmp_path / "out"
    prep_cli.main([*command, str(bids), str(out), "--participant-label", "11",
                   "--skip-bids-validation", *options])
    written = list((out / tree).rglob("*_nirs.snirf"))
    assert written
    for path in written:
        assert path.parent.parent.name == "ses-1", path
        assert "_ses-1_" in path.name


def test_a_bare_subject_id_in_the_pairs_table_reads_as_its_sub_label(tmp_path):
    csv = tmp_path / "pairs.csv"
    csv.write_text("group_id,subject_id,task\nG1,101,draw\nG1,sub-201,draw\n")
    (members,) = parse_group_csv(csv).values()
    assert [m.subject_id for m in members] == ["sub-101", "sub-201"]
