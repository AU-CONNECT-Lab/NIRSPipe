"""A recording numbered run-00 keeps its run in every name."""

from bids.layout.utils import PaddedInt

from nirspipe.io.bids import bids_label


def test_run_zero_is_a_run():
    assert bids_label("01", {"task": "rest", "run": PaddedInt("00")}) == "sub-01_task-rest_run-00"


def test_an_absent_entity_is_left_out():
    assert bids_label("01", {"session": None, "task": "rest", "acquisition": ""}) == "sub-01_task-rest"
