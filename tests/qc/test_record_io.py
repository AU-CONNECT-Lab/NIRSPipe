"""A record's per-channel values go to a table and come back as they went in."""

import numpy as np
import pytest

from fnirs_pipe.qc.subject.record_io import read_record, write_record

_PATH = "sub-01_task-tap_desc-sqmraw_qc.json"


def _record(per_channel):
    return {"Sources": [], "raw": {"sci_mean": 0.9}, "per_channel": per_channel}


def test_per_channel_values_round_trip(tmp_path):
    per_channel = {"raw": {"sci_per_channel": {"S1_D1 760": 0.91, "S1_D1 850": np.float64(0.5)}},
                   "rawhaemo": {"hbo_hbr_corr_per_channel": {"S1_D1": -0.4}}}
    back = read_record(write_record(tmp_path / _PATH, _record(per_channel)))
    assert back["per_channel"] == {"raw": {"sci_per_channel": {"S1_D1 760": 0.91,
                                                               "S1_D1 850": 0.5}},
                                   "rawhaemo": {"hbo_hbr_corr_per_channel": {"S1_D1": -0.4}}}


def test_a_nested_value_is_refused_when_written(tmp_path):
    nested = {"raw": {"share_by_condition": {"rest": {"S1_D1 760": 1.0}}}}
    with pytest.raises(ValueError, match="raw.share_by_condition"):
        write_record(tmp_path / _PATH, _record(nested))
    assert not (tmp_path / _PATH).exists()
