"""Two conditions whose file names would be the same are refused, not overwritten.

A condition reaches a file name through a slug that keeps letters and digits only, so
"game 1" and "game1" name one file, and so do a second "game1" block ("game1#2") and a
condition called "game12". The later file replaced the earlier one without a word. The dyad
tables also reserve ``cond-all`` for the file holding every condition.
"""

import mne
import numpy as np
import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.qc.common.windows import condition_windows


def _raw(descriptions, block=100.0):
    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], 5.0, ["hbo", "hbr"])
    n = int(5.0 * block * (len(descriptions) + 1))
    raw = mne.io.RawArray(np.zeros((2, n)), info, verbose="ERROR")
    onsets = [block * i for i in range(len(descriptions))]
    raw.set_annotations(mne.Annotations(onsets, [block] * len(descriptions), descriptions))
    return raw


def test_distinct_conditions_are_kept():
    labels = [w[0] for w in condition_windows(_raw(["rest", "talk"]), min_duration=10.0)]
    assert labels == ["rest", "talk"]


def test_two_spellings_of_one_name_are_refused():
    with pytest.raises(StageError, match="game 1.*game1|game1.*game 1"):
        condition_windows(_raw(["game 1", "game1"]), min_duration=10.0)


def test_a_numbered_repeat_colliding_with_another_condition_is_refused():
    with pytest.raises(StageError, match="game12"):
        condition_windows(_raw(["game1", "game1", "game12"]), min_duration=10.0)


def test_a_condition_named_all_is_refused_by_the_dyad_report(tmp_path):
    from fnirs_pipe.pipeline.hyper import GroupEntry
    from fnirs_pipe.qc.hyper.hyper_report import build_hyper_post_report

    raws = {sid: _raw(["rest", "all"]) for sid in ("sub-01", "sub-02")}
    with pytest.raises(StageError, match="all"):
        build_hyper_post_report(
            group_id="G1", task="tap",
            group=[GroupEntry("G1", s, "tap") for s in raws],
            aligned_raws=raws, offsets={s: 0.0 for s in raws}, output_dir=tmp_path,
            wtc_fmin=0.02, wtc_fmax=0.2, wtc_chroma=("hbo",), wtc_by_condition=True)
