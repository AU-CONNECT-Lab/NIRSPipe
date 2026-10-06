"""Every table the package writes codes a missing value the way BIDS requires."""

import numpy as np
import pandas as pd

from fnirs_pipe.io.tables import write_tsv


def test_a_missing_value_is_written_n_a_and_no_index_by_default(tmp_path):
    path = write_tsv(pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]}), tmp_path / "t.tsv")
    assert path.read_text(encoding="utf-8").splitlines() == ["a\tb", "1.0\tx", "n/a\tn/a"]


def test_an_index_is_written_only_when_asked(tmp_path):
    frame = pd.DataFrame({"S1_D1": [1.0]}, index=["S1_D1"])
    path = write_tsv(frame, tmp_path / "t.tsv", index=True, index_label="channel")
    assert path.read_text(encoding="utf-8").splitlines()[0] == "channel\tS1_D1"
