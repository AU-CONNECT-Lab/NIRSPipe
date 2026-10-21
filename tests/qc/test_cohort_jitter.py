"""The cohort strip's jitter is the same in every process, so two reports of one cohort match."""

import json
import os
import subprocess
import sys

_STRIP_X = """
import json
import pandas as pd
from nirspipe.qc.figures.subject.group_figures import _group_box
df = pd.DataFrame({"raw_long_sci_mean": [0.9, 0.8, 0.7], "raw_short_sci_mean": [0.6, 0.5, 0.4]})
fig = _group_box("SCI", list(df.columns), df, ["01", "02", "03"])
print(json.dumps([list(t.x) for t in fig.data if t.type == "scatter"]))
"""


def _strip_x(hash_seed: str) -> list:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run([sys.executable, "-c", _STRIP_X], env=env, capture_output=True,
                         text=True, check=True)
    return json.loads(out.stdout)


def test_the_jitter_does_not_follow_the_process_hash_seed():
    first = _strip_x("1")
    assert len(first) == 2 and all(len(x) == 3 for x in first)
    assert _strip_x("2") == first
