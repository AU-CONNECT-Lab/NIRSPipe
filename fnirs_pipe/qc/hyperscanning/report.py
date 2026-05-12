from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from fnirs_pipe.qc.hyperscanning.io import GroupEntry


def build_hyper_report(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    iqm_data: dict[str, dict],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    coherence_df: pd.DataFrame,
    output_dir: Path,
    sci_threshold: float = 0.75,
    coherence_fmin: float = 0.01,
    coherence_fmax: float = 0.10,
) -> Path:
    raise NotImplementedError
