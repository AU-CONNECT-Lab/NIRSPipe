"""Figure builders for the hyperscanning post-processing QC report.

Planned figures
---------------
build_wtc_channel       Per-channel wavelet transform coherence (time × frequency)
build_wtc_roi           ROI-level WTC (channels averaged within each ROI first)
build_isc_matrix        Inter-brain correlation heatmap (Sub1 channels × Sub2 channels)
build_connectivity_circle  Inter-brain connectogram (left = Sub1, right = Sub2)
"""

from __future__ import annotations

import mne
import pandas as pd
import plotly.graph_objects as go

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.hyper_post")


def build_wtc_channel(
    aligned_raws: dict[str, mne.io.Raw],
    ch_pair: str,
    subject_ids: list[str],
    markers_list: list[dict],
    cond_colors: dict[str, str],
    fmin: float = 0.004,
    fmax: float = 0.20,
) -> go.Figure | None:
    """Per-channel WTC time-frequency plot between two subjects.

    Uses pycwt Morlet wavelet. Condition epochs highlighted as shaded regions.
    Returns a Plotly figure with a heatmap (time × log-frequency, colorscale = coherence 0–1).
    """
    raise NotImplementedError


def build_wtc_roi(
    aligned_raws: dict[str, mne.io.Raw],
    roi_map: dict[str, list[str]],
    subject_ids: list[str],
    markers_list: list[dict],
    cond_colors: dict[str, str],
    fmin: float = 0.004,
    fmax: float = 0.20,
) -> dict[str, go.Figure]:
    """ROI-level WTC: average HbO within each ROI, then compute WTC.

    roi_map: {"PFC_left": ["S1-D1", "S1-D2", ...], ...}
    Returns {roi_label: Figure}.
    """
    raise NotImplementedError


def build_isc_matrix(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
) -> go.Figure | None:
    """Inter-brain correlation heatmap.

    Rows = Sub1 channels, columns = Sub2 channels.
    Cell value = Pearson r of HbO time series.
    """
    raise NotImplementedError


def build_connectivity_circle(
    isc_df: pd.DataFrame,
    subject_ids: list[str],
    threshold: float = 0.3,
) -> go.Figure | None:
    """Inter-brain connectogram.

    Left semicircle = Sub1 channels, right semicircle = Sub2 channels.
    Arc drawn only when mean ISC >= threshold; arc width/colour encodes strength.

    isc_df columns: ch_name, sub1, sub2, isc (mean ISC value per channel pair).
    """
    raise NotImplementedError
