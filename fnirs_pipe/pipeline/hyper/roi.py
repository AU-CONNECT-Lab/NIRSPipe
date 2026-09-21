"""Grouping per-channel coherence into regions.

  roi_mean_of_channels     The band means, averaged per region pair.
  roi_mean_of_homologous   The same, restricted to a region against itself in the partner.
  roi_maps_from_channels   The maps grouped the same way, so a figure matches the table.

ROI coherence is always computed per channel pair and then averaged. Averaging the signals
into one ROI trace first is a different and less sensitive number, and the studies that
compared the two report the channel route detecting effects the signal route misses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fnirs_pipe.pipeline.hyper._helpers import _fisher_z
from fnirs_pipe.pipeline.hyper.wtc import WTCResult, _circular_stats
from fnirs_pipe.utils.logging import get_logger

# the caller's logger name, kept so existing log filters still match
logger = get_logger("pipeline.hyperscanning")


def _mean_phase(phases: "list[np.ndarray]") -> "np.ndarray | None":
    """Circular mean of several phase maps, cell by cell.

    e.g. [179 deg, -179 deg] gives 180 deg, not the 0 deg an arithmetic mean would.
    """
    if not phases:
        return None
    stack = np.stack([np.asarray(p, dtype=float) for p in phases])
    return np.angle(np.exp(1j * stack).mean(axis=0)).astype(np.float32)


def roi_maps_from_channels(
    result: WTCResult,
    roi_map: dict[str, list[str]],
) -> WTCResult:
    """Average the channel-pair WTC maps cell by cell into one map per ROI pair.

    The ROI number the field reports is a mean over channel-pair coherences, so the figure
    that belongs beside it is the mean over those channels' maps rather than a separate
    transform on ROI-averaged signals. Averaging the maps first and the band second gives the
    same number as averaging the band first and the channels second, so the picture and the
    table finally agree.

    ::

      {"S1_D1": map, "S1_D2": map}  + {"lPFC": ["S1_D1", "S1_D2"]}  ->  {"lPFC": mean map}

    The COI depends only on record length and sampling rate, so every member shares one and
    it is carried through unchanged. ``sig`` is dropped: a Monte Carlo level belongs to the
    pair it was computed for and does not average.

    Phase averages as a direction, not as a number: the arithmetic mean of 179 degrees and
    -179 degrees is 0, the one direction neither member points in. The members are summed as
    unit vectors and the angle read off the sum, so a set of members that disagree gives a
    short resultant and, drawn as an arrow, a direction no more confident than they were.
    """
    ch_to_roi = {ch: roi for roi, chs in roi_map.items() for ch in chs}

    pairs: dict = {}
    for pair_key, labels in result.pairs.items():
        bucket: dict = {}
        for label, data in labels.items():
            if data is None:
                continue
            label1, label2 = label if isinstance(label, tuple) else (label, label)
            roi1, roi2 = ch_to_roi.get(label1), ch_to_roi.get(label2)
            if roi1 is None or roi2 is None:
                continue
            key = (roi1, roi2) if isinstance(label, tuple) else roi1
            bucket.setdefault(key, []).append(data)
        pairs[pair_key] = {
            key: {"wtc": np.mean([d["wtc"] for d in members], axis=0),
                  "coi": members[0]["coi"], "sig": None,
                  "phase": _mean_phase([d["phase"] for d in members
                                       if d.get("phase") is not None])}
            for key, members in bucket.items()
        }

    return WTCResult(pairs=pairs, freqs=result.freqs, times=result.times)


def roi_mean_of_channels(
    band_df: pd.DataFrame,
    roi_map: dict[str, list[str]],
    min_channels: int = 2,
) -> pd.DataFrame:
    """Average channel-level band means within each ROI: the ROI number the WTC literature reports.

    The field computes coherence per channel pair and averages those values into ROI
    clusters. The alternative, one WTC on the ROI-averaged signal, is a different number
    because coherence is bounded and nonlinear, and it is the less sensitive of the two.

    ``band_df`` is what :func:`wtc_band_mean` returns for a channel-level result; a crossed
    one carries ``label2`` and is grouped on both sides into the ROI-by-ROI matrix. ``n_ch``
    counts the channel pairs behind each mean, so an ROI thinned by rejection is visible.
    Channels no ROI lists are dropped.

    ``min_channels`` drops a cell resting on fewer than that many channel pairs, the rule the
    published pipelines use to stop one surviving optode from standing in for a region. It
    counts pairs, so on a crossed frame a cell needs ``min_channels`` combinations rather than
    that many channels on each side.

    ``coherence_z`` is recomputed from the averaged coherence rather than averaged itself, so
    it stays the Fisher z of the number in the same row.

    ``phase_angle`` is averaged circularly, an arithmetic mean of angles being wrong at the
    wrap. **Its ``phase_sd`` changes meaning here**: at channel level it is the spread of the
    cells behind one pairing, and here it is the spread of the pairings behind one ROI, so a
    tight channel angle that disagrees with its neighbours' gives a small one there and a
    large one here. ``phase_n`` stays a cell count and is summed. The columns are absent when
    the frame has none, so a caller that built one by hand is unaffected.
    """
    ch_to_roi = {ch: roi for roi, chs in roi_map.items() for ch in chs}
    df = band_df.copy()
    label_cols = ["label"] + (["label2"] if "label2" in df.columns else [])
    for col in label_cols:
        df[col] = df[col].map(ch_to_roi)
    df = df.dropna(subset=label_cols)

    keys = ["sub1", "sub2"] + label_cols
    out = (
        df.groupby(keys, sort=False)
          .agg(coherence=("coherence", "mean"),
               n_valid_frac=("n_valid_frac", "mean"),
               n_ch=("coherence", "count"))
          .reset_index()
    )
    if min_channels > 1:
        thin = out["n_ch"] < min_channels
        if thin.any():
            logger.info("ROI means: %d cell(s) under %d channel pairs, dropped",
                        int(thin.sum()), min_channels)
        out = out[~thin].reset_index(drop=True)
    out.insert(out.columns.get_loc("n_valid_frac"), "coherence_z",
               out["coherence"].map(_fisher_z))

    if "phase_angle" in df.columns:
        def _roi_phase(g: pd.DataFrame) -> pd.Series:
            angle, spread, _ = _circular_stats(np.radians(g["phase_angle"].to_numpy()))
            return pd.Series({"phase_angle": np.degrees(angle),
                              "phase_sd": np.degrees(spread),
                              "phase_n": int(g["phase_n"].sum())})
        phase = (df.groupby(keys, sort=False)[["phase_angle", "phase_n"]]
                   .apply(_roi_phase).reset_index())
        out = out.merge(phase, on=keys, how="left")
    return out


def roi_mean_of_homologous(
    band_df: pd.DataFrame,
    roi_map: dict[str, list[str]],
    min_channels: int = 2,
) -> pd.DataFrame:
    """Average an ROI's *homologous* channel pairs: one value per ROI, however the run was made.

    ::

      right_pfc holds S1_D1, S1_D2, S2_D1, S2_D2
      -> the mean of the four (S1_D1, S1_D1), (S1_D2, S1_D2), ... coherences

    :func:`roi_mean_of_channels` groups whatever it is handed, so on a crossed frame its
    ``(roi, roi)`` diagonal is the mean of every pairing inside the ROI, sixteen of them here,
    of which four are homologous. That is a different quantity from the one the literature
    reports and from the one an uncrossed run produces, and a flag whose job is to add the
    off-diagonal cells should not silently redefine the diagonal. This function is the
    reported number: the homologous mean, identical whether or not the run crossed.

    It is also the only ROI value the homologous phase-scrambled null can rank, since the null
    draws exactly these pairings.
    """
    df = band_df
    if "label2" in df.columns:
        df = df[df["label"] == df["label2"]].drop(columns=["label2"])
    return roi_mean_of_channels(df, roi_map, min_channels=min_channels)
