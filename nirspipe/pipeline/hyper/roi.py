"""Grouping per-channel coherence into regions.

  roi_mean_of_channels     The band means, averaged per region pair.
  roi_mean_of_homologous   The same, restricted to a region against itself in the partner.
  roi_maps_from_channels   The maps grouped the same way, so a figure matches the table.

ROI coherence is always computed per channel pair and then averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nirspipe.pipeline.hyper.wtc import WTCResult, _circular_stats
from nirspipe.utils import ROI_MIN_CHANNELS, bare_roi_map, fisher_r_to_z
from nirspipe.utils.logging import get_logger

logger = get_logger("pipeline.roi")


def _mean_phase(phases: "list[np.ndarray]") -> "np.ndarray | None":
    """Circular mean of several phase maps, cell by cell.

    e.g. [179 deg, -179 deg] gives 180 deg, not the 0 deg an arithmetic mean would.
    """
    if not phases:
        return None
    stack = np.stack([np.asarray(p, dtype=float) for p in phases])
    return np.angle(np.exp(1j * stack).mean(axis=0)).astype(np.float32)


def _rois_of(roi_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """Channel -> every ROI that lists it, so a channel two ROIs share counts in both.

    {"L": ["S1_D1", "S1_D2"], "R": ["S1_D2"]}  ->  {"S1_D1": ["L"], "S1_D2": ["L", "R"]}
    """
    out: dict[str, list[str]] = {}
    # a map passed in directly may still carry " hbo" / " hbr"; the labels here are pairs
    for roi, chs in bare_roi_map(roi_map).items():
        for ch in chs:
            if roi not in out.setdefault(ch, []):
                out[ch].append(roi)
    return out


def roi_maps_from_channels(
    result: WTCResult,
    roi_map: dict[str, list[str]],
    min_channels: int = ROI_MIN_CHANNELS,
    levels: "dict | None" = None,
) -> WTCResult:
    """Average the channel-pair WTC maps cell by cell into one map per ROI pair.

    The ROI number is a mean over channel-pair coherences, so the figure beside it is the
    mean over those channels' maps. Averaging the maps first and the band second gives the
    same number as averaging the band first and the channels second, so the picture and the
    table agree.

    ::

      {"S1_D1": map, "S1_D2": map}  + {"lPFC": ["S1_D1", "S1_D2"]}  ->  {"lPFC": mean map}

    The COI depends only on record length and sampling rate, so every member shares one and
    it is carried through unchanged. ``sig`` is dropped: a Monte Carlo level belongs to the
    pair it was computed for and does not average. ``levels``, ``{pairing: {roi key:
    ndarray}}``, puts back the ROI maps' own phase-scrambled level, matched on the exact key.

    Phase averages as a direction, not as a number: the arithmetic mean of 179 degrees and
    -179 degrees is 0, the one direction neither member points in. The members are summed as
    unit vectors and the angle read off the sum, so a set of members that disagree gives a
    short resultant and, drawn as an arrow, a direction no more confident than they were.

    ``min_channels`` leaves out a cell where either member contributes fewer channels, the
    rule :func:`roi_mean_of_channels` blanks the same cell by, so no map stands for a region
    the table leaves empty.
    """
    rois_of = _rois_of(roi_map)

    pairs: dict = {}
    levels = levels or {}
    for pair_key, labels in result.pairs.items():
        bucket: dict = {}
        # the channels behind each cell, per member, for the minimum
        sides: dict = {}
        for label, data in labels.items():
            if data is None:
                continue
            crossed = isinstance(label, tuple)
            label1, label2 = label if crossed else (label, label)
            for roi1 in rois_of.get(label1, ()):
                for roi2 in (rois_of.get(label2, ()) if crossed else (roi1,)):
                    key = (roi1, roi2) if crossed else roi1
                    bucket.setdefault(key, []).append(data)
                    s1, s2 = sides.setdefault(key, (set(), set()))
                    s1.add(label1)
                    s2.add(label2)
        pairs[pair_key] = {
            key: {"wtc": np.mean([d["wtc"] for d in members], axis=0),
                  "coi": members[0]["coi"], "sig": None,
                  "phase": _mean_phase([d["phase"] for d in members
                                       if d.get("phase") is not None])}
            for key, members in bucket.items()
            if min(len(s) for s in sides[key]) >= min_channels
        }
        for key, level in levels.get(pair_key, {}).items():
            data = pairs[pair_key].get(key)
            if data is not None and len(level) == len(result.freqs):
                data.update(sig=level, sig_source="null")

    return WTCResult(pairs=pairs, freqs=result.freqs, times=result.times)


def roi_mean_of_channels(
    band_df: pd.DataFrame,
    roi_map: dict[str, list[str]],
    min_channels: int = ROI_MIN_CHANNELS,
) -> pd.DataFrame:
    """Average channel-level band means within each ROI.

    ``band_df`` is what :func:`wtc_band_mean` returns for a channel-level result; a crossed
    one carries ``label2`` and is grouped on both sides into the ROI-by-ROI matrix. ``n_ch``
    counts the channel pairs behind each mean, so an ROI thinned by rejection is visible.
    Channels no ROI lists are dropped.

    ``min_channels`` blanks a cell where either member contributes fewer than that many
    channels with a value; at one, only a member with no channel left blanks it. On a
    crossed frame each side is counted on its own: two channels against one is two pairings
    but one channel of the second member, and is blanked. A blanked cell keeps its row and
    its ``n_ch``, with NaN in every measured column, as a channel with no map does in
    :func:`wtc_band_mean`.

    ``coherence_z`` is recomputed from the averaged coherence rather than averaged itself, so
    it stays the Fisher z of the number in the same row.

    ``phase_angle`` is averaged circularly, an arithmetic mean of angles being wrong at the
    wrap. **Its ``phase_sd`` changes meaning here**: at channel level it is the spread of the
    cells behind one pairing, and here it is the spread of the pairings behind one ROI, so a
    tight channel angle that disagrees with its neighbours' gives a small one there and a
    large one here. ``phase_n`` stays a cell count and is summed. The columns are absent when
    the frame has none, so a caller that built one by hand is unaffected.
    """
    rois_of = _rois_of(roi_map)
    df = band_df.copy()
    label_cols = ["label"] + (["label2"] if "label2" in df.columns else [])
    # each side's channel names, kept before they become ROI names, blank where no value
    sides = [f"_side{i}" for i in range(len(label_cols))]
    for col, side in zip(label_cols, sides):
        df[side] = df[col].where(df["coherence"].notna())
    for col in label_cols:
        # one row per ROI the channel is in; a channel no ROI lists drops out
        df[col] = df[col].map(rois_of)
        df = df.dropna(subset=[col]).explode(col, ignore_index=True)

    keys = ["sub1", "sub2"] + label_cols
    grouped = df.groupby(keys, sort=False)
    out = (
        grouped.agg(coherence=("coherence", "mean"),
                    n_valid_frac=("n_valid_frac", "mean"),
                    n_ch=("coherence", "count"))
               .reset_index()
    )
    thin = np.zeros(len(out), dtype=bool)
    if min_channels > 1:
        per_side = grouped[sides].nunique().reset_index(drop=True)
        thin = (per_side < min_channels).any(axis=1).to_numpy()
        if thin.any():
            logger.info("ROI means: %d cell(s) with a member under %d channels, left blank",
                        int(thin.sum()), min_channels)
        out.loc[thin, ["coherence", "n_valid_frac"]] = np.nan
    out.insert(out.columns.get_loc("n_valid_frac"), "coherence_z",
               fisher_r_to_z(out["coherence"]))

    if "phase_angle" in df.columns:
        def _roi_phase(g: pd.DataFrame) -> pd.Series:
            angle, spread, _ = _circular_stats(np.radians(g["phase_angle"].to_numpy()))
            return pd.Series({"phase_angle": np.degrees(angle),
                              "phase_sd": np.degrees(spread),
                              "phase_n": int(g["phase_n"].sum())})
        phase = (df.groupby(keys, sort=False)[["phase_angle", "phase_n"]]
                   .apply(_roi_phase).reset_index())
        out = out.merge(phase, on=keys, how="left")
        out.loc[thin, ["phase_angle", "phase_sd"]] = np.nan
    return out


def roi_mean_of_homologous(
    band_df: pd.DataFrame,
    roi_map: dict[str, list[str]],
    min_channels: int = ROI_MIN_CHANNELS,
) -> pd.DataFrame:
    """Average an ROI's *homologous* channel pairs: one value per ROI, however the run was made.

    ::

      right_pfc holds S1_D1, S1_D2, S2_D1, S2_D2
      -> the mean of the four (S1_D1, S1_D1), (S1_D2, S1_D2), ... coherences

    :func:`roi_mean_of_channels` groups whatever it is handed, so on a crossed frame its
    ``(roi, roi)`` diagonal is the mean of every pairing inside the ROI, sixteen of them here,
    of which four are homologous. That is a different quantity from the one an uncrossed run
    produces. This function is the homologous mean, identical whether or not the run crossed.

    It is also the only ROI value the homologous phase-scrambled null can rank, since the null
    draws exactly these pairings.
    """
    df = band_df
    if "label2" in df.columns:
        df = df[df["label"] == df["label2"]].drop(columns=["label2"])
    return roi_mean_of_channels(df, roi_map, min_channels=min_channels)
