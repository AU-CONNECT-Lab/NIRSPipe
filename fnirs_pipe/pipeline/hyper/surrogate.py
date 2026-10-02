"""Surrogate coherence: what a dyad's value is read against.

  phase_scramble             One surrogate trace, the spectrum kept and the phases randomised.
  compute_wtc_phase_null     A dyad against a phase-scrambled partner, computed inside the
                             per-dyad pass.
  compute_wtc_pair_null      The accounting the re-paired null needs, whose draws are real
                             recordings rather than scrambled ones and therefore arrive from
                             outside.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd

from fnirs_pipe.pipeline.hyper._helpers import _long_signals, long_axis_over
from fnirs_pipe.pipeline.hyper.roi import roi_mean_of_homologous
from fnirs_pipe.pipeline.hyper.wtc import (
    WTCResult,
    _ChannelWavelet,
    _in_coi,
    _wtc_over_pairs,
    wtc_band_mean,
    window_result,
)
from fnirs_pipe.utils import fisher_r_to_z
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.surrogate")


def phase_scramble(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate with the same power spectrum as ``sig`` and its phases randomised.

    Scrambling one side's phases destroys every temporal relationship while leaving each
    signal's own spectrum and autocorrelation intact, so coherence computed against the
    surrogate is the coherence two unrelated recordings of this kind produce.

    Phases are randomised under Hermitian symmetry, so the inverse transform is real and the
    magnitude spectrum is preserved exactly. DC and, on an even-length record, Nyquist have
    no mirror partner and keep their sign instead of taking a phase.
    """
    n = len(sig)
    spectrum = np.fft.rfft(sig)
    scrambled = np.abs(spectrum) * np.exp(1j * rng.uniform(-np.pi, np.pi, size=spectrum.shape))
    # carried over rather than rebuilt: both are real, and |x| would flip a negative one,
    # which for DC means flipping the sign of the mean
    scrambled[0] = spectrum[0]
    if n % 2 == 0:
        scrambled[-1] = spectrum[-1]
    return np.fft.irfft(scrambled, n=n)


# ---- Phase-scrambled null ----


@dataclass
class NullDraws:
    """One chromophore's null: the draws, the levels, and the summary of both.

    Carries either null. They differ in how a draw is made, phase randomisation against
    re-pairing, and in nothing after that, so both summarise through the same code and their
    tables subtract from the same real table.

    The draws are kept rather than averaged on the spot, since a real value is ranked inside
    them by a step that runs after this one. :meth:`summarise` is that step's half.
    """

    draws: "list[pd.DataFrame]"
    cond_draws: "list[pd.DataFrame]"
    keys: "list[str]"
    # (sub1, sub2, label) -> the coherence a cell clears at each frequency to beat the null
    levels: dict
    # who each draw was against, re-pairing only: its pool is finite and named, so the
    # sidecar can say which recordings the null was built from rather than only how many
    partners: "list[str] | None" = None
    # what each entry of cond_draws came from, one id per frame: the stand-in for a
    # re-paired draw, the iteration for a scrambled one. Kept so the draws can be written
    # out one row per draw
    cond_draw_ids: "list[str] | None" = None
    # condition -> (sub1, sub2, label) -> level, re-pairing only: its draws are conditions,
    # so its level is counted per condition and there is no whole-run one
    cond_levels: "dict | None" = None

    def summarise(self, real: "pd.DataFrame | None" = None,
                  real_by_cond: "pd.DataFrame | None" = None,
                  ) -> "tuple[pd.DataFrame, pd.DataFrame | None]":
        """The whole-run and per-condition null tables, ranked against ``real`` if given.

        ``real`` is the true-dyad band-mean table this null sits beside, matched on the key
        columns, and turns on the ``percentile`` column: the share of a cell's draws its real
        value beat. Exact rather than interpolated from the stored quantiles, and the same
        definition :func:`screening_coherence` uses, so "above the null" means one thing
        across the report. ``real_by_cond`` does the same for the per-condition table.
        """
        return (
            (_average_iterations(self.draws, self.keys, real=real) if self.draws else None),
            (_average_iterations(self.cond_draws, ["condition"] + self.keys,
                                 real=real_by_cond) if self.cond_draws else None),
        )

    def summarise_roi(self, roi_map: dict[str, list[str]],
                      real: "pd.DataFrame | None" = None,
                      real_by_cond: "pd.DataFrame | None" = None,
                      min_channels: int = 2,
                      ) -> "tuple[pd.DataFrame, pd.DataFrame | None]":
        """The same two tables at ROI level, for :func:`roi_mean_of_homologous`.

        **Each iteration is grouped into ROIs before the iterations are summarised**, so the
        null of an ROI mean keeps how its channels' draws move together within an iteration.

        The draws being averaged are the ones already taken for the channel table; no
        surrogate is transformed twice.

        Only the homologous ROI value can be ranked this way. A crossed ``(roi, roi)`` cell
        also holds the within-region cross pairings, which a homologous null never draws.
        """
        keys = ["sub1", "sub2", "label"]
        whole = [roi_mean_of_homologous(f, roi_map, min_channels=min_channels)
                 for f in self.draws]
        cond = None
        if self.cond_draws:
            cond = []
            for frame in self.cond_draws:
                for condition, part in frame.groupby("condition", sort=False):
                    grouped = roi_mean_of_homologous(part, roi_map,
                                                     min_channels=min_channels)
                    grouped.insert(0, "condition", condition)
                    cond.append(grouped)
        return (
            (_average_iterations(whole, keys, real=real) if whole else None),
            (_average_iterations(cond, ["condition"] + keys, real=real_by_cond)
             if cond else None),
        )


# Quantile of the surrogate coherence a cell has to clear before its phase arrow is drawn.
# 0.95 is the alpha NULL_ALPHA_PCT grades on, so "above the null" reads the same on a
# coherence map as on the screening panel.
NULL_ARROW_QUANTILE = 0.95

# Bins the surrogate coherences are counted into, per frequency. Coherence is bounded on
# [0, 1], so a fixed grid is exact to 1/_NULL_HIST_BINS and costs the same whatever n_iter is.
_NULL_HIST_BINS = 1000


def _accumulate_null_hist(hists: dict, result: "WTCResult", mask_coi: bool) -> None:
    """Count one iteration's surrogate coherences into a per-pair, per-frequency histogram.

    ::

      a 51 x 3962 surrogate map -> 51 rows of counts, added to whatever earlier iterations left

    Pooled over time as well as over iterations, because the null being estimated is the
    distribution of a single cell at that frequency and every in-cone cell of a surrogate map
    is a draw from it. ``hists`` is mutated in place, keyed the way ``result.pairs`` is.
    """
    freqs = np.asarray(result.freqs, dtype=float)
    for (sub1, sub2), labels in result.pairs.items():
        for label, data in labels.items():
            if data is None:
                continue
            wtc = np.asarray(data["wtc"], dtype=float)
            if wtc.shape[0] != len(freqs):
                continue
            keep = _in_coi(freqs, data["coi"]) if mask_coi else np.ones(wtc.shape, dtype=bool)
            if not keep.any():
                continue
            # one bincount over the whole map rather than one per row: the row index is
            # folded into the bin index, so the flat counts reshape straight back
            col = np.clip((wtc * _NULL_HIST_BINS).astype(np.int64), 0, _NULL_HIST_BINS - 1)
            row = np.broadcast_to(np.arange(wtc.shape[0])[:, None], wtc.shape)
            flat = row[keep] * _NULL_HIST_BINS + col[keep]
            counts = np.bincount(flat, minlength=wtc.shape[0] * _NULL_HIST_BINS)
            key = (sub1, sub2, label)
            if key not in hists:
                hists[key] = np.zeros((wtc.shape[0], _NULL_HIST_BINS), dtype=np.int64)
            hists[key] += counts.reshape(hists[key].shape)


def _null_level(hist: np.ndarray, quantile: float = NULL_ARROW_QUANTILE) -> np.ndarray:
    """The quantile of each frequency's accumulated surrogate counts, as a coherence.

    ::

      a 51 x 1000 count matrix -> ndarray(51,), the level a cell clears to beat the null

    A frequency no surrogate cell ever landed on, every one of its cells having been outside
    the cone, gets NaN rather than 0: no level was measured there, and a 0 would pass every
    cell as significant.
    """
    total = hist.sum(axis=1)
    cum = np.cumsum(hist, axis=1)
    out = np.full(hist.shape[0], np.nan)
    for i in np.flatnonzero(total):
        j = int(np.searchsorted(cum[i], quantile * total[i], side="left"))
        # bin centre, so the level sits inside the bin its count was recorded in
        out[i] = (min(j, hist.shape[1] - 1) + 0.5) / hist.shape[1]
    return out


def _collect_draw(
    result: WTCResult,
    frames: "list[pd.DataFrame]",
    cond_frames: "list[pd.DataFrame]",
    hists: "dict[tuple, np.ndarray]",
    *,
    band_fmin: float,
    band_fmax: float,
    mask_coi: bool,
    windows: "list[tuple[str, float, float]] | None",
    analysis_window: "tuple[float, float] | None",
) -> None:
    """Fold one surrogate WTC run into the draws the null is summarised from.

    Shared by both nulls, so each reads its cells off the map by the same rule as the table
    it is subtracted from.
    """
    # --tstart/--tend, read off this draw's transform the way the real table reads it off
    # its own
    run_result = (result if analysis_window is None
                  else window_result(result, *analysis_window))
    frames.append(wtc_band_mean(run_result, band_fmin, band_fmax, mask_coi=mask_coi))
    # counted off the whole-run transform, not the windowed read: a condition is a slice
    # of the same map, so its cells are draws from the same per-frequency null
    _accumulate_null_hist(hists, result, mask_coi)
    # windowed off this draw's own transform, never recomputed on the cut: the real table
    # is windowed the same way
    for label, tstart, tstop in (windows or []):
        part = wtc_band_mean(window_result(result, tstart, tstop),
                             band_fmin, band_fmax, mask_coi=mask_coi)
        part.insert(0, "condition", label)
        cond_frames.append(part)


def compute_wtc_phase_null(
    raws: dict[str, mne.io.Raw],
    band_fmin: float,
    band_fmax: float,
    n_iter: int = 100,
    fmin: float = 0.004,
    fmax: float = 0.20,
    seed: int | None = None,
    cross: bool = False,
    limit_scales: bool = True,
    mask_coi: bool = True,
    ch_type: str = "hbo",
    sep_bands=None,
    windows: "list[tuple[str, float, float]] | None" = None,
    analysis_window: "tuple[float, float] | None" = None,
) -> "NullDraws":
    """Phase-scrambled band means: WTC against a phase-scrambled partner, averaged over ``n_iter``.

    One subject's signals are replaced by surrogates and the whole pairwise WTC is rerun, once
    per iteration; the band means are averaged across iterations. The result has the columns
    ``wtc_band_mean`` returns, so a true-dyad table and this one subtract or test cell by cell,
    plus ``null_sd``, ``null_p95`` and ``n_iter``.

    Returns a :class:`NullDraws`, which holds the per-iteration draws as well as their
    summary: ranking a real value inside its null needs the draws, and the caller that has
    the real table to rank runs after this one.

    Cost is ``n_iter`` times a full WTC run. Significance contours are never computed here:
    this table *is* the null. The
    surrogate maps are not saved either; each one is counted into a per-frequency histogram on the way past, and ``NullDraws.levels`` is
    ``{(sub1, sub2, label): ndarray(n_freqs,)}``, the coherence a cell has to clear at each
    frequency to beat the null. That is what the phase arrows are drawn against, and it is
    per frequency because surrogate coherence is not flat in frequency.

    ``seed`` drives the phase randomisation and nothing else. Passing the same value as the
    real run is what makes the pair reproducible together.

    ``ch_type`` runs the null on one chromophore, and has to match the real table it is
    compared against: a null computed on HbO says nothing about an HbR coupling.

    ``windows`` is ``[(label, tstart, tstop)]``, and turns the second return value into a
    per-condition null with a ``condition`` column. Each window is read off the same
    whole-run transform the whole-run table is averaged from, by :func:`window_result`, so
    one pass of ``n_iter`` transforms produces both tables and the null is windowed exactly
    the way the real table it is compared against was.

    ``analysis_window`` is ``--tstart``/``--tend``, and does the same job for the whole-run
    row that ``windows`` does for the per-condition ones: the real whole-run table describes
    that stretch, so its null has to as well.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}")

    subject_ids = list(raws.keys())
    if len(subject_ids) != 2:
        # refused, not warned: a third member would leave real pairs in a table labelled null
        raise ValueError(
            f"phase-scrambled WTC needs exactly 2 subjects, got {len(subject_ids)}: "
            f"{subject_ids}. Only one side is scrambled, so a larger group would leave "
            "real-against-real pairs in a table labelled null. Run it per dyad."
        )

    true_signals = {sid: _long_signals(raw, ch_type, sep_bands) for sid, raw in raws.items()}
    # scramble the second subject only, so a real recording is tested against a surrogate
    scrambled_id = subject_ids[1]
    rng = np.random.default_rng(seed)

    frames: list[pd.DataFrame] = []
    cond_frames: list[pd.DataFrame] = []
    cond_draw_ids: list[str] = []
    hists: dict[tuple, np.ndarray] = {}
    # the unscrambled side is the same signal in every iteration, so its transforms are
    # computed once and reused; one montage of transforms stays resident, whatever n_iter
    cache1: dict[tuple[str, str], _ChannelWavelet] = {}
    for i in range(n_iter):
        signals = dict(true_signals)
        signals[scrambled_id] = {
            label: phase_scramble(sig, rng)
            for label, sig in true_signals[scrambled_id].items()
        }
        result = _wtc_over_pairs(
            raws, signals, fmin, fmax, significance=False, seed=None,
            cross=cross, limit_scales=limit_scales, cache1=cache1,
            # the same axis the real table is built on, without which the null cannot be
            # subtracted from it row by row
            axis=long_axis_over(raws.values(), ch_type, sep_bands))
        before = len(cond_frames)
        _collect_draw(result, frames, cond_frames, hists,
                      band_fmin=band_fmin, band_fmax=band_fmax,
                      mask_coi=mask_coi, windows=windows,
                      analysis_window=analysis_window)
        cond_draw_ids.extend([f"iter-{i:04d}"] * (len(cond_frames) - before))
        if (i + 1) % 10 == 0:
            logger.info("phase-scrambled WTC: %d/%d iterations", i + 1, n_iter)

    keys = ["sub1", "sub2", "label"] + (["label2"] if "label2" in frames[0].columns else [])
    return NullDraws(draws=frames, cond_draws=cond_frames, keys=keys,
                      levels={key: _null_level(hist) for key, hist in hists.items()},
                      cond_draw_ids=cond_draw_ids)


def compute_wtc_pair_null(
    draws: "Iterable[tuple[str, dict[str, mne.io.Raw]]]",
    true_pair: "tuple[str, str]",
    axis: "list[str]",
    band_fmin: float,
    band_fmax: float,
    fmin: float = 0.004,
    fmax: float = 0.20,
    cross: bool = False,
    limit_scales: bool = True,
    mask_coi: bool = True,
    ch_type: str = "hbo",
    sep_bands=None,
    windows: "list[tuple[str, float, float]] | None" = None,
    analysis_window: "tuple[float, float] | None" = None,
    on_draw: "Callable[[str, dict], None] | None" = None,
) -> "NullDraws":
    """Re-paired band means: WTC of one member against people they never interacted with.

    ``draws`` yields ``(partner_id, aligned_raws)``, one false pair per draw, each already
    cut onto the clock the real table was computed on. Building them is the caller's job:
    it needs the pairs table and the derivatives tree, neither of which this module reads.

    The difference from :func:`compute_wtc_phase_null` is what the surrogate keeps: a
    re-paired partner is a real recording of the same task, so each member's own time-locked
    response to the task survives. Everything downstream of the transform is the other null's
    code, so the two tables subtract from the same real table and from each other.

    Both the pair key and the per-frequency histogram are relabelled to ``true_pair``. Each
    draw carries a different partner, and the summary groups by ``sub1``/``sub2``, so without
    this every draw would be its own group of one and the table would report ``n_iter`` 1
    against a null it never averaged.

    ``axis`` is the real dyad's channel axis, passed in rather than recomputed: a partner with
    a different montage would otherwise move the rows and the null could no longer be
    subtracted from the real table row by row. A label the partner lacks lands as a blank row,
    the same shape a rejected channel leaves.

    The pool is finite, unlike phase randomisation's, so the number of draws is a property of
    the cohort rather than a setting. That is what limits the resolution of ``percentile``.
    """
    # no whole-run list: a re-paired draw is one condition, cut from each side's own marker
    cond_frames: list[pd.DataFrame] = []
    cond_draw_ids: list[str] = []
    cond_hists: dict[str, dict[tuple, np.ndarray]] = {}
    partners: list[str] = []
    # Keyed per condition and per segment, not per condition alone: a draw pads the condition
    # with whatever both recordings can spare either side, so the fixed member's stretch is
    # the same only across draws that got the same pad
    caches: dict[tuple, dict[tuple[str, str], _ChannelWavelet]] = {}
    # a fifth element is the pair the coherence reads, whitened, where --wtc-whiten is on
    for partner_id, label, pair, inner, *whitened in draws:
        # other metrics on the same re-paired pool ride on this draw rather than a second pass
        if on_draw is not None:
            on_draw(partner_id, label, pair, inner)
        wtc_pair = whitened[0] if whitened else pair
        signals = {sid: _long_signals(raw, ch_type, sep_bands)
                   for sid, raw in wtc_pair.items()}
        segment = float(min(raw.times[-1] for raw in pair.values()))
        cache_key = (label, round(float(inner[0]), 3), round(segment, 3))
        result = _wtc_over_pairs(
            wtc_pair, signals, fmin, fmax, significance=False, seed=None,
            cross=cross, limit_scales=limit_scales,
            cache1=caches.setdefault(cache_key, {}), axis=axis)
        result = WTCResult(pairs={true_pair: next(iter(result.pairs.values()))},
                           freqs=result.freqs, times=result.times)
        # The segment carries context either side of the condition, so the pad is dropped
        # here: the transform used it, the band mean must not. There is no whole-run draw to
        # collect, the two recordings being alignable one condition at a time or not at all.
        result = window_result(result, *inner)
        before = len(cond_frames)
        _collect_draw(result, [], cond_frames, cond_hists.setdefault(label, {}),
                      band_fmin=band_fmin, band_fmax=band_fmax,
                      mask_coi=mask_coi, windows=[(label, *inner)],
                      analysis_window=None)
        cond_draw_ids.extend([partner_id] * (len(cond_frames) - before))
        if partner_id not in partners:
            partners.append(partner_id)
        logger.info("re-paired WTC: %s against %s", label, partner_id)

    if not cond_frames:
        raise ValueError(
            "no usable partner was drawn, so there is no null. Every candidate was refused: "
            "the log says which test each one failed.")

    keys = ["sub1", "sub2", "label"] + (["label2"] if "label2" in cond_frames[0].columns else [])
    return NullDraws(draws=[], cond_draws=cond_frames, keys=keys, levels={},
                     partners=partners, cond_draw_ids=cond_draw_ids,
                     cond_levels={label: {key: _null_level(hist) for key, hist in hists.items()}
                                  for label, hists in cond_hists.items()})


def _average_iterations(frames: "list[pd.DataFrame]", keys: "list[str]",
                        real: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """One band-mean table summarising the iterations that produced it, distribution and all.

    ::

      [iter1 rows, iter2 rows], ["sub1", "sub2", "label"]  ->  one row per channel pair

    ``null_mean`` is the mean of the draws and ``null_mean_z`` its Fisher z, taken from the
    averaged value rather than averaged itself. **Not called ``coherence``**, which is what
    every other table's measured value is called: this column is the null's own centre.
    ``null_sd``, ``null_p95`` and ``n_iter`` describe the spread it came out of.

    ``real`` is the true-dyad table, matched on ``keys``, and adds ``percentile``: the share
    of a cell's draws its real value beat. A cell the real table has no row for, or whose
    real value is NaN, gets NaN rather than a rank against nothing.
    """
    stacked = pd.concat(frames, ignore_index=True)
    grouped = stacked.groupby(keys, sort=False)
    out = (grouped.agg(null_mean=("coherence", "mean"),
                       null_sd=("coherence", "std"),
                       n_iter=("coherence", "count"),
                       n_valid_frac=("n_valid_frac", "mean"))
                  .reset_index())
    draws = {key: part.to_numpy(dtype=float) for key, part in grouped["coherence"]}
    out.insert(out.columns.get_loc("null_sd"), "null_mean_z",
               fisher_r_to_z(out["null_mean"]))
    out.insert(out.columns.get_loc("n_iter"), "null_p95",
               [_p95(draws[k]) for k in _row_keys(out, keys)])
    if real is not None:
        out.insert(out.columns.get_loc("n_iter"), "percentile",
                   _null_percentile(out, keys, draws, real))
    return out


def _p95(draw: np.ndarray) -> float:
    """95th percentile of the draws that exist. A pair blank in every iteration has none."""
    finite = draw[np.isfinite(draw)]
    return float(np.percentile(finite, 95)) if finite.size else float("nan")


def _row_keys(frame: pd.DataFrame, keys: "list[str]") -> list:
    """The groupby key of each row, shaped the way pandas hands it back: scalar for one key."""
    if len(keys) == 1:
        return frame[keys[0]].tolist()
    return list(map(tuple, frame[keys].to_numpy()))


def _null_percentile(out: pd.DataFrame, keys: "list[str]", draws: dict,
                     real: pd.DataFrame) -> "list[float]":
    """Where each real value falls among its cell's own surrogate draws, as a percentage.

    ::

      real 0.31 against draws [0.22, 0.25, 0.29, 0.33]  ->  75.0

    Counting rather than interpolating a stored quantile.

    A homologous null against a crossed real table ranks the crossed table's diagonal: the
    rows where ``label`` and ``label2`` agree are the pairings the null was drawn for.
    """
    missing = [k for k in keys if k not in real.columns]
    if missing:
        logger.warning("null percentile skipped: the real table has no %s column(s)",
                       ", ".join(missing))
        return [float("nan")] * len(out)
    # a crossed table holds one row per label pair, so keying on `label` alone would take
    # whichever pairing that label came first in rather than its homologous one
    if "label2" in real.columns and "label2" not in keys:
        real = real[real["label"] == real["label2"]]
    truth = real.set_index(keys)["coherence"]
    if truth.index.has_duplicates:
        logger.warning("null percentile: %d real row(s) share a key, ranking against the "
                       "first of each", int(truth.index.duplicated().sum()))
        truth = truth[~truth.index.duplicated()]
    values = []
    for key in _row_keys(out, keys):
        real_value = truth.get(key, float("nan"))
        draw = draws.get(key)
        # a NaN draw compares False, so a pair blank in every iteration would rank 0th:
        # "the real value beat none of them", which is a claim about draws that do not exist
        usable = draw is not None and np.isfinite(draw).any()
        values.append(float((draw < real_value).mean() * 100)
                      if usable and np.isfinite(real_value) else float("nan"))
    return values
