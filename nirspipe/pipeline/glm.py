from __future__ import annotations
import gc
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import mne
import mne.io

from nirspipe.io.tables import read_table, write_tsv
from nirspipe.pipeline.denoise import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_FILTER_ORDER,
    filter_array,
)
from nirspipe.exceptions import StageError
from nirspipe.utils import correct_p, is_marker
from nirspipe.pipeline.chromophores import DEFAULT_CHROMOPHORES, chroma_of, hbt_names, hbt_of
from nirspipe.pipeline.glm_censored import (censor_summary, condition_columns, events_left,
                                           fit_censored, kept_frames, refuse_too_few_frames,
                                           silent_conditions)
from nirspipe.pipeline.glm_worsley import NOISE_PREFIX as WORSLEY_PREFIX, fit_worsley
from nirspipe.pipeline.glm_whitening import (WHITENING_LOWBAND, WHITENING_PSD_SEGMENT_S,
                                            whitening_tables)
from nirspipe.utils.lineage import stamp
from nirspipe.utils.spans import bad_spans
from nirspipe.utils.logging import get_logger
from nirspipe.io.auxiliary import TIME_COLUMN, read_aux_table, resample_to_grid
from nirspipe.io.derivatives import entity_of, write_step_sidecar
from nirspipe.io.naming import derivative_path

logger = get_logger("post.glm")

HRFModel   = Literal[
    "spm",
    "spm + derivative",
    "spm + derivative + dispersion",
    "glover",
    "glover + derivative",
    "glover + derivative + dispersion",
    "fir",
]
# arN (e.g. "ar2", "ar3") is also valid but cannot be expressed as a Literal
NoiseModel = Literal["ols", "ar1", "ar2", "ar3", "ar4", "ar5", "auto", "ar_irls", "worsley"]
DriftModel = Literal["cosine", "polynomial", "none"]
SCRStrategy = Literal["mean", "pca"]

# the prefix every short-channel confound column carries, whatever the strategy built it
SHORT_CH_PREFIX = "short_ch_"
# AR(1) coefficients are truncated onto a 1/bins grid; arN uses min(bins, n_channels) clusters
AR_BINS_MIN = 1000


def _short_channel_regressors(
    haemo: mne.io.Raw, strategy: SCRStrategy, sep_bands=None,
) -> dict[str, np.ndarray]:
    from nirspipe.qc.metrics._helpers import long_short_channels

    # a --config TOML can write `short_channel = true`, which reaches this past the CLI's choices
    if strategy is True:
        strategy = "mean"
    if strategy not in ("mean", "pca"):
        raise ValueError(
            f"short-channel strategy must be 'mean' or 'pca', got {strategy!r}.")
    # not mne_nirs' get_short_channels: that reads distance 0 as short, so a montage with no
    # registered positions would build these out of every channel
    short_names = long_short_channels(haemo, sep_bands)[1]
    if not short_names:
        # refusing rather than skipping: the methods text names the regressors that were
        # asked for, so a silent skip publishes a claim the residual does not support
        from nirspipe.qc.metrics._helpers import separation_bands
        picks = mne.pick_types(haemo.info, meg=False, fnirs=True, exclude=[])
        dists = mne.preprocessing.nirs.source_detector_distances(haemo.info, picks=picks)
        positive = [d for d in dists if d > 0]
        nearest = min(positive) if positive else None
        short_max = (sep_bands or separation_bands())[0]
        raise StageError(
            "--short-channel was asked for but this montage has no channel at or under "
            f"{short_max * 1000:.0f} mm"
            + (f"; the shortest is {nearest * 1000:.1f} mm" if nearest else "")
            + ". Raise --short-max-dist if those are meant to be the short channels, or "
            "drop --short-channel."
        )
    short = haemo.copy().pick(short_names)
    # a rejected short channel would otherwise enter the regressor, and the regressor
    # is in the design matrix, so one bad channel would reach every channel's fit
    good_hbo = mne.pick_types(short.info, fnirs="hbo", exclude="bads")
    good_hbr = mne.pick_types(short.info, fnirs="hbr", exclude="bads")
    # picking an empty selection raises rather than returning nothing, so the emptiness
    # has to be caught here or a subject whose short channels were all rejected kills the run
    if not len(good_hbo) or not len(good_hbr):
        logger.warning("every short channel of a chromophore is rejected, so this run gets "
                       "no short-channel regressor; its record and its methods text say so. "
                       "The run is not comparable with the subjects that got one.")
        return {}
    empty = sole_regressor_channels(haemo, strategy, sep_bands)
    if strategy == "pca" and empty:
        logger.warning(
            "the pca regressor keeps every short-channel component, so it spans each good "
            "short channel and their own residuals are empty: %s. Their GLM fit, ALFF and FC "
            "are left blank; the long channels are unaffected.", ", ".join(empty))
    else:
        for lone in empty:
            logger.warning(
                "%s is the only short channel of its chromophore that passed screening, so "
                "the regressor built from it is that channel; its own residual is therefore "
                "empty and its GLM fit, ALFF and FC are left blank. Every other channel is "
                "unaffected.", lone)
    hbo_data = short.get_data(picks=good_hbo)  # (n_channels, n_times)
    hbr_data = short.get_data(picks=good_hbr)
    n_dropped = len(short.ch_names) - len(hbo_data) - len(hbr_data)
    if n_dropped:
        logger.info("short-channel regressors: %d of %d short channels excluded as bad",
                    n_dropped, len(short.ch_names))
    if strategy == "pca":
        return _short_channel_basis(np.vstack([hbo_data, hbr_data]))
    return {
        "short_ch_hbo_mean": hbo_data.mean(axis=0),
        "short_ch_hbr_mean": hbr_data.mean(axis=0),
    }


def sole_regressor_channels(
    haemo: mne.io.Raw, strategy: "SCRStrategy | None", sep_bands=None,
) -> list[str]:
    """Short channels that would be regressed out of themselves, leaving nothing.

    ``mean``: ``[]`` for a montage with several good short channels a chromophore,
    ``["S5_D6 hbo"]`` for one whose only surviving short channel is that one.
    ``pca``: every good short channel of both chromophores.

    The ``mean`` regressor is the mean of the good short channels of a chromophore. Where
    one of them survives screening, that mean **is** that channel, sample for sample, and
    the design matrix is applied to every channel including the ones it was built from: the
    channel is fitted against a copy of itself and its residual is numerically zero. With
    two or more the residual is the channel minus their mean, which is a real if small
    quantity, so the line is at one and not at some share.

    ``pca`` keeps every component, so its basis spans each good short channel and all of
    them are fitted against themselves, however many survive.

    A chromophore with no good short channel builds no regressor at all, so nothing is
    named then.
    """
    if not strategy:
        return []
    from nirspipe.qc.metrics._helpers import long_short_channels

    short_names = long_short_channels(haemo, sep_bands)[1]
    if not short_names:
        return []
    short = haemo.copy().pick(short_names)
    good = [[short.ch_names[i] for i in mne.pick_types(short.info, fnirs=chromo, exclude="bads")]
            for chromo in ("hbo", "hbr")]
    if not all(good):
        return []
    if strategy == "pca":
        return [ch for names in good for ch in names]
    return [names[0] for names in good if len(names) == 1]


def _short_channel_basis(data: np.ndarray) -> dict[str, np.ndarray]:
    """An orthonormal basis of every short channel, both chromophores in one decomposition.

    ::

        6 short channels x 2 chromophores, 2000 samples
          -> {"short_ch_pca01": ..., ..., "short_ch_pca12": ...}

    ``data`` is ``(n_channels, n_times)``, chromophores stacked.

    **Every component is kept.** A least-squares fit depends only on the column space of its
    design matrix, so an orthonormal basis of a full-rank block spans what the raw channels
    spanned and leaves the residual identical to the last bit. The decomposition is there so
    the columns are not collinear.

    Steps:

    1. transpose to samples-by-channels and remove each channel's mean, so the first
       direction describes covariation rather than the offset the intercept already carries
    2. take the left singular vectors, dropping the numerically zero ones. Two short
       channels carrying the same signal make the block rank deficient, and those
       directions are numerical noise rather than components
    3. scale each to unit variance, so the columns sit beside the drift basis on one scale
    """
    centred = data.T - data.T.mean(axis=0)
    u, singular, _ = np.linalg.svd(centred, full_matrices=False)
    # the numerical rank, spelled the way `orth` spells it: anything below this is a
    # direction the data does not actually carry
    tol = max(centred.shape) * np.finfo(float).eps * (singular[0] if singular.size else 0.0)
    keep = int((singular > tol).sum())
    if keep < len(singular):
        logger.info("short-channel regressors: %d of %d directions are rank deficient and "
                    "were dropped", len(singular) - keep, len(singular))
    return {f"short_ch_pca{i + 1:02d}": u[:, i] / u[:, i].std()
            for i in range(keep) if u[:, i].std() > 0}


def _band_fraction(x: np.ndarray, sfreq: float,
                   l_freq: float | None, h_freq: float | None) -> float:
    """Share of a signal's variance sitting inside ``l_freq``-``h_freq``, from its spectrum.

    Read off the spectrum rather than off the filter's output: a narrow cutoff needs a long
    filter, so on a short recording much of the filtered series is edge transient and its
    variance is not a measure of what was in the band.

        sin(2 pi 0.05 t) over 0.01-0.2 Hz -> 1.0
        sin(2 pi 2.0  t) over 0.01-0.2 Hz -> 0.0
    """
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    freqs = np.fft.rfftfreq(len(x), 1.0 / sfreq)
    inside = np.ones(len(freqs), dtype=bool)
    if l_freq is not None:
        inside &= freqs >= l_freq
    if h_freq is not None:
        inside &= freqs <= h_freq
    return float(spectrum[inside].sum() / total)


def _aux_regressors(
    haemo: mne.io.Raw,
    aux_path: Path | str,
    channels: list[str] | None,
    data_band: tuple[float | None, float | None] | None,
    filter_method: str = DEFAULT_FILTER_METHOD,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> dict[str, np.ndarray]:
    """External confound columns from the aux table preprocessing wrote.

    Three steps, and the order is the whole point:

      1. onto the data's own time axis, anti-aliased (`io.auxiliary.resample_to_grid`)
      2. through the same bandpass the data went through
      3. z-scored

    Step 2 is what short channels get for free by riding through the filter with the data.
    Aux comes from outside the recording, so without it the regressor would carry variance
    the data no longer has and put that variance back into the residual.

    The drift columns cover the equivalent mismatch below the high-pass cutoff; nothing in
    the design matrix spans what sits above the low-pass, so this filter is not optional.
    """
    table = read_aux_table(Path(aux_path))
    available = [c for c in table.columns if c != TIME_COLUMN]
    if channels:
        wanted = [c for c in channels if c in available]
        missing = sorted(set(channels) - set(available))
        if missing:
            logger.warning("aux channels not in %s: %s (have %s)",
                           Path(aux_path).name, missing, available)
        available = wanted
    if not available:
        logger.warning("no usable aux channels in %s, skipping aux regressors", Path(aux_path).name)
        return {}

    t_aux = table[TIME_COLUMN].to_numpy(dtype=float)
    t_dst = haemo.times
    # np.interp holds the end values rather than extrapolating, so a short aux record would
    # silently contribute a constant tail instead of failing
    if t_aux[-1] < t_dst[-1] - 1.0:
        logger.warning("aux table ends at %.1f s but the data runs to %.1f s; "
                       "the last %.1f s of every aux regressor is held constant",
                       t_aux[-1], t_dst[-1], t_dst[-1] - t_aux[-1])

    l_freq, h_freq = data_band or (None, None)
    out: dict[str, np.ndarray] = {}
    for name in available:
        column = resample_to_grid(t_aux, table[name].to_numpy(dtype=float), t_dst)
        if l_freq is not None or h_freq is not None:
            kept = _band_fraction(column, haemo.info["sfreq"], l_freq, h_freq)
            # the same design bandpass_filter used, so regressor and data see one filter
            column = filter_array(
                column[None, :], haemo.info["sfreq"], l_freq, h_freq,
                method=filter_method, order=filter_order,
            )[0]
            # the z-score below rescales whatever survives to unit variance, so a channel
            # with nothing inside the band would arrive as a unit-variance regressor made
            # of filter residue and cost a degree of freedom for it
            if kept < 0.01:
                logger.warning("aux channel %s keeps %.1f%% of its variance inside "
                               "%s-%s Hz; it contributes little but a lost degree of freedom",
                               name, 100 * kept, l_freq, h_freq)
        sd = column.std()
        out[f"aux_{name}"] = (column - column.mean()) / sd if sd > 0 else column - column.mean()

    logger.info("aux regressors: %s", ", ".join(out))
    return out


IMPULSE_REASON = ("a zero-duration event is modelled as a single sample, so its beta scales with "
                  "the sampling rate and is not in concentration units. Give each event its real "
                  "duration (a button press can take 0.1 s).")


def refuse_impulse_events(events: pd.DataFrame) -> None:
    """Stop on any task event lasting 0 s or less, naming its conditions."""
    zero = events[events["duration"] <= 0]
    if zero.empty:
        return
    names = ", ".join(sorted({str(t) for t in zero["trial_type"]}))
    raise ValueError(f"{len(zero)} event(s) of {names} last 0 s: {IMPULSE_REASON}")


def _annotation_events(raw: mne.io.Raw, stim_dur: float | None) -> pd.DataFrame:
    """The recording's own annotations as design events, each lasting ``stim_dur`` seconds."""
    if stim_dur is None:
        raise ValueError("stim_dur required when no events DataFrame provided")
    # BAD_/EDGE_ spans mark unusable frames, not conditions
    keep = [i for i, d in enumerate(raw.annotations.description) if is_marker(d)]
    conditions = raw.annotations.description[keep]
    onsets = raw.annotations.onset[keep] - raw.first_time
    duration = stim_dur * np.ones(len(conditions))
    return pd.DataFrame({"trial_type": conditions, "onset": onsets, "duration": duration})


def build_design_matrix(
    raw: mne.io.Raw,
    stim_dur: float | None,
    hrf_model: HRFModel,
    drift_model: DriftModel,
    high_pass: float | None,
    drift_order: int | None,
    fir_delays: tuple[int, ...] = (0,),
    add_regs: pd.DataFrame | None = None,
    add_reg_names: list[str] | None = None,
    min_onset: float = -24,
    oversampling: int = 50,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a GLM design matrix, wrapping nilearn's make_first_level_design_matrix.

    Modified from mne_nirs.experimental_design.make_first_level_design_matrix to accept
    external events and confounds, and to fall back to snirf annotations when events is None.

    Conditional/non-obvious parameters::

      stim_dur     required when events is None (annotation fallback)
      high_pass    only used when drift_model='cosine'
      drift_order  only used when drift_model='polynomial'
      events       columns 'onset', 'duration', 'trial_type'; None → read from snirf
                   annotations, skipping the BAD_/EDGE_ spans
    """

    from nilearn.glm.first_level import make_first_level_design_matrix

    frame_times = raw.times

    if events is None:
        events = _annotation_events(raw, stim_dur)

    # nilearn reads these four and warns about every other column a BIDS events.tsv carries
    events = events[[c for c in ("trial_type", "onset", "duration", "modulation")
                     if c in events.columns]]
    # nilearn sums events sharing these three into one doubled regressor; a repeated
    # hardware trigger is the usual cause, so keep one copy
    duplicated = events.duplicated(subset=["trial_type", "onset", "duration"])
    if duplicated.any():
        logger.warning("dropped %d duplicated event(s) (same trial_type, onset and duration)",
                       int(duplicated.sum()))
        events = events[~duplicated]

    return make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model=hrf_model,
        drift_model=drift_model if drift_model != "none" else None,
        high_pass=high_pass if drift_model == "cosine" else None,
        drift_order=drift_order if drift_model == "polynomial" else None,
        fir_delays=fir_delays,
        add_regs=add_regs,
        add_reg_names=add_reg_names,
        min_onset=min_onset,
        oversampling=oversampling,
    )


def fit_glm(
    haemo: mne.io.Raw,
    design_matrix: pd.DataFrame,
    noise_model: NoiseModel = "ar1",
    bins: int | None = None,
) -> Any:
    from mne_nirs.statistics import run_glm
    if noise_model.startswith(WORSLEY_PREFIX):
        return fit_worsley(haemo, design_matrix, noise_model)[0]
    if not noise_model.startswith("ar_irls"):
        bins = bins or max(len(haemo.ch_names), AR_BINS_MIN)
        return run_glm(haemo, design_matrix, noise_model=noise_model, bins=bins)
    return _fit_glm_ar_irls(haemo, design_matrix, noise_model)


def constant_channels(haemo: mne.io.Raw, keep: np.ndarray) -> list[str]:
    """Channels holding one value on every kept frame, e.g. a wholly non-finite one prep zeroed."""
    return [ch for ch, x in zip(haemo.ch_names, haemo.get_data()) if np.ptp(x[keep]) == 0]


def _fit_one_model(haemo: mne.io.Raw, design_matrix: pd.DataFrame, noise_model: str,
                   keep: np.ndarray, bins: int) -> tuple[Any, list[str]]:
    """The fit ``noise_model`` names, censored where ``keep`` drops frames, and Worsley's fallbacks."""
    if noise_model.startswith(WORSLEY_PREFIX):
        return fit_worsley(haemo, design_matrix, noise_model,
                           keep=None if keep.all() else keep)
    if keep.all():
        return fit_glm(haemo, design_matrix, noise_model=noise_model, bins=bins), []
    return fit_censored(haemo, design_matrix, keep, noise_model), []


def fit_noise_model(haemo: mne.io.Raw, design_matrix: pd.DataFrame, noise_model: str,
                    keep: np.ndarray) -> tuple[Any, list[str], list[str]]:
    """(results, Worsley's channels fitted as auto, constant channels fitted by OLS)."""
    from mne_nirs.statistics._glm_level_first import RegressionResults

    # from the whole run, so the other channels' AR(1) grid is the one they had with it
    bins = max(len(haemo.ch_names), AR_BINS_MIN)
    # a constant channel's residual is all zeros, from which no AR model can be estimated
    constant = constant_channels(haemo, keep) if noise_model != "ols" else []
    if not constant:
        return *_fit_one_model(haemo, design_matrix, noise_model, keep, bins), []
    logger.info("%d constant channel(s) fitted by OLS, no AR model being estimable from "
                "them: %s", len(constant), constant)
    varying = [ch for ch in haemo.ch_names if ch not in set(constant)]
    fits, fitted_as_auto = {}, []
    if varying:
        est, fitted_as_auto = _fit_one_model(haemo.copy().pick(varying), design_matrix,
                                             noise_model, keep, bins)
        fits.update(est.data)
    fits.update(_fit_one_model(haemo.copy().pick(constant), design_matrix, "ols", keep,
                               bins)[0].data)
    # keyed by the info's own name objects: the container checks identity, not equality
    return (RegressionResults(haemo.info, {ch: fits[ch] for ch in haemo.ch_names},
                              design_matrix),
            fitted_as_auto, constant)


def _fit_glm_ar_irls(haemo: mne.io.Raw, design_matrix: pd.DataFrame, spec: str) -> Any:
    """`run_glm`'s own container, filled channel by channel from the robust solver.

    nilearn's `run_glm` has no robust norm, so the loop here replaces it; what goes into
    `RegressionResults` is the same per-channel object either way, which is what lets the
    contrast, the tidy frame and the residual extraction stay ignorant of the difference.
    """
    from mne_nirs.statistics._glm_level_first import RegressionResults

    from nirspipe.pipeline.ar_irls import fit_channel, resolve_pmax

    pmax = resolve_pmax(spec, haemo.info["sfreq"])
    logger.debug("ar_irls: pmax %d over %d channels", pmax, len(haemo.ch_names))
    design = design_matrix.values
    results = {}
    for ch in haemo.ch_names:
        results[ch] = fit_channel(haemo.get_data(picks=[ch])[0], design, pmax)
        # the robust fits leave reference cycles holding one design-sized array each, and
        # the default thresholds let them pile up, and one collection per channel is cheap
        gc.collect(0)
    return RegressionResults(haemo.info, results, design_matrix)


def read_contrasts(path: str | Path) -> dict[str, dict[str, float]]:
    """A ``--contrast-file`` as ``{contrast: {condition: weight}}``: a TOML file, or a TSV/CSV table.

    A table holds one row per contrast, its name in a ``contrast`` column and a weight under
    each condition's column; an empty cell leaves the condition out, as an absent TOML key does::

        contrast            Tapping_Left  Tapping_Right  Control
        tapping_vs_control  0.5           0.5            -1
        left_minus_right    1             -1

    gives ``{"tapping_vs_control": {"Tapping_Left": 0.5, "Tapping_Right": 0.5, "Control": -1.0},
    "left_minus_right": {"Tapping_Left": 1.0, "Tapping_Right": -1.0}}``. ValueError names the
    line of a cell that is not a number, a name given twice, or a row that weights nothing.
    """
    path = Path(path)
    if path.suffix.lower() == ".toml":
        from nirspipe.utils import load_toml
        return load_toml(path)
    table = read_table(path, dtype=str).fillna("")
    if "contrast" not in table.columns:
        raise ValueError(f"--contrast-file {path} needs a contrast column, then one column "
                         f"per condition")
    contrasts: dict[str, dict[str, float]] = {}
    for line, row in enumerate(table.to_dict("records"), start=2):
        name = str(row.pop("contrast")).strip()
        where = f"--contrast-file {path} line {line}"
        if not name:
            raise ValueError(f"{where}: no contrast name")
        if name in contrasts:
            raise ValueError(f"{where}: contrast {name!r} is named twice")
        weights = {}
        for condition, cell in row.items():
            cell = str(cell).strip()
            if not cell:
                continue
            try:
                weights[condition] = float(cell)
            except ValueError:
                raise ValueError(f"{where}: {condition} {cell!r} is not a number") from None
        if not weights:
            raise ValueError(f"{where}: contrast {name!r} weights no condition")
        contrasts[name] = weights
    return contrasts


def compute_contrasts(
    glm_est: Any,
    contrast_def: dict[str, Any],
    design_matrix: pd.DataFrame,
) -> dict[str, Any]:
    """Each ``{condition: weight}`` mapping as a weight vector over the design's columns.

    ``{"left_minus_right": {"Tapping_Left": 1, "Tapping_Right": -1}}`` on a design carrying
    ``[Tapping_Left, Tapping_Right, constant]``  ->  ``[1, -1, 0]``, then the fit's own
    contrast of that vector.

    A name the design does not carry raises rather than weighting nothing: a contrast file
    is written by hand against condition labels, and a typo there would otherwise be a
    silently empty contrast that still lands in the `desc-contrast_nirsmap.tsv` table.
    """
    columns = list(design_matrix.columns)
    results = {}
    for name, weights in contrast_def.items():
        missing = sorted(set(weights) - set(columns))
        if missing:
            raise StageError(
                f"contrast {name!r} names {missing}, which the design matrix does not "
                f"carry. Its columns are {columns}")
        vector = np.array([float(weights.get(column, 0.0)) for column in columns])
        results[name] = glm_est.compute_contrast(vector)
    return results

def run_glm_pipeline(
    haemo: mne.io.Raw,
    stim_dur: float | None,
    hrf_model: HRFModel,
    noise_model: NoiseModel,
    drift_model: DriftModel,
    high_pass: float | None,
    drift_order: int | None,
    fir_delays: tuple[int, ...] | None,
    events_path: str | None = None,
    events: pd.DataFrame | None = None,
    short_channel: bool | SCRStrategy | None = None,
    # the strategy was chosen by --short-channel auto rather than typed
    short_channel_auto: bool = False,
    aux_path: str | Path | None = None,
    aux_channels: list[str] | None = None,
    # the bandpass applied to `haemo`, not the drift cutoff `high_pass` above. Only the aux
    # regressors need it, and only to be filtered to the same band as the data. The design
    # travels with the band because two designs over one band are not the same filter.
    data_band: tuple[float | None, float | None] | None = None,
    data_filter_method: str = DEFAULT_FILTER_METHOD,
    data_filter_order: int = DEFAULT_FILTER_ORDER,
    contrast_def: dict[str, Any] | None = None,
    output_dir: str | None = None,
    source_path: str | None = None,
    sep_bands=None,
    chromophores: "tuple[str, ...]" = DEFAULT_CHROMOPHORES,
    p_correction: str = "none",
) -> tuple:
    if output_dir and not source_path:
        raise ValueError(
            "GLM outputs take their file name and their Sources entry from the file the data "
            "was read from, and none is known. Pass source_path (to run_post: the "
            "desc-preproc file the haemoglobin was read from)."
        )
    # explicit events take precedence; then external TSV; then snirf annotations
    from_annotations = events is None and not events_path
    if events is None and events_path:
        events = read_table(events_path)
        # a BAD_ row in the table marks unusable time, not a condition
        events = events[events["trial_type"].map(is_marker)]
    if events is None:
        events = _annotation_events(haemo, stim_dur)
    refuse_impulse_events(events)
    # recorded so a reader of the residual can tell a task GLM from a confound regression
    conditions = sorted({str(t) for t in events["trial_type"]})
    # and, where there is a task model, what shaped its regressors
    design_params: dict = {}
    if conditions:
        if from_annotations:
            design_params["stim_dur"] = stim_dur
        elif events_path:
            design_params["event_table"] = True
        if hrf_model == "fir":
            design_params["fir_delays"] = [int(d) for d in (fir_delays or (0,))]

    # short-channel confounds come from `haemo` itself, so they have been through whatever
    # filter it has and cannot re-inject variance the filter removed. External confounds
    # have not, hence `_aux_regressors`
    confound_cols = (_short_channel_regressors(haemo, short_channel, sep_bands)
                     if short_channel else {})
    if aux_path:
        confound_cols.update(_aux_regressors(haemo, aux_path, aux_channels, data_band,
                                             data_filter_method, data_filter_order))
    confounds = pd.DataFrame(confound_cols) if confound_cols else None
    # what was built, not what was asked for. The methods sentence is generated from the
    # sidecar, so a regressor that could not be made must not be named in it: a subject
    # whose short channels were all rejected keeps running, and its text has to say so.
    short_channel_used = short_channel if any(
        c.startswith(SHORT_CH_PREFIX) for c in confound_cols) else None
    auto_params = {"short_channel_auto": True} if short_channel_used and short_channel_auto else {}

    dm = build_design_matrix(
        raw=haemo,
        stim_dur=stim_dur,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        drift_order=drift_order,
        fir_delays=fir_delays,
        add_regs=confounds,
        events=events,
    )
    logger.debug("design matrix: %d scans x %d regressors - %s", dm.shape[0], dm.shape[1], list(dm.columns))

    keep = kept_frames(haemo)
    censor_params: dict = censor_summary(haemo, keep)
    if conditions:
        censor_params["events_left"] = events_left(events, bad_spans(haemo))
    worsley = noise_model.startswith(WORSLEY_PREFIX)
    if not keep.all():
        name = Path(source_path).name if source_path else "the recording"
        dropped = silent_conditions(dm, conditions, keep)
        if dropped:
            logger.warning("%s: no kept frame carries any signal of %s; dropped from the "
                           "design", name, ", ".join(dropped))
            dm = dm.drop(columns=[c for cond in dropped for c in condition_columns(dm, cond)])
            conditions = [c for c in conditions if c not in dropped]
        censor_params["dropped_conditions"] = dropped
        refuse_too_few_frames(keep, dm.shape[1], name)
        left = censor_params.get("events_left", {})
        logger.info("GLM fitted on %d of %d frames (%.0f%% unselected, %.0f%% corrupted, %d "
                    "BAD_ span(s))%s", keep.sum(), len(keep),
                    100 * censor_params["censored_unselected_frac"],
                    100 * censor_params["censored_corrupted_frac"],
                    censor_params["censored_n_spans"],
                    "; events left " + ", ".join(f"{c} {n['left']}/{n['total']}"
                                                 for c, n in left.items()) if left else "")
    # a contrast on a dropped condition has nothing to estimate; the rest of the fit stands
    dropped = set(censor_params.get("dropped_conditions") or [])
    skipped = {name: sorted(set(weights) & dropped)
               for name, weights in (contrast_def or {}).items() if set(weights) & dropped}
    if skipped:
        for name, lost in skipped.items():
            logger.warning("contrast %r not computed: no kept frame carries %s", name,
                           ", ".join(lost))
        contrast_def = {k: v for k, v in contrast_def.items() if k not in skipped} or None
        censor_params["skipped_contrasts"] = skipped
    glm_est, fitted_as_auto, fitted_as_ols = fit_noise_model(haemo, dm, noise_model, keep)
    resid_data = _data_residual(glm_est, haemo, dm, keep)
    raw_resid = haemo.copy()
    raw_resid._data[:] = resid_data
    # spelled drift_high_pass, not high_pass: the sidecar merges these with the bandpass
    # parameters and the two cutoffs would otherwise collide
    if drift_model == "cosine":
        drift_params = {"drift_high_pass": high_pass}
    elif drift_model == "polynomial":
        drift_params = {"drift_order": drift_order}
    else:
        drift_params = {}
    # measured confounds stamped alongside the frequency ones, so a downstream step can read
    # off this file whether the systemic component was regressed out
    # the channels whose bias-corrected AR could not be had at the order, fitted as auto
    noise_params = {"worsley_fitted_as_auto": fitted_as_auto} if worsley else {}
    if fitted_as_ols:
        noise_params["constant_fitted_as_ols"] = fitted_as_ols
    stamp(raw_resid, stage="errts", step="glm_residuals", source=haemo,
          noise_model=noise_model, **noise_params, drift_model=drift_model,
          short_channel=short_channel_used, **auto_params,
          aux_regressors=sorted(k for k in confound_cols if k.startswith("aux_")),
          conditions=conditions,
          # the HRF only shapes a design that holds conditions
          **({"hrf_model": hrf_model} if conditions else {}),
          **design_params, **drift_params, **censor_params)

    # what the tables and the report show: HbT, when asked for, fitted as a series of its own
    # beside the pair, whose residual alone is the one returned and written
    tables_est, tables_raw, tables_resid = glm_est, haemo, resid_data
    self_fitted = sole_regressor_channels(haemo, short_channel_used, sep_bands)
    table_noise = dict(noise_params)
    if output_dir and "hbt" in chromophores:
        from mne_nirs.statistics._glm_level_first import RegressionResults

        hbt = hbt_of(haemo)
        hbt_est, hbt_auto, hbt_ols = fit_noise_model(hbt, dm, noise_model, keep)
        tables_raw = haemo.copy().add_channels([hbt])
        fits = {**glm_est.data, **hbt_est.data}
        # keyed by the info's own name objects: the container checks identity, not equality
        tables_est = RegressionResults(tables_raw.info,
                                       {ch: fits[ch] for ch in tables_raw.ch_names}, dm)
        tables_resid = np.vstack([resid_data, _data_residual(hbt_est, hbt, dm, keep)])
        self_fitted = self_fitted + hbt_names(self_fitted)
        if worsley:
            table_noise["worsley_fitted_as_auto"] = fitted_as_auto + hbt_auto
        if fitted_as_ols or hbt_ols:
            table_noise["constant_fitted_as_ols"] = fitted_as_ols + hbt_ols
    contrasts = compute_contrasts(tables_est, contrast_def, dm) if contrast_def else None

    long_channels: list[str] = []
    if output_dir:
        from nirspipe.qc.metrics._helpers import long_short_channels

        long_channels = long_short_channels(tables_raw, sep_bands)[0]
    # only a task GLM's t values lean on the whitening being complete
    whitening = (_whitening_of(tables_est, tables_raw, tables_resid, keep, long_channels,
                               self_fitted)
                 if output_dir and conditions else None)

    if output_dir:
        _save_glm_outputs(tables_est, dm, Path(output_dir), contrasts=contrasts,
                          contrast_def=contrast_def, source_path=source_path,
                          inputs=glm_inputs(events_path if not from_annotations else None, aux_path),
                          bads=list(tables_raw.info["bads"]),
                          self_fitted=self_fitted, chromophores=chromophores,
                          long_channels=long_channels, p_correction=p_correction,
                          censored=~keep, whitening=whitening, **censor_params,
                          hrf_model=hrf_model, conditions=conditions, **design_params,
                          noise_model=noise_model, **table_noise, drift_model=drift_model,
                          drift_high_pass=high_pass, drift_order=drift_order,
                          short_channel=short_channel_used, **auto_params,
                          aux_regressors=sorted(k for k in confound_cols if k.startswith("aux_")))

    return haemo, tables_est, dm, raw_resid


def _data_residual(glm_est: Any, haemo: mne.io.Raw, design_matrix: pd.DataFrame,
                   keep: np.ndarray) -> np.ndarray:
    """The fit's residual in the data's units, full length; nilearn's `.residuals` subtracts the whitened design's fit instead."""
    if keep.all():
        # nilearn stores per-channel arrays as (n_times, 1); squeeze removes the trailing dim
        return np.array([
            np.asarray(res.Y) - np.asarray(res.model.design) @ np.asarray(res.theta)
            for res in (glm_est.data[ch] for ch in glm_est.ch_names)
        ]).squeeze(-1)
    theta = np.column_stack([np.asarray(glm_est.data[ch].theta).ravel()
                             for ch in haemo.ch_names])
    # estimated on the kept rows, applied to the whole run, so the residual is full length
    return haemo.get_data() - (design_matrix.values @ theta).T


def _whitening_of(glm_est: Any, haemo: mne.io.Raw, residual: np.ndarray, keep: np.ndarray,
                  long_channels: "list[str]",
                  self_fitted: "list[str]") -> tuple[pd.DataFrame, pd.DataFrame]:
    return whitening_tables(
        glm_est, residual, keep, float(haemo.info["sfreq"]),
        [chroma_of(ch) for ch in haemo.ch_names],
        long_channels=set(long_channels),
        bads=set(haemo.info["bads"]), self_fitted=set(self_fitted))


_GLM_ID_COLUMNS = frozenset({"Condition", "ContrastType", "contrast", "Source", "Detector",
                             "Chroma", "ch_name", "bad"})


def blank_self_fitted(df: pd.DataFrame, channels: "list[str]") -> pd.DataFrame:
    """Every statistic of a channel fitted against a copy of itself set to n/a, rows kept."""
    rows = df["ch_name"].isin(channels) if "ch_name" in df.columns else None
    if rows is None or not rows.any():
        return df
    stats = [c for c in df.columns if c not in _GLM_ID_COLUMNS]
    for col in stats:
        # floats take n/a as they are; any other dtype needs room for it
        if not pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(object)
        df.loc[rows, col] = np.nan
    return df


def glm_inputs(events_path: "str | Path | None", aux_path: "str | Path | None") -> list[Path]:
    """The tables a regression read besides the haemoglobin, for the Sources of what it writes."""
    return [Path(p) for p in (events_path, aux_path) if p]


def _glm_name(source_path: "str | None", suffix: str, **extra) -> str:
    """One GLM output's filename, carrying the entities of the recording it was fitted on.

    ``(".../sub-01_task-tapping_desc-resampled_nirs.snirf", "design")``
        -> ``"sub-01_task-tapping_design.tsv"``

    The input's own ``desc`` is dropped: it describes the recording, not the fit. Anything
    in ``extra`` is what distinguishes these outputs from each other, so the caller passes
    ``desc="glm"`` or ``desc="contrast"``.
    """
    stem = Path(source_path).name if source_path else ""
    carried = {key: entity_of(stem, short)
               for key, short in (("subject", "sub"), ("session", "ses"),
                                  ("task", "task"), ("acquisition", "acq"), ("run", "run"))}
    return derivative_path("", suffix, ".tsv", **carried, **extra).name


def _save_glm_outputs(
    glm_est: Any,
    design_matrix: pd.DataFrame,
    output_dir: Path,
    contrasts: dict[str, Any] | None = None,
    contrast_def: dict[str, Any] | None = None,
    source_path: str | None = None,
    inputs: "list[str | Path]" = (),
    bads: list[str] | None = None,
    self_fitted: "list[str]" = (),
    censored: "np.ndarray | None" = None,
    whitening: "tuple[pd.DataFrame, pd.DataFrame] | None" = None,
    chromophores: "tuple[str, ...]" = DEFAULT_CHROMOPHORES,
    long_channels: "list[str]" = (),
    p_correction: str = "none",
    **params: Any,
) -> None:
    bads = bads or []
    table_params = {**params, "chromophores": list(chromophores)}
    p_params = {**table_params, "p_correction": p_correction}
    if p_correction != "none" and not long_channels:
        logger.warning("--p-correction %s: no channel was told long, so no GLM p value was "
                       "corrected", p_correction)

    def _mark_bads(df: pd.DataFrame) -> pd.DataFrame:
        # the fit runs on every channel; the tables carry the chromophores asked for
        if "Chroma" in df.columns:
            df = df[df["Chroma"].isin(chromophores)].reset_index(drop=True)
        # mne-nirs' flag is p < 0.05 uncorrected; a corrected p comes from --p-correction
        df = df.drop(columns="Significant", errors="ignore")
        # the rejected ones are kept and flagged rather than dropped: removing rows would
        # change the shape group analysis expects
        if "ch_name" in df.columns:
            df["bad"] = df["ch_name"].isin(bads)
        return blank_self_fitted(df, self_fitted)

    def _corrected(df: pd.DataFrame, test: str) -> pd.DataFrame:
        """``family`` and the corrected p beside ``p_value``, one family per test and chromophore."""
        if "p_value" not in df.columns:
            return df
        tested = df["ch_name"].isin(long_channels) & ~df["bad"]
        out = correct_p(df, [test, "Chroma"], p_correction, p="p_value", eligible=tested)
        cols = list(df.columns)
        at = cols.index("p_value") + 1
        return out[cols[:at] + [c for c in out.columns if c not in cols] + cols[at:]]

    output_dir.mkdir(parents=True, exist_ok=True)

    def _named(suffix: str, **extra) -> Path:
        """The scheme's name for one GLM output, dropped into the directory given here.

        The name comes from the config so it cannot drift; the directory does not, because
        this function is handed a subject's ``nirs/`` rather than the tree root.
        """
        return output_dir / _glm_name(source_path, suffix, **extra)

    dm_path = _named("design")
    # full length: the rows the fit left out are flagged, not removed
    flags = np.zeros(len(design_matrix), dtype=int) if censored is None else censored.astype(int)
    write_tsv(design_matrix.assign(censored=flags), dm_path)
    write_step_sidecar(dm_path, "design_matrix", source_path, bads, inputs=inputs, **params)

    res_path = _named("nirsmap", desc="glm")
    write_tsv(_corrected(_mark_bads(glm_est.to_dataframe()), "Condition"), res_path)
    write_step_sidecar(res_path, "glm_fit", source_path, bads, inputs=inputs, **p_params)

    if contrasts:
        frames = []
        for name, result in contrasts.items():
            df = _mark_bads(result.to_dataframe())
            df.insert(0, "contrast", name)
            frames.append(df)
        con_path = _named("nirsmap", desc="contrast")
        write_tsv(_corrected(pd.concat(frames, ignore_index=True), "contrast"), con_path)
        # the weights as the file gave them: the TSV names each contrast, nothing else does
        write_step_sidecar(con_path, "contrasts", source_path, bads, inputs=inputs,
                           contrast_weights=contrast_def, **p_params)

    if whitening is not None:
        channels, spectrum = whitening
        channels = channels[channels["Chroma"].isin(chromophores)]
        spectrum = spectrum[[c for c in spectrum.columns
                             if c == "freq" or c.split("_")[0] in chromophores]]
        spectrum_params = {**table_params, "whitening_lowband": list(WHITENING_LOWBAND),
                           "whitening_psd_segment_s": WHITENING_PSD_SEGMENT_S}
        for table, path in zip((channels, spectrum),
                               (_named("nirsmap", desc="whitening"),
                                _named("timeseries", desc="whitening", statistic="psd"))):
            write_tsv(table, path)
            write_step_sidecar(path, "glm_whitening", source_path, bads, inputs=inputs,
                               **spectrum_params)

    logger.info("GLM outputs written to %s", output_dir)
