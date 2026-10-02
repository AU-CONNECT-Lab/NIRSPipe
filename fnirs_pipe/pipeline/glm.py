from __future__ import annotations
import gc
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import mne
import mne.io

from fnirs_pipe.io.tables import read_table
from fnirs_pipe.pipeline.denoise import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_FILTER_ORDER,
    filter_array,
)
from fnirs_pipe.exceptions import StageError
from fnirs_pipe.utils.lineage import stamp
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.io.auxiliary import TIME_COLUMN, read_aux_table, resample_to_grid
from fnirs_pipe.io.derivatives import entity_of, write_step_sidecar
from fnirs_pipe.io.naming import derivative_path

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
NoiseModel = Literal["ols", "ar1", "ar2", "ar3", "ar4", "ar5", "auto", "ar_irls"]
DriftModel = Literal["cosine", "polynomial", "none"]
SCRStrategy = Literal["mean", "pca"]

# the prefix every short-channel confound column carries, whatever the strategy built it
SHORT_CH_PREFIX = "short_ch_"


def _short_channel_regressors(
    haemo: mne.io.Raw, strategy: SCRStrategy, sep_bands=None,
) -> dict[str, np.ndarray]:
    from fnirs_pipe.qc.metrics._helpers import long_short_channels

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
        from fnirs_pipe.qc.metrics._helpers import separation_bands
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
    for lone in sole_regressor_channels(haemo, strategy, sep_bands):
        logger.warning(
            "%s is the only short channel of its chromophore that passed screening, so the "
            "regressor built from it is that channel; its own residual is therefore empty "
            "and its ALFF and FC are left blank. Every other channel is unaffected.", lone)
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

    ``[] `` for a montage with several good short channels a chromophore, ``["S5_D6 hbo"]``
    for one whose only surviving short channel is that one.

    The ``mean`` regressor is the mean of the good short channels of a chromophore. Where
    one of them survives screening, that mean **is** that channel, sample for sample, and
    the design matrix is applied to every channel including the ones it was built from: the
    channel is fitted against a copy of itself and its residual is numerically zero. With
    two or more the residual is the channel minus their mean, which is a real if small
    quantity, so the line is at one and not at some share.

    ``pca`` spans the short channels rather than averaging them, and a single channel's
    basis is again that channel, so it falls the same way.
    """
    if not strategy:
        return []
    from fnirs_pipe.qc.metrics._helpers import long_short_channels

    short_names = long_short_channels(haemo, sep_bands)[1]
    if not short_names:
        return []
    short = haemo.copy().pick(short_names)
    lone = []
    for chromo in ("hbo", "hbr"):
        picks = mne.pick_types(short.info, fnirs=chromo, exclude="bads")
        if len(picks) == 1:
            lone.append(short.ch_names[picks[0]])
    return lone


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

    Conditional/non-obvious parameters:
      stim_dur     required when events is None (annotation fallback)
      high_pass    only used when drift_model='cosine'
      drift_order  only used when drift_model='polynomial'
      events       columns 'onset', 'duration', 'trial_type'; None → read from snirf
                   annotations, skipping the BAD_/EDGE_ spans
    """

    from nilearn.glm.first_level import make_first_level_design_matrix

    frame_times = raw.times

    if events is None:
        if stim_dur is None:
            raise ValueError("stim_dur required when no events DataFrame provided")
        # BAD_/EDGE_ spans mark unusable frames, not conditions
        keep = [i for i, d in enumerate(raw.annotations.description)
                if not str(d).lower().startswith(("bad", "edge"))]
        conditions = raw.annotations.description[keep]
        onsets = raw.annotations.onset[keep] - raw.first_time
        duration = stim_dur * np.ones(len(conditions))
        events = pd.DataFrame({"trial_type": conditions, "onset": onsets, "duration": duration})

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
) -> Any:
    from mne_nirs.statistics import run_glm
    if not noise_model.startswith("ar_irls"):
        return run_glm(haemo, design_matrix, noise_model=noise_model)
    return _fit_glm_ar_irls(haemo, design_matrix, noise_model)


def _fit_glm_ar_irls(haemo: mne.io.Raw, design_matrix: pd.DataFrame, spec: str) -> Any:
    """`run_glm`'s own container, filled channel by channel from the robust solver.

    nilearn's `run_glm` has no robust norm, so the loop here replaces it; what goes into
    `RegressionResults` is the same per-channel object either way, which is what lets the
    contrast, the tidy frame and the residual extraction stay ignorant of the difference.
    """
    from mne_nirs.statistics._glm_level_first import RegressionResults

    from fnirs_pipe.pipeline.ar_irls import fit_channel, resolve_pmax

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
) -> tuple:
    if output_dir and not source_path:
        raise ValueError(
            "GLM outputs take their file name and their Sources entry from the file the data "
            "was read from, and none is known. Pass source_path (to run_post: the "
            "desc-preproc file the haemoglobin was read from)."
        )
    # explicit events take precedence; then external TSV; then snirf annotations
    if events is None:
        events = read_table(events_path) if events_path else None

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

    glm_est = fit_glm(haemo, dm, noise_model=noise_model)

    # the residual in the data's own units. nilearn's `.residuals` subtracts the whitened
    # design's fit instead, which differs from this under any AR noise model
    # nilearn stores per-channel arrays as (n_times, 1); squeeze removes the trailing dim
    resid_data = np.array([
        np.asarray(res.Y) - np.asarray(res.model.design) @ np.asarray(res.theta)
        for res in (glm_est.data[ch] for ch in glm_est.ch_names)
    ]).squeeze(-1)
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
    stamp(raw_resid, stage="errts", step="glm_residuals", source=haemo,
          noise_model=noise_model, drift_model=drift_model,
          short_channel=short_channel_used,
          aux_regressors=sorted(k for k in confound_cols if k.startswith("aux_")),
          **drift_params)

    contrasts = compute_contrasts(glm_est, contrast_def, dm) if contrast_def else None

    if output_dir:
        _save_glm_outputs(glm_est, dm, Path(output_dir), contrasts=contrasts,
                          source_path=source_path, bads=list(haemo.info["bads"]),
                          hrf_model=hrf_model,
                          noise_model=noise_model, drift_model=drift_model,
                          drift_high_pass=high_pass, drift_order=drift_order,
                          short_channel=short_channel_used,
                          aux_regressors=sorted(k for k in confound_cols if k.startswith("aux_")))

    return haemo, glm_est, dm, raw_resid


_TAB = "\t"


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
                                  ("task", "task"), ("run", "run"))}
    return derivative_path("", suffix, ".tsv", **carried, **extra).name


def _save_glm_outputs(
    glm_est: Any,
    design_matrix: pd.DataFrame,
    output_dir: Path,
    contrasts: dict[str, Any] | None = None,
    source_path: str | None = None,
    bads: list[str] | None = None,
    **params: Any,
) -> None:
    bads = bads or []

    def _mark_bads(df: pd.DataFrame) -> pd.DataFrame:
        # the fit runs on every channel, so the rejected ones are kept and flagged rather
        # than dropped: removing rows would change the shape group analysis expects
        if "ch_name" in df.columns:
            df["bad"] = df["ch_name"].isin(bads)
        return df

    output_dir.mkdir(parents=True, exist_ok=True)

    def _named(suffix: str, **extra) -> Path:
        """The scheme's name for one GLM output, dropped into the directory given here.

        The name comes from the config so it cannot drift; the directory does not, because
        this function is handed a subject's ``nirs/`` rather than the tree root.
        """
        return output_dir / _glm_name(source_path, suffix, **extra)

    dm_path = _named("design")
    design_matrix.to_csv(dm_path, index=False, sep=_TAB)
    write_step_sidecar(dm_path, "design_matrix", source_path, bads, **params)

    res_path = _named("nirsmap", desc="glm")
    _mark_bads(glm_est.to_dataframe()).to_csv(res_path, index=False, sep=_TAB)
    write_step_sidecar(res_path, "glm_fit", source_path, bads, **params)

    if contrasts:
        frames = []
        for name, result in contrasts.items():
            df = _mark_bads(result.to_dataframe())
            df.insert(0, "contrast", name)
            frames.append(df)
        con_path = _named("nirsmap", desc="contrast")
        pd.concat(frames, ignore_index=True).to_csv(con_path, index=False, sep=_TAB)
        write_step_sidecar(con_path, "contrasts", source_path, bads, **params)

    logger.info("GLM outputs written to %s", output_dir)
