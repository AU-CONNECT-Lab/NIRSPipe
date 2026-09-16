from __future__ import annotations
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
NoiseModel = Literal["ols", "ar1", "ar2", "ar3", "ar4", "ar5", "auto"]
DriftModel = Literal["cosine", "polynomial", "none"]
SCRStrategy = Literal["mean"]

# May add motion parameters (if available) and/or other confounds in the future

def _short_channel_regressors(
    haemo: mne.io.Raw, strategy: SCRStrategy, sep_bands=None,
) -> dict[str, np.ndarray]:
    from fnirs_pipe.qc.metrics._helpers import long_short_channels

    # a --config TOML reaches this past the CLI's own choices
    if strategy is not True and strategy != "mean":
        raise ValueError(
            f"short-channel strategy must be 'mean', got {strategy!r}. The 'pca' strategy "
            "was removed: no reference implementation regresses short channels on a "
            "principal component, and the first one tracks whichever short channel has the "
            "most variance rather than what they share.")
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
    hbo_data = short.get_data(picks=good_hbo)  # (n_channels, n_times)
    hbr_data = short.get_data(picks=good_hbr)
    n_dropped = len(short.ch_names) - len(hbo_data) - len(hbr_data)
    if n_dropped:
        logger.info("short-channel regressors: %d of %d short channels excluded as bad",
                    n_dropped, len(short.ch_names))
    return {
        "short_ch_hbo_mean": hbo_data.mean(axis=0),
        "short_ch_hbr_mean": hbr_data.mean(axis=0),
    }

def _band_fraction(x: np.ndarray, sfreq: float,
                   l_freq: float | None, h_freq: float | None) -> float:
    """Share of a signal's variance sitting inside ``l_freq``-``h_freq``, from its spectrum.

    Measured on the signal rather than on the filter's output, because the two disagree on
    a short recording. A 0.01 Hz cutoff at 10 Hz builds a 3301-tap FIR, so on anything under
    an hour most of the filtered series is edge transient: a 2 Hz tone put through a
    0.01-0.2 Hz band keeps 21% of its variance over 120 s and 8.7% over 400 s, none of it
    signal. The spectrum returns ~0 at every length.

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

    Step 2 is what short channels get for free. They are channels of the same recording, so
    they ride through the filter with everything else and regressor and target end up in one
    frequency band. Aux comes from outside that recording and gets none of it, so it would
    otherwise arrive carrying variance the data no longer has anywhere. That inflates the
    denominator of every beta it appears in, under-correcting inside the band, and puts the
    same out-of-band variance back into the residual the filter had just cleaned.

    The drift columns of the design matrix cover the equivalent mismatch below the high-pass
    cutoff, since they span exactly the frequencies the high-pass removed. Nothing in the
    design matrix spans what sits above the low-pass, which for a motion sensor is most of
    its power, so this filter is not optional.
    """
    from fnirs_pipe.io.auxiliary import TIME_COLUMN, read_aux_table, resample_to_grid

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

    Refs:
      https://mne.tools/mne-nirs/dev/_modules/mne_nirs/experimental_design/_experimental_design.html#make_first_level_design_matrix
      https://nilearn.github.io/dev/modules/generated/nilearn.glm.first_level.make_first_level_design_matrix.html
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
    return run_glm(haemo, design_matrix, noise_model=noise_model)


def compute_contrasts(
    glm_est: Any,
    contrast_def: dict[str, Any],
) -> dict[str, Any]:
    from mne_nirs.statistics import compute_contrast
    return {name: compute_contrast(glm_est, weights) for name, weights in contrast_def.items()}

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
    # explicit events take precedence; then external TSV; then snirf annotations
    if events is None:
        events = read_table(events_path) if events_path else None

    # Short-channel confounds are pulled from `haemo`, the same data the design matrix is fit
    # against. When that data has been bandpass-filtered upstream, the short channels ride
    # through the same filter, so regressors and data live in the same frequency band. That
    # is what keeps the regression from re-injecting out-of-band variance the filter removed
    # (the spectral-misspecification problem of Hallquist 2013; the accepted fix is to filter
    # data and confounds with the same filter before regressing, which holds here implicitly
    # because both derive from one filtered recording). External confounds do not get that
    # for free, which is what `_aux_regressors` filters them for.
    confound_cols = (_short_channel_regressors(haemo, short_channel, sep_bands)
                     if short_channel else {})
    if aux_path:
        confound_cols.update(_aux_regressors(haemo, aux_path, aux_channels, data_band,
                                             data_filter_method, data_filter_order))
    confounds = pd.DataFrame(confound_cols) if confound_cols else None
    # what was built, not what was asked for. The methods sentence is generated from the
    # sidecar, so a regressor that could not be made must not be named in it: a subject
    # whose short channels were all rejected keeps running, and its text has to say so.
    short_channel_used = short_channel if "short_ch_hbo_mean" in confound_cols else None

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

    # nilearn stores residuals as (n_times, 1) per channel; squeeze removes the trailing dim
    resid_data = np.array([glm_est.data[ch].residuals for ch in glm_est.ch_names]).squeeze(-1)
    raw_resid = haemo.copy()
    raw_resid._data[:] = resid_data
    stamp(raw_resid, stage="errts", step="glm_residuals", source=haemo,
          noise_model=noise_model, drift_model=drift_model)

    contrasts = compute_contrasts(glm_est, contrast_def) if contrast_def else None

    if output_dir:
        _save_glm_outputs(glm_est, dm, Path(output_dir), contrasts=contrasts,
                          source_path=source_path, bads=list(haemo.info["bads"]),
                          hrf_model=hrf_model,
                          noise_model=noise_model, drift_model=drift_model,
                          drift_high_pass=high_pass, drift_order=drift_order,
                          short_channel=short_channel_used,
                          aux_regressors=sorted(k for k in confound_cols if k.startswith("aux_")))

    return haemo, glm_est, dm, raw_resid


def _entity_prefix(source_path: str | None) -> str:
    """BIDS entity prefix carried over from the input file, so per-task outputs do not collide.

    .../sub-01_task-tapping_desc-resampled_nirs.snirf -> "sub-01_task-tapping_"
    unknown source                                    -> ""
    """
    if not source_path:
        return ""
    stem = Path(source_path).name.split(".")[0]
    # drop the desc- entity (it describes the input, not these outputs) and the suffix
    tokens = [t for t in stem.split("_") if not t.startswith("desc-")][:-1]
    return "_".join(tokens) + "_" if tokens else ""


def _save_glm_outputs(
    glm_est: Any,
    design_matrix: pd.DataFrame,
    output_dir: Path,
    contrasts: dict[str, Any] | None = None,
    source_path: str | None = None,
    bads: list[str] | None = None,
    **params: Any,
) -> None:
    from fnirs_pipe import __version__
    from fnirs_pipe.io.derivatives import write_sidecar_json

    bads = bads or []

    def _sidecar(path: Path, step: str) -> None:
        write_sidecar_json(path, {
            "pipeline_version": __version__,
            "step": step,
            "Sources": [source_path] if source_path else [],
            "parameters": params,
            "bad_channels": bads,
        })

    def _mark_bads(df: pd.DataFrame) -> pd.DataFrame:
        # the fit runs on every channel, so the rejected ones are kept and flagged rather
        # than dropped: removing rows would change the shape group analysis expects
        if "ch_name" in df.columns:
            df["bad"] = df["ch_name"].isin(bads)
        return df

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _entity_prefix(source_path)
    dm_path = output_dir / f"{prefix}design_matrix.csv"
    design_matrix.to_csv(dm_path, index=False)
    _sidecar(dm_path, "design_matrix")

    res_path = output_dir / f"{prefix}glm_results.csv"
    _mark_bads(glm_est.to_dataframe()).to_csv(res_path, index=False)
    _sidecar(res_path, "glm_fit")
    # glm_est.save(str(output_dir / "glm.h5"), overwrite=True)

    if contrasts:
        frames = []
        for name, result in contrasts.items():
            df = _mark_bads(result.to_dataframe())
            df.insert(0, "contrast", name)
            frames.append(df)
        con_path = output_dir / f"{prefix}contrasts.csv"
        pd.concat(frames, ignore_index=True).to_csv(con_path, index=False)
        _sidecar(con_path, "contrasts")

    logger.info("GLM outputs written to %s", output_dir)
