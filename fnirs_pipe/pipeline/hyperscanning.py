"""Hyperscanning pipeline utilities: group IO, alignment, and signal metrics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import coherence

from fnirs_pipe.exceptions import AlignmentError, GroupCSVError, MissingDerivativesError
from fnirs_pipe.io.derivatives import find_preproc_snirf
from fnirs_pipe.io.snirf import read_snirf
from fnirs_pipe.utils import load_toml
from fnirs_pipe.utils.lineage import path_from
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.hyperscanning")


def _hyper_sidecar(path: Path, step: str, sources: list[str], **params) -> None:
    from fnirs_pipe import __version__
    from fnirs_pipe.io.derivatives import write_sidecar_json
    write_sidecar_json(path, {
        "pipeline_version": __version__,
        "step": step,
        "Sources": sources,
        "parameters": params,
    })


# ---- Data management: group definition & IO ----


@dataclass
class GroupEntry:
    group_id: str
    subject_id: str
    task: str


def parse_group_csv(csv_path: Path) -> dict[tuple[str, str], list[GroupEntry]]:
    """Parse group CSV into {(group_id, task): [GroupEntry, ...]}."""
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        raise GroupCSVError(f"Cannot read CSV {csv_path}: {exc}") from exc

    missing = {"group_id", "subject_id", "task"} - set(df.columns)
    if missing:
        raise GroupCSVError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=["group_id", "subject_id", "task"])
    if df.empty:
        raise GroupCSVError("CSV contains no valid rows after dropping NaN values")

    result: dict[tuple[str, str], list[GroupEntry]] = {}
    for _, row in df.iterrows():
        key = (str(row["group_id"]).strip(), str(row["task"]).strip())
        entry = GroupEntry(
            group_id=str(row["group_id"]).strip(),
            subject_id=str(row["subject_id"]).strip(),
            task=str(row["task"]).strip(),
        )
        result.setdefault(key, []).append(entry)

    for (gid, task), members in result.items():
        if len(members) < 2:
            raise GroupCSVError(
                f"Group '{gid}' task '{task}' has only {len(members)} subject — need at least 2"
            )

    return result


def load_group_haemo(output_dir: Path, group: list[GroupEntry]) -> dict[str, mne.io.Raw]:
    """Load desc-preproc haemo SNIRF for each subject.

    Returns {subject_id: raw_haemo}.
    Raises MissingDerivativesError if any SNIRF is absent.
    """
    result: dict[str, mne.io.Raw] = {}
    for entry in group:
        snirf_path = find_preproc_snirf(output_dir, entry.subject_id, entry.task)
        result[entry.subject_id] = read_snirf(snirf_path, verbose=False)
    return result


def load_group_raw_bids(bids_dir: Path, group: list[GroupEntry]) -> dict[str, mne.io.Raw]:
    """Load raw CW-amplitude SNIRF from BIDS for each group member.

    Returns {subject_id: raw_intensity}.
    Raises MissingDerivativesError if no SNIRF is found for any member.
    """
    from fnirs_pipe.io.bids import get_layout, get_nirs_files

    layout = get_layout(bids_dir, validate=False)
    result: dict[str, mne.io.Raw] = {}
    for entry in group:
        sub_label = entry.subject_id.removeprefix("sub-")
        files = get_nirs_files(layout, subject=sub_label, task=entry.task)
        if not files:
            raise MissingDerivativesError(
                f"No SNIRF found in BIDS for {entry.subject_id} task-{entry.task}"
            )
        result[entry.subject_id] = read_snirf(files[0], verbose=False)
    return result


def _raw_to_haemo(raw: mne.io.Raw, dpf: list[float]) -> mne.io.Raw:
    raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
    # Single value applies to both wavelengths; a list gives one PPF per wavelength.
    ppf = dpf[0] if len(dpf) == 1 else dpf
    return mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=ppf)


# ---- Computation: raw-level QC ----


def compute_group_sqm_raw(
    group: list[GroupEntry],
    raws: dict[str, mne.io.Raw],
    sci_threshold: float,
    output_dir: Path,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> dict[str, dict]:
    """Compute raw-level SQM (SCI, bad channels) for each group member.

    Writes two TSVs to output_dir following BIDS-derivatives conventions:
      group-{gid}_task-{task}_hyper-raw_sqm.tsv      — one row per subject (scalars)
      group-{gid}_task-{task}_hyper-raw_channels.tsv  — one row per subject × channel

    Returns {subject_id: sqm_dict} for use in the HTML report.
    """
    from fnirs_pipe.qc.quantitative_metrics import compute_raw_sqm

    gid  = group[0].group_id
    task = group[0].task
    output_dir.mkdir(parents=True, exist_ok=True)

    sqm_data: dict[str, dict] = {}
    scalar_rows: list[dict] = []
    channel_rows: list[dict] = []

    for entry in group:
        raw = raws[entry.subject_id]

        try:
            raw_od  = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
            sci_arr = mne.preprocessing.nirs.scalp_coupling_index(
                raw_od, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq, verbose=False)
            sci_cw  = {ch: float(sci_arr[i]) for i, ch in enumerate(raw.ch_names)}
        except Exception:
            sci_cw = {ch: float("nan") for ch in raw.ch_names}

        sci_scores: dict[str, float] = {}
        for ch, val in sci_cw.items():
            pair = ch.rsplit(" ", 1)[0]
            sci_scores[f"{pair} hbo"] = val
            sci_scores[pair] = val

        bad_channels = [
            ch for ch, s in sci_cw.items() if s < sci_threshold
        ]

        try:
            sqm = compute_raw_sqm(raw, sci_cw, bad_channels, cardiac_l_freq, cardiac_h_freq)
        except Exception:
            sqm = {}

        sqm["sci_per_channel"] = sci_scores
        sqm["bad_channels"]    = bad_channels
        sqm_data[entry.subject_id] = sqm

        scalar_rows.append({
            "group_id":               gid,
            "subject_id":             entry.subject_id,
            "task":                   task,
            "sci_mean":               sqm.get("sci_mean"),
            "n_bad_channels":         len(bad_channels),
            "channel_retention_rate": sqm.get("channel_retention_rate"),
        })

        for ch, sci_val in sci_cw.items():
            channel_rows.append({
                "group_id":   gid,
                "subject_id": entry.subject_id,
                "task":       task,
                "channel":    ch,
                "sci":        sci_val,
                "is_bad":     ch in bad_channels,
            })

    stem = f"group-{gid}_task-{task}_hyper-raw"
    sources = [p for p in (path_from(raws[e.subject_id]) for e in group) if p]

    scalar_path = output_dir / f"{stem}_sqm.tsv"
    pd.DataFrame(scalar_rows).to_csv(scalar_path, sep="\t", index=False)
    _hyper_sidecar(scalar_path, "group_sqm_raw", sources,
                   sci_threshold=sci_threshold,
                   cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq)

    channel_path = output_dir / f"{stem}_channels.tsv"
    pd.DataFrame(channel_rows).to_csv(channel_path, sep="\t", index=False)
    _hyper_sidecar(channel_path, "group_sqm_raw_channels", sources,
                   sci_threshold=sci_threshold,
                   cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq)

    return sqm_data


# ---- Data management: alignment & signal preprocessing ----


def align_recordings(
    raws: dict[str, mne.io.Raw],
    task: str,
) -> tuple[dict[str, mne.io.Raw], dict[str, float]]:
    """Align N recordings by the first shared annotation trigger.

    Crops each raw from its first occurrence of a common trigger description,
    then trims all to the same duration (shortest post-crop).

    Returns (aligned_raws, {subject_id: crop_offset_seconds}).
    Raises AlignmentError if no shared trigger exists across all subjects.
    """
    # Collect each subject's trigger descriptions, dropping BAD_* motion annotations.
    desc_sets: dict[str, set[str]] = {}
    for sub_id, raw in raws.items():
        desc_sets[sub_id] = {
            ann["description"]
            for ann in raw.annotations
            if not str(ann["description"]).upper().startswith("BAD")
        }

    if not any(desc_sets.values()):
        raise AlignmentError(
            "No non-BAD annotations found in any recording. "
            "Cannot align without shared trigger events."
        )

    # A trigger usable for alignment must be present in every subject.
    common: set[str] = set.intersection(*desc_sets.values()) if desc_sets else set()
    if not common:
        all_descs = {sub: sorted(d) for sub, d in desc_sets.items()}
        raise AlignmentError(
            f"No shared trigger descriptions across all participants. "
            f"Per-subject descriptions: {all_descs}. "
            "Note: alignment requires a shared hardware trigger."
        )

    # Offset = onset of each subject's earliest common trigger (sort by onset).
    offsets: dict[str, float] = {}
    for sub_id, raw in raws.items():
        for ann in sorted(raw.annotations, key=lambda a: float(a["onset"])):
            if ann["description"] in common:
                offsets[sub_id] = float(ann["onset"])
                break
        if sub_id not in offsets:
            raise AlignmentError(f"Subject {sub_id}: no common trigger found (unexpected state)")

    # Crop each recording to start at its trigger, then clip all to a shared length.
    aligned: dict[str, mne.io.Raw] = {}
    for sub_id, raw in raws.items():
        aligned[sub_id] = raw.copy().crop(tmin=offsets[sub_id])

    min_duration = min(r.times[-1] for r in aligned.values())
    for sub_id in list(aligned):
        aligned[sub_id].crop(tmax=min_duration)

    return aligned, offsets


def trim_to_shortest(
    raws: dict[str, mne.io.Raw],
) -> tuple[dict[str, mne.io.Raw], dict[str, float]]:
    """Trim all recordings to the shortest duration without trigger-based alignment.

    Use for resting-state data where no shared trigger exists.
    Returns (trimmed_raws, {subject_id: 0.0}).
    """
    min_duration = min(r.times[-1] for r in raws.values())
    trimmed = {sid: raw.copy().crop(tmax=min_duration) for sid, raw in raws.items()}
    offsets = {sid: 0.0 for sid in raws}
    return trimmed, offsets


def normalize_raws(raws: dict[str, mne.io.Raw]) -> dict[str, mne.io.Raw]:
    """Z-score each channel independently per subject (mean=0, std=1 across time).

    Applied after alignment so all subjects share the same time axis.
    Channels with near-zero variance are left unchanged (divided by 1.0).

    Affects signal overlay display only; coherence and ISC values are scale-invariant
    and unchanged. SCI / bad-channel detection run on raw CW data before normalization.
    """
    result: dict[str, mne.io.Raw] = {}
    for sid, raw in raws.items():
        r = raw.copy()
        data = r.get_data()
        mu = data.mean(axis=1, keepdims=True)
        sd = data.std(axis=1, keepdims=True)
        r._data[:] = (data - mu) / np.where(sd < 1e-12, 1.0, sd)
        result[sid] = r
    return result


# ---- Computation: synchrony metrics (WTC / coherence) ----


@dataclass
class WTCResult:
    """Pairwise wavelet transform coherence per HbO channel.

    ::

      pairs[(sub1, sub2)][ch_name] = {"wtc": ndarray(n_freqs, n_times),
                                       "coi": ndarray(n_times)}

    freqs: ascending Hz.  times: decimated aligned time axis (seconds).
    """
    pairs: dict
    freqs: np.ndarray
    times: np.ndarray


def _pairwise_wtc(
    sig1: np.ndarray,
    sig2: np.ndarray,
    dt: float,
    step: int,
    fmin: float,
    fmax: float,
    significance: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    r"""Morlet wavelet transform coherence for one signal pair (0 = independent, 1 = locked).

    .. math::

        R^2(f, t) = \frac{\left| S(W_{xy}) \right|^2}{S(|W_x|^2)\, S(|W_y|^2)}

    :math:`W_x, W_y` are the continuous wavelet transforms, :math:`W_{xy} = W_x W_y^{*}`
    the cross-wavelet spectrum, and :math:`S` a smoothing operator in time and scale.
    Output is trimmed of pycwt zero-padding, sorted to ascending frequency, then
    band-limited to ``[fmin, fmax]`` and decimated by ``step``. With ``significance``
    the fourth return is the per-frequency Monte Carlo significance level (else None).

    Backend: `pycwt.wct <https://pycwt.readthedocs.io/en/development/reference/#pycwt.wct>`_.
    """
    import pycwt

    WCT, _, coi, freqs, signif = pycwt.wct(
        sig1, sig2, dt=dt,
        dj=1.0 / 12,  # 12 sub-octaves per octave (pycwt default; frequency-axis resolution)
        sig=significance, normalize=True,
    )
    n_sig = len(sig1)
    WCT   = WCT[:, :n_sig]
    coi   = coi[:n_sig]

    order      = np.argsort(freqs)
    freqs_s    = freqs[order]
    WCT_s      = WCT[order]
    band       = (freqs_s >= fmin) & (freqs_s <= fmax)
    WCT_band   = WCT_s[band][:, ::step].astype(np.float32)
    freqs_band = freqs_s[band]
    coi_dec    = coi[::step].astype(np.float32)

    # Per-frequency significance is constant over time: reorder + band-limit only, no decimation.
    sig_band = None
    if significance and np.ndim(signif) == 1 and len(signif) == len(freqs):
        sig_band = np.asarray(signif)[order][band].astype(np.float32)
    return WCT_band, freqs_band, coi_dec, sig_band


def _wtc_over_pairs(
    raws: dict[str, mne.io.Raw],
    signals: dict[str, dict[str, np.ndarray]],
    labels: list[str],
    fmin: float,
    fmax: float,
    significance: bool = False,
) -> WTCResult:
    """Run pairwise Morlet WTC over precomputed per-subject {label: signal} maps.

    Signals are matched by label; a label absent for either subject yields None for that pair.
    Time axis decimated to ~1 Hz for display; frequency axis filtered to [fmin, fmax] Hz.
    significance adds a per-frequency Monte Carlo level to each pair (slow; ~300 surrogate runs).
    """
    subject_ids = list(raws.keys())
    ref_raw = raws[subject_ids[0]]
    sfreq   = float(ref_raw.info["sfreq"])
    dt      = 1.0 / sfreq
    step    = max(1, int(round(sfreq)))

    result_pairs: dict = {}
    shared_freqs: np.ndarray | None = None
    shared_times: np.ndarray | None = None

    for sub1, sub2 in combinations(subject_ids, 2):
        sig_map1, sig_map2 = signals[sub1], signals[sub2]
        pair_data: dict[str, dict | None] = {}
        for label in labels:
            sig1, sig2 = sig_map1.get(label), sig_map2.get(label)
            if sig1 is None or sig2 is None:
                pair_data[label] = None
                continue
            try:
                WCT_band, freqs_band, coi_dec, sig_band = _pairwise_wtc(
                    sig1, sig2, dt, step, fmin, fmax, significance)
                if shared_freqs is None:
                    shared_freqs = freqs_band
                    shared_times = ref_raw.times[::step]
                pair_data[label] = {"wtc": WCT_band, "coi": coi_dec, "sig": sig_band}
            except Exception as exc:
                logger.warning("WTC failed %s-%s label %s: %s", sub1, sub2, label, exc)
                pair_data[label] = None
        result_pairs[(sub1, sub2)] = pair_data

    return WTCResult(
        pairs=result_pairs,
        freqs=shared_freqs if shared_freqs is not None else np.array([]),
        times=shared_times if shared_times is not None else np.array([]),
    )


def compute_wtc(
    raws: dict[str, mne.io.Raw],
    fmin: float = 0.004,
    fmax: float = 0.20,
    significance: bool = False,
) -> WTCResult:
    """Compute pairwise WTC per HbO channel using pycwt Morlet wavelet.

    Channels are matched by S-D label across subjects; time axis decimated to ~1 Hz.
    significance adds a Monte Carlo significance level per pair (slow; see _wtc_over_pairs).
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for WTC")

    signals: dict[str, dict[str, np.ndarray]] = {}
    for sid, raw in raws.items():
        picks = mne.pick_types(raw.info, fnirs="hbo")
        signals[sid] = {
            raw.ch_names[p].rsplit(" ", 1)[0]: raw.get_data(picks=[p])[0].astype(np.float64)
            for p in picks
        }

    return _wtc_over_pairs(
        raws, signals, list(signals[subject_ids[0]]), fmin, fmax, significance)


def _roi_averaged_signals(
    raw: mne.io.Raw,
    roi_map: dict[str, list[str]],
    bad_pairs: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Average HbO channels within each ROI → {roi_name: 1D signal}.

    bad_pairs (S-D labels without suffix) are excluded from the average.
    ROIs left with no usable channel are skipped.
    """
    bad_pairs = bad_pairs or set()
    picks = mne.pick_types(raw.info, fnirs="hbo")
    label_to_data = {
        raw.ch_names[p].rsplit(" ", 1)[0]: raw.get_data(picks=[p])[0].astype(np.float64)
        for p in picks
    }
    out: dict[str, np.ndarray] = {}
    for roi, chs in roi_map.items():
        rows = [label_to_data[c] for c in chs if c in label_to_data and c not in bad_pairs]
        if rows:
            out[roi] = np.mean(rows, axis=0)
    return out


def compute_wtc_roi(
    raws: dict[str, mne.io.Raw],
    roi_map: dict[str, list[str]],
    bad_channels: dict[str, list[str]] | None = None,
    fmin: float = 0.004,
    fmax: float = 0.20,
    significance: bool = False,
) -> WTCResult:
    """Compute pairwise WTC on ROI-averaged HbO signals.

    Averages each ROI's HbO channels into one representative signal per subject,
    excluding the union of all subjects' bad_channels so every subject's ROI signal
    is built from the same channel set. Then runs the same Morlet WTC as compute_wtc.
    Returned WTCResult.pairs is keyed by ROI name instead of channel.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for WTC")

    bad_channels = bad_channels or {}
    bad_pairs_union = {
        ch.rsplit(" ", 1)[0]
        for sid in subject_ids
        for ch in bad_channels.get(sid, [])
    }
    signals = {
        sid: _roi_averaged_signals(raw, roi_map, bad_pairs_union)
        for sid, raw in raws.items()
    }

    return _wtc_over_pairs(raws, signals, list(roi_map), fmin, fmax, significance)


def compute_pairwise_coherence(
    raws: dict[str, mne.io.Raw],
    fmin: float = 0.01,
    fmax: float = 0.10,
) -> pd.DataFrame:
    r"""Magnitude-squared coherence per HbO channel, averaged over [fmin, fmax] Hz.

    .. math::

        C_{xy}(f) = \frac{|P_{xy}(f)|^2}{P_{xx}(f)\, P_{yy}(f)}

    :math:`P_{xy}` is the cross-spectral density and :math:`P_{xx}, P_{yy}` the auto-spectra
    (Welch). The scalar per channel pair is :math:`C_{xy}` averaged over the band.
    Channels are matched by index; all subjects must share the same channel layout.
    Returns a DataFrame with columns: ch_name, sub1, sub2, coherence.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for pairwise coherence")

    ref_raw = raws[subject_ids[0]]
    sfreq = ref_raw.info["sfreq"]
    nperseg = min(512, max(64, ref_raw.n_times // 4))

    rows: list[dict] = []
    for sub1, sub2 in combinations(subject_ids, 2):
        raw1, raw2 = raws[sub1], raws[sub2]
        picks1 = mne.pick_types(raw1.info, fnirs="hbo")
        picks2 = mne.pick_types(raw2.info, fnirs="hbo")
        data1 = raw1.get_data(picks=picks1)
        data2 = raw2.get_data(picks=picks2)
        ch_names1 = [raw1.ch_names[p] for p in picks1]
        ch_names2 = [raw2.ch_names[p] for p in picks2]

        for i in range(min(len(picks1), len(picks2))):
            ch_label = ch_names1[i].rsplit(" ", 1)[0] if " " in ch_names1[i] else ch_names1[i]
            freqs, coh = coherence(data1[i], data2[i], fs=sfreq, nperseg=nperseg)
            mask = (freqs >= fmin) & (freqs <= fmax)
            mean_coh = float(np.mean(coh[mask])) if mask.any() else float("nan")
            rows.append({"ch_name": ch_label, "sub1": sub1, "sub2": sub2, "coherence": mean_coh})

    return pd.DataFrame(rows, columns=["ch_name", "sub1", "sub2", "coherence"])


# ---- Data management: derivatives IO ----


def load_group_sqm(output_dir: Path, group: list[GroupEntry]) -> dict[str, dict]:
    """Load per-subject SQM scalars and channel metrics from derivatives.

    Returns {subject_id: sqm_dict}.
    """
    result: dict[str, dict] = {}
    for entry in group:
        nirs_dir = output_dir / entry.subject_id / "nirs"
        toml_path = nirs_dir / f"{entry.subject_id}_sqm.toml"
        csv_path  = nirs_dir / f"{entry.subject_id}_channel_metrics.csv"

        sqm: dict = {}
        if toml_path.exists():
            sqm.update(load_toml(toml_path))

        if csv_path.exists():
            try:
                ch_df = pd.read_csv(csv_path)
                if {"name", "sci"}.issubset(ch_df.columns):
                    sqm["sci_per_channel"] = dict(
                        zip(ch_df["name"].astype(str),
                            pd.to_numeric(ch_df["sci"], errors="coerce"))
                    )
                if "is_bad" in ch_df.columns:
                    sqm["bad_channels"] = ch_df.loc[
                        ch_df["is_bad"].astype(str).str.lower().isin({"true", "1"}), "name"
                    ].tolist()
            except Exception:
                pass

        result[entry.subject_id] = sqm

    return result
