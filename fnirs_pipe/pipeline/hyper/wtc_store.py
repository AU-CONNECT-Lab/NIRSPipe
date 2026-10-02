"""Save the WTC time-frequency maps to disk, and re-average a saved one over a new band.

The band mean written beside the archive is the cheap half of the calculation, so the maps
saved here let a new band be averaged offline, without a new wavelet transform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import json

from fnirs_pipe.pipeline.hyper.alignment import alignment_params
from fnirs_pipe.pipeline.hyper.group_io import _hyper_sidecar
from fnirs_pipe.pipeline.hyper.wtc import WTCResult, wtc_band_mean, wtc_grid_params
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.io.derivatives import read_json
from fnirs_pipe.io.naming import bids_label, derivative_path, parse_path
from fnirs_pipe.utils.lineage import paths_from

logger = get_logger("pipeline.wtc_store")

_SEP = "\x1f"  # not legal in a BIDS label, so it cannot collide with a channel or ROI name


def _flatten_key(sub1: str, sub2: str, label) -> str:
    """One npz key per stored map. ``("S1_D1", "S2_D2")`` keeps both halves of a crossed pair."""
    label1, label2 = label if isinstance(label, tuple) else (label, "")
    return _SEP.join((sub1, sub2, label1, label2))


def _restore_key(key: str) -> tuple[str, str, "str | tuple[str, str]"]:
    sub1, sub2, label1, label2 = key.split(_SEP)
    return sub1, sub2, ((label1, label2) if label2 else label1)


def save_wtc(result: WTCResult, path: Path) -> Path:
    """Write every pair's coherence map, its cone of influence and the axes to one npz.

    Stored as float32, which is what the maps already are.

    The relative phase is stored beside each map when the pair carries one, so the arrows can
    be redrawn from a saved run. A pair stored without one gets a phase of None from
    :func:`load_wtc`.
    """
    arrays: dict[str, np.ndarray] = {
        "freqs": np.asarray(result.freqs, dtype=np.float32),
        "times": np.asarray(result.times, dtype=np.float32),
    }
    n_pairs = 0
    for (sub1, sub2), labels in result.pairs.items():
        for label, data in labels.items():
            if data is None:
                continue
            key = _flatten_key(sub1, sub2, label)
            arrays[f"wtc{_SEP}{key}"] = np.asarray(data["wtc"], dtype=np.float32)
            arrays[f"coi{_SEP}{key}"] = np.asarray(data["coi"], dtype=np.float32)
            if data.get("phase") is not None:
                arrays[f"phase{_SEP}{key}"] = np.asarray(data["phase"], dtype=np.float32)
            n_pairs += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    logger.info("WTC maps saved: %s (%d pairs, %.1f MB)",
                path, n_pairs, path.stat().st_size / 1e6)
    return path


def load_wtc(path: Path) -> WTCResult:
    """Read back what :func:`save_wtc` wrote, as the same WTCResult the pipeline builds."""
    with np.load(path) as npz:
        freqs, times = npz["freqs"], npz["times"]
        pairs: dict = {}
        for name in npz.files:
            if not name.startswith(f"wtc{_SEP}"):
                continue
            key = name.split(_SEP, 1)[1]
            sub1, sub2, label = _restore_key(key)
            phase_key = f"phase{_SEP}{key}"
            pairs.setdefault((sub1, sub2), {})[label] = {
                "wtc": npz[name],
                "coi": npz[f"coi{_SEP}{key}"],
                "sig": None,
                "phase": npz[phase_key] if phase_key in npz.files else None,
            }
    return WTCResult(pairs=pairs, freqs=freqs, times=times)


def save_null_levels(levels: dict, path: Path) -> Path:
    """Write the phase-scrambled null's per-frequency levels, keyed the way the maps are.

    ::

      {("sub-01", "sub-02", "S1_D1"): ndarray(51,)} -> one npz of 51-long float32 arrays

    Tiny beside the maps: one row per pair, not one map per pair, so it is written whether or
    not ``--wtc-save-maps`` was asked for. The report needs it to draw arrows against the
    null.
    """
    arrays = {_flatten_key(sub1, sub2, label): np.asarray(level, dtype=np.float32)
              for (sub1, sub2, label), level in levels.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    logger.info("WTC null levels saved: %s (%d pairs)", path, len(arrays))
    return path


def load_null_levels(path: Path) -> dict:
    """Read back what :func:`save_null_levels` wrote, keyed as ``result.pairs`` is."""
    with np.load(path) as npz:
        out: dict = {}
        for name in npz.files:
            sub1, sub2, label = _restore_key(name)
            out.setdefault((sub1, sub2), {})[label] = npz[name]
    return out


def save_cond_null_levels(levels: dict, path: Path) -> Path:
    """Write per-condition levels, ``{condition: {(sub1, sub2, label): ndarray}}``, to one npz."""
    arrays = {_SEP.join((cond, _flatten_key(sub1, sub2, label))): np.asarray(level,
                                                                          dtype=np.float32)
              for cond, per_key in levels.items()
              for (sub1, sub2, label), level in per_key.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    logger.info("WTC null levels per condition saved: %s (%d conditions)", path, len(levels))
    return path


def load_cond_null_levels(path: Path) -> dict:
    """Read back :func:`save_cond_null_levels`, as ``{condition: {(sub1, sub2): {label: level}}}``."""
    with np.load(path) as npz:
        out: dict = {}
        for name in npz.files:
            cond, key = name.split(_SEP, 1)
            sub1, sub2, label = _restore_key(key)
            out.setdefault(cond, {}).setdefault((sub1, sub2), {})[label] = npz[name]
    return out


def level_params(raws: dict, *, wtc_fmin: float, wtc_fmax: float, mask_coi: bool,
                 whiten_s: float = 0.0) -> dict:
    """What a per-frequency level depends on, so a report can tell whether one still fits it.

    The writer of a level and the report that thresholds against it both call this on their
    own recordings, and a level is used only when the two dicts agree. The band is not in
    it: a level is per frequency, and the band only chooses which frequencies are averaged.
    Nor is ``--tstart``/``--tend``: the phase-scrambled level is counted over the whole
    record, and a re-paired one is checked condition by condition against its spans.
    """
    return {"wtc_fmin": wtc_fmin, "wtc_fmax": wtc_fmax, "mask_coi": bool(mask_coi),
            "wtc_whiten_s": float(whiten_s or 0.0),
            **wtc_grid_params(raws), **alignment_params(raws)}


def level_mismatch(path: Path, expected: dict, raws: dict) -> "str | None":
    """Why the level at ``path`` cannot be used for these recordings, or None if it can.

    ::

      a level written on desc-preproc, read by a run on desc-errts -> "its Sources differ"
    """
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return f"{sidecar.name} is missing, so nothing says what the level was drawn on"
    try:
        side = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{sidecar.name} is unreadable: {exc}"
    sources = sorted(paths_from(raws.values()))
    if sorted(side.get("Sources") or []) != sources:
        return "its Sources differ from the recordings this run read"
    recorded = side.get("parameters") or {}
    # through JSON so a tuple here and the list it was written as compare equal
    wanted = json.loads(json.dumps(expected))
    differ = sorted(k for k in set(wanted) | {k for k in recorded if k in _LEVEL_KEYS}
                    if recorded.get(k) != wanted.get(k))
    if differ:
        return f"it was drawn with different {', '.join(differ)}"
    return None


# every key level_params can write, so a key the writer recorded and this run lacks counts
_LEVEL_KEYS = {"wtc_fmin", "wtc_fmax", "mask_coi", "wtc_whiten_s", "wtc_dj",
               "wtc_time_step_s", "wtc_scale_smooth_dj0", "aligned", "align_step",
               "align_trigger", "align_offset_s", "aligned_duration_s"}


def reband(path: Path, fmin: float, fmax: float, mask_coi: bool = True) -> pd.DataFrame:
    """Band means over a new band, from saved maps rather than a new wavelet transform.

    The band has to sit inside the range the maps were computed over: what was filtered out
    before saving cannot be recovered here, and :func:`wtc_band_mean` says so if it is empty.
    """
    return wtc_band_mean(load_wtc(path), fmin, fmax, mask_coi=mask_coi)


def _maps_params(npz_path: Path) -> dict:
    """The parameters a saved map's sidecar records, or none for a map saved without one."""
    return read_json(npz_path.with_suffix(".json")).get("parameters") or {}


def reband_tree(
    output_dir: Path, fmin: float, fmax: float, suffix: str | None = None,
    mask_coi: bool = True,
) -> list[Path]:
    """Re-average every saved map under output_dir, writing one TSV beside each npz.

    The new tables are named after the band so they sit next to the original without
    overwriting it.

    An archive carrying a ``chromo-`` entity gets its ``chromophore`` column back, so the
    re-banded table has the shape ``fnirs-hyper`` writes.
    """
    # the band- value, so letters and digits only whatever the caller typed
    tag = bids_label(suffix) if suffix else f"{fmin:g}to{fmax:g}".replace(".", "p")
    written: list[Path] = []
    for npz_path in sorted(output_dir.rglob("*_stat-wtc_relmat.npz")):
        entities = parse_path(npz_path.name)
        # the null levels sit under the same entities but hold one row per pair, not a map
        if entities.get("desc") == "level":
            continue
        try:
            df = reband(npz_path, fmin, fmax, mask_coi=mask_coi)
        except Exception as exc:
            logger.warning("skipping %s: %s", npz_path.name, exc)
            continue
        # the archive is per chromophore and says so in its name; the column puts it back,
        # so a re-banded table has the same shape as the one `fnirs-hyper` wrote
        ch_type = entities.get("chromophore")
        if ch_type:
            df.insert(0, "chromophore", ch_type)
        # the band is in the name here and nowhere else in the scheme: these tables exist to
        # sit beside the one the run wrote, and the band is the only thing telling them apart
        out_path = derivative_path(
            output_dir, "relmat", ".tsv",
            **{k: v for k, v in entities.items()
               if k not in ("suffix", "extension", "datatype")},
            band=tag)
        df.to_csv(out_path, sep="\t", index=False)
        # everything the maps were computed with, which a new band changes none of
        _hyper_sidecar(out_path, "wtc_reband", [str(npz_path)],
                       **{**_maps_params(npz_path),
                          "band_fmin": fmin, "band_fmax": fmax, "mask_coi": mask_coi})
        logger.info("reband -> %s (%d rows)", out_path, len(df))
        written.append(out_path)
    return written
