"""Save the WTC time-frequency maps to disk, and re-average a saved one over a new band.

The band mean written to ``hyper-wtc.tsv`` is the cheap half of the calculation: the wavelet
transform costs minutes per dyad, the averaging costs milliseconds. Keeping only the mean
therefore made "would the result hold over 0.05 to 0.20 Hz?" a question that had to be
answered by recomputing everything. The maps saved here answer it offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fnirs_pipe.pipeline.hyper.synchrony import WTCResult, wtc_band_mean
from fnirs_pipe.utils.logging import get_logger

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

    Stored as float32, which is what the maps already are: a coherence is bounded in [0, 1]
    and nothing downstream reads more than three decimals of it.

    The relative phase is stored beside each map when the pair carries one, so the arrows can
    be redrawn from a saved run. A file written before phase existed simply has none, and
    :func:`load_wtc` gives those pairs a phase of None.
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

      {("sub-p1", "sub-p2", "S1_D1"): ndarray(51,)} -> one npz of 51-long float32 arrays

    Tiny beside the maps: one row per pair, not one map per pair, so it is written whether or
    not ``--wtc-save-maps`` was asked for. The report needs it to draw arrows against the
    null, and re-running the null to recover it costs hours.
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


def reband(path: Path, fmin: float, fmax: float, mask_coi: bool = True) -> pd.DataFrame:
    """Band means over a new band, from saved maps rather than a new wavelet transform.

    The band has to sit inside the range the maps were computed over: what was filtered out
    before saving cannot be recovered here, and :func:`wtc_band_mean` says so if it is empty.
    """
    return wtc_band_mean(load_wtc(path), fmin, fmax, mask_coi=mask_coi)


def reband_tree(
    output_dir: Path, fmin: float, fmax: float, suffix: str | None = None,
    mask_coi: bool = True,
) -> list[Path]:
    """Re-average every saved map under output_dir, writing one TSV beside each npz.

    The new tables are named after the band so they sit next to the original without
    overwriting it, which is the point: the comparison is between them.

    An archive whose name ends in the chromophore gets its ``chromophore`` column back, so
    the re-banded table has the shape ``fnirs-hyper run`` writes. An archive written before
    the chromophore was in the name has no column, and its rows are HbO.
    """
    from fnirs_pipe import __version__
    from fnirs_pipe.io.derivatives import write_sidecar_json

    tag = suffix or f"band{fmin:g}-{fmax:g}".replace(".", "p")
    written: list[Path] = []
    for npz_path in sorted(output_dir.rglob("*_hyper-wtc*.npz")):
        # the null levels sit under the same prefix but hold one row per pair, not a map;
        # without this they would be opened, found to have no map in them, and warned about
        if "-nulllevel-" in npz_path.stem:
            continue
        try:
            df = reband(npz_path, fmin, fmax, mask_coi=mask_coi)
        except Exception as exc:
            logger.warning("skipping %s: %s", npz_path.name, exc)
            continue
        # the archive is per chromophore and says so in its name; the column puts it back,
        # so a re-banded table has the same shape as the one `fnirs-hyper run` wrote
        ch_type = next((c for c in ("hbo", "hbr")
                        if npz_path.stem.endswith(f"-{c}")), None)
        if ch_type:
            df.insert(0, "chromophore", ch_type)
        out_path = npz_path.with_name(f"{npz_path.stem}-{tag}.tsv")
        df.to_csv(out_path, sep="\t", index=False)
        write_sidecar_json(out_path, {
            "pipeline_version": __version__,
            "step": "wtc_reband",
            "Sources": [str(npz_path)],
            "parameters": {"band_fmin": fmin, "band_fmax": fmax, "mask_coi": mask_coi},
        })
        logger.info("reband -> %s (%d rows)", out_path, len(df))
        written.append(out_path)
    return written
