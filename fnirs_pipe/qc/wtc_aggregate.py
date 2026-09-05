"""Merge the per-dyad WTC band-mean tables into one long table per study.

``fnirs-hyper run`` writes one ``group-<id>_task-<task>_hyper-wtc.tsv`` per dyad and
task. A study with twenty dyads and three conditions therefore ends up with sixty files that
a group analysis has to stitch together by hand, and the stitching is where the mistakes
live. This produces the stitched table instead, with ``group_id`` and ``task`` carried as
columns so nothing about a row depends on the filename it came from.

The merge refuses more than it warns. A coherence value only means something alongside the
band it was averaged over, and nothing downstream of a concatenated TSV can recover which
band a given row used, so a disagreement here is a stop rather than a caveat.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.wtc_aggregate")

# parameters that have to match across every file in a merge, and why they cannot be mixed
# n_iter only ever appears on a pseudo-dyad sidecar, and a file without a key carries no
# opinion, so listing it here guards the null merge without touching the real tables
_MUST_AGREE = ("band_fmin", "band_fmax", "mask_coi", "n_iter")

_KINDS = {
    "wtc":         "group_hyper_wtc",
    "wtc-roichan": "group_hyper_wtc_roichan",
    "wtc-pseudo":  "group_hyper_wtc_pseudo",
}


def _entities(name: str, kind: str) -> tuple[str, str] | None:
    """('group-07_task-rest_hyper-wtc.tsv', 'wtc') -> ('07', 'rest')."""
    match = re.fullmatch(rf"group-([A-Za-z0-9]+)_task-([A-Za-z0-9]+)_hyper-{kind}\.tsv", name)
    return match.groups() if match else None


def _band_params(tsv_path: Path) -> dict:
    sidecar = tsv_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text()).get("parameters", {})
    except (OSError, json.JSONDecodeError):
        logger.warning("unreadable sidecar, its band goes unchecked: %s", sidecar)
        return {}


def _refuse_mixed_bands(seen: dict[str, dict]) -> None:
    """Raise if the files disagree on the band their coherence was averaged over.

    ``seen`` maps a display name to that file's sidecar parameters. A file whose sidecar is
    missing carries no opinion and does not block the merge; the caller has already been
    warned about it.
    """
    for key in _MUST_AGREE:
        values = {name: params[key] for name, params in seen.items() if key in params}
        distinct = set(values.values())
        if len(distinct) > 1:
            spread = "\n".join(f"  {name}: {key}={value}"
                               for name, value in sorted(values.items()))
            raise ValueError(
                f"the WTC tables disagree on {key}, so their coherence columns are not "
                f"comparable and merging them would hide it:\n{spread}\n"
                f"Re-run `fnirs-hyper run` for the odd ones out with a matching band, or aggregate "
                f"them separately."
            )


def _refuse_mixed_shapes(frames: dict[str, pd.DataFrame]) -> None:
    """Raise if some tables are crossed and some are not.

    A crossed table carries ``label2``; a homologous one does not. Concatenating the two
    leaves half a column empty, and an empty ``label2`` is indistinguishable from a genuinely
    missing value. ``--wtc-channel-cross`` is what crosses them, at channel level and, since
    the ROI numbers are grouped from the channel ones, at ROI level too; it can be on for
    some dyads only.
    """
    crossed = {name for name, df in frames.items() if "label2" in df.columns}
    if crossed and len(crossed) != len(frames):
        plain = sorted(set(frames) - crossed)
        raise ValueError(
            f"some tables are crossed (they carry label2) and some are not: "
            f"crossed={sorted(crossed)}, homologous={plain}. Merging them would leave "
            f"label2 half empty, which reads as missing data rather than as a different "
            f"analysis. Aggregate the two sets separately."
        )


def aggregate_wtc(output_dir: Path, kind: str = "wtc") -> pd.DataFrame:
    """Concatenate every per-dyad WTC band-mean table under output_dir.

    kind is "wtc" for the channel-level tables, "wtc-roichan" for the ROI-level ones or
    "wtc-pseudo" for the phase-scrambled null. Returns
    an empty frame when nothing matches, so a study that never ran WTC is not an error.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {kind!r}")

    frames: dict[str, pd.DataFrame] = {}
    params: dict[str, dict] = {}
    for tsv_path in sorted(output_dir.rglob(f"*_hyper-{kind}.tsv")):
        ents = _entities(tsv_path.name, kind)
        if ents is None:
            continue
        group_id, task = ents
        try:
            df = pd.read_csv(tsv_path, sep="\t")
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("skipping unreadable table %s: %s", tsv_path, exc)
            continue
        if df.empty:
            continue
        df.insert(0, "task", task)
        df.insert(0, "group_id", group_id)
        frames[tsv_path.name] = df
        params[tsv_path.name] = _band_params(tsv_path)

    if not frames:
        logger.info("no hyper-%s tables under %s", kind, output_dir)
        return pd.DataFrame()

    _refuse_mixed_bands(params)
    _refuse_mixed_shapes(frames)

    merged = pd.concat(frames.values(), ignore_index=True)
    sort_cols = [c for c in ("group_id", "task", "sub1", "sub2", "label", "label2")
                 if c in merged.columns]
    return merged.sort_values(sort_cols, ignore_index=True)


def write_aggregate_wtc(output_dir: Path, kind: str = "wtc") -> Path | None:
    """Write the merged table to output_dir, with a sidecar naming its inputs.

    Returns the path, or None when there was nothing to merge.
    """
    from fnirs_pipe import __version__
    from fnirs_pipe.io.derivatives import write_sidecar_json

    merged = aggregate_wtc(output_dir, kind=kind)
    if merged.empty:
        return None

    out_path = output_dir / f"{_KINDS[kind]}.tsv"
    merged.to_csv(out_path, sep="\t", index=False)

    sources = sorted(p for p in output_dir.rglob(f"*_hyper-{kind}.tsv")
                     if _entities(p.name, kind))
    # the band is uniform by the time we get here, so one file's parameters describe them all
    band = next((_band_params(p) for p in sources), {})
    write_sidecar_json(out_path, {
        "pipeline_version": __version__,
        "step": _KINDS[kind],
        "Sources": [p.as_posix() for p in sources],
        "parameters": {
            "n_tables": len(sources),
            "n_dyads": int(merged["group_id"].nunique()),
            "tasks": sorted(merged["task"].unique()),
            **{k: band[k] for k in _MUST_AGREE if k in band},
        },
    })
    logger.info("merged %d %s tables -> %s", len(sources), kind, out_path)
    return out_path
