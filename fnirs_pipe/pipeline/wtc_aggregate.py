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

logger = get_logger("pipeline.wtc_aggregate")

# parameters that have to match across every file in a merge, and why they cannot be mixed.
# n_iter, null_kind and pair_pool only ever appear on a null's sidecar, and a file without a
# key carries no opinion, so listing them guards the null merges without touching the real
# tables. null_kind is what keeps the two nulls apart if one is renamed onto the other's
# path: they answer different questions and a table holding both answers neither
# n_iter is not here: it is a column of the table, not a property of one, so mixing it
# leaves every row readable and separable. `_warn_mixed_iterations` says what it costs
_MUST_AGREE = ("band_fmin", "band_fmax", "mask_coi", "null_kind", "pair_pool")

_KINDS = {
    "wtc":                "group_hyper_wtc",
    "wtc-roichan":        "group_hyper_wtc_roichan",
    # the homologous ROI mean and its null: four rows a condition, the number to report
    "wtc-roihom":         "group_hyper_wtc_roihom",
    "wtc-roihom-phasenull":  "group_hyper_wtc_roihom_phasenull",
    "wtc-phasenull":         "group_hyper_wtc_phasenull",
    # --wtc-by-condition writes these beside the whole-run pair above; they carry a
    # `condition` column and are merged separately, never into the whole-run table
    "wtcbycond":          "group_hyper_wtc_bycondition",
    "wtcbycond-roichan":  "group_hyper_wtc_bycondition_roichan",
    "wtcbycond-roihom":   "group_hyper_wtc_bycondition_roihom",
    "wtcbycond-roihom-phasenull": "group_hyper_wtc_bycondition_roihom_phasenull",
    # the null for the pair above, windowed off the same transform they are
    "wtcbycond-phasenull":   "group_hyper_wtc_bycondition_phasenull",
    # the re-paired null, drawn across the cohort rather than inside one dyad. Merged apart
    # from the phase-scrambled tables on purpose: same columns, different question
    "wtc-pairnull":              "group_hyper_wtc_pairnull",
    "wtc-roihom-pairnull":       "group_hyper_wtc_roihom_pairnull",
    "wtcbycond-pairnull":        "group_hyper_wtc_bycondition_pairnull",
    "wtcbycond-roihom-pairnull": "group_hyper_wtc_bycondition_roihom_pairnull",
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


def _warn_mixed_iterations(seen: dict[str, dict]) -> None:
    """Warn, not refuse, when the groups' nulls rest on different numbers of draws.

    Unlike a band, ``n_iter`` is a column of the merged table, so a reader can see what each
    row rests on and split or weight by it. What it costs is the resolution of ``percentile``:
    a null of 22 draws ranks a real value to about 5%, one of 2 draws to 50%, and one of a
    single draw has two possible answers. Comparing percentiles across groups without looking
    at ``n_iter`` treats those as the same number.

    For the re-paired null it is worse than uneven, it is **systematic**. The pool is the
    other groups whose recording reaches this one's length, so a long recording has few
    stand-ins and a short one has many, and the resolution of the null ends up correlated
    with duration. Duration also moves coherence, since a shorter transform loses a larger
    share of its band to the cone. So the groups with the coarsest null are not a random
    subset. Report ``n_iter`` beside any percentile drawn from this table.
    """
    counts = {name: params["n_iter"] for name, params in seen.items() if "n_iter" in params}
    if len(set(counts.values())) <= 1:
        return
    spread = ", ".join(f"{name.split('_task-')[0]}={n}" for name, n in sorted(counts.items()))
    logger.warning(
        "the groups' nulls rest on %d to %d draws, so their percentile columns do not have "
        "one resolution: %s. The merged table keeps n_iter per row; read it beside any "
        "percentile, and see that a re-paired null's draw count tracks recording length.",
        min(counts.values()), max(counts.values()), spread)


def _warn_mixed_chromophores(frames: dict[str, pd.DataFrame]) -> None:
    """Warn, not refuse, when the tables do not all carry the same chromophores.

    Unlike a band or a crossing, a chromophore is a *row* label, so mixing does not corrupt
    a column: the rows stay readable and separable. It does mean the merged table has a
    different dyad count per chromophore, which a group model will silently absorb, so it is
    worth a line in the log.
    """
    seen = {name: tuple(sorted(df["chromophore"].dropna().unique()))
            for name, df in frames.items() if "chromophore" in df.columns}
    missing = sorted(set(frames) - set(seen))
    if len(set(seen.values())) > 1 or (seen and missing):
        spread = ", ".join(f"{name}={'+'.join(chroma) or 'none'}"
                           for name, chroma in sorted(seen.items()))
        logger.warning(
            "the tables do not all carry the same chromophores, so the merged table has a "
            "different number of dyads per chromophore: %s%s. Nothing is corrupted, since "
            "the chromophore is a row label, but a group model will not notice.",
            spread,
            f"; no chromophore column in {', '.join(missing)}" if missing else "")


def aggregate_wtc(output_dir: Path, kind: str = "wtc") -> pd.DataFrame:
    """Concatenate every per-dyad WTC band-mean table under output_dir.

    kind is "wtc" for the channel-level tables, "wtc-roichan" for the ROI-level ones,
    "wtc-phasenull" for the phase-scrambled null, or the "wtcbycond" trio for what
    ``--wtc-by-condition`` wrote, its ROI means and its own null. Returns an empty frame
    when nothing matches, so a study that never ran WTC is not an error.
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
    _warn_mixed_iterations(params)
    _warn_mixed_chromophores(frames)

    merged = pd.concat(frames.values(), ignore_index=True)
    sort_cols = [c for c in ("group_id", "task", "chromophore", "sub1", "sub2",
                             "label", "label2")
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
