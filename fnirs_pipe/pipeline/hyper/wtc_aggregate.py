"""Merge the per-dyad WTC band-mean tables into one long table per study.

``fnirs-hyper run`` writes one ``group-<id>_task-<task>_stat-wtc_relmat.tsv`` per dyad and
task, and a dozen more beside it for the ROI means, the conditions and the nulls. A study
with twenty dyads and three conditions ends up with hundreds of files that a group analysis
has to stitch together by hand, and the stitching is where the mistakes live. This produces
the stitched tables instead, with ``group_id`` and ``task`` carried as columns so nothing
about a row depends on the filename it came from.

What counts as one kind is not a list kept here: it is the set of entities a file carries
besides its group and its task. Two files merge together exactly when everything but those
two agrees, which is the rule a merged name follows as well -- the merged table is the
inputs' own name with ``group-`` and ``task-`` taken out, so it lands at the root, where
having no analysis unit in the name is what marks a table as cross-dyad.

The merge refuses more than it warns. A coherence value only means something alongside the
band it was averaged over, and nothing downstream of a concatenated TSV can recover which
band a given row used, so a disagreement here is a stop rather than a caveat.
"""

from __future__ import annotations

import json
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

# never merged, whatever else they carry. The draws are the same null at full detail and
# would double every row of it; the per-scale phase table is a different measurement that
# happens to share these entities. Both are per-dyad files a study reads one at a time.
_NOT_MERGED = frozenset({"draws"})


def _kind_of(path: Path) -> "tuple | None":
    """The entity set that decides which merged table a per-dyad file belongs to.

    ::

      group-07_task-rest_cond-all_stat-wtc_relmat.tsv
        -> (("condition", "all"), ("statistic", "wtc"))

    The group and the task come out, being what varies across the files of one merge; they
    survive as columns. Everything else is what makes two files the same kind. A file this
    module has no business merging answers None.
    """
    from fnirs_pipe.io.naming import parse_path

    entities = parse_path(path.name)
    if entities.get("suffix") != "relmat" or entities.get("desc") in _NOT_MERGED:
        return None
    if not entities.get("group") or not entities.get("task"):
        return None
    dropped = {"group", "task", "suffix", "extension", "datatype", "subject"}
    return tuple(sorted((k, str(v)) for k, v in entities.items() if k not in dropped))


def _merged_path(output_dir: Path, path: Path) -> Path:
    """Where one per-dyad table's merge lands: its own name with group and task taken out."""
    from fnirs_pipe.io.naming import derivative_path, parse_path

    entities = {k: v for k, v in parse_path(path.name).items()
                if k not in ("group", "task", "suffix", "extension", "datatype")}
    return derivative_path(output_dir, "relmat", ".tsv", **entities)


def merge_kinds(output_dir: Path) -> "dict[tuple, list[Path]]":
    """Every per-dyad table under output_dir, grouped into the merges it would produce.

    The discovery half of :func:`write_aggregate_wtc`, separate so a command can report what
    is on disk without merging it.
    """
    kinds: dict[tuple, list[Path]] = {}
    for path in sorted(Path(output_dir).rglob("group-*/**/*_relmat.tsv")):
        kind = _kind_of(path)
        if kind is not None:
            kinds.setdefault(kind, []).append(path)
    return kinds


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


def aggregate_wtc(output_dir: Path, sources: "list[Path]") -> pd.DataFrame:
    """Concatenate the per-dyad tables of one kind, as :func:`merge_kinds` grouped them.

    Returns an empty frame when every input was unreadable or empty, so a study that never
    ran WTC is not an error.
    """
    from fnirs_pipe.io.naming import parse_path

    frames: dict[str, pd.DataFrame] = {}
    params: dict[str, dict] = {}
    for tsv_path in sources:
        entities = parse_path(tsv_path.name)
        group_id, task = entities.get("group"), entities.get("task")
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


def write_aggregate_wtc(output_dir: Path, sources: "list[Path]") -> Path | None:
    """Write one kind's merged table to output_dir, with a sidecar naming its inputs.

    Returns the path, or None when there was nothing to merge.
    """
    from fnirs_pipe import __version__
    from fnirs_pipe.io.derivatives import write_sidecar_json
    from fnirs_pipe.io.naming import parse_path

    merged = aggregate_wtc(output_dir, sources)
    if merged.empty:
        return None

    out_path = _merged_path(output_dir, sources[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep="\t", index=False)

    # the band is uniform by the time we get here, so one file's parameters describe them all
    band = next((_band_params(p) for p in sources), {})
    write_sidecar_json(out_path, {
        "pipeline_version": __version__,
        "step": "hyper_merge",
        "Sources": [p.as_posix() for p in sources],
        "parameters": {
            "n_tables": len(sources),
            "n_dyads": int(merged["group_id"].nunique()),
            "tasks": sorted(merged["task"].unique()),
            "entities": {k: str(v) for k, v in sorted(parse_path(out_path.name).items())
                         if k not in ("suffix", "extension")},
            **{k: band[k] for k in _MUST_AGREE if k in band},
        },
    })
    logger.info("merged %d tables -> %s", len(sources), out_path.name)
    return out_path


def write_all_aggregates(output_dir: Path) -> "list[Path]":
    """Every merge the tree supports, one table per kind. The paths written."""
    return [path for sources in merge_kinds(output_dir).values()
            if (path := write_aggregate_wtc(output_dir, sources)) is not None]
