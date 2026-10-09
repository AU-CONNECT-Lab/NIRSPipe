"""Merge the per-dyad WTC band-mean tables into one long table per study.

``fnirs-hyper`` writes one ``group-<id>_task-<task>_stat-wtc_relmat.tsv`` per dyad and
task, and a dozen more beside it for the ROI means, the conditions and the nulls. This
produces the stitched tables, with ``group_id`` and ``task`` carried as columns so nothing
about a row depends on the filename it came from.

What counts as one kind is not a list kept here: it is the set of entities a file carries
besides its group and its task. Two files merge together exactly when everything but those
two agrees, which is the rule a merged name follows as well -- the merged table is the
inputs' own name with ``group-`` and ``task-`` taken out, so it lands at the root, where
having no analysis unit in the name is what marks a table as cross-dyad.

The merge refuses more than it warns: nothing downstream of a concatenated TSV can recover
which band a given row used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.pipeline.hyper.group_io import _hyper_sidecar
from fnirs_pipe.io.naming import parse_path, derivative_path
from fnirs_pipe.io.tables import write_tsv

logger = get_logger("pipeline.wtc_aggregate")

# parameters that have to match across every file in a merge. null_kind and pair_pool only
# ever appear on a null's sidecar, and a file without a key carries no opinion, so listing
# them guards the null merges without touching the real tables.
# n_iter is not here: it is a column of the table, not a property of one, so mixing it
# leaves every row readable and separable. `_warn_mixed_iterations` says what it costs
_MUST_AGREE = ("band_fmin", "band_fmax", "mask_coi", "null_kind", "pair_pool", "wtc_whiten_s",
               "desc", "bads_scope", "roi_min_channels")
# an unwhitened table carries no wtc_whiten_s, and that absence is an opinion: it was not
# whitened, so it must not merge with one that was
_ABSENT_MEANS = {"wtc_whiten_s": 0.0}

# never merged: the draws are the same null at full detail and would double every row of it
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
    entities = parse_path(path.name)
    if entities.get("suffix") != "relmat" or entities.get("desc") in _NOT_MERGED:
        return None
    if not entities.get("group") or not entities.get("task"):
        return None
    dropped = {"group", "session", "task", "suffix", "extension", "datatype", "subject"}
    return tuple(sorted((k, str(v)) for k, v in entities.items() if k not in dropped))


def _merged_path(output_dir: Path, path: Path) -> Path:
    """Where one per-dyad table's merge lands: its own name with group and task taken out."""
    entities = {k: v for k, v in parse_path(path.name).items()
                if k not in ("group", "session", "task", "suffix", "extension", "datatype")}
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
        values = {name: params.get(key, _ABSENT_MEANS.get(key)) for name, params in seen.items()
                  if key in params or key in _ABSENT_MEANS}
        distinct = set(values.values())
        if len(distinct) > 1:
            spread = "\n".join(f"  {name}: {key}={value}"
                               for name, value in sorted(values.items()))
            raise ValueError(
                f"the WTC tables disagree on {key}, so their coherence columns are not "
                f"comparable and merging them would hide it:\n{spread}\n"
                f"Re-run `fnirs-hyper` for the odd ones out with matching settings, or "
                f"aggregate them separately."
            )


def _refuse_mixed_shapes(frames: dict[str, pd.DataFrame]) -> None:
    """Raise if some tables are crossed and some are not.

    A crossed table carries ``label2``; a homologous one does not. Concatenating the two
    leaves half a column empty, and an empty ``label2`` is indistinguishable from a genuinely
    missing value. ``--channel-cross`` is what crosses them, at channel level and, since
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


# A wide matrix says so in its first column: `write_isc_matrix` writes the index under one
# of these, and everything after it is a channel or a region rather than a variable.
_MATRIX_INDEX = ("channel", "roi")


def _refuse_ragged_matrices(frames: dict[str, pd.DataFrame]) -> None:
    """Raise if the wide matrices being merged do not share a set of columns.

    A long table survives a column its neighbour lacks: the value is NaN in those rows and a
    reader can see which rows and why. A wide matrix does not, because its columns *are* the
    data: two dyads screened onto different channel sets concatenate to a grid whose blanks
    are indistinguishable from a correlation that was measured and came out empty.

    Only the matrices. The long tables have their own two guards above, one of which
    deliberately tolerates a missing chromophore column.
    """
    # `group_id` and `task` are already on the front by now, so the index column is the
    # first one that is neither
    wide = {name: df for name, df in frames.items()
            if next((c for c in df.columns if c not in ("group_id", "session", "task")), None)
            in _MATRIX_INDEX}
    if len(wide) < 2:
        return
    shapes = {name: tuple(df.columns) for name, df in wide.items()}
    if len(set(shapes.values())) == 1:
        return
    common = set.intersection(*(set(c) for c in shapes.values()))
    spread = "\n".join(
        f"  {name}: {len(cols)} column(s), {sorted(set(cols) - common) or 'none'} not shared"
        for name, cols in sorted(shapes.items()))
    raise ValueError(
        f"these matrices do not cover the same channels, and a matrix's columns are its "
        f"data, so merging them would fill the gaps with blanks nothing can tell from a "
        f"correlation that came out empty:\n{spread}\n"
        f"Aggregate the sets separately, or screen the dyads onto one channel set."
    )


def _warn_mixed_iterations(seen: dict[str, dict]) -> None:
    """Warn, not refuse, when the groups' nulls rest on different numbers of draws.

    Unlike a band, ``n_iter`` is a column of the merged table, so a reader can see what each
    row rests on and split or weight by it. What it costs is the resolution of ``percentile``:
    a null of 20 draws ranks a real value to about 5%, one of 2 draws to 50%, and one of a
    single draw has two possible answers.

    For the re-paired null the pool is the other groups whose own block of the condition is
    at least as long as this one's, so its draw count tracks block length.
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
    different dyad count per chromophore.
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
    frames: dict[str, pd.DataFrame] = {}
    params: dict[str, dict] = {}
    # a column only where some group was recorded per session, so a tree without sessions
    # merges exactly as before
    sessions = any(parse_path(p.name).get("session") for p in sources)
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
        if sessions:
            df.insert(0, "session", entities.get("session"))
        df.insert(0, "group_id", group_id)
        frames[tsv_path.name] = df
        params[tsv_path.name] = _band_params(tsv_path)

    if not frames:
        return pd.DataFrame()

    _refuse_mixed_bands(params)
    _refuse_mixed_shapes(frames)
    _refuse_ragged_matrices(frames)
    _warn_mixed_iterations(params)
    _warn_mixed_chromophores(frames)

    merged = pd.concat(frames.values(), ignore_index=True)
    sort_cols = [c for c in ("group_id", "session", "task", "chromophore", "sub1", "sub2",
                             "label", "label2")
                 if c in merged.columns]
    return merged.sort_values(sort_cols, ignore_index=True)


def write_aggregate_wtc(output_dir: Path, sources: "list[Path]") -> Path | None:
    """Write one kind's merged table to output_dir, with a sidecar naming its inputs.

    Returns the path, or None when there was nothing to merge.
    """
    merged = aggregate_wtc(output_dir, sources)
    if merged.empty:
        return None

    out_path = _merged_path(output_dir, sources[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(merged, out_path)

    # the band is uniform by the time we get here, so one file's parameters describe them all
    band = next((_band_params(p) for p in sources), {})
    _hyper_sidecar(
        out_path, "hyper_merge", [p.as_posix() for p in sources],
        n_tables=len(sources),
        n_dyads=int(merged["group_id"].nunique()),
        tasks=sorted(merged["task"].unique()),
        entities={k: str(v) for k, v in sorted(parse_path(out_path.name).items())
                  if k not in ("suffix", "extension")},
        **{k: band[k] for k in _MUST_AGREE if k in band},
    )
    logger.info("merged %d tables -> %s", len(sources), out_path.name)
    return out_path


def write_all_aggregates(output_dir: Path) -> "list[Path]":
    """Every merge the tree supports, one table per kind. The paths written."""
    return [path for sources in merge_kinds(output_dir).values()
            if (path := write_aggregate_wtc(output_dir, sources)) is not None]
