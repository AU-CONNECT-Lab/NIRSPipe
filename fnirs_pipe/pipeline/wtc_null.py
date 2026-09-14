"""The pseudo-dyad null, computed and written on its own.

Called from ``fnirs-hyper run`` when ``--wtc-pseudo`` is given, so it inherits that run's
stage, band and window by construction: a null averaged over a different band is not the
null for the table it sits beside. Crossing is the one thing it does not inherit. The null
used to take ``--wtc-channel-cross`` from the real run, so asking for the exploratory
196-pair channel table also multiplied the surrogate cost by 14; it is now the null's own
decision, ``--wtc-pseudo-cross``, and defaults to off.

The chromophores are inherited, unlike crossing: a null missing one leaves that half of the
real table with nothing to be tested against, which is not a saving worth offering.

Two halves, and the report sits between them. :func:`run_wtc_null` draws the surrogates and
writes the per-frequency level the phase arrows are drawn against, which has to exist before
the figures are built. :func:`write_wtc_null` ranks those draws against the real band means,
which the same report writes. One call could only ever have satisfied one of the two.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run_wtc_null(
    group_id: str,
    task: str,
    aligned_raws: dict,
    output_dir: Path,
    n_iter: int = 100,
    wtc_fmin: float = 0.004,
    wtc_fmax: float = 0.20,
    band_fmin: float | None = None,
    band_fmax: float | None = None,
    seed: int | None = None,
    cross: bool = False,
    limit_scales: bool = True,
    mask_coi: bool = True,
    chroma: "tuple[str, ...] | list[str]" = ("hbo", "hbr"),
    sep_bands=None,
    windows: "list[tuple[str, float, float]] | None" = None,
    analysis_window: "tuple[float, float] | None" = None,
) -> dict:
    """Draw the null for one dyad and write the level its phase arrows are read against.

    The expensive half: ``n_iter`` full WTC runs per chromophore. Writes
    ``...hyper-wtc-nulllevel-<chroma>.npz``, the coherence a cell has to clear at each
    frequency to beat the null, and returns ``{chromophore: PseudoNull}`` for
    :func:`write_wtc_null` to rank once the real tables exist.

    ``chroma`` has to cover the real run's chromophores: a null computed on HbO says nothing
    about an HbR coupling, so a table missing one chromophore leaves that half of the real
    table with nothing to be tested against.

    ``analysis_window`` is ``--tstart``/``--tend``. The whole-run table has to be windowed
    the same way the real whole-run table was, or the two describe different spans of the
    recording and the comparison between them is not one.

    ``windows`` is what ``--wtc-by-condition`` read its conditions out of, and carries through
    to the second table. Both tables come out of this one pass of ``n_iter`` transforms.
    """
    # imported in the call, not at module load: the wiring tests patch these on the module
    # that defines them, which only a lookup made at call time can see
    from fnirs_pipe.io.derivatives import group_data_dir
    from fnirs_pipe.pipeline.hyperscanning import compute_wtc_pseudo
    from fnirs_pipe.pipeline.wtc_store import save_null_levels

    band_fmin = band_fmin if band_fmin is not None else wtc_fmin
    band_fmax = band_fmax if band_fmax is not None else wtc_fmax

    chroma = tuple(dict.fromkeys(chroma))
    if not chroma or set(chroma) - {"hbo", "hbr"}:
        raise ValueError(f"chroma must be some of ('hbo', 'hbr'), got {chroma!r}")

    logger.info(
        "Pseudo-dyad WTC: %d phase-scrambled iterations, %s pairs, one full WTC run each, "
        "per chromophore (%s).",
        n_iter, "crossed" if cross else "homologous", "+".join(chroma))

    data_dir = group_data_dir(output_dir, group_id)
    stem = f"group-{group_id}_task-{task}_hyper"
    nulls: dict = {}
    for ch_type in chroma:
        null = compute_wtc_pseudo(
            aligned_raws, band_fmin, band_fmax, n_iter=n_iter,
            fmin=wtc_fmin, fmax=wtc_fmax, seed=seed, cross=cross,
            limit_scales=limit_scales, mask_coi=mask_coi, ch_type=ch_type,
            sep_bands=sep_bands, windows=windows, analysis_window=analysis_window)
        nulls[ch_type] = null
        if getattr(null, "levels", None):
            save_null_levels(null.levels,
                             data_dir / f"{stem}-wtc-nulllevel-{ch_type}.npz")
    return nulls


def write_wtc_null(
    nulls: dict,
    group_id: str,
    task: str,
    aligned_raws: dict,
    output_dir: Path,
    n_iter: int = 100,
    wtc_fmin: float = 0.004,
    wtc_fmax: float = 0.20,
    band_fmin: float | None = None,
    band_fmax: float | None = None,
    seed: int | None = None,
    cross: bool = False,
    mask_coi: bool = True,
    windows: "list[tuple[str, float, float]] | None" = None,
    analysis_window: "tuple[float, float] | None" = None,
) -> Path:
    """Rank what :func:`run_wtc_null` drew against the real band means, and write it.

    Writes ``group-<id>_task-<task>_hyper-wtc-pseudo.tsv``, one table with a ``chromophore``
    column, matching the real band-mean tables. The sidecar additionally records ``n_iter``,
    ``cross`` and ``chroma``, without which a 5-iteration probe and a 100-iteration null are
    indistinguishable on disk.

    ``windows`` adds a second table, ``...hyper-wtcbycond-pseudo.tsv``, with a ``condition``
    column: the null for what ``--wtc-by-condition`` wrote. It mirrors the real side, where
    the whole-run and per-condition tables are also two files merged separately. Returns the
    whole-run path either way; the per-condition one sits beside it.

    The real tables are read back off disk rather than passed in, the report step having just
    written them, and are what gives both tables their ``percentile`` column. They are only
    ever a few hundred rows, and reading them here keeps the report's own signature out of
    the null's business. A tree without them still gets a null, just one nothing has been
    ranked against yet.
    """
    from fnirs_pipe.io.derivatives import group_data_dir
    from fnirs_pipe.pipeline.hyperscanning import _hyper_sidecar, alignment_params
    from fnirs_pipe.pipeline.synchrony import wtc_grid_params
    from fnirs_pipe.utils.lineage import path_from

    band_fmin = band_fmin if band_fmin is not None else wtc_fmin
    band_fmax = band_fmax if band_fmax is not None else wtc_fmax
    chroma = tuple(nulls)

    data_dir = group_data_dir(output_dir, group_id)
    stem = f"group-{group_id}_task-{task}_hyper"
    real = _real_table(data_dir / f"{stem}-wtc.tsv")
    real_by_cond = _real_table(data_dir / f"{stem}-wtcbycond.tsv")

    frames, cond_frames = [], []
    for ch_type, null in nulls.items():
        part, cond_part = null.summarise(real=_for_chroma(real, ch_type),
                                         real_by_cond=_for_chroma(real_by_cond, ch_type))
        # tagged after the averaging, which groups on the columns it knows and drops the
        # rest, and on a copy, since the frame is not ours to mutate
        part = part.copy()
        part.insert(0, "chromophore", ch_type)
        frames.append(part)
        if cond_part is not None:
            cond_part = cond_part.copy()
            cond_part.insert(0, "chromophore", ch_type)
            cond_frames.append(cond_part)

    sources = [p for p in (path_from(r) for r in aligned_raws.values()) if p]
    params = dict(
        band_fmin=band_fmin, band_fmax=band_fmax, mask_coi=mask_coi,
        **({"analysis_window_s": [round(t, 3) for t in analysis_window]}
           if analysis_window is not None else {}),
        wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax, n_iter=n_iter, cross=cross, seed=seed,
        chroma=list(chroma), **wtc_grid_params(aligned_raws),
        # the null is subtracted from the real table row by row, so the two have to say
        # they were built on the same clock for that subtraction to mean anything
        **alignment_params(aligned_raws),
    )
    out_path = data_dir / f"{stem}-wtc-pseudo.tsv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, sep="\t", index=False)
    _hyper_sidecar(out_path, "hyper_wtc_pseudo", sources, **params)
    logger.info("Pseudo-dyad WTC band means saved: %s", out_path)

    if cond_frames:
        cond_path = data_dir / f"{stem}-wtcbycond-pseudo.tsv"
        pd.concat(cond_frames, ignore_index=True).to_csv(cond_path, sep="\t", index=False)
        _hyper_sidecar(cond_path, "hyper_wtc_bycondition_pseudo", sources,
                       conditions=[w[0] for w in (windows or [])], **params)
        logger.info("Pseudo-dyad WTC per condition saved: %s", cond_path)

    return out_path


def _real_table(path: Path) -> "pd.DataFrame | None":
    """The true-dyad band means the null is ranked against, or None with a line in the log."""
    if not path.exists():
        logger.info("no %s to rank the null against; writing it without a percentile column",
                    path.name)
        return None
    return pd.read_csv(path, sep="\t")


def _for_chroma(table: "pd.DataFrame | None", ch_type: str) -> "pd.DataFrame | None":
    """One chromophore's rows of a real table: the null is drawn one chromophore at a time."""
    if table is None or "chromophore" not in table.columns:
        return table
    return table[table["chromophore"] == ch_type]
