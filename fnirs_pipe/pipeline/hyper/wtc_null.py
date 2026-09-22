"""The phase-scrambled null, computed and written on its own.

Called from ``fnirs-hyper run`` when ``--wtc-phase-null`` is given, so it inherits that run's
stage, band and window by construction: a null averaged over a different band is not the
null for the table it sits beside. Crossing is the one thing it does not inherit. The null
used to take ``--wtc-channel-cross`` from the real run, so asking for the exploratory
196-pair channel table also multiplied the surrogate cost by 14; it is now the null's own
decision, ``--wtc-phase-null-cross``, and defaults to off.

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


def write_tsv(frame: pd.DataFrame, path: Path) -> Path:
    """One spelling for the tab-separated write every table here does."""
    frame.to_csv(path, sep="\t", index=False)
    return path


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
    frequency to beat the null, and returns ``{chromophore: NullDraws}`` for
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
    from fnirs_pipe.io.derivatives import group_data_dir, hyper_stem
    from fnirs_pipe.pipeline.hyper import compute_wtc_phase_null
    from fnirs_pipe.pipeline.hyper.wtc_store import save_null_levels

    band_fmin = band_fmin if band_fmin is not None else wtc_fmin
    band_fmax = band_fmax if band_fmax is not None else wtc_fmax

    chroma = tuple(dict.fromkeys(chroma))
    if not chroma or set(chroma) - {"hbo", "hbr"}:
        raise ValueError(f"chroma must be some of ('hbo', 'hbr'), got {chroma!r}")

    logger.info(
        "Phase-scrambled WTC: %d phase-scrambled iterations, %s pairs, one full WTC run each, "
        "per chromophore (%s).",
        n_iter, "crossed" if cross else "homologous", "+".join(chroma))

    data_dir = group_data_dir(output_dir, group_id)
    stem = hyper_stem(group_id, task)
    nulls: dict = {}
    for ch_type in chroma:
        null = compute_wtc_phase_null(
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
    roi_map: "dict[str, list[str]] | None" = None,
) -> Path:
    """Rank what :func:`run_wtc_null` drew against the real band means, and write it.

    Writes ``group-<id>_task-<task>_hyper-wtc-phasenull.tsv``, one table with a ``chromophore``
    column, matching the real band-mean tables. The sidecar additionally records ``n_iter``,
    ``cross`` and ``chroma``, without which a 5-iteration probe and a 100-iteration null are
    indistinguishable on disk.

    ``roi_map`` adds ``...hyper-wtc-roihom-phasenull.tsv``, the null for the homologous ROI
    means in ``...hyper-wtc-roihom.tsv``. It is free: the iterations are grouped into
    regions before they are summarised, so no surrogate is transformed a second time, and
    grouping inside the iteration is what makes it the null of the ROI mean rather than a
    bracket around it. The crossed ``-roichan`` matrix has no null and cannot get one from
    here; see :func:`~fnirs_pipe.pipeline.hyper.roi.roi_mean_of_homologous`.

    ``windows`` adds a second table, ``...hyper-wtcbycond-phasenull.tsv``, with a ``condition``
    column: the null for what ``--wtc-by-condition`` wrote. It mirrors the real side, where
    the whole-run and per-condition tables are also two files merged separately. Returns the
    whole-run path either way; the per-condition one sits beside it.

    The real tables are read back off disk rather than passed in, the report step having just
    written them, and are what gives both tables their ``percentile`` column. They are only
    ever a few hundred rows, and reading them here keeps the report's own signature out of
    the null's business. A tree without them still gets a null, just one nothing has been
    ranked against yet.
    """
    from fnirs_pipe.io.derivatives import group_data_dir, hyper_stem
    from fnirs_pipe.pipeline.hyper import _hyper_sidecar, alignment_params
    from fnirs_pipe.pipeline.hyper.wtc import wtc_grid_params
    from fnirs_pipe.utils.lineage import path_from

    band_fmin = band_fmin if band_fmin is not None else wtc_fmin
    band_fmax = band_fmax if band_fmax is not None else wtc_fmax
    chroma = tuple(nulls)

    data_dir = group_data_dir(output_dir, group_id)
    stem = hyper_stem(group_id, task)
    real = _real_table(data_dir / f"{stem}-wtc.tsv")
    real_by_cond = _real_table(data_dir / f"{stem}-wtcbycond.tsv")
    real_roi = _real_table(data_dir / f"{stem}-wtc-roihom.tsv")
    real_roi_by_cond = _real_table(data_dir / f"{stem}-wtcbycond-roihom.tsv")

    frames, cond_frames = [], []
    roi_frames, roi_cond_frames = [], []
    draw_frames: list = []
    for ch_type, null in nulls.items():
        # every iteration kept, not only their summary: a test that averages the draws over
        # channels before ranking cannot be rebuilt from null_mean and null_p95
        for draw_id, frame in zip(null.cond_draw_ids or [], null.cond_draws):
            draw_frames.append(frame.assign(chromophore=ch_type, draw=draw_id))
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
        if roi_map:
            roi_part, roi_cond_part = null.summarise_roi(
                roi_map, real=_for_chroma(real_roi, ch_type),
                real_by_cond=_for_chroma(real_roi_by_cond, ch_type))
            roi_part = roi_part.copy()
            roi_part.insert(0, "chromophore", ch_type)
            roi_frames.append(roi_part)
            if roi_cond_part is not None:
                roi_cond_part = roi_cond_part.copy()
                roi_cond_part.insert(0, "chromophore", ch_type)
                roi_cond_frames.append(roi_cond_part)

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
    out_path = data_dir / f"{stem}-wtc-phasenull.tsv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, sep="\t", index=False)
    _hyper_sidecar(out_path, "hyper_wtc_phasenull", sources, **params)
    logger.info("Phase-scrambled WTC band means saved: %s", out_path)

    if cond_frames:
        cond_path = data_dir / f"{stem}-wtcbycond-phasenull.tsv"
        pd.concat(cond_frames, ignore_index=True).to_csv(cond_path, sep="\t", index=False)
        _hyper_sidecar(cond_path, "hyper_wtc_bycondition_phasenull", sources,
                       conditions=[w[0] for w in (windows or [])], **params)
        logger.info("Phase-scrambled WTC per condition saved: %s", cond_path)

    for group, stem_suffix, step, extra in (
            (draw_frames, "-wtcbycond-phasenull-draws",
             "hyper_wtc_bycondition_phasenull_draws",
             {"conditions": [w[0] for w in (windows or [])]}),
            (roi_frames, "-wtc-roihom-phasenull", "hyper_wtc_roihom_phasenull", {}),
            (roi_cond_frames, "-wtcbycond-roihom-phasenull",
             "hyper_wtc_bycondition_roihom_phasenull",
             {"conditions": [w[0] for w in (windows or [])]})):
        if not group:
            continue
        path = data_dir / f"{stem}{stem_suffix}.tsv"
        write_tsv(pd.concat(group, ignore_index=True), path)
        _hyper_sidecar(path, step, sources, **extra, **params)
        logger.info("Phase-scrambled WTC ROI means saved: %s", path)

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
