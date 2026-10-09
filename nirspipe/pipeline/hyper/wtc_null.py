"""The phase-scrambled null, computed and written on its own.

Called from ``fnirs-hyper`` when ``--wtc-phase-null`` is given, so it inherits that run's
stage, band, window and ROI minimum by construction. Crossing follows the real table unless
``--wtc-phase-null-cross`` says otherwise, and n channels crossed give n^2 pairings and so n
times the surrogate cost of the n homologous ones.

The chromophores are inherited, unlike crossing, so no half of the real table is left with
nothing to be tested against.

Two halves, and the report sits between them. :func:`run_wtc_null` draws the surrogates and
writes the per-frequency level the phase arrows are drawn against, which has to exist before
the figures are built. :func:`write_wtc_null` ranks those draws against the real band means,
which the same report writes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fnirs_pipe.io.derivatives import group_output_path
from fnirs_pipe.io.tables import write_tsv
from fnirs_pipe.utils import ROI_MIN_CHANNELS, bare_roi_map

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
    whiten_s: float = 0.0,
    session: "str | None" = None,
    roi_map: "dict[str, list[str]] | None" = None,
    roi_map_name: str = "custom",
    roi_min_channels: int = ROI_MIN_CHANNELS,
) -> dict:
    """Draw the null for one dyad and write the level its phase arrows are read against.

    The expensive half: ``n_iter`` full WTC runs per chromophore. Writes
    ``..._chromo-<chroma>_null-phase_stat-wtc_desc-level_relmat.npz``, the coherence a cell has to clear at each
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

    ``roi_map`` adds the ROI maps' own level, :func:`roi_level_path`, from the same pass.
    """
    # imported in the call, not at module load: the wiring tests patch these on the module
    # that defines them, which only a lookup made at call time can see
    from fnirs_pipe.io.derivatives import group_output_path
    from fnirs_pipe.pipeline.hyper import _hyper_sidecar, compute_wtc_phase_null
    from fnirs_pipe.pipeline.hyper._helpers import _chroma_tuple
    from fnirs_pipe.pipeline.hyper.whiten import whiten_raws
    from fnirs_pipe.pipeline.hyper.wtc_store import level_params, save_null_levels
    from fnirs_pipe.utils.lineage import paths_from

    band_fmin = band_fmin if band_fmin is not None else wtc_fmin
    band_fmax = band_fmax if band_fmax is not None else wtc_fmax

    chroma = _chroma_tuple(chroma, "chroma")

    logger.info(
        "Phase-scrambled WTC: %d phase-scrambled iterations, %s pairs, one full WTC run each, "
        "per chromophore (%s).",
        n_iter, "crossed" if cross else "homologous", "+".join(chroma))

    # whitened before it is scrambled, the order the real table's transform sees it in
    wtc_raws = whiten_raws(aligned_raws, whiten_s, sep_bands) if whiten_s else aligned_raws
    nulls: dict = {}
    for ch_type in chroma:
        null = compute_wtc_phase_null(
            wtc_raws, band_fmin, band_fmax, n_iter=n_iter,
            fmin=wtc_fmin, fmax=wtc_fmax, seed=seed, cross=cross,
            limit_scales=limit_scales, mask_coi=mask_coi, ch_type=ch_type,
            sep_bands=sep_bands, windows=windows, analysis_window=analysis_window,
            roi_map=roi_map, roi_min_channels=roi_min_channels)
        nulls[ch_type] = null
        if getattr(null, "levels", None):
            path = save_null_levels(null.levels, group_output_path(
                output_dir, group_id,
                {"task": task, "chromophore": ch_type, "nulldist": "phase",
                 "statistic": "wtc", "desc": "level"}, "relmat", ".npz", session=session))
            # what the report checks before it thresholds its arrows against this level
            _hyper_sidecar(path, "hyper_wtc_phasenull_level",
                           paths_from(aligned_raws.values()),
                           **level_params(aligned_raws, wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax,
                                          mask_coi=mask_coi, whiten_s=whiten_s),
                           n_iter=n_iter, seed=seed)
        if getattr(null, "roi_levels", None):
            path = save_null_levels(null.roi_levels, roi_level_path(
                output_dir, group_id, task, ch_type, roi_map_name, session))
            _hyper_sidecar(path, "hyper_wtc_phasenull_level",
                           paths_from(aligned_raws.values()),
                           **level_params(aligned_raws, wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax,
                                          mask_coi=mask_coi, whiten_s=whiten_s),
                           **roi_level_params(roi_map, roi_min_channels, cross),
                           n_iter=n_iter, seed=seed)
    return nulls


def roi_level_path(output_dir: Path, group_id: str, task: str, ch_type: str,
                   roi_map_name: str, session: "str | None" = None) -> Path:
    """Where the ROI maps' phase-scrambled level goes, beside the channels' own."""
    return group_output_path(
        output_dir, group_id,
        {"task": task, "segmentation": roi_map_name, "aggregation": "roi",
         "chromophore": ch_type, "nulldist": "phase", "statistic": "wtc", "desc": "level"},
        "relmat", ".npz", session=session)


def roi_level_params(roi_map: dict, roi_min_channels: int, cross: bool) -> dict:
    """What an ROI level depends on beyond a channel level: the map, its minimum, the crossing.

    A crossed (R, R) map averages every pairing inside R and an uncrossed R only the
    same-channel ones, so a level of one is not a level of the other.
    """
    return {"roi_map": {k: list(v) for k, v in bare_roi_map(roi_map).items()},
            "roi_min_channels": int(roi_min_channels), "channel_cross": bool(cross)}


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
    roi_map_name: str = "custom",
    roi_min_channels: int = ROI_MIN_CHANNELS,
    whiten_s: float = 0.0,
    session: "str | None" = None,
) -> Path:
    """Rank what :func:`run_wtc_null` drew against the real band means, and write it.

    Writes ``..._null-phase_stat-wtc_relmat.tsv``, one table with a ``chromophore``
    column, matching the real band-mean tables. The sidecar additionally records ``n_iter``,
    ``cross`` and ``chroma``, without which a 5-iteration probe and a 100-iteration null are
    indistinguishable on disk.

    ``roi_map`` adds the ``agg-homologous`` twin, the null for the homologous ROI
    means beside it. It is free: the iterations are grouped into
    regions before they are summarised, so no surrogate is transformed a second time, and
    grouping inside the iteration is what makes it the null of the ROI mean rather than a
    bracket around it. A null drawn crossed also writes the ``agg-roi`` twin, the null for
    the crossed ROI x ROI matrix, when the real ROI table is crossed too; a homologous null
    never drew the off-diagonal pairings that matrix averages. Both group their cells under
    ``roi_min_channels``, the rule the real ROI tables were blanked by.

    ``windows`` adds a second table, the ``cond-all`` twin, with a ``condition``
    column: the null for what ``--wtc-by-condition`` wrote. It mirrors the real side, where
    the whole-run and per-condition tables are also two files merged separately. Returns the
    whole-run path either way; the per-condition one sits beside it.

    The real tables are read back off disk rather than passed in, the report step having just
    written them, and are what gives both tables their ``percentile`` column. A tree without
    them still gets a null, just one nothing has been ranked against yet.
    """
    from fnirs_pipe.pipeline.hyper import _hyper_sidecar, alignment_params
    from fnirs_pipe.pipeline.hyper.wtc import wtc_grid_params
    from fnirs_pipe.utils.lineage import paths_from

    band_fmin = band_fmin if band_fmin is not None else wtc_fmin
    band_fmax = band_fmax if band_fmax is not None else wtc_fmax
    chroma = tuple(nulls)

    roi_entities = {"segmentation": roi_map_name, "aggregation": "homologous"}
    cross_entities = {"segmentation": roi_map_name, "aggregation": "roi"}

    def _path(entities: dict, extension: str = ".tsv") -> Path:
        return group_output_path(output_dir, group_id, {"task": task, **entities},
                                 "relmat", extension, session=session)

    # the real tables this null is ranked against, each the same name minus `null-`
    real = _real_table(_path({"statistic": "wtc"}))
    real_by_cond = _real_table(_path({"condition": "all", "statistic": "wtc"}))
    real_roi = _real_table(_path({**roi_entities, "statistic": "wtc"}))
    real_roi_by_cond = _real_table(
        _path({**roi_entities, "condition": "all", "statistic": "wtc"}))
    real_cross = real_cross_by_cond = None
    if roi_map and cross:
        real_cross = _real_table(_path({**cross_entities, "statistic": "wtc"}))
        real_cross_by_cond = _real_table(
            _path({**cross_entities, "condition": "all", "statistic": "wtc"}))
    # an uncrossed run's ROI diagonal is the homologous mean, not what a crossed null groups
    roi_crossed = any(t is not None and "label2" in t.columns
                      for t in (real_cross, real_cross_by_cond))
    if roi_map and cross and not roi_crossed:
        logger.info("crossed null over an uncrossed ROI table: no crossed ROI null written")

    frames, cond_frames = [], []
    roi_frames, roi_cond_frames = [], []
    cross_frames, cross_cond_frames = [], []
    draw_frames: list = []

    def _put(bucket: list, part: "pd.DataFrame | None", ch_type: str) -> None:
        # tagged after the averaging, which groups on the columns it knows and drops the
        # rest, and on a copy, since the frame is not ours to mutate
        if part is not None:
            part = part.copy()
            part.insert(0, "chromophore", ch_type)
            bucket.append(part)

    for ch_type, null in nulls.items():
        # every iteration kept, not only their summary: a test that averages the draws over
        # channels before ranking cannot be rebuilt from null_mean and null_p95
        for draw_id, frame in zip(null.cond_draw_ids or [], null.cond_draws):
            draw_frames.append(frame.assign(chromophore=ch_type, draw=draw_id))
        part, cond_part = null.summarise(real=_for_chroma(real, ch_type),
                                         real_by_cond=_for_chroma(real_by_cond, ch_type))
        _put(frames, part, ch_type)
        _put(cond_frames, cond_part, ch_type)
        if roi_map:
            roi_part, roi_cond_part = null.summarise_roi(
                roi_map, real=_for_chroma(real_roi, ch_type),
                real_by_cond=_for_chroma(real_roi_by_cond, ch_type),
                min_channels=roi_min_channels)
            _put(roi_frames, roi_part, ch_type)
            _put(roi_cond_frames, roi_cond_part, ch_type)
        if roi_crossed:
            cross_part, cross_cond_part = null.summarise_roi(
                roi_map, real=_for_chroma(real_cross, ch_type),
                real_by_cond=_for_chroma(real_cross_by_cond, ch_type),
                min_channels=roi_min_channels, crossed=True)
            _put(cross_frames, cross_part, ch_type)
            _put(cross_cond_frames, cross_cond_part, ch_type)

    sources = paths_from(aligned_raws.values())
    params = dict(
        band_fmin=band_fmin, band_fmax=band_fmax, mask_coi=mask_coi,
        **({"analysis_window_s": [round(t, 3) for t in analysis_window]}
           if analysis_window is not None else {}),
        wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax, n_iter=n_iter, cross=cross, seed=seed,
        **({"wtc_whiten_s": float(whiten_s)} if whiten_s else {}),
        **({"roi_min_channels": int(roi_min_channels)} if roi_map else {}),
        chroma=list(chroma), **wtc_grid_params(aligned_raws),
        # the null is subtracted from the real table row by row, so the two have to say
        # they were built on the same clock for that subtraction to mean anything
        **alignment_params(aligned_raws),
    )
    out_path = _path({"nulldist": "phase", "statistic": "wtc"})
    write_tsv(pd.concat(frames, ignore_index=True), out_path)
    _hyper_sidecar(out_path, "hyper_wtc_phasenull", sources, **params)
    logger.info("Phase-scrambled WTC band means saved: %s", out_path)

    if cond_frames:
        cond_path = _path({"condition": "all", "nulldist": "phase",
                           "statistic": "wtc"})
        write_tsv(pd.concat(cond_frames, ignore_index=True), cond_path)
        _hyper_sidecar(cond_path, "hyper_wtc_bycondition_phasenull", sources,
                       conditions=[w[0] for w in (windows or [])], **params)
        logger.info("Phase-scrambled WTC per condition saved: %s", cond_path)

    for group, entities, step, extra in (
            (draw_frames,
             {"condition": "all", "nulldist": "phase", "statistic": "wtc",
              "desc": "draws"},
             "hyper_wtc_bycondition_phasenull_draws",
             {"conditions": [w[0] for w in (windows or [])]}),
            (roi_frames, {**roi_entities, "nulldist": "phase", "statistic": "wtc"},
             "hyper_wtc_roihom_phasenull", {}),
            (roi_cond_frames,
             {**roi_entities, "condition": "all", "nulldist": "phase",
              "statistic": "wtc"},
             "hyper_wtc_bycondition_roihom_phasenull",
             {"conditions": [w[0] for w in (windows or [])]}),
            (cross_frames, {**cross_entities, "nulldist": "phase", "statistic": "wtc"},
             "hyper_wtc_roichan_phasenull", {}),
            (cross_cond_frames,
             {**cross_entities, "condition": "all", "nulldist": "phase",
              "statistic": "wtc"},
             "hyper_wtc_bycondition_roichan_phasenull",
             {"conditions": [w[0] for w in (windows or [])]})):
        if not group:
            continue
        path = _path(entities)
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
