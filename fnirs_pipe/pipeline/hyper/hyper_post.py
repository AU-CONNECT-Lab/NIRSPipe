"""The analysis behind ``fnirs-hyper post``: the WTC and ISC runs, and the tables they write.

Nothing here draws. The transform covers every pairing at once and a figure is of one
pairing, so the two are separated; :mod:`fnirs_pipe.qc.hyper.hyper_report` draws the
:class:`HyperPostResult` this returns, and skips the transform when handed one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fnirs_pipe.utils import ROI_MIN_CHANNELS, bare_roi_map
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.exceptions import StageError
from fnirs_pipe.io.derivatives import group_output_path
from fnirs_pipe.pipeline.hyper._helpers import _chroma_tuple
from fnirs_pipe.pipeline.hyper.roi import roi_mean_of_channels, roi_mean_of_homologous
from fnirs_pipe.pipeline.hyper.isc import compute_isc_pairs, roi_mean_of_isc
from fnirs_pipe.pipeline.hyper.whiten import whiten_order, whiten_raws
from fnirs_pipe.qc.common.figure_io import _pair_fname, get_channel_pairs, pair_slug
from fnirs_pipe.qc.common.windows import condition_windows, refuse_colliding_labels

logger = get_logger("pipeline.hyper_post")


@dataclass
class HyperPostConfig:
    """Every ``--wtc-*`` option of ``fnirs-hyper post``, with the derived values resolved once.

    Field names are the CLI's. ``band_fmin``, ``band_fmax``, ``chroma`` and ``cond_pad_s``
    are the resolved forms, in one place so the tables and the figures cannot fall back
    differently.
    """

    wtc_fmin: float = 0.004
    wtc_fmax: float = 0.20
    wtc_band_fmin: float | None = None
    wtc_band_fmax: float | None = None
    wtc_significance: bool = False
    wtc_seed: int | None = None
    wtc_mc_count: int = 300
    wtc_channel_cross: bool = False
    wtc_by_condition: bool = False
    wtc_window_s: "float | None" = None
    wtc_cond_pad_s: float | None = None
    wtc_limit_scales: bool = True
    wtc_save_maps: bool = False
    wtc_mask_coi: bool = True
    wtc_roi_min_channels: int = ROI_MIN_CHANNELS
    wtc_chroma: Any = ("hbo", "hbr")
    # seconds of AR order for prewhitening before the coherence, 0 for none
    wtc_whiten_s: float = 0.0
    isc_whiten: int = 0
    isc_max_lag_s: float = 0.0
    isc_phase_null: int = 0
    isc_band: "tuple[float | None, float | None] | None" = None
    roi_map: dict | None = None
    roi_map_name: str = "custom"
    sep_bands: Any = None
    analysis_window: "tuple[float, float] | None" = None
    # the stage and the rejection scope the recordings were loaded with, for the sidecars
    desc: str | None = None
    bads_scope: str | None = None

    # resolved in __post_init__, never passed in
    chroma: tuple = field(init=False)
    band_fmin: float = field(init=False)
    band_fmax: float = field(init=False)
    cond_pad_s: "float | None" = field(init=False)

    def __post_init__(self) -> None:
        self.chroma = _chroma_tuple(self.wtc_chroma, "wtc_chroma")
        self.band_fmin = (self.wtc_band_fmin if self.wtc_band_fmin is not None
                          else self.wtc_fmin)
        self.band_fmax = (self.wtc_band_fmax if self.wtc_band_fmax is not None
                          else self.wtc_fmax)
        if self.wtc_band_fmin is None or self.wtc_band_fmax is None:
            logger.info(
                "WTC band mean taken over the whole %.4f-%.4f Hz axis; name a narrower band "
                "to average over the frequencies the task lives in",
                self.band_fmin, self.band_fmax)
        self.cond_pad_s = (None if self.wtc_cond_pad_s is None
                           else max(0.0, float(self.wtc_cond_pad_s)))


@dataclass
class HyperPostResult:
    """What one dyad's post run produced: the transforms, the numbers, and where they went.

    ``passes`` is keyed ``hbo`` / ``hbr``, each holding the whole-run ``WTCResult`` under
    ``result``, its band means under ``chan`` and ``roichan``, and the windows under
    ``cond_wtc`` / ``cond_bands`` positionally over ``cond_windows`` -- positional so the
    chromophores line up even where a guard failed on one.

    ``isc`` is ``{pairing: {label or None: {chromophore: (matrix, channel names)}}}``, and
    ``isc_roi`` the same shape over the ROI means of those matrices.
    ``chan_axis`` and ``roi_labels`` are the axes every panel and matrix is indexed by; they
    ride here so a panel cannot be drawn on a different axis than the table beside it.
    """

    subject_ids: list[str]
    chroma: tuple
    band_fmin: float
    band_fmax: float
    cond_windows: list
    # how each condition was read: None means windowed out of the whole-run transform, a
    # number means transformed on its own over a cut padded by that many seconds. The report
    # points a window at the run's own ROI map only in the first case
    cond_pad_s: "float | None"
    chan_axis: list[str]
    roi_labels: list[str]
    roi_rows: list[dict]
    passes: dict
    chan_band_df: Any = None
    roi_band_df: Any = None
    isc: dict = field(default_factory=dict)
    isc_roi: dict = field(default_factory=dict)
    # {pairing: {label: {chromophore: ndarray or None}}}, the per-cell level a connectogram
    # chord is drawn against when --isc-phase-null ran. Its own field rather than a third slot
    # in `isc`, so nothing reading that pair has to learn a new shape
    isc_levels: dict = field(default_factory=dict)
    # {pairing: {condition: {chromophore: partner count}}} where a chord level came from the
    # re-paired null rather than the phase-scrambled one, so the wording can say which
    isc_level_sources: dict = field(default_factory=dict)
    align_info: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)


def _level_source(result) -> "str | list[str]":
    """Which per-frequency level a map's phase cells were gated by, for a sidecar.

    ::

      maps thresholded against the re-paired null -> "pair"
      no level at all                             -> "none", only the cone applied
    """
    found = {({"null": "phase", "pair": "pair"}.get(data.get("sig_source"))
              or ("montecarlo" if data.get("sig") is not None else "none"))
             for labels in (result.pairs.values() if result is not None else [])
             for data in labels.values() if data is not None}
    return found.pop() if len(found) == 1 else sorted(found)


def write_isc_matrix(
    tsv_path: Path,
    isc_mat,
    ch_names: list[str],
    ch_type: str,
    sources: list[str],
    subject_ids: list[str],
    align: "dict | None" = None,
    step: str = "hyper_isc",
    index_label: str = "channel",
    **params,
) -> None:
    """Write the matrix the ISC panel is drawn from, so the numbers can leave the report.

    Both axes carry the montage's channel labels, which is how
    :func:`~fnirs_pipe.pipeline.hyper.isc.compute_isc` pairs the
    two brains: cell (i, j) is the first subject's channel i against the other's channel j.
    Rejected channels are blank rather than absent, so the file's shape is the montage's
    however many channels a given dyad lost.

    ``step`` and ``index_label`` are what let this serve the ROI means of those matrices
    too, which are the same square shape over regions instead of channels.

    ``align`` is what :func:`~fnirs_pipe.pipeline.hyper.alignment_params` returned, so the
    file says which alignment it got.

    A failure here costs the file and not the panel: the report is still readable without it.
    """
    from fnirs_pipe.pipeline.hyper import _hyper_sidecar

    try:
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(isc_mat, index=ch_names, columns=ch_names).to_csv(
            tsv_path, sep="\t", index_label=index_label)
        _hyper_sidecar(tsv_path, step, sources,
                       chromophore=ch_type, subjects=subject_ids,
                       **params, **(align or {}))
        logger.info("ISC matrix saved: %s", tsv_path)
    except Exception as exc:
        logger.warning("ISC matrix (%s) not written: %s", ch_type, exc)


def run_hyper_post(
    group_id: str,
    task: str,
    aligned_raws: dict,
    output_dir: Path,
    config: HyperPostConfig,
    *,
    subject_ids: list[str],
    pairings: list,
    align_info: dict,
    cond_windows: "list[tuple[str, float, float]] | None" = None,
    errors: list | None = None,
    notes: list | None = None,
    scope: str = "",
) -> HyperPostResult:
    """Run the coherence and the correlation for one dyad, and write every table they make.

    ``cond_windows`` supplies the per-condition windows instead of resolving them here: the
    caller passes the same list to the phase-scrambled null, and the two tables are only
    subtractable row by row if they describe the same windows.

    ``errors`` and ``notes`` are the report's own lists, so a guard that fails here reaches
    the footer of the page this result is drawn on and not only the run log.
    """
    from fnirs_pipe.pipeline.hyper import _hyper_sidecar
    # straight from the modules that define them, not the package re-exports
    from fnirs_pipe.pipeline.hyper._helpers import long_axis_over
    from fnirs_pipe.pipeline.hyper.wtc import (
        WTCResult,
        compute_wtc,
        window_result,
        wtc_band_mean,
        wtc_grid_params,
        wtc_phase_by_scale,
    )
    from fnirs_pipe.qc.common.report_shell import guard, note
    from fnirs_pipe.utils.lineage import paths_from

    errors = errors if errors is not None else []
    notes  = notes  if notes  is not None else []
    tables: dict = {}

    # under the names the blocks below were written against, so the transform reads one
    # resolved value and never a raw option beside it
    wtc_fmin, wtc_fmax     = config.wtc_fmin, config.wtc_fmax
    band_fmin, band_fmax   = config.band_fmin, config.band_fmax
    wtc_significance       = config.wtc_significance
    wtc_seed, wtc_mc_count = config.wtc_seed, config.wtc_mc_count
    wtc_channel_cross      = config.wtc_channel_cross
    wtc_by_condition       = config.wtc_by_condition
    wtc_limit_scales       = config.wtc_limit_scales
    wtc_save_maps          = config.wtc_save_maps
    wtc_mask_coi           = config.wtc_mask_coi
    wtc_roi_min_channels   = config.wtc_roi_min_channels
    isc_whiten, isc_phase_null = config.isc_whiten, config.isc_phase_null
    isc_max_lag_s          = config.isc_max_lag_s
    isc_band               = config.isc_band
    chroma, cond_pad_s     = config.chroma, config.cond_pad_s
    roi_map, sep_bands     = config.roi_map, config.sep_bands
    # the entities every ROI table carries, so one tree can hold two ROI definitions
    roi_entities           = {"segmentation": config.roi_map_name}
    analysis_window        = config.analysis_window
    wtc_whiten_s           = float(config.wtc_whiten_s or 0.0)
    # the coherence alone reads the whitened copies; the correlation has its own whitening
    wtc_raws = (whiten_raws(aligned_raws, wtc_whiten_s, sep_bands) if wtc_whiten_s
                else aligned_raws)

    def _whiten_params() -> dict:
        return ({"wtc_whiten_s": wtc_whiten_s,
                 "wtc_whiten_order": whiten_order(aligned_raws, wtc_whiten_s)}
                if wtc_whiten_s else {})

    ref_raw = aligned_raws.get(subject_ids[0]) if subject_ids else None
    # the channel axis follows the first chromophore; the labels do not carry one, so one
    # pass serves both
    fig_chroma = chroma[0]

    def _write_df_tsv(df, entities: dict, step: str, **extra) -> Path:
        """One long-format table, named by what distinguishes it from the others.

        No chromophore entity: these are long tables with a ``chromophore`` column, and no
        band entity: the band is a parameter of the measurement rather than something that
        tells two files apart, so it goes in the sidecar below. ``fnirs-hyper band`` is the
        one writer that needs it in a name, and only because its output sits beside this one.
        """
        tsv_path = group_output_path(output_dir, group_id, {"task": task, **entities},
                                     "relmat", ".tsv")
        df.to_csv(tsv_path, sep="\t", index=False)
        _hyper_sidecar(tsv_path, step,
                       paths_from(aligned_raws.values()),
                       **_wtc_params(), **extra)
        tables[tsv_path.name] = tsv_path
        return tsv_path

    def _wtc_params() -> dict:
        """What every WTC table and map records about how it was computed."""
        return dict(
            band_fmin=band_fmin, band_fmax=band_fmax, mask_coi=wtc_mask_coi,
            # --tstart/--tend, so a table of a stretch is not read as the whole recording
            **({"analysis_window_s": [round(x, 3) for x in analysis_window]}
               if analysis_window is not None else {}),
            # the window grid, without which the null cannot resolve the same one
            **({"wtc_window_s": round(float(config.wtc_window_s), 3)}
               if config.wtc_window_s else {}),
            wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax, chroma=list(chroma),
            # None: conditions windowed out of the whole-run transform
            wtc_cond_pad_s=None if cond_pad_s is None else round(cond_pad_s, 3),
            # the Monte Carlo level, only when one was drawn
            **({"wtc_mc_count": int(wtc_mc_count), "wtc_seed": wtc_seed}
               if wtc_significance else {}),
            # what the re-paired null and the Methods read back rather than take again
            channel_cross=bool(wtc_channel_cross),
            roi_min_channels=int(wtc_roi_min_channels),
            **({"desc": config.desc} if config.desc else {}),
            **({"bads_scope": config.bads_scope} if config.bads_scope else {}),
            **wtc_grid_params(aligned_raws), **align_info, **_whiten_params(),
        )

    def _transform_condition(tstart: float, tstop: float, ch_type: str):
        """One condition's own transform, taken over a padded cut and windowed back.

        ::

          condition [3580, 3880] with cond_pad_s 47
            -> transform [3533, 3927], then window the result to [3580, 3880]

        The alternative to reading the window out of the whole-run transform, for a caller
        who wants each condition transformed on its own. What the padding buys is the cone:
        with a margin past :func:`~fnirs_pipe.pipeline.hyper.wtc.cone_margin_s` it lands
        outside the condition instead of eating its edges.

        ``cond_pad_s`` of 0 does cut to the boundaries.

        Costs one transform per condition per chromophore on top of the whole-run pass.
        """
        lo = max(0.0, tstart - cond_pad_s)
        hi = min(min(float(r.times[-1]) for r in wtc_raws.values()), tstop + cond_pad_s)
        cut = {sid: raw.copy().crop(tmin=lo, tmax=hi)
               for sid, raw in wtc_raws.items()}
        res = compute_wtc(cut, fmin=wtc_fmin, fmax=wtc_fmax,
                          significance=wtc_significance, seed=wtc_seed,
                          mc_count=wtc_mc_count, cross=wtc_channel_cross,
                          limit_scales=wtc_limit_scales, ch_type=ch_type,
                          sep_bands=sep_bands)
        # the cut's own clock starts at zero, so the condition sits `tstart - lo` into it
        return window_result(res, tstart - lo, tstart - lo + (tstop - tstart))

    def _tag(df, ch_type: str):
        """The column saying which chromophore a row is, added after every aggregation.

        ``roi_mean_of_channels`` and ``wtc_band_mean`` group on the columns they know and
        drop the rest, so tagging before them loses the tag. Same reason ``condition`` is
        inserted after the band mean rather than carried into it.
        """
        df.insert(0, "chromophore", ch_type)
        return df

    def _save_maps(result, ch_type: str) -> None:
        """The full time-frequency maps beside the table, one archive per chromophore.

        It carries ``chromo-`` where the TSV beside it does not: the TSV is long format and
        holds both chromophores in a column, and an archive cannot.
        """
        from fnirs_pipe.pipeline.hyper.wtc_store import save_wtc
        npz_path = group_output_path(output_dir, group_id,
                                     {"task": task, "chromophore": ch_type,
                                      "statistic": "wtc"}, "relmat", ".npz")
        with guard(f"Saving WTC maps ({ch_type})", errors, scope):
            save_wtc(result, npz_path)
            # what `fnirs-hyper band` carries onto the tables it re-averages from this
            _hyper_sidecar(npz_path, "hyper_wtc_maps",
                           paths_from(aligned_raws.values()),
                           **_wtc_params())

    def _level_path(ch_type: str, nulldist: str) -> Path:
        return group_output_path(output_dir, group_id,
                                 {"task": task, "chromophore": ch_type,
                                  **({"condition": "all"} if nulldist == "pair" else {}),
                                  "nulldist": nulldist, "statistic": "wtc", "desc": "level"},
                                 "relmat", ".npz")

    def _usable_level(path: Path, what: str) -> bool:
        """Whether a level on disk was drawn on these recordings with these settings."""
        from fnirs_pipe.pipeline.hyper.wtc_store import level_mismatch, level_params
        if not path.exists():
            return False
        why = level_mismatch(path, level_params(aligned_raws, wtc_fmin=wtc_fmin,
                                                wtc_fmax=wtc_fmax, mask_coi=wtc_mask_coi,
                                                whiten_s=wtc_whiten_s),
                             aligned_raws)
        if why:
            note(notes, scope, f"{path.name} is on disk but was not used for the {what}: "
                               f"{why}. Rerun the null that wrote it on this tree.")
            logger.warning("%s | %s not used: %s", scope, path.name, why)
        return why is None

    def _set_level(data: dict, level, source: str, n_freqs: int) -> bool:
        # a level of the wrong length would be ignored by the arrows yet named in the caption
        if data is None or level is None or len(level) != n_freqs:
            return False
        data["sig"] = level
        data["sig_source"] = source
        return True

    def _apply_null_level(result, ch_type: str) -> None:
        """Put the phase-scrambled null's per-frequency level on each pair, where one was drawn.

        ``sig`` is whatever the phase arrows are thresholded against, and it already holds
        pycwt's Monte Carlo level when --wtc-significance ran. The null's level is the same
        shape, so it goes in the same slot rather than a second one the figures would have to
        choose between.

        Absent unless --wtc-phase-null ran for this dyad, which is the usual case: the maps then
        keep whatever they had, and the arrows fall back to the flat --wtc-arrow-min. A level
        drawn on other recordings or settings is refused rather than used.
        """
        from fnirs_pipe.pipeline.hyper.wtc_store import load_null_levels
        npz_path = _level_path(ch_type, "phase")
        if result is None or not _usable_level(npz_path, f"{ch_type} phase arrows"):
            return
        with guard(f"WTC null level ({ch_type})", errors, scope):
            levels = load_null_levels(npz_path)
            for pair_key, labels in result.pairs.items():
                for label, data in labels.items():
                    _set_level(data, levels.get(pair_key, {}).get(label), "null",
                               len(result.freqs))
            logger.info("%s | phase arrows drawn against the phase-scrambled null (%s)",
                        scope, ch_type)

    pair_levels: dict = {}

    def _apply_pair_level(cond_wtc, label: str, tstart: float, tstop: float,
                          ch_type: str) -> None:
        """Threshold one condition's arrows against the re-paired null, where one fits it.

        The re-paired null is drawn one condition at a time, so it has a level per condition
        and none for the whole run: the whole-run page keeps what :func:`_apply_null_level`
        gave it. A condition whose span on this run's clock differs from the one the null was
        drawn on keeps that too.
        """
        from fnirs_pipe.pipeline.hyper.wtc_store import load_cond_null_levels
        if ch_type not in pair_levels:
            path = _level_path(ch_type, "pair")
            pair_levels[ch_type] = None
            if _usable_level(path, f"{ch_type} condition arrows"):
                side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                pair_levels[ch_type] = (load_cond_null_levels(path),
                                        side["parameters"].get("condition_windows_s") or {})
        if cond_wtc is None or pair_levels[ch_type] is None:
            return
        levels, spans = pair_levels[ch_type]
        if spans.get(label) != [round(float(tstart), 3), round(float(tstop), 3)]:
            logger.info("%s | condition %s: no re-paired level for its span", scope, label)
            return
        used = sum(_set_level(data, levels.get(label, {}).get(pair_key, {}).get(ch), "pair",
                              len(cond_wtc.freqs))
                   for pair_key, labels in cond_wtc.pairs.items() for ch, data in labels.items())
        if used:
            logger.info("%s | condition %s: phase arrows drawn against the re-paired null "
                        "(%s, %d channel pairs)", scope, label, ch_type, used)

    def _phase_scale(result, scope_name: str, ch_type: str):
        """The per-frequency phase for one scope, which is where an angle becomes a delay."""
        if result is None or not result.pairs:
            return None
        df = None
        with guard(f"WTC phase by scale ({scope_name} {ch_type})", errors, scope):
            df = wtc_phase_by_scale(result, band_fmin, band_fmax, mask_coi=wtc_mask_coi)
        return df

    def _band_means(result, kind: str, ch_type: str):
        if result is None or not result.pairs:
            return None
        df = None
        with guard(f"WTC band mean ({kind} {ch_type})", errors, scope):
            df = wtc_band_mean(result, band_fmin, band_fmax, mask_coi=wtc_mask_coi)
        return df

    # the channel selector and the ROI grouping table describe the montage, so they are the
    # same whichever chromophore is drawn and are built once
    ch_pairs_post: list[str] = get_channel_pairs(ref_raw) if ref_raw else []

    # ---- the axis both channel panels are indexed by ----
    # The union of the members' long channels, rejections kept, which is the axis the
    # cross matrix is drawn on: selector and matrix have to name the same set or a reader
    # cannot find a matrix cell in the selector. `ch_pairs_post` above is the whole montage
    # including the short channels, which have no coherence and no place on this axis.
    # Labels do not carry the chromophore, so one pass serves both.
    _members = [aligned_raws[s] for s in subject_ids if s in aligned_raws]
    chan_axis: list[str] = (long_axis_over(_members, fig_chroma, sep_bands) if _members
                            else ch_pairs_post)
    roi_rows: list[dict] = []
    roi_labels: list[str] = list(roi_map.keys()) if roi_map else []
    if roi_map:
        roi_map = bare_roi_map(roi_map)
        assigned = {ch for chs in roi_map.values() for ch in chs}
        for roi_name, chs in roi_map.items():
            roi_rows.append({"roi": roi_name, "channels": chs})
        unassigned = [c for c in ch_pairs_post if c not in assigned]
        if unassigned:
            roi_rows.append({"roi": "Unassigned", "channels": unassigned})

    # the task windows are a property of the annotations, not of the chromophore
    if not wtc_by_condition:
        cond_windows = []
    else:
        if cond_windows is None:
            cond_windows = (condition_windows(ref_raw, min_duration=1.0 / wtc_fmin)
                            if ref_raw else [])
        # after any split into equal windows, which makes labels of its own
        refuse_colliding_labels([w[0] for w in cond_windows], reserved=("all",))
        if not cond_windows:
            note(notes, scope,
                 "--wtc-by-condition asked for, but no annotation window is long enough "
                 f"for one cycle of {wtc_fmin:.4f} Hz: no per-condition figures")
        else:
            logger.info("--wtc-by-condition: %d window(s), each read off the whole-run "
                        "transform", len(cond_windows))

    def _roi_band(chan_band_df, ch_type: str, what: str):
        """The ROI number: coherence per channel pair, averaged.

        A table rather than a figure, and derived from a frame carrying every pairing in the
        group, so it is computed once per chromophore beside the channel means instead of
        once per pairing alongside the pictures.
        """
        if not roi_map or chan_band_df is None:
            return None
        out = None
        with guard(f"ROI mean of channel WTC ({what}, {ch_type})", errors, scope):
            out = roi_mean_of_channels(chan_band_df, roi_map,
                                       min_channels=wtc_roi_min_channels)
        return out

    def _roi_hom_band(chan_band_df, ch_type: str, what: str):
        """The homologous ROI mean, the one with a null.

        Separate from `_roi_band` rather than a column beside it, because a crossed run
        writes a 4x4 matrix and this writes four rows: one file, one quantity. See
        `roi_mean_of_homologous` for why the crossed diagonal is not this number.
        """
        if not roi_map or chan_band_df is None:
            return None
        out = None
        with guard(f"Homologous ROI mean ({what}, {ch_type})", errors, scope):
            out = roi_mean_of_homologous(chan_band_df, roi_map,
                                         min_channels=wtc_roi_min_channels)
        return out

    def _wtc_pass(ch_type: str) -> dict:
        """The whole WTC analysis for one chromophore: rows for the tables, and figures.

        HbO and HbR are two parallel runs of the same code. A member's HbO pairs only with
        the other member's HbO, the two are never mixed and never averaged, and nothing
        about the statistic changes, so this is a loop over ``--wtc-chroma`` rather than a
        branch anywhere inside it. Cost is one full multiple per chromophore.

        Returns the rows each table wants, untagged, and the transforms they came off.
        The caller concatenates the rows across chromophores and writes one table per kind:
        the band-mean tables are long-format, so a ``chromophore`` column keeps their
        filenames stable.

        **No figures here.** The transform covers every pairing in the group at once and is
        the expensive step, while a figure is of one pairing, so the two are separated: this
        runs once per chromophore and :func:`_figures_for` runs once per chromophore per
        pairing off what it returns.

        ``cond_wtc`` and ``cond_bands`` are positional over the windows, so the chromophores'
        lists line up even where a guard failed on one of them. ``cond_bands`` holds each
        window's untagged band means, which is what the cross matrices are drawn from: they
        carry both chromophores on one pair of axes, so they cannot be built inside this pass.
        """
        out: dict = {"chan": None, "roichan": None, "roihom": None,
                     "cond_chan": [], "cond_roi": [], "cond_roihom": [],
                     "phasescale": None, "cond_phasescale": [],
                     "result": None, "cond_wtc": [], "cond_bands": []}

        wtc_result: WTCResult | None = None
        with guard(f"WTC computation ({ch_type})", errors, scope):
            logger.info("Computing WTC on %s for %d subjects...", ch_type, len(subject_ids))
            wtc_result = compute_wtc(
                wtc_raws, fmin=wtc_fmin, fmax=wtc_fmax, significance=wtc_significance,
                seed=wtc_seed, mc_count=wtc_mc_count, cross=wtc_channel_cross,
                limit_scales=wtc_limit_scales, ch_type=ch_type, sep_bands=sep_bands)

        # --tstart/--tend: the window is read out of the transform, never cut from the
        # recording, so the whole-run pass below is the window's and the conditions inside
        # it are windows of the same transform. One route for both.
        if wtc_result is not None and analysis_window is not None:
            with guard(f"Analysis window ({ch_type})", errors, scope):
                wtc_result = window_result(wtc_result, *analysis_window)

        _apply_null_level(wtc_result, ch_type)

        chan_band_df = _band_means(wtc_result, "wtc", ch_type)
        out["chan"] = chan_band_df
        if chan_band_df is not None and wtc_save_maps:
            _save_maps(wtc_result, ch_type)

        out["phasescale"] = _phase_scale(wtc_result, "whole run", ch_type)
        out["result"] = wtc_result
        out["roichan"] = _roi_band(chan_band_df, ch_type, "whole run")
        out["roihom"] = _roi_hom_band(chan_band_df, ch_type, "whole run")

        # ---- per condition ----
        # each task window read out of the whole-run pass, not recomputed; see `window_result`
        for label, tstart, tstop in cond_windows:
            out["cond_wtc"].append(None)
            out["cond_bands"].append({"chan": None, "roichan": None,
                                      "roihom": None})
            cond_chan = cond_wtc = None
            with guard(f"Condition {label}: WTC ({ch_type})", errors, scope):
                if cond_pad_s is None:
                    if wtc_result is None:
                        raise StageError("the whole-run WTC failed, so no window can be "
                                         "read out of it")
                    cond_wtc = window_result(wtc_result, tstart, tstop)
                else:
                    cond_wtc = _transform_condition(tstart, tstop, ch_type)
                _apply_pair_level(cond_wtc, label, tstart, tstop, ch_type)
                cond_chan = wtc_band_mean(cond_wtc, band_fmin, band_fmax,
                                          mask_coi=wtc_mask_coi)
            if cond_chan is None:
                continue

            out["cond_wtc"][-1] = cond_wtc

            cond_roi = _roi_band(cond_chan, ch_type, f"condition {label}")
            out["cond_bands"][-1] = {"chan": cond_chan, "roichan": cond_roi}
            if cond_roi is not None and not cond_roi.empty:
                cond_roi = cond_roi.copy()
                cond_roi.insert(0, "condition", label)
                out["cond_roi"].append(_tag(cond_roi, ch_type))

            cond_hom = _roi_hom_band(cond_chan, ch_type, f"condition {label}")
            out["cond_bands"][-1]["roihom"] = cond_hom
            if cond_hom is not None and not cond_hom.empty:
                cond_hom = cond_hom.copy()
                cond_hom.insert(0, "condition", label)
                out["cond_roihom"].append(_tag(cond_hom, ch_type))

            cond_chan = cond_chan.copy()
            cond_chan.insert(0, "condition", label)
            out["cond_chan"].append(_tag(cond_chan, ch_type))

            cond_scale = _phase_scale(cond_wtc, f"condition {label}", ch_type)
            if cond_scale is not None and not cond_scale.empty:
                cond_scale.insert(0, "condition", label)
                out["cond_phasescale"].append(_tag(cond_scale, ch_type))

        return out

    if wtc_significance:
        logger.warning("WTC significance on: %d Monte Carlo surrogates per channel pair, "
                       "this is slow.", wtc_mc_count)
    if wtc_channel_cross:
        logger.warning("WTC channel crossing on: every long channel against every other, "
                       "so the pair count is squared and so is the runtime.")
    if len(chroma) > 1:
        logger.info("--wtc-chroma %s: %d full WTC passes, one per chromophore",
                    "+".join(chroma), len(chroma))

    passes = {ch_type: _wtc_pass(ch_type) for ch_type in chroma}

    # which level decided the cells every phase column below was averaged over, since the
    # column cannot say so itself and two runs of one dyad can differ only in this
    run_level = {c: _level_source(passes[c]["result"]) for c in chroma}
    cond_level = {c: {label: _level_source(got)
                      for (label, _, _), got in zip(cond_windows, passes[c]["cond_wtc"])}
                  for c in chroma}

    def _stack(key: str):
        """One kind's rows from every chromophore, tagged, or None when nothing ran."""
        frames = []
        for ch_type, result in passes.items():
            df = result[key]
            if df is not None and not df.empty:
                frames.append(_tag(df.copy(), ch_type))
        return pd.concat(frames, ignore_index=True) if frames else None

    chan_band_df = _stack("chan")
    if chan_band_df is not None:
        logger.info("WTC band means saved: %s",
                    _write_df_tsv(chan_band_df, {"statistic": "wtc"}, "hyper_wtc", phase_level_source=run_level))
    roi_band_df = _stack("roichan")
    if roi_band_df is not None:
        logger.info("WTC ROI means from channels saved: %s",
                    _write_df_tsv(roi_band_df,
                                  {**roi_entities, "aggregation": "roi",
                                   "statistic": "wtc"}, "hyper_wtc_roichan", phase_level_source=run_level))
    roi_hom_df = _stack("roihom")
    if roi_hom_df is not None:
        logger.info("WTC homologous ROI means saved: %s",
                    _write_df_tsv(roi_hom_df,
                                  {**roi_entities, "aggregation": "homologous",
                                   "statistic": "wtc"}, "hyper_wtc_roihom", phase_level_source=run_level))

    phase_scale_df = _stack("phasescale")
    if phase_scale_df is not None:
        logger.info("WTC phase per scale saved: %s",
                    _write_df_tsv(phase_scale_df, {"statistic": "wtcphase"},
                                  "hyper_wtc_phasescale", phase_level_source=run_level))

    # the windows are the one thing a reader cannot reconstruct from the table
    spans = {label: [round(t0, 3), round(t1, 3)] for label, t0, t1 in cond_windows}
    cond_chan_frames = [f for r in passes.values() for f in r["cond_chan"]]
    cond_roi_frames  = [f for r in passes.values() for f in r["cond_roi"]]
    if cond_chan_frames:
        logger.info("WTC band means per condition saved: %s",
                    _write_df_tsv(pd.concat(cond_chan_frames, ignore_index=True),
                                  {"condition": "all", "statistic": "wtc"},
                                  "hyper_wtc_bycondition",
                                  condition_windows_s=spans,
                                  phase_level_source=cond_level))
    if cond_roi_frames:
        logger.info("WTC ROI means per condition saved: %s",
                    _write_df_tsv(pd.concat(cond_roi_frames, ignore_index=True),
                                  {**roi_entities, "aggregation": "roi",
                                   "condition": "all", "statistic": "wtc"},
                                  "hyper_wtc_bycondition_roichan",
                                  condition_windows_s=spans,
                                  phase_level_source=cond_level))
    cond_hom_frames = [f for r in passes.values() for f in r["cond_roihom"]]
    if cond_hom_frames:
        logger.info("WTC homologous ROI means per condition saved: %s",
                    _write_df_tsv(pd.concat(cond_hom_frames, ignore_index=True),
                                  {**roi_entities, "aggregation": "homologous",
                                   "condition": "all", "statistic": "wtc"},
                                  "hyper_wtc_bycondition_roihom",
                                  condition_windows_s=spans,
                                  phase_level_source=cond_level))

    cond_scale_frames = [f for r in passes.values() for f in r["cond_phasescale"]]
    if cond_scale_frames:
        logger.info("WTC phase per scale per condition saved: %s",
                    _write_df_tsv(pd.concat(cond_scale_frames, ignore_index=True),
                                  {"condition": "all", "statistic": "wtcphase"},
                                  "hyper_wtc_bycondition_phasescale",
                                  condition_windows_s=spans,
                                  phase_level_source=cond_level))

    # ---- ISC, which is a cut and not a slice ----
    # Always both chromophores, not --wtc-chroma: its matrix cannot share a file the way the
    # long-format WTC tables can. A window here is a real cut, unlike the coherence, which is
    # sliced out of the whole-run transform; see `compute_isc` and `window_result`.
    isc_pair_frames: list = []

    def _isc_params() -> dict:
        """What the correlation was computed on, for the sidecar to carry."""
        return {"isc_band_hz": list(isc_band) if isc_band else None,
                "isc_whiten_max_order": isc_whiten,
                "isc_max_lag_s": isc_max_lag_s,
                "isc_phase_null_iter": isc_phase_null}

    def _apply_pair_isc_levels(isc: dict, isc_levels: dict) -> dict:
        """Put each condition's re-paired |r| level in place of its chord level, where it fits.

        Written by `fnirs-hyper-pairnull` for the members it re-paired, one condition at a
        time, so only that pairing's condition pages can take it. A table drawn with other ISC
        settings, on other recordings, or over a condition span this run does not share is
        left unused and the chords keep the phase-scrambled level or the fallback.
        """
        from fnirs_pipe.pipeline.hyper.wtc import wtc_grid_params
        from fnirs_pipe.pipeline.hyper.wtc_store import level_mismatch
        path = group_output_path(output_dir, group_id,
                                 {"task": task, "condition": "all", "nulldist": "pair",
                                  "statistic": "isc"}, "relmat", ".tsv")
        expected = {**wtc_grid_params(aligned_raws), **align_info,
                    "isc_whiten_max_order": isc_whiten, "isc_max_lag_s": isc_max_lag_s,
                    "isc_band_hz": list(isc_band) if isc_band else None}
        if not path.exists():
            return {}
        why = level_mismatch(path, expected, aligned_raws)
        if why:
            note(notes, scope, f"{path.name} is on disk but was not used for the chords: "
                               f"{why}. Rerun fnirs-hyper-pairnull on this tree.")
            return {}
        side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["parameters"]
        spans = side.get("condition_windows_s") or {}
        n_partners = int(side.get("n_iter") or 0)
        table = pd.read_csv(path, sep="	")
        sources: dict = {}
        for pr, by_label in isc.items():
            for label, t0, t1 in cond_windows:
                if spans.get(label) != [round(float(t0), 3), round(float(t1), 3)]:
                    continue
                for c, (mat, names) in (by_label.get(label) or {}).items():
                    rows = table[(table["chromophore"] == c) & (table["condition"] == label)
                                 & (table["sub1"] == pr[0]) & (table["sub2"] == pr[1])]
                    if mat is None or rows.empty:
                        continue
                    level = (rows.pivot_table(index="label", columns="label2",
                                              values="null_abs_p95", aggfunc="first")
                             .reindex(index=names, columns=names).to_numpy(dtype=float))
                    if not np.isfinite(level).any():
                        continue
                    isc_levels[pr][label][c] = level
                    sources.setdefault(pr, {}).setdefault(label, {})[c] = n_partners
        return sources

    def _isc_of(ch_type: str, label, window, pair) -> tuple:
        """One scope's ISC: ``((matrix, channels), (matrix, regions), arc level)``."""
        what = f"condition {label}" if label else "whole run"
        channel_level = roi_level = (None, None)
        arc_level = None
        with guard(f"ISC ({what}, {ch_type})", errors, scope):
            pair_ids = list(pair) if pair else subject_ids
            isc_mat, isc_ch_names, pairs_df, arc_level = compute_isc_pairs(
                aligned_raws, pair_ids, ch_type, sep_bands, window=window,
                whiten=isc_whiten, max_lag_s=isc_max_lag_s,
                n_null=isc_phase_null, seed=wtc_seed, band=isc_band)
            if isc_mat is None:
                return channel_level, roi_level, arc_level
            channel_level = (isc_mat, isc_ch_names)
            # the same numbers one row per pairing, which is the shape the z, the AR order
            # and the null columns fit in and the shape a group analysis reads
            pairs_df.insert(0, "chromophore", ch_type)
            if label:
                pairs_df.insert(0, "condition", label)
            isc_pair_frames.append(pairs_df)
            sources = paths_from(aligned_raws.values())
            # the cond- entity a condition's page takes, so its table is named the way its
            # page is and a reader can pair the two without a rule of their own
            common = {"task": task,
                      "pairing": pair_slug(pair, len(pairings)).lstrip("_") or None,
                      "chromophore": ch_type,
                      "condition": _pair_fname(label) if label else None,
                      "statistic": "isc"}
            write_isc_matrix(
                group_output_path(output_dir, group_id, common, "relmat", ".tsv"),
                isc_mat, isc_ch_names, ch_type, sources, pair_ids, align=align_info,
                **_isc_params(),
            )
            if roi_map:
                roi_mat, roi_names = roi_mean_of_isc(
                    isc_mat, isc_ch_names, roi_map, min_channels=wtc_roi_min_channels)
                if roi_mat is not None:
                    roi_level = (roi_mat, roi_names)
                    write_isc_matrix(
                        group_output_path(
                            output_dir, group_id,
                            {**common, **roi_entities, "aggregation": "roi"},
                            "relmat", ".tsv"),
                        roi_mat, roi_names, ch_type, sources, pair_ids, align=align_info,
                        step="hyper_isc_roichan", index_label="roi",
                    )
        return channel_level, roi_level, arc_level

    isc: dict = {}
    isc_roi: dict = {}
    isc_levels: dict = {}
    for pr in pairings:
        scopes = [(None, analysis_window)] + [(lab, (a, b)) for lab, a, b in cond_windows]
        both = {lab: {c: _isc_of(c, lab, window, pr) for c in ("hbo", "hbr")}
                for lab, window in scopes}
        isc[pr] = {lab: {c: got[0] for c, got in by_chroma.items()}
                   for lab, by_chroma in both.items()}
        isc_roi[pr] = {lab: {c: got[1] for c, got in by_chroma.items()}
                       for lab, by_chroma in both.items()}
        isc_levels[pr] = {lab: {c: got[2] for c, got in by_chroma.items()}
                          for lab, by_chroma in both.items()}

    isc_level_sources = _apply_pair_isc_levels(isc, isc_levels)

    if isc_pair_frames:
        # its own writer rather than _write_df_tsv: that one stamps the WTC band and grid on
        # everything it writes, and a correlation was averaged over no band at all
        with guard("ISC pair table", errors, scope):
            tsv_path = group_output_path(output_dir, group_id,
                                         {"task": task, "statistic": "isc"},
                                         "relmat", ".tsv")
            pd.concat(isc_pair_frames, ignore_index=True).to_csv(tsv_path, sep="\t",
                                                                 index=False)
            _hyper_sidecar(
                tsv_path, "hyper_isc_pairs",
                paths_from(aligned_raws.values()),
                **_isc_params(), seed=wtc_seed,
                chroma=["hbo", "hbr"], conditions=[w[0] for w in cond_windows],
                **align_info,
            )
            tables[tsv_path.name] = tsv_path
            logger.info("ISC pair table saved: %s", tsv_path)

    return HyperPostResult(
        subject_ids=subject_ids,
        chroma=chroma,
        band_fmin=band_fmin,
        band_fmax=band_fmax,
        cond_windows=cond_windows,
        cond_pad_s=cond_pad_s,
        chan_axis=chan_axis,
        roi_labels=roi_labels,
        roi_rows=roi_rows,
        passes=passes,
        chan_band_df=chan_band_df,
        roi_band_df=roi_band_df,
        isc=isc,
        isc_roi=isc_roi,
        isc_levels=isc_levels,
        isc_level_sources=isc_level_sources,
        align_info=align_info,
        tables=tables,
    )
