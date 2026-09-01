"""Generate hyperscanning group-level raw QC HTML report."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from fnirs_pipe.pipeline.hyperscanning import GroupEntry
from fnirs_pipe.qc.figure_io import extract_markers, get_channel_pairs
from fnirs_pipe.qc.figures.hyper_figures import _cond_colors
from fnirs_pipe.qc.hyper_raw_writer import _process_hyper_raw_group
from fnirs_pipe.utils.lineage import path_from
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.hyper_report")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_BASE_CSS     = (_TEMPLATE_DIR / "_base.css").read_text(encoding="utf-8")


def write_isc_matrix(
    tsv_path: Path,
    isc_mat,
    ch_names: list[str],
    ch_type: str,
    sources: list[str],
    subject_ids: list[str],
) -> None:
    """Write the matrix the ISC panel is drawn from, so the numbers can leave the report.

    Both axes carry the first subject's channel labels, which is how :func:`compute_isc`
    pairs the two brains: cell (i, j) is that subject's channel i against the other's
    channel j. Rejected channels are blank rather than absent, so the file's shape is the
    montage's however many channels a given dyad lost.

    A failure here costs the file and not the panel: the report is still readable without it.
    """
    from fnirs_pipe.pipeline.hyperscanning import _hyper_sidecar

    try:
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(isc_mat, index=ch_names, columns=ch_names).to_csv(
            tsv_path, sep="\t", index_label="channel")
        _hyper_sidecar(tsv_path, "hyper_isc", sources,
                       chromophore=ch_type, subjects=subject_ids)
        logger.info("ISC matrix saved: %s", tsv_path)
    except Exception as exc:
        logger.warning("ISC matrix (%s) not written: %s", ch_type, exc)


def build_hyper_report(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    sqm_data: dict[str, dict],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    coherence_df: pd.DataFrame,
    output_dir: Path,
    raw_raws: dict[str, mne.io.Raw] | None = None,
    session: str | None = None,
    sci_threshold: float = 0.8,
    cardiac_l_freq: float | None = None,
    cardiac_h_freq: float | None = None,
    coherence_fmin: float = 0.01,
    coherence_fmax: float = 0.10,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    coherence_window_s: float = 30.0,
    coherence_step_s: float = 5.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _process_hyper_raw_group(
        group_id=group_id, task=task, group=group,
        sqm_data=sqm_data, aligned_raws=aligned_raws, offsets=offsets,
        coherence_df=coherence_df, output_dir=output_dir,
        raw_raws=raw_raws, session=session, sci_threshold=sci_threshold,
        cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq,
        coherence_fmin=coherence_fmin, coherence_fmax=coherence_fmax,
        epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
        coherence_window_s=coherence_window_s, coherence_step_s=coherence_step_s,
    )

    sci_per_subject = {
        sid: sqm_data.get(sid, {}).get("sci_per_channel", {})
        for sid in meta["subject_ids"]
    }

    name_parts = [f"group-{group_id}"]
    if session:
        name_parts.append(f"ses-{session}")
    name_parts.append(f"task-{task}")
    output_path = output_dir / ("_".join(name_parts) + "_desc-hyperraw_nirs.html")

    env  = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    html = env.get_template("hyper_report.html.j2").render(
        base_css=_BASE_CSS,
        group_id=group_id,
        task=task,
        subject_ids=meta["subject_ids"],
        sci_threshold=sci_threshold,
        coherence_fmin=coherence_fmin,
        coherence_fmax=coherence_fmax,
        alignment_json=json.dumps(meta["alignment"]),
        ch_pairs_json=json.dumps(meta["ch_pairs"]),
        sqm_json=json.dumps(meta["sqm"], default=str),
        figure_paths=meta["figure_paths"],
        ch_detail_template_json=json.dumps(meta["figure_paths"].get("ch_detail_template")),
        sci_per_subject_json=json.dumps(sci_per_subject),
    )
    output_path.write_text(html, encoding="utf-8")
    logger.info("Hyper raw report saved: %s", output_path)
    return output_path


def build_hyper_post_report(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    output_dir: Path,
    roi_map: dict[str, list[str]] | None = None,
    bad_channels: dict[str, list[str]] | None = None,
    wtc_fmin: float = 0.004,
    wtc_fmax: float = 0.20,
    wtc_band_fmin: float | None = None,
    wtc_band_fmax: float | None = None,
    wtc_significance: bool = False,
    wtc_seed: int | None = None,
    wtc_mc_count: int = 300,
    wtc_roi_cross: bool = False,
    wtc_channel_cross: bool = False,
    wtc_limit_scales: bool = True,
    wtc_save_maps: bool = False,
    isc_threshold: float = 0.3,
) -> Path:
    """Build hyperscanning post-QC report.

    Sections:
      1. Per-channel WTC  — Morlet wavelet coherence, one heatmap per channel
      2. Per-ROI WTC      — WTC on HbO averaged within each ROI (when roi_map given)
      3. ISC matrix       — inter-brain Pearson r heatmap (channel × channel)
      4. ISC connectogram — inter-brain arcs filtered by isc_threshold

    Each WTC map is also collapsed to one number per channel over
    [wtc_band_fmin, wtc_band_fmax] and written as a TSV beside the HTML, so a group
    analysis reads the same values the figures were drawn from.

    ``wtc_roi_cross`` crosses each subject's ROIs with the other's, so the TSV gains the
    off-diagonal pairs. Those extra pairs stay out of the time-frequency selector, which
    keeps showing the homologous ones: every map carried in the page is a full
    frequency × time array, and a report that embedded all of them would be too large to
    open. They arrive instead as one ROI × ROI matrix of band means, one cell per pair.

    ``wtc_channel_cross`` does the same at channel level, 14 channels giving 196 rows in
    ``hyper-wtc.tsv`` instead of 14. Same treatment: the extra pairs reach the TSV, the
    heatmap selector keeps the homologous ones.

    ``wtc_save_maps`` writes the full time-frequency maps beside each TSV as ``.npz``, so a
    different band can be averaged later without a second wavelet transform. See
    :mod:`fnirs_pipe.qc.wtc_store`. ``wtc_limit_scales`` computes only the scales inside
    ``[wtc_fmin, wtc_fmax]`` plus margin, which is most of the runtime and, given that the
    scales land on pycwt's own grid and the margin exceeds its scale-smoothing window,
    reproduces the unrestricted coherences bit for bit.
    """
    from fnirs_pipe.pipeline.hyperscanning import (
        WTCResult,
        _hyper_sidecar,
        compute_wtc,
        compute_wtc_roi,
        roi_mean_of_channels,
        wtc_band_mean,
    )
    from fnirs_pipe.qc.figures.hyper_post_figures import (
        build_isc_panel,
        build_wtc_channel,
        build_wtc_roi_matrix,
        compute_isc,
    )

    subject_ids  = [e.subject_id for e in group]
    ref_raw      = aligned_raws.get(subject_ids[0]) if subject_ids else None
    markers_list = extract_markers(ref_raw) if ref_raw else []
    all_descs    = list(dict.fromkeys(m["description"] for m in markers_list))
    cond_colors_ = _cond_colors(all_descs)

    alignment_rows = [
        {
            "subject_id": sid,
            "offset_s":   round(offsets.get(sid, 0.0), 3),
            "duration_s": round(aligned_raws[sid].times[-1], 1)
                          if sid in aligned_raws else None,
        }
        for sid in subject_ids
    ]

    band_fmin = wtc_band_fmin if wtc_band_fmin is not None else wtc_fmin
    band_fmax = wtc_band_fmax if wtc_band_fmax is not None else wtc_fmax
    if wtc_band_fmin is None or wtc_band_fmax is None:
        logger.info(
            "WTC band mean taken over the whole %.4f-%.4f Hz axis; name a narrower band "
            "to average over the frequencies the task lives in", band_fmin, band_fmax)

    def _write_band_tsv(result: "WTCResult | None", kind: str, step: str):
        if result is None or not result.pairs:
            return None
        try:
            df = wtc_band_mean(result, band_fmin, band_fmax)
        except Exception as exc:
            logger.warning("WTC band mean (%s) failed: %s", kind, exc)
            return None
        tsv_path = _write_df_tsv(df, kind, step)
        if wtc_save_maps:
            from fnirs_pipe.qc.wtc_store import save_wtc
            try:
                save_wtc(result, tsv_path.with_suffix(".npz"))
            except Exception as exc:
                logger.warning("saving WTC maps (%s) failed: %s", kind, exc)
        logger.info("WTC band means saved: %s", tsv_path)
        return df

    def _write_df_tsv(df, kind: str, step: str) -> Path:
        tsv_path = output_dir / f"group-{group_id}_task-{task}_hyper-{kind}.tsv"
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tsv_path, sep="\t", index=False)
        _hyper_sidecar(
            tsv_path, step,
            [p for p in (path_from(r) for r in aligned_raws.values()) if p],
            band_fmin=band_fmin, band_fmax=band_fmax, mask_coi=True,
            wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax,
        )
        return tsv_path

    def _safe_post(name: str, fn, *args):
        try:
            fig = fn(*args)
            return fig.to_dict() if fig is not None else None
        except Exception as exc:
            logger.warning("%s figure failed: %s", name, exc)
            return None

    # Compute WTC
    wtc_result: WTCResult | None = None
    try:
        if wtc_significance:
            logger.warning("WTC significance on: %d Monte Carlo surrogates per channel pair, "
                           "this is slow.", wtc_mc_count)
        if wtc_channel_cross:
            logger.warning("WTC channel crossing on: every long channel against every "
                           "other, so the pair count is squared and so is the runtime.")
        logger.info("Computing WTC for %d subjects...", len(subject_ids))
        wtc_result = compute_wtc(
            aligned_raws, fmin=wtc_fmin, fmax=wtc_fmax, significance=wtc_significance,
            seed=wtc_seed, mc_count=wtc_mc_count, cross=wtc_channel_cross,
            limit_scales=wtc_limit_scales)
    except Exception as exc:
        logger.warning("WTC computation failed: %s", exc)

    chan_band_df = _write_band_tsv(wtc_result, "wtc", "hyper_wtc")

    pair_key   = next(iter(wtc_result.pairs)) if wtc_result and wtc_result.pairs else None
    pair_label = f"{pair_key[0]} × {pair_key[1]}" if pair_key else ""

    # Build per-channel WTC figures
    ch_pairs_post: list[str] = get_channel_pairs(ref_raw) if ref_raw else []
    per_channel_post: dict[str, dict] = {}
    for pair in ch_pairs_post:
        wtc_fig = None
        if wtc_result and pair_key:
            # crossing keys every pair, so the homologous one is the (ch, ch) cell
            ch_key  = (pair, pair) if wtc_channel_cross else pair
            ch_data = wtc_result.pairs.get(pair_key, {}).get(ch_key)
            wtc_fig = _safe_post(
                "wtc", build_wtc_channel,
                ch_data, wtc_result.freqs, wtc_result.times,
                pair_label, markers_list, cond_colors_,
            )
        per_channel_post[pair] = {"wtc": wtc_fig}

    # Compute ISC panels (HbO and HbR)
    def _isc_panel(ch_type: str) -> str:
        try:
            isc_mat, isc_ch_names = compute_isc(
                aligned_raws, subject_ids, ch_type, bad_channels=bad_channels
            )
            if isc_mat is None:
                return ""
            write_isc_matrix(
                output_dir / f"group-{group_id}_task-{task}_hyper-isc-{ch_type}.tsv",
                isc_mat, isc_ch_names, ch_type,
                [p for p in (path_from(r) for r in aligned_raws.values()) if p],
                subject_ids,
            )
            return build_isc_panel(
                isc_mat, isc_ch_names, subject_ids,
                ch_type=ch_type, isc_threshold=isc_threshold,
            )
        except Exception as exc:
            logger.warning("ISC panel (%s) failed: %s", ch_type, exc)
            return ""

    isc_panel_hbo_b64 = _isc_panel("hbo")
    isc_panel_hbr_b64 = _isc_panel("hbr")

    roi_rows: list[dict] = []
    roi_labels: list[str] = []
    per_roi_post: dict[str, dict] = {}
    roi_matrix_fig = None
    if roi_map:
        assigned = {ch for chs in roi_map.values() for ch in chs}
        for roi_name, chs in roi_map.items():
            roi_rows.append({"roi": roi_name, "channels": chs})
        unassigned = [c for c in ch_pairs_post if c not in assigned]
        if unassigned:
            roi_rows.append({"roi": "Unassigned", "channels": unassigned})

        roi_wtc: WTCResult | None = None
        try:
            roi_wtc = compute_wtc_roi(
                aligned_raws, roi_map, bad_channels=bad_channels,
                fmin=wtc_fmin, fmax=wtc_fmax, significance=wtc_significance,
                seed=wtc_seed, mc_count=wtc_mc_count, cross=wtc_roi_cross,
                limit_scales=wtc_limit_scales,
            )
        except Exception as exc:
            logger.warning("ROI WTC computation failed: %s", exc)

        roi_band_df = _write_band_tsv(roi_wtc, "wtc-roi", "hyper_wtc_roi")

        # the ROI number the WTC literature reports; wtc-roi above is the other algorithm
        if chan_band_df is not None:
            try:
                path = _write_df_tsv(
                    roi_mean_of_channels(chan_band_df, roi_map),
                    "wtc-roichan", "hyper_wtc_roichan")
                logger.info("WTC ROI means from channels saved: %s", path)
            except Exception as exc:
                logger.warning("ROI mean of channel WTC failed: %s", exc)

        roi_pair_key = next(iter(roi_wtc.pairs)) if roi_wtc and roi_wtc.pairs else None
        roi_labels   = list(roi_map.keys())
        if roi_band_df is not None and wtc_roi_cross:
            roi_matrix_fig = _safe_post(
                "wtc-roi-matrix", build_wtc_roi_matrix,
                roi_band_df, roi_labels, subject_ids, band_fmin, band_fmax,
            )
        for roi_name in roi_labels:
            roi_fig = None
            if roi_wtc and roi_pair_key:
                # crossing keys every pair, so the homologous one is the (roi, roi) cell
                roi_key  = (roi_name, roi_name) if wtc_roi_cross else roi_name
                roi_data = roi_wtc.pairs.get(roi_pair_key, {}).get(roi_key)
                roi_fig = _safe_post(
                    "wtc-roi", build_wtc_channel,
                    roi_data, roi_wtc.freqs, roi_wtc.times,
                    pair_label, markers_list, cond_colors_,
                )
            per_roi_post[roi_name] = {"wtc": roi_fig}

    bad_pairs_all: set[str] = set()
    if bad_channels:
        for chs in bad_channels.values():
            bad_pairs_all |= {c.rsplit(" ", 1)[0] for c in chs}

    output_path = output_dir / f"group-{group_id}_task-{task}_hyper-post.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env  = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    html = env.get_template("hyper_post_report.html.j2").render(
        base_css=_BASE_CSS,
        group_id=group_id,
        task=task,
        subject_ids=subject_ids,
        wtc_fmin=wtc_fmin,
        wtc_fmax=wtc_fmax,
        isc_threshold=isc_threshold,
        alignment_json=json.dumps(alignment_rows),
        per_channel_post_json=json.dumps(per_channel_post),
        ch_pairs_post_json=json.dumps(ch_pairs_post),
        bad_pairs_json=json.dumps(sorted(bad_pairs_all)),
        isc_panel_hbo_b64=isc_panel_hbo_b64,
        isc_panel_hbr_b64=isc_panel_hbr_b64,
        roi_rows=roi_rows,
        roi_labels_json=json.dumps(roi_labels),
        per_roi_post_json=json.dumps(per_roi_post),
        wtc_roi_matrix_json=json.dumps(roi_matrix_fig),
    )
    output_path.write_text(html, encoding="utf-8")
    logger.info("Hyper post report saved: %s", output_path)
    return output_path
