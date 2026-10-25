"""Preprocessing pipeline: all prep steps in a single module.

Step outputs written to output_dir/sub-XX/[ses-YY/]nirs/:
  desc-od            raw intensity → ΔOD
  desc-sci           SCI-pruned OD
  desc-motcorrected  motion-corrected OD
  desc-preproc       final HbO/HbR  (Beer-Lambert output)
  desc-aux           the recording's auxiliary channels, if it has any (tsv.gz)

Each snirf is accompanied by a JSON provenance sidecar.

motion correction lives in pipeline/motion.py.

"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import mne
import mne.io
import numpy as np

from nirspipe import __version__
from nirspipe.io.auxiliary import aux_table_path, write_aux_table
from nirspipe.io.derivatives import (build_output_path, carry_entities, data_state,
                                     nirs_required_fields, write_sidecar_json)
from nirspipe.io.head_template import FSAVERAGE, space_of
from nirspipe.io.optode_space import Placement, hold_placement, place_optodes, write_channel_coords
from nirspipe.io.snirf import write_snirf
from nirspipe.pipeline.dpf import as_ppf, dpf_from_age, resolve_dpf
from nirspipe.pipeline.motion import (  # noqa: F401  MotionMethod re-exported
    MotionMethod,
    correct_motion,
    motion_settings,
    overmarked,
)
from nirspipe.exceptions import StageError
from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.metrics._helpers import GOOD_FRAC_PASS, require_cardiac_below_nyquist
from nirspipe.utils import is_optical_density, pair_of
from nirspipe.utils.lineage import Recorder, lineage_of, stage_of, stamp
from nirspipe.utils.logging import get_logger
from nirspipe.qc.metrics import (require_screened, resolve_cutoffs, screen_channels,
                                 screening_scores)
from nirspipe.qc.metrics.windowed import SCREEN_WINDOW_S, counted_screen_windows
from nirspipe.qc.common.screen_scope import resolve_screen_scope
from nirspipe.utils.spans import (EDGE, add_bad_spans, bad_spans, excluded_spans,
                                  excluded_time, mark_manual, mark_unselected, measured_from)
from nirspipe.pipeline.censor_fill import repair_dark_edges

logger = get_logger("pipeline.prep")

# Step 1: OD conversion
def intensity_to_od(raw: mne.io.Raw) -> mne.io.Raw:
    """Convert raw intensity signal to optical density."""
    types = set(raw.get_channel_types())
    if "fnirs_cw_amplitude" not in types:
        raise StageError(
            f"OD conversion needs raw intensity (fnirs_cw_amplitude channels), got "
            f"{', '.join(sorted(types))}; data already in optical density goes straight to "
            f"screening, which run_prep does on its own."
        )
    od = mne.preprocessing.nirs.optical_density(raw)
    return stamp(od, stage="od", step="od_conversion", source=raw)

def _zero_non_finite(raw: mne.io.Raw) -> dict[str, int]:
    """Zero every NaN / inf sample in place and return how many each affected channel had.

    _zero_non_finite(od_with_nan_stretch) -> {"S1_D1 760": 1000, "S1_D1 850": 1000}
    """
    data = raw.load_data()._data
    bad = ~np.isfinite(data)
    data[bad] = 0.0
    return {raw.ch_names[i]: int(n) for i, n in enumerate(bad.sum(axis=1)) if n}


# Step 2: SCI / bad channel pruning
def compute_sci(raw_od: mne.io.Raw, cardiac_l_freq: float, cardiac_h_freq: float) -> dict[str, float]:
    """Return SCI score per channel name."""
    from mne.preprocessing.nirs import scalp_coupling_index
    scores = scalp_coupling_index(raw_od, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq)
    return dict(zip(raw_od.ch_names, scores))


def _expand_bad_pairs(raw: mne.io.Raw, labels: list[str]) -> list[str]:
    """Match S-D pair labels (or full channel names) to channels present in raw.

    e.g. ["S1_D1"] and ["S1_D1 760"] both give ["S1_D1 760", "S1_D1 850"] on an OD
    recording. A label is reduced to its S-D pair, so naming one wavelength marks the other
    with it: the two are one measurement, and a pair whose 760 nm is unusable has no usable
    760/850 ratio either. Beer-Lambert then renames the marks to "S1_D1 hbo" / "S1_D1 hbr",
    which is what keeps a manual rejection from surviving into only one chromophore.
    """
    wanted = {pair_of(lbl) for lbl in labels}
    return [ch for ch in raw.ch_names if pair_of(ch) in wanted]


def mark_bad_channels(
    raw_od: mne.io.Raw,
    threshold: float,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    psp_threshold: float | None = None,
    min_good_frac: float | None = None,
    screen_scope: str = "run",
    keep: "list[str] | tuple" = (),
) -> tuple[mne.io.Raw, list[str], dict[str, float], dict[str, float]]:
    """Screen channels into raw.info['bads'].

    The criteria are :data:`nirspipe.qc.metrics.screening.CRITERIA` and a channel is
    rejected if it fails any criterion that screens, so what gets pruned is decided by that
    table rather than here. ``threshold`` is the SCI line a window has to clear, and
    ``psp_threshold`` a PSP line it has to clear as well, None for none; ``min_good_frac`` is
    the share of windows that has to pass, and it is the line that rejects, None keeping the
    criterion's own default. ``screen_scope`` is
    "run" or "task" and decides which windows are counted; see :func:`resolve_screen_scope`.
    Windows touching a ``BAD_`` span are never counted, and fewer than two left stops the run.

    Returns raw (modified in-place), the rejected channel names, the SCI scores, and the
    passing-window shares, the last two because the report and the quality record both want
    them on their own. ``keep`` (pairs a reviewer kept) does not change the rejections
    returned; it only stops a run whose every channel failed from being refused here.
    Raises StageError if the criteria leave no usable channel and nothing is kept.
    """
    require_cardiac_below_nyquist(raw_od.info["sfreq"], cardiac_h_freq)
    cutoffs = resolve_cutoffs(sci=threshold, psp=psp_threshold, good_frac=min_good_frac)
    scope = resolve_screen_scope(raw_od, screen_scope)
    name = raw_od.filenames[0].name if raw_od.filenames and raw_od.filenames[0] else "run"
    _, counted = counted_screen_windows(raw_od, scope)
    if counted.sum() < 2:
        found = (f"channel screening has {int(counted.sum())} window(s) of "
                 f"{SCREEN_WINDOW_S:g} s to count and needs 2")
        if bad_spans(raw_od):
            raise StageError(
                f"{name}: with its BAD_ spans left out"
                f"{' and only the task blocks counted' if scope else ''}, {found} "
                f"({excluded_time(raw_od)['kept_s']:.0f} s of the recording "
                f"are outside every BAD_ span). Widen --keep-spans or shorten --bad-spans for "
                f"this recording.")
        raise StageError(
            f"{name}: {'with only the task blocks counted, ' if scope else ''}{found} "
            f"({raw_od.times[-1]:.0f} s recorded). "
            f"{'Use --screen-scope run, or leave' if scope else 'Leave'} this recording out.")
    sci_scores = compute_sci(raw_od, cardiac_l_freq, cardiac_h_freq)
    scores = screening_scores(raw_od, cardiac_l_freq, cardiac_h_freq,
                              have={"sci": sci_scores}, cutoffs=cutoffs, scope=scope)
    require_screened(scores, name)
    bad_chs, why = screen_channels(scores, cutoffs)
    # without this the run dies four steps later inside Beer-Lambert, which reports only
    # that it found no optical density data and never mentions the screening
    if sci_scores and len(release_kept(bad_chs, list(keep))[0]) == len(sci_scores):
        failed = sorted({c for reasons in why.values() for c in reasons})
        lines = ", ".join(f"{k} {v}" for k, v in sorted(cutoffs.items()) if v is not None)
        raise StageError(
            f"every channel failed screening on {', '.join(failed)} (cutoffs {lines}; "
            f"best SCI {max(sci_scores.values()):.3f}). Lower --min-good-frac, or the "
            f"--sci-threshold / --psp-threshold a window has to clear, or check the "
            f"recording for scalp coupling."
        )
    raw_od.info["bads"] = bad_chs
    stamp(raw_od, stage="sci", step="sci_pruning", source=raw_od, threshold=threshold,
          psp_threshold=cutoffs["psp"], min_good_frac=cutoffs["good_frac"],
          screen_scope=screen_scope, screen_scope_windows=len(scope or []))
    return raw_od, bad_chs, sci_scores, scores.get("good_frac") or {}

def release_kept(screened: list[str], keep: list[str]) -> tuple[list[str], list[str]]:
    """Screening's rejections minus the pairs a reviewer kept, as ``(still bad, kept)``.

    ``(["S1_D1 760", "S1_D1 850", "S2_D2 760", "S2_D2 850"], ["S2_D2"])``
        -> ``(["S1_D1 760", "S1_D1 850"], ["S2_D2 760", "S2_D2 850"])``
    """
    wanted = {pair_of(c) for c in keep}
    kept = [c for c in screened if pair_of(c) in wanted]
    return [c for c in screened if c not in kept], kept


# Step 4: Beer-Lambert
def od_to_haemo(raw_od: mne.io.Raw, dpf: list[float],
                age: "float | None" = None) -> mne.io.Raw:
    """Convert OD to haemoglobin concentration via Beer-Lambert law; ``age`` is what an age DPF came from."""
    from mne.preprocessing.nirs import beer_lambert_law
    haemo = beer_lambert_law(raw_od, ppf=as_ppf(dpf))
    return stamp(haemo, stage="preproc", step="beer_lambert", source=raw_od, dpf=dpf,
                 **({"dpf_age": age} if age is not None else {}))

# Pipeline orchestration
@dataclass
class PrepResult:
    """Outputs from run_prep() needed for reporting and downstream use."""
    raw_haemo: mne.io.Raw
    sci_scores: dict[str, float]
    bad_channels: list[str]
    censor_spans: list[tuple[float, float]] = field(default_factory=list)

@dataclass
class PrepConfig:
    subject: str
    dpf: list                               # required; one value, one per wavelength, or ["age"]
    sci_threshold: float                    # required; e.g. 0.8
    cardiac_l_freq: float
    cardiac_h_freq: float
    resp_l_freq: float
    resp_h_freq: float
    session: str | None = None
    age: float | None = None                # years, from participants.tsv, for --dpf age
    template: str = FSAVERAGE               # the head the optodes are placed on, by --template
    psp_threshold: float | None = None      # a second per-window line, off unless given
    min_good_frac: float = GOOD_FRAC_PASS   # share of passing windows a kept channel needs
    screen_scope: str = "run"               # count those windows over "run" or "task" only
    qc_window_s: float = 10.0               # sliding-window length (s) for windowed SCI/PSP/GVTD
    # trial window for the report's epoch figures and per-trial scoring; None means the
    # report's own -5 to 25 s for the figures and each event's own duration for the scoring
    epoch_tmin: float | None = None
    epoch_tmax: float | None = None
    # cut each task annotation into trials this long first, for a block design
    epoch_chunk_duration: float | None = None
    # the channel set to censor on, or None for no censoring: "long", "short" or "all".
    # It is one field rather than a flag plus a set because --gvtd-censor carries both
    gvtd_censor: str | None = None
    gvtd_censor_n_std: float = 10.0
    gvtd_min_epoch_s: float = 30.0
    motion_correction: str | None = None
    # None takes the method's default; a method that does not read one refuses it
    motion_std_thresh: float | None = None
    wavelet_iqr: float | None = None
    # separation bands, in metres; None each keeps the package default, and a None
    # long_max_dist means no upper bound at all
    short_max_dist: float | None = None
    long_min_dist: float | None = None
    long_max_dist: float | None = None
    bad_channels: list[str] = field(default_factory=list)
    # the reviewer's keeps: screening's rejections of these come back out; marks by hand still win
    keep_channels: list[str] = field(default_factory=list)
    # the channels the run's BIDS channels.tsv marks bad in its status column
    channels_tsv_bads: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)
    # (onset, duration) stretches on the recording's own axis; empty uses the whole recording
    keep_spans: list[tuple[float, float]] = field(default_factory=list)
    # the marker keep_spans are measured from instead, or None
    keep_spans_from: str | None = None
    # (onset, duration) stretches the user marked corrupted, on the same axis
    bad_spans: list[tuple[float, float]] = field(default_factory=list)

def run_prep(
    raw: mne.io.Raw,
    config: PrepConfig,
    output_dir: Path,
    source_entities: dict[str, str] | None = None,
    source_path: Path | None = None,
    source_metadata: dict | None = None,
) -> PrepResult:
    """Run the full preprocessing pipeline in locked step order.

    Steps (order is fixed; BIDS validation handled upstream before this call):
      1. OD conversion     -> desc-od_nirs.snirf
      2. SCI channel marking -> desc-sci_nirs.snirf
      3. Motion correction -> desc-motcorrected_nirs.snirf
      4. Beer-Lambert      -> desc-preproc_nirs.snirf
      5. Aux extraction    -> desc-aux_timeseries.tsv.gz  (only if the recording has aux)

    source_path is the BIDS file *raw* was read from; it becomes the Sources
    entry of the first output, and step 5 reads the aux channels back out of it.
    source_metadata is that file's sidecar, inheritance applied; every output carries
    its REQUIRED _nirs.json fields on.
    """
    entities_base = carry_entities(source_entities)
    ses = config.session
    # off the recording's own probe: a derivative's landmarks are the ones mne-nirs wrote.
    # Held on the input, so the report drawn from it later uses the same head
    space = space_of(config.template)
    placement = (place_optodes(source_path, space) if source_path is not None
                 else Placement("none", note="no source file to read the probe from", space=space))
    if not placement.placed:
        logger.warning("sub-%s | optodes not placed on %s: %s", config.subject, space,
                       placement.note)
    hold_placement(raw, placement)

    input_spans = bad_spans(raw)
    if config.keep_spans_from:
        # onto the recording's own axis, the one every sidecar records them on
        config = replace(config, keep_spans=measured_from(raw, config.keep_spans,
                                                          config.keep_spans_from))
    if config.keep_spans:
        raw = mark_unselected(raw.copy(), config.keep_spans)
    if config.bad_spans:
        raw = mark_manual(raw.copy(), config.bad_spans)
    # before OD conversion, where one non-finite sample would take its whole channel
    raw, (lead, trail) = repair_dark_edges(raw)
    if lead or trail:
        logger.warning("sub-%s | %d frame(s) at the start and %d at the end measured no light "
                       "in most channels; held at the nearest lit frame and marked %s",
                       config.subject, lead, trail, EDGE)

    rec = Recorder()
    if source_path is not None:
        rec.register_input(source_path, raw)

    def _save(raw_step: mne.io.Raw, desc: str, extra_provenance: dict | None = None,
              extra_parameters: dict | None = None) -> Path:
        lin = lineage_of(raw_step)
        if lin is None or lin.stage != desc:
            raise StageError(f"_save({desc!r}) got an object stamped {stage_of(raw_step)!r}")
        path = build_output_path(
            output_dir=output_dir,
            subject=config.subject,
            entities={**entities_base, "desc": desc},
            suffix="nirs",
            extension=".snirf",
            session=ses,
        )
        write_snirf(raw_step, path)
        write_sidecar_json(path, {
            **nirs_required_fields(raw_step, source_metadata or {}),
            "pipeline_version": __version__,
            "step": lin.step,
            "Sources": rec.sources_of(raw_step),
            # the settings this step used; the run record holds the whole configuration
            "parameters": extra_parameters or {},
            "data": data_state(raw_step),
            # read back by read_snirf: SNIRF itself cannot carry the marks
            "bad_channels": list(raw_step.info["bads"]),
            "excluded_spans": excluded_spans(raw_step, input_spans),
            **(extra_provenance or {}),
        })
        return rec.written(path, raw_step)

    # step 1: OD conversion (skip if input is already optical density)
    if is_optical_density(raw):
        logger.warning(
            "sub-%s | input is already optical density; skipping OD conversion "
            "and raw-intensity QC", config.subject,
        )
        raw_od = stamp(raw.copy(), stage="od", step="od_passthrough", source=raw)
    else:
        logger.info("sub-%s | step 1: OD conversion (%d ch)", config.subject, len(raw.ch_names))
        raw_od = intensity_to_od(raw)
    # left in, a non-finite sample survives screening in a rejected channel and crashes the
    # GLM later; zeroed and forced bad here, the channel is dropped like any other
    non_finite = _zero_non_finite(raw_od)
    if non_finite:
        logger.warning("sub-%s | non-finite samples zeroed and their channels marked bad: %s",
                       config.subject, non_finite)
    _save(raw_od, "od", extra_provenance={
        **({"non_finite_samples": non_finite} if non_finite else {}),
        **({"dark_edge_frames": {"start": lead, "end": trail}} if lead or trail else {}),
    } or None)

    # step 2: SCI channel marking
    logger.info("sub-%s | step 2: SCI marking (threshold=%.2f, %d ch)", config.subject, config.sci_threshold, len(raw_od.ch_names))
    raw_od, bad_chs, sci_scores, good_frac_scores = mark_bad_channels(
        raw_od, threshold=config.sci_threshold, psp_threshold=config.psp_threshold,
        min_good_frac=config.min_good_frac, screen_scope=config.screen_scope,
        cardiac_l_freq=config.cardiac_l_freq, cardiac_h_freq=config.cardiac_h_freq,
        keep=config.keep_channels)
    kept: list[str] = []
    if config.keep_channels:
        both = sorted({pair_of(c) for c in config.keep_channels}
                      & {pair_of(c) for c in config.bad_channels})
        if both:
            raise StageError(f"--bad-channels both marks bad and keeps {', '.join(both)} for "
                             f"this recording; settle each in the rating viewer")
        bad_chs, kept = release_kept(bad_chs, config.keep_channels)
        if kept:
            logger.info("sub-%s | kept on review despite screening: %s", config.subject, kept)
        raw_od.info["bads"] = bad_chs
    forced = _expand_bad_pairs(raw_od, list(non_finite))
    if config.bad_channels:
        manual = _expand_bad_pairs(raw_od, config.bad_channels)
        if not manual:
            logger.warning("sub-%s | --bad-channels matched no channels: %s", config.subject, config.bad_channels)
        else:
            logger.info("sub-%s | manual bad channels: %s", config.subject, manual)
        forced += manual
    if config.channels_tsv_bads:
        listed = _expand_bad_pairs(raw_od, config.channels_tsv_bads)
        unmatched = sorted({pair_of(c) for c in config.channels_tsv_bads}
                           - {pair_of(c) for c in listed})
        if unmatched:
            logger.warning("sub-%s | channels.tsv marks bad channels the recording does not "
                           "have: %s", config.subject, unmatched)
        if listed:
            logger.info("sub-%s | bad channels from channels.tsv: %s", config.subject, listed)
        forced += listed
    if forced:
        bad_chs = sorted(set(bad_chs) | set(forced))
        overruled = sorted({pair_of(c) for c in config.keep_channels} & {pair_of(c) for c in forced})
        if overruled:
            logger.warning("sub-%s | kept on review but marked bad by channels.tsv or for "
                           "non-finite samples, so still bad: %s", config.subject, overruled)
            kept = [c for c in kept if pair_of(c) not in overruled]
        if len(bad_chs) == len(sci_scores):
            why = [w for w, on in (("--bad-channels", config.bad_channels),
                                   ("channels.tsv", config.channels_tsv_bads),
                                   ("non-finite samples", non_finite)) if on]
            raise StageError(
                f"{' and '.join(why)} leave{'' if len(why) > 1 else 's'} no usable channel: "
                f"all {len(bad_chs)} are marked bad."
            )
        raw_od.info["bads"] = bad_chs
    n_bad, n_total = len(bad_chs), len(sci_scores)
    logger.info(
        "sub-%s | bad channels: %d/%d%s",
        config.subject, n_bad, n_total,
        f" ({bad_chs})" if bad_chs else "",
    )
    # before desc-sci is written, so that file carries the marks and every
    # derivative taken from it inherits them
    censor_spans: list[tuple[float, float]] = []
    censor_metrics: dict = {}
    if config.gvtd_censor:
        logger.info("sub-%s | GVTD censoring on %s channels (n_std=%s, min epoch %.0fs)",
                    config.subject, config.gvtd_censor, config.gvtd_censor_n_std,
                    config.gvtd_min_epoch_s)
        from nirspipe.qc.metrics import gvtd_censor_spans
        from nirspipe.qc.metrics._helpers import bands_to_record, separation_bands
        censor_spans, censor_metrics = gvtd_censor_spans(
            raw_od, n_std=config.gvtd_censor_n_std,
            min_epoch_s=config.gvtd_min_epoch_s,
            sep_bands=separation_bands(config),
            channel_set=config.gvtd_censor)
        # BAD_ annotations, so the spans travel with the data instead of being cut out of
        # it: MNE's reject_by_annotation drops the epochs they overlap, and a continuous
        # analysis can pick the surviving stretches
        add_bad_spans(raw_od, censor_spans, "BAD_gvtd")

    # These go in the sidecar because the SQM record is assembled from disk after the run,
    # and they are the inputs to it that no output file carries. `good_frac_scores` is the
    # second of them and the more important one: it is the line a channel is rejected on,
    # and unlike SCI it cannot be recomputed from the OD file alone, since the count depends
    # on both thresholds and on the screening scope this run used.
    # what the screening counted over, which a "task" request falls back from without blocks
    counted = ("task" if (lineage_of(raw_od).params or {}).get("screen_scope_windows")
               else "run")
    _save(raw_od, "sci", extra_provenance={
        "bad_channels": bad_chs,
        "sci_scores": {k: float(v) for k, v in sci_scores.items()},
        "good_frac_scores": {k: float(v) for k, v in good_frac_scores.items()},
        **({"gvtd_censor": censor_metrics} if censor_metrics else {}),
        "excluded_time": excluded_time(raw_od),
    }, extra_parameters={
        **_screening_parameters(config),
        "screen_scope_counted": counted,
        # what the Methods sentence on reviewer keeps is written from
        "kept_channels": kept,
        # what the Methods sentence on GVTD censoring is written from
        **({"gvtd_censor": config.gvtd_censor,
            **{k: censor_metrics.get(k) for k in ("gvtd_censor_n_std", "gvtd_censor_min_epoch_s",
                                                  "gvtd_censor_channel_set")},
            # the bands decide which channels the censoring read
            **bands_to_record(separation_bands(config))}
           if config.gvtd_censor else {}),
    })

    # step 3: motion correction (spike/step artifact repair)
    logger.info("sub-%s | step 3: motion correction (%s)", config.subject, config.motion_correction)
    # no copy kept: desc-sci on disk is this same object, and the report and the record
    # both read the correction's two sides from there
    raw_od = correct_motion(raw_od, method=config.motion_correction,
                            std_thresh=config.motion_std_thresh, iqr_factor=config.wavelet_iqr)
    tpca = (lineage_of(raw_od).params or {}).get("tpca")
    spline = (lineage_of(raw_od).params or {}).get("spline")
    spline_sg = (lineage_of(raw_od).params or {}).get("spline_sg")
    spline_wavelet = (lineage_of(raw_od).params or {}).get("spline_wavelet")
    _save(raw_od, "motcorrected",
          extra_provenance={k: v for k, v in (("tpca", tpca), ("spline", spline),
                                              ("spline_sg", spline_sg),
                                              ("spline_wavelet", spline_wavelet)) if v} or None,
          extra_parameters={"motion_correction": config.motion_correction,
                            **motion_settings(config.motion_correction, config.motion_std_thresh,
                                              config.wavelet_iqr)})
    summary = (lineage_of(raw_od).params or {}).get(config.motion_correction)
    if overmarked(config.motion_correction, summary):
        logger.warning("sub-%s | %s", config.subject, section_note(
            "caveat.motion_overmarked", method=config.motion_correction,
            share=summary["motion_share"], sfreq=raw_od.info["sfreq"],
            thresh=motion_settings(config.motion_correction,
                                   config.motion_std_thresh)["motion_std_thresh"]))

    # step 4: Beer-Lambert
    factors = resolve_dpf(config.dpf, raw_od, config.age)
    logger.info("sub-%s | step 4: Beer-Lambert (dpf=%s)", config.subject,
                ", ".join(f"{d:.3f}" for d in factors))
    aged = config.age if dpf_from_age(config.dpf) else None
    raw_haemo = od_to_haemo(raw_od, dpf=factors, age=aged)
    preproc_path = _save(raw_haemo, "preproc", extra_parameters={
        "dpf": factors, **({"dpf_age": aged} if aged is not None else {})})
    # named by the haemoglobin channels, as every analysis table after this one is
    write_channel_coords(raw_haemo, placement,
                         build_output_path(output_dir, config.subject,
                                           {**entities_base, "space": space}, "channels",
                                           ".tsv", session=ses),
                         sources=[source_path] if source_path is not None else [])

    # step 5: the aux channels, if the recording has any. They never enter an mne object, so
    # this is the only chance to carry them forward: the derivative snirfs do not hold them.
    _save_aux_table(source_path, preproc_path, config)

    # SQM is not computed here. It is assembled per run from the files this pipeline left
    # on disk, once post-processing has also finished; see qc/sqm_record.py.

    # The windowed series are not computed here either. They are assembled from
    # desc-motcorrected once both passes have finished, and the report reads them back from
    # the record; see qc/sqm_record.py.
    return PrepResult(
        raw_haemo=raw_haemo,
        sci_scores=sci_scores,
        bad_channels=bad_chs,
        censor_spans=censor_spans,
    )


def _save_aux_table(source_path: Path | None, preproc_path: Path, config: PrepConfig) -> Path | None:
    """Copy the recording's aux channels out of the source snirf, beside the preproc output.

    Nothing downstream can recover them otherwise: MNE never reads them, so they are not in
    the Raw the pipeline carries, and `write_snirf` therefore cannot put them in any
    derivative. Failure is logged and swallowed, since a missing confound table is not a
    reason to lose a finished preprocessing run.
    """
    if source_path is None:
        logger.debug("sub-%s | no source path, skipping aux extraction", config.subject)
        return None

    out_path = aux_table_path(preproc_path)
    try:
        written = write_aux_table(Path(source_path), out_path)
    except Exception:
        logger.exception("sub-%s | aux extraction failed", config.subject)
        return None

    if written is None:
        logger.info("sub-%s | recording carries no aux channels", config.subject)
        return None

    table, facts = written
    logger.info("sub-%s | step 5: aux table (%d channels at %.2f Hz)",
                config.subject, len(table.columns) - 1, facts["SamplingFrequency"])
    write_sidecar_json(out_path, {
        "pipeline_version": __version__,
        "step": "aux_extract",
        "Sources": [Path(source_path).as_posix()],
        "parameters": {},
        **facts,
    })
    return out_path


def _screening_parameters(config: PrepConfig) -> dict:
    """What desc-sci records: screening settings, span and bad-channel inputs, the SQM band and window."""
    return {
        "sci_threshold": config.sci_threshold,
        "psp_threshold": config.psp_threshold,
        "min_good_frac": config.min_good_frac,
        "screen_scope": config.screen_scope,
        "cardiac_l_freq": config.cardiac_l_freq,
        "cardiac_h_freq": config.cardiac_h_freq,
        "resp_l_freq": config.resp_l_freq,
        "resp_h_freq": config.resp_h_freq,
        "qc_window_s": config.qc_window_s,
        "bad_channels": config.bad_channels,
        "keep_channels": config.keep_channels,
        "channels_tsv_bads": config.channels_tsv_bads,
        # read by no step, but it tells an empty channels_tsv_bads from a skipped status column
        "ignore": config.ignore,
        "keep_spans": [list(span) for span in config.keep_spans],
        "keep_spans_from": config.keep_spans_from,
        "bad_spans": [list(span) for span in config.bad_spans],
    }
