"""Preprocessing pipeline — all prep steps in a single module.

Step outputs written to output_dir/sub-XX/[ses-YY/]nirs/:
  desc-od            raw intensity → ΔOD
  desc-sci           SCI-pruned OD
  desc-motcorrected  motion-corrected OD
  desc-preproc       final HbO/HbR  (Beer-Lambert output)
  desc-aux           the recording's auxiliary channels, if it has any (tsv.gz)

Each snirf is accompanied by a JSON provenance sidecar.

motion correction lives in pipeline/motion.py.

"""

from dataclasses import dataclass, field
from pathlib import Path

import mne
import mne.io

from fnirs_pipe import __version__
from fnirs_pipe.io.auxiliary import aux_table_path, write_aux_table
from fnirs_pipe.io.derivatives import build_output_path, carry_entities, data_state, write_sidecar_json
from fnirs_pipe.io.snirf import write_snirf
from fnirs_pipe.pipeline.motion import MotionMethod, correct_motion  # noqa: F401  re-exported
from fnirs_pipe.exceptions import StageError
from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.lineage import Recorder, lineage_of, stage_of, stamp
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.prep")

# Step 1: OD conversion
def intensity_to_od(raw: mne.io.Raw) -> mne.io.Raw:
    """Convert raw intensity signal to optical density."""
    od = mne.preprocessing.nirs.optical_density(raw)
    return stamp(od, stage="od", step="od_conversion", source=raw)

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
    wanted = {lbl.rsplit(" ", 1)[0] for lbl in labels}
    return [ch for ch in raw.ch_names if ch.rsplit(" ", 1)[0] in wanted]


def mark_bad_channels(
    raw_od: mne.io.Raw,
    threshold: float,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> tuple[mne.io.Raw, list[str], dict[str, float]]:
    """Mark channels below SCI threshold into raw.info['bads'].

    Returns raw (modified in-place), list of bad channel names, and SCI scores dict.
    Raises StageError if the threshold leaves no usable channel.
    """
    sci_scores = compute_sci(raw_od, cardiac_l_freq, cardiac_h_freq)
    bad_chs = [ch for ch, score in sci_scores.items() if score < threshold]
    # without this the run dies four steps later inside Beer-Lambert, which reports only
    # that it found no optical density data and never mentions the threshold
    if sci_scores and len(bad_chs) == len(sci_scores):
        best = max(sci_scores.values())
        raise StageError(
            f"every channel scored below the SCI threshold {threshold}; the best channel "
            f"scored {best:.3f}. Lower --sci-threshold, or check the recording for "
            f"scalp coupling."
        )
    raw_od.info["bads"] = bad_chs
    stamp(raw_od, stage="sci", step="sci_pruning", source=raw_od, threshold=threshold)
    return raw_od, bad_chs, sci_scores

# Step 4: Beer-Lambert
def od_to_haemo(raw_od: mne.io.Raw, dpf: list[float]) -> mne.io.Raw:
    """Convert OD to haemoglobin concentration via Beer-Lambert law."""
    from mne.preprocessing.nirs import beer_lambert_law
    ppf = dpf[0] if len(dpf) == 1 else dpf
    haemo = beer_lambert_law(raw_od, ppf=ppf)
    return stamp(haemo, stage="preproc", step="beer_lambert", source=raw_od, dpf=dpf)

# Pipeline orchestration
@dataclass
class PrepResult:
    """Outputs from run_prep() needed for reporting and downstream use."""
    raw_haemo: mne.io.Raw
    sci_scores: dict[str, float]
    bad_channels: list[str]

@dataclass
class PrepConfig:
    subject: str
    dpf: list[float]                        # required; one value or one per wavelength
    sci_threshold: float                    # required; e.g. 0.8
    cardiac_l_freq: float
    cardiac_h_freq: float
    resp_l_freq: float
    resp_h_freq: float
    session: str | None = None
    qc_window_s: float = 10.0               # sliding-window length (s) for windowed SCI/PSP/GVTD
    motion_correction: str | None = None
    bad_channels: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)

def run_prep(
    raw: mne.io.Raw,
    config: PrepConfig,
    output_dir: Path,
    source_entities: dict[str, str] | None = None,
    work_dir: Path | None = None,
    source_path: Path | None = None,
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
    """
    entities_base = carry_entities(source_entities)
    ses = config.session

    rec = Recorder()
    if source_path is not None:
        rec.register_input(source_path, raw)

    def _save(raw_step: mne.io.Raw, desc: str, extra_provenance: dict | None = None) -> Path:
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
            "pipeline_version": __version__,
            "step": lin.step,
            "Sources": rec.sources_of(raw_step),
            "parameters": _config_dict(config),
            "data": data_state(raw_step),
            # read back by read_snirf: SNIRF itself cannot carry the marks
            "bad_channels": list(raw_step.info["bads"]),
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
    _save(raw_od, "od")

    # step 2: SCI channel marking
    logger.info("sub-%s | step 2: SCI marking (threshold=%.2f, %d ch)", config.subject, config.sci_threshold, len(raw_od.ch_names))
    raw_od, bad_chs, sci_scores = mark_bad_channels(
        raw_od, threshold=config.sci_threshold,
        cardiac_l_freq=config.cardiac_l_freq, cardiac_h_freq=config.cardiac_h_freq)
    if config.bad_channels:
        manual = _expand_bad_pairs(raw_od, config.bad_channels)
        if not manual:
            logger.warning("sub-%s | --bad-channels matched no channels: %s", config.subject, config.bad_channels)
        else:
            logger.info("sub-%s | manual bad channels: %s", config.subject, manual)
        bad_chs = sorted(set(bad_chs) | set(manual))
        if len(bad_chs) == len(sci_scores):
            raise StageError(
                f"--bad-channels leaves no usable channel: all {len(bad_chs)} are marked bad."
            )
        raw_od.info["bads"] = bad_chs
    n_bad, n_total = len(bad_chs), len(sci_scores)
    logger.info(
        "sub-%s | bad channels: %d/%d%s",
        config.subject, n_bad, n_total,
        f" — {bad_chs}" if bad_chs else "",
    )
    # sci_scores go in the sidecar because the SQM record is assembled from disk after the
    # run, and SCI is the one input to it that no output file carries
    _save(raw_od, "sci", extra_provenance={
        "bad_channels": bad_chs,
        "sci_scores": {k: float(v) for k, v in sci_scores.items()},
    })

    # step 3: motion correction (spike/step artifact repair)
    logger.info("sub-%s | step 3: motion correction (%s)", config.subject, config.motion_correction)
    # no copy kept: desc-sci on disk is this same object, and the report and the record
    # both read the correction's two sides from there
    raw_od = correct_motion(raw_od, method=config.motion_correction)
    _save(raw_od, "motcorrected")

    # step 4: Beer-Lambert
    logger.info("sub-%s | step 4: Beer-Lambert (dpf=%s)", config.subject, config.dpf)
    raw_haemo = od_to_haemo(raw_od, dpf=config.dpf)
    preproc_path = _save(raw_haemo, "preproc")

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
        "parameters": _config_dict(config),
        **facts,
    })
    return out_path


def _config_dict(config: PrepConfig) -> dict:
    return {
        "subject": config.subject,
        "session": config.session,
        "dpf": config.dpf,
        "motion_correction": config.motion_correction,
        "sci_threshold": config.sci_threshold,
        "cardiac_l_freq": config.cardiac_l_freq,
        "cardiac_h_freq": config.cardiac_h_freq,
        "resp_l_freq": config.resp_l_freq,
        "resp_h_freq": config.resp_h_freq,
        "qc_window_s": config.qc_window_s,
        "bad_channels": config.bad_channels,
        "ignore": config.ignore,
    }
