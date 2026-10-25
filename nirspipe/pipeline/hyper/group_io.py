"""Reading a hyperscanning group's files off disk; who is in it is ``nirspipe.io.groups``.

Finds and loads; does not align and does not judge. Depends on no other group module.
"""

from __future__ import annotations

from pathlib import Path

import mne

from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import find_preproc_snirf
from nirspipe.io.derivatives import subject_nirs_dirs
from nirspipe.io.derivatives import write_sidecar_json
from nirspipe.io.groups import GroupEntry, member_snirfs
from nirspipe.io.snirf import read_snirf
from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.metrics._helpers import as_optical_density
from nirspipe.utils import is_optical_density
from nirspipe.utils.lineage import lineage_of
from nirspipe.utils.lineage import stage_of
from nirspipe.utils.logging import get_logger
from nirspipe import __version__

logger = get_logger("pipeline.group_io")


def _hyper_sidecar(path: Path, step: str, sources: list[str], **params) -> None:
    write_sidecar_json(path, {
        "pipeline_version": __version__,
        "step": step,
        "Sources": sources,
        "parameters": params,
    })


def load_group_raw_bids(bids_dir: Path, group: list[GroupEntry]) -> dict[str, mne.io.Raw]:
    """Load raw CW-amplitude SNIRF from BIDS for each group member.

    Returns {subject_id: raw_intensity}.
    Raises MissingDerivativesError if no SNIRF is found for any member.
    """
    return {sid: read_snirf(path, verbose=False)
            for sid, path in member_snirfs(bids_dir, group).items()}


def load_group_haemo(
    output_dir: Path,
    group: list[GroupEntry],
    desc: str = "preproc",
) -> dict[str, mne.io.Raw]:
    """Load one haemoglobin stage per subject, selected by its desc entity.

    "preproc" is Beer-Lambert output, the stage every montage has. Anything the post
    pipeline wrote is equally valid input here: "filtered", "resampled", "errts" (the
    confound-regression residual).

    Returns {subject_id: raw_haemo}.
    Raises MissingDerivativesError if any SNIRF is absent, StageError if one holds optical
    density rather than concentration.
    """
    result: dict[str, mne.io.Raw] = {}
    for entry in group:
        snirf_path = find_preproc_snirf(output_dir, entry.subject_id, entry.task, desc=desc,
                                        session=entry.session, run=entry.run)
        raw = read_snirf(snirf_path, verbose=False)
        if is_optical_density(raw):
            raise StageError(
                f"{snirf_path.name} holds optical density, not haemoglobin concentration. "
                f"desc-{desc} is a pre-Beer-Lambert stage; pick one at or after it "
                "(preproc, filtered, resampled, errts, errtsbroad)."
            )
        result[entry.subject_id] = raw
    return result


def load_group_stage(
    output_dir: Path, group: list[GroupEntry], desc: str,
) -> dict[str, mne.io.Raw]:
    """Each member's ``desc-<desc>`` derivative, for the members that have one.

    ::

        load_group_stage(out, group, "motcorrected")  ->  {"sub-01": Raw, ...}

    Optional by design: the dyad raw report runs off BIDS and a member who has never been
    through ``nirspipe`` simply contributes nothing here, which the caller draws as a
    panel with no corrected side rather than as a failure. The task entity is matched, so a
    subject with five tasks does not hand back another task's recording.
    """
    out: dict[str, mne.io.Raw] = {}
    for entry in group:
        paths = _for_task(
            _member_sqm_files(output_dir, entry,
                              f"{entry.subject_id}*_desc-{desc}_nirs.snirf"),
            entry.task)
        if not paths:
            continue
        try:
            out[entry.subject_id] = read_snirf(paths[0], verbose=False)
        except Exception:
            logger.warning("%s: desc-%s could not be read", entry.subject_id, desc,
                           exc_info=True)
    return out


def _raw_to_haemo(raw: mne.io.Raw, dpf: list, age: "float | None" = None) -> mne.io.Raw:
    from nirspipe.pipeline.dpf import as_ppf, resolve_dpf

    raw_od = as_optical_density(raw)
    return mne.preprocessing.nirs.beer_lambert_law(raw_od,
                                                   ppf=as_ppf(resolve_dpf(dpf, raw_od, age)))


def _for_task(paths: list[Path], task: str) -> list[Path]:
    """The subset of paths belonging to one task.

    ``[sub-01_task-rest_x, sub-01_task-game_x], "rest"`` -> the rest one
    ``[sub-01_x], "rest"``                               -> that one (nothing is labelled)
    ``[sub-01_task-game_x], "rest"``                     -> nothing

    The second case covers derivatives with no task entity at all, where the single
    unlabelled file is the right answer. The third returns empty on purpose: substituting
    another task's file is the failure this function exists to prevent.
    """
    matched = [p for p in paths if f"_task-{task}_" in p.name]
    if matched or any("_task-" in p.name for p in paths):
        return matched
    return paths


def _member_sqm_files(output_dir: Path, entry: GroupEntry, pattern: str) -> list[Path]:
    """The member's quality files matching ``pattern``, session folders included.

    ::

      sub-01/ses-a/nirs/sub-01_ses-a_task-hold_desc-sci_nirs.json, entry with no session
        -> that file, rather than nothing

    Reads the same folders :func:`~nirspipe.io.derivatives.find_preproc_snirf` reads the
    recording itself from.
    """
    return sorted((f for d in subject_nirs_dirs(output_dir, entry.subject_id, entry.session)
                   for f in d.glob(pattern)), key=lambda f: f.name)


def _low_edge(params: dict) -> "tuple[float | None, str]":
    """The lowest frequency a file still carries, and which setting put it there.

    _low_edge({"high_pass": 0.01}) -> (0.01, "0.01 Hz high-pass")

    Two settings can empty the low end and a run may use either or both: the bandpass, and
    a cosine drift basis, which spans everything below its cutoff and so is a high-pass by
    projection. The binding one is whichever sits higher.
    """
    low = params.get("high_pass")
    drift = params.get("drift_high_pass") if params.get("drift_model") == "cosine" else None
    if drift is not None and (low is None or drift > low):
        return drift, f"{drift} Hz cosine drift basis"
    if low is not None:
        return low, f"{low} Hz high-pass"
    return None, ""


def warn_outside_passband(raws: dict[str, mne.io.Raw], fmin: float, fmax: float) -> None:
    """Warn when a requested frequency range reaches past the band the files record.

    Reported rather than enforced: a wider range is occasionally deliberate. The band comes
    from the sidecar, so it is what the file went through rather than what was asked for.
    """
    for subject_id, raw in raws.items():
        lin = lineage_of(raw)
        params = (lin.params if lin else None) or {}
        low, low_label = _low_edge(params)
        high = params.get("low_pass")
        outside = []
        if low is not None and fmin < low:
            outside.append(f"{fmin} Hz is below its {low_label}")
        if high is not None and fmax > high:
            outside.append(f"{fmax} Hz is above its {high} Hz low-pass")
        if outside:
            logger.warning("sub-%s | requested %s-%s Hz but %s. Those scales carry what the "
                           "filter removed, not signal",
                           subject_id, fmin, fmax, " and ".join(outside))


def unfiltered_stage_note(
    raws: dict[str, mne.io.Raw],
    isc_band: "tuple[float | None, float | None] | None" = None,
) -> "str | None":
    """A sentence for the ISC panel when the files record no bandpass, else None.

    ISC is a whole-record zero-lag correlation and so has no frequency axis to keep drift
    out of; on an unfiltered stage it is dominated by the slowest component present. WTC is
    unaffected, since its band mean averages only the cells inside the requested band.

    ``--desc`` defaults to ``preproc``, which is Beer-Lambert output and is not bandpassed,
    so the default is the case this warns about.

    The low edge comes from the sidecar through the lineage stamp, the same route
    :func:`warn_outside_passband` reads, so a file whose sidecar is missing looks the same as
    one that was never filtered. The wording says "record no bandpass" rather than "are
    unfiltered" for that reason. A cosine drift basis counts: it empties the band below its
    cutoff just as the filter does, so a run that used one and no bandpass is not warned about.
    Nor is one whose ``isc_band`` has a low edge: the correlation is then high-passed itself.
    """
    if isc_band and isc_band[0] is not None:
        return None
    unrecorded = []
    for subject_id, raw in sorted(raws.items()):
        lin = lineage_of(raw)
        if _low_edge((lin.params if lin else None) or {})[0] is None:
            unrecorded.append(subject_id)
    if not unrecorded:
        return None
    stages = sorted({stage_of(raw) or "?" for raw in raws.values()})
    return section_note("caveat.isc_unfiltered", subjects=", ".join(unrecorded),
                        stages=", ".join(repr(s) for s in stages))
