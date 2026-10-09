"""Reading a hyperscanning group off disk: who is in it, and which files are theirs.

Finds and loads; does not align and does not judge. Depends on no other group module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne

from fnirs_pipe.exceptions import GroupCSVError
from fnirs_pipe.exceptions import StageError
from fnirs_pipe.io.derivatives import find_preproc_snirf
from fnirs_pipe.io.derivatives import subject_nirs_dirs
from fnirs_pipe.io.derivatives import write_sidecar_json
from fnirs_pipe.io.derivatives import select_one_run
from fnirs_pipe.io.snirf import read_snirf
from fnirs_pipe.io.tables import read_table
from fnirs_pipe.qc.boilerplate.notes import section_note
from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.lineage import lineage_of
from fnirs_pipe.utils.lineage import stage_of
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe import __version__

logger = get_logger("pipeline.group_io")


def _hyper_sidecar(path: Path, step: str, sources: list[str], **params) -> None:
    write_sidecar_json(path, {
        "pipeline_version": __version__,
        "step": step,
        "Sources": sources,
        "parameters": params,
    })


@dataclass
class GroupEntry:
    group_id: str
    subject_id: str
    task: str
    # optional CSV columns, present when (group_id, subject_id, task) alone does not name
    # one recording: a subject recorded over several sessions, or several runs of one task
    session: str | None = None
    run: str | None = None


def parse_group_csv(csv_path: Path) -> "dict[tuple[str, str, str | None], list[GroupEntry]]":
    """Parse group CSV into {(group_id, task, session): [GroupEntry, ...]}.

    ``session`` is None for a row without one, so a CSV with no session column keys every
    group as before; one group recorded in two sessions is two groups, never four members.

    An ``occasion`` column names the sitting when members carry different session labels for
    it; the key's third element is then the occasion (a blank one falls back to the row's
    session), while each entry keeps its own session to find its file::

        G1,sub-A,chat,3,1  +  G1,sub-B,chat,1,1  ->  {("G1", "chat", "1"): [A ses 3, B ses 1]}
    """
    try:
        df = read_table(csv_path, dtype=str)
    except Exception as exc:
        raise GroupCSVError(f"Cannot read CSV {csv_path}: {exc}") from exc

    missing = {"group_id", "subject_id", "task"} - set(df.columns)
    if missing:
        raise GroupCSVError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=["group_id", "subject_id", "task"])
    if df.empty:
        raise GroupCSVError("CSV contains no valid rows after dropping NaN values")

    def optional(row, column: str) -> str | None:
        """A session, occasion or run label if the CSV carries one for this row, else None."""
        if column not in df.columns:
            return None
        value = str(row[column]).strip()
        return value or None if value.lower() not in ("nan", "none", "") else None

    result: dict[tuple[str, str], list[GroupEntry]] = {}
    for _, row in df.iterrows():
        session = optional(row, "session")
        occasion = optional(row, "occasion")
        # it becomes the dyad's ses- entity, so it has to be a label a filename can carry
        if occasion is not None and not occasion.isalnum():
            raise GroupCSVError(
                f"occasion {occasion!r} is not a BIDS label: letters and digits only, "
                "without a 'ses-' prefix")
        key = (str(row["group_id"]).strip(), str(row["task"]).strip(), occasion or session)
        entry = GroupEntry(
            group_id=str(row["group_id"]).strip(),
            subject_id=str(row["subject_id"]).strip(),
            task=str(row["task"]).strip(),
            session=session,
            run=optional(row, "run"),
        )
        result.setdefault(key, []).append(entry)

    for (gid, task, ses), members in result.items():
        if len(members) < 2:
            where = f" session '{ses}'" if ses else ""
            split = sorted(s or "(none)" for g, t, s in result if (g, t) == (gid, task) and s != ses)
            hint = (f" Its other rows sit under session(s) {', '.join(split)}: if they are "
                    "one sitting whose members carry different session labels, name it in "
                    "an 'occasion' column." if split else "")
            raise GroupCSVError(
                f"Group '{gid}' task '{task}'{where} has only {len(members)} subject, "
                f"need at least 2.{hint}"
            )

    return result


def load_group_raw_bids(bids_dir: Path, group: list[GroupEntry]) -> dict[str, mne.io.Raw]:
    """Load raw CW-amplitude SNIRF from BIDS for each group member.

    Returns {subject_id: raw_intensity}.
    Raises MissingDerivativesError if no SNIRF is found for any member.
    """
    return {sid: read_snirf(path, verbose=False)
            for sid, path in member_snirfs(bids_dir, group).items()}


def member_snirfs(bids_dir: Path, group: list[GroupEntry],
                  validate: bool = False) -> dict[str, Path]:
    """The one raw SNIRF each member's CSV row names, ``{subject_id: path}``.

    One lookup for everything that follows a member's recording: loading it, writing its
    aligned copy under the same stem, and naming the session its channel decisions sit in.
    Separate lookups that dropped the session could, on a two-session tree, give the aligned
    copy its name and sidecars from the other session.
    """
    from fnirs_pipe.io.bids import get_layout, get_nirs_files

    layout = get_layout(bids_dir, validate=validate)
    out: dict[str, Path] = {}
    for entry in group:
        sub_label = entry.subject_id.removeprefix("sub-")
        files = get_nirs_files(layout, subject=sub_label, session=entry.session,
                               task=entry.task)
        out[entry.subject_id] = Path(select_one_run(
            sorted(files), what="raw SNIRF in BIDS", subject_id=entry.subject_id,
            task=entry.task, session=entry.session, run=entry.run,
        ))
    return out


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
    through ``fnirs-pipe`` simply contributes nothing here, which the caller draws as a
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


def _raw_to_haemo(raw: mne.io.Raw, dpf: list[float]) -> mne.io.Raw:
    raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
    # Single value applies to both wavelengths; a list gives one PPF per wavelength.
    ppf = dpf[0] if len(dpf) == 1 else dpf
    return mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=ppf)


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

    Reads the same folders :func:`~fnirs_pipe.io.derivatives.find_preproc_snirf` reads the
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
