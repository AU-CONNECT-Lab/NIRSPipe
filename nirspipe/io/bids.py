"""BIDS dataset querying and participants.tsv helpers."""

import json
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from bids import BIDSLayout

from nirspipe.utils.logging import get_logger

logger = get_logger("io.bids")

_VALIDATOR_INSTALL = "deno install -A -g -n bids-validator jsr:@bids/validator"


def get_layout(bids_dir: Path, validate: bool = True) -> BIDSLayout:
    """Return a pybids BIDSLayout for the dataset."""
    return BIDSLayout(str(bids_dir), validate=validate)


def _validator_command() -> "list[str] | None":
    """The installed BIDS validator, as an argv prefix, or None."""
    exe = shutil.which("bids-validator")
    if exe:
        return [exe]
    deno = shutil.which("deno")
    if deno:
        return [deno, "run", "-A", "jsr:@bids/validator"]
    return None


def _validator_errors(report: dict) -> "dict[str, list[str]]":
    """``{error code: [locations]}`` from a validator's JSON report, either report format.

    The schema validator lists every issue with a severity; the older one keeps errors in
    their own list, each carrying the files it was found in.
    """
    issues = report.get("issues") or {}
    errors: dict[str, list[str]] = defaultdict(list)
    for issue in issues.get("issues") or []:
        if issue.get("severity") == "error":
            errors[issue.get("code", "?")].append(issue.get("location") or "")
    for issue in issues.get("errors") or []:
        for f in issue.get("files") or [{}]:
            path = ((f or {}).get("file") or {}).get("relativePath", "")
            errors[issue.get("key", "?")].append(path)
    return dict(errors)


def validate_bids(bids_dir: Path) -> None:
    """Check the input dataset with the BIDS validator, exiting non-zero on any error.

    Carries on with a warning when no validator is installed, so a machine without one can
    still run; ``--skip-bids-validation`` is how a user proceeds past a failing dataset.
    """
    cmd = _validator_command()
    if cmd is None:
        logger.warning("bids-validator not found, so the input was not validated; install it "
                       "with `%s`, or pass --skip-bids-validation to silence this",
                       _VALIDATOR_INSTALL)
        return

    logger.info("validating %s with bids-validator", bids_dir)
    proc = subprocess.run([*cmd, str(bids_dir), "--json"], capture_output=True, text=True)
    try:
        errors = _validator_errors(json.loads(proc.stdout))
    except (json.JSONDecodeError, AttributeError):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise SystemExit("\n  ".join(
            [f"Error: bids-validator did not produce a report (exit {proc.returncode}):", *tail])
            + "\nPass --skip-bids-validation to run without it.")
    if not errors:
        return

    lines = [f"Error: {bids_dir} is not valid BIDS "
             f"({sum(map(len, errors.values()))} errors from bids-validator):"]
    for code, where in sorted(errors.items()):
        shown = ", ".join(w for w in where[:3] if w)
        more = f" and {len(where) - 3} more" if len(where) > 3 else ""
        lines.append(f"  {code} x{len(where)}: {shown}{more}")
    lines.append("Fix the dataset, or pass --skip-bids-validation to run on it anyway.")
    raise SystemExit("\n".join(lines))


def write_bids_from_snirf(
    input_file: Path,
    bids_dir: Path,
    subject: str,
    task: str,
    session: str | None = None,
    run: str | None = None,
    overwrite: bool = False,
    optode_frame: str = "unknown",
) -> None:
    """Convert one raw snirf file into a BIDS nirs entry.

    `optode_frame` names the space the SNIRF's optode coordinates live in. SNIRF does not
    record it, so MNE reads them as "unknown" and mne-bids then writes no `_optodes.tsv` or
    `_coordsystem.json`, both of which BIDS requires: it will not guess a frame on the
    author's behalf. Pass "head" when the positions were digitised against the nasion and
    the two preauricular points, "mri" when they are in a subject's MRI space.
    """
    import mne
    import mne_bids

    raw = mne.io.read_raw_snirf(str(input_file), optode_frame=optode_frame, preload=False)
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        task=task,
        session=session,
        run=run,
        root=bids_dir,
        datatype="nirs",
        suffix="nirs",
        extension=".snirf",
    )
    mne_bids.write_raw_bids(raw, bids_path=bids_path, overwrite=overwrite, format="auto")


def get_participant_age(
    layout: BIDSLayout,
    subject: str,
    fallback: float | None = None,
) -> float:
    """Resolve participant age from participants.tsv; use fallback if missing.

    Raises ValueError if age is not found and no fallback is provided.
    """
    from nirspipe.io.tables import read_table

    tsv_path = Path(layout.root) / "participants.tsv"
    if tsv_path.exists():
        df = read_table(tsv_path, dtype={"participant_id": str})
        # with or without the prefix, as the --bad-channels table is read
        wanted = subject.removeprefix("sub-")
        row = df[df["participant_id"].str.removeprefix("sub-") == wanted]
        if not row.empty and "age" in df.columns:
            return float(row["age"].iloc[0])

    if fallback is not None:
        return float(fallback)

    raise ValueError(
        f"Age not found for sub-{subject} in participants.tsv and no --age fallback given."
    )


def get_nirs_files(
    layout: BIDSLayout,
    subject: str,
    session: str | None = None,
    task: str | None = None,
    filter_file: Path | None = None,
) -> list[Path]:
    """Return paths to all snirf files for a given subject/session/task.

    task: if None, all tasks are returned.
    filter_file: optional JSON with extra pybids query kwargs, e.g. {"run": "01"}.
    """
    kwargs: dict = {"subject": subject, "extension": ".snirf"}

    if session is not None:
        kwargs["session"] = session

    if task is not None:
        kwargs["task"] = task

    if filter_file is not None:
        extra = json.loads(filter_file.read_text())
        kwargs.update(extra)

    return [Path(f.path) for f in layout.get(**kwargs)]


def bids_label(subject: str, entities: dict) -> str:
    """Build a BIDS filename stem (sub-.._ses-.._task-.._run-..) from parsed entities."""
    parts = [f"sub-{subject}"]
    for key, prefix in (("session", "ses"), ("task", "task"), ("run", "run")):
        val = entities.get(key)
        if val:
            parts.append(f"{prefix}-{val}")
    return "_".join(parts)


def iter_run_files(
    layout: BIDSLayout,
    subjects: list[str],
    sessions: list[str | None],
    task: str | None,
) -> Iterator[tuple[Path, str]]:
    """Yield (snirf_path, bids_label) for each subject × session run matching task."""
    for sub in subjects:
        for ses in sessions:
            for f in get_nirs_files(layout, subject=sub, session=ses, task=task):
                entities = layout.parse_file_entities(str(f))
                yield f, bids_label(sub, entities)
