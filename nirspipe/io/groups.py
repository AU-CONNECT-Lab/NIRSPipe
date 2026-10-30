"""The pairs CSV: which recordings make up each hyperscanning group."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nirspipe.exceptions import GroupCSVError
from nirspipe.io.derivatives import select_one_run
from nirspipe.io.naming import is_bids_label
from nirspipe.io.tables import read_table


@dataclass
class GroupEntry:
    group_id: str
    subject_id: str
    task: str
    # optional CSV columns, present when (group_id, subject_id, task) alone does not name
    # one recording: a subject recorded over several sessions, or several runs of one task
    session: str | None = None
    run: str | None = None
    # the part the member plays (teacher, student), from an optional role column
    role: str | None = None


def parse_group_csv(csv_path: Path) -> "dict[tuple[str, str, str | None], list[GroupEntry]]":
    """Parse group CSV into {(group_id, task, session): [GroupEntry, ...]}.

    ``session`` is None for a row without one, so a CSV with no session column keys every
    group as before; one group recorded in two sessions is two groups, never four members.

    An ``occasion`` column names the sitting when members carry different session labels for
    it; the key's third element is then the occasion (a blank one falls back to the row's
    session), while each entry keeps its own session to find its file::

        G1,sub-A,chat,3,1  +  G1,sub-B,chat,1,1  ->  {("G1", "chat", "1"): [A ses 3, B ses 1]}

    A ``role`` column names each member's part, on every row or on none; members keep the
    rows' order either way.
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
        if occasion is not None and not is_bids_label(occasion):
            raise GroupCSVError(
                f"occasion {occasion!r} is not a BIDS label: letters and digits only, "
                "without a 'ses-' prefix")
        key = (str(row["group_id"]).strip(), str(row["task"]).strip(), occasion or session)
        role = optional(row, "role")
        if "role" in df.columns and role is None:
            # a member without a part would match nobody's, or everybody's
            raise GroupCSVError(
                f"row for {row['subject_id']} in group {key[0]} has no role. The role "
                "column names every member's part or none: fill it in, or drop the column")
        # it becomes part of a level's name, so it has to read as one word there
        if role is not None and not is_bids_label(role):
            raise GroupCSVError(f"role {role!r} is not a label: letters and digits only")
        entry = GroupEntry(
            group_id=str(row["group_id"]).strip(),
            # one spelling downstream: files are found by the sub- label, written either way
            subject_id="sub-" + str(row["subject_id"]).strip().removeprefix("sub-"),
            task=str(row["task"]).strip(),
            session=session,
            run=optional(row, "run"),
            role=role,
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


def member_snirfs(bids_dir: Path, group: list[GroupEntry],
                  validate: bool = False) -> dict[str, Path]:
    """The one raw SNIRF each member's CSV row names, ``{subject_id: path}``.

    One lookup for everything that follows a member's recording: loading it, writing its
    aligned copy under the same stem, and naming the session its channel decisions sit in.
    Separate lookups that dropped the session could, on a two-session tree, give the aligned
    copy its name and sidecars from the other session.
    """
    from nirspipe.io.bids import get_layout, get_nirs_files

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
