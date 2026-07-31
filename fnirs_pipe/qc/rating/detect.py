import re
from pathlib import Path


def detect_sessions(output_dir: Path, subject: str) -> list[str | None]:
    """Return sorted session list from desc-preproc snirf filenames, or [None]."""
    nirs_dir = output_dir / f"sub-{subject}" / "nirs"
    if not nirs_dir.exists():
        return [None]
    sessions: set[str | None] = set()
    for f in nirs_dir.glob("*.snirf"):
        if "desc-preproc" not in f.name:
            continue
        m = re.search(r"_ses-([^_]+)_", f.name)
        sessions.add(m.group(1) if m else None)
    non_none = sorted(s for s in sessions if s is not None)
    return non_none if non_none else [None]


def detect_glm(output_dir: Path, subject: str) -> bool:
    nirs_dir = output_dir / f"sub-{subject}" / "nirs"
    return nirs_dir.exists() and any(nirs_dir.glob("*glm_results.csv"))
