"""Subject-level landing page: one row per run, linking to that run's QC report.

Every QC figure is per run, so the subject report is an index rather than a report. The
rows are read back off disk from the SQM records the run already wrote, so this needs
nothing held in memory and can be rebuilt for a past output tree.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from fnirs_pipe.qc.boilerplate import collect_software_versions
from fnirs_pipe.qc.sqm_record import entities_of
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.subject_index")
_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Column -> (heading, "section_metric" in the SQM record, format spec). The five that say
# whether a run is usable at a glance; everything else stays in the run's own report.
_COLUMNS = (
    ("Channels kept", "raw_channel_retention_rate", "{:.0%}"),
    ("SCI mean",      "raw_sci_mean",               "{:.2f}"),
    ("GVTD p95",      "raw_gvtd_p95",               "{:.2e}"),
    ("Motion corr.",  "motion_motion_corrected_pct", "{:.1f}%"),
    ("HbO-HbR corr",  "preproc_hbo_hbr_corr_mean",  "{:+.2f}"),
)


def _flat(record: dict) -> dict[str, float]:
    """Flatten the sectioned record to ``section_metric`` keys, as the group TSV names them."""
    out: dict[str, float] = {}
    for section, body in record.items():
        if isinstance(body, dict):
            out.update({f"{section}_{k}": v for k, v in body.items()
                        if isinstance(v, (int, float))})
    return out


def _shape(nirs_dir: Path, label: str) -> dict:
    """Channel count, rate and length, from whichever stage sidecar of this run has them."""
    for desc in ("preproc", "od"):
        path = nirs_dir / f"{label}_desc-{desc}_nirs.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")).get("data") or {}
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("sfreq"):
            return data
    return {}


def collect_runs(sub_dir: Path) -> list[dict]:
    """One row per run under sub_dir, newest BIDS entity order, for the index table."""
    nirs_dir = sub_dir / "nirs"
    rows: list[dict] = []
    for sqm_path in sorted(nirs_dir.glob("*_desc-sqm_nirs.json")):
        label = sqm_path.name[: sqm_path.name.index("_desc-sqm")]
        try:
            record = json.loads(sqm_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skip %s: %s", sqm_path.name, exc)
            continue
        flat = _flat(record)
        shape = _shape(nirs_dir, label)
        report = sub_dir / f"{label}_qc.html"
        rows.append({
            "label": label,
            "entities": {k: v for k, v in entities_of(label).items() if v},
            "href": report.name if report.exists() else None,
            "n_channels": shape.get("n_channels"),
            "n_bad": shape.get("n_bad"),
            "duration_s": shape.get("duration_s"),
            "sfreq": shape.get("sfreq"),
            "metrics": [
                (head, fmt.format(flat[key]) if key in flat else "n/a")
                for head, key, fmt in _COLUMNS
            ],
        })
    return rows


def write_subject_index(
    subject: str,
    sub_dir: Path,
    run_command: str,
    mode: str | None = None,
) -> Path | None:
    """Render sub-<subject>_qc.html as the index over the subject's per-run reports."""
    rows = collect_runs(sub_dir)
    if not rows:
        logger.warning("sub-%s | no SQM records found, index not written", subject)
        return None

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    html = env.get_template("subject_index.html.j2").render(
        subject=subject,
        rows=rows,
        columns=[head for head, _, _ in _COLUMNS],
        run_command=run_command,
        mode=mode or "",
        run_date=date.today().isoformat(),
        versions=collect_software_versions(),
    )
    out_path = sub_dir / f"sub-{subject}_qc.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("sub-%s | run index saved: %s", subject, out_path)
    return out_path
