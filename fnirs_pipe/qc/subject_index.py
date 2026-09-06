"""Subject-level landing page: one row per run, linking to that run's QC report.

Every QC figure is per run, so the subject report is an index rather than a report. What
it adds on top of the links is the two things no single run can show: which run stands
apart from the subject's others on each metric, and which channels were rejected in which
run, which is the set ``--bads-scope subject`` unions.

The rows are read back off disk from the records the run already wrote, so this needs
nothing held in memory and can be rebuilt for a past output tree.
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from fnirs_pipe.qc.boilerplate import collect_software_versions
from fnirs_pipe.qc.sqm_record import entities_of
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.subject_index")
_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Column -> (heading, "section_metric" keys in preference order, format spec). The five that
# say whether a run is usable at a glance; everything else stays in the run's own report.
#
# Long channels first, all channels as the fallback, which is the same preference the run's
# own report and the hyperscanning tables use. Without it this table read `raw_*` while the
# report beside it read `raw_long_*`, so one label named two different numbers. The last two
# have no long-channel form: those sections are not split by separation.
_COLUMNS = (
    ("Channels kept", ("raw_long_channel_retention_rate", "raw_channel_retention_rate"), "{:.0%}"),
    ("SCI mean",      ("raw_long_sci_mean", "raw_sci_mean"),                             "{:.2f}"),
    ("GVTD p95",      ("raw_long_gvtd_p95", "raw_gvtd_p95"),                             "{:.2e}"),
    ("Motion corr.",  ("motion_motion_corrected_pct",),                                  "{:.1f}%"),
    ("HbO-HbR corr",  ("preproc_hbo_hbr_corr_mean",),                                    "{:+.2f}"),
)


# Other products of a run, as (link text, path relative to sub_dir). Only the ones on disk
# reach the page; the run report links itself and is not repeated here.
_ARTEFACTS = (
    ("MNE",        "{label}_qc_mne.html"),
    ("provenance", "figures/{label}/provenance.png"),
    ("channels",   "nirs/{label}_channel_metrics.csv"),
    ("aux",        "nirs/{label}_desc-aux_timeseries.tsv.gz"),
)

# How far from the subject's own median a run has to sit before the cell is marked.
_OUTLIER_Z = 3.5


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


def _outlier_flags(values: list[float | None]) -> list[bool]:
    """Which runs sit apart from the subject's own runs on one metric.

    Scaled by the median absolute deviation, so the run being looked for cannot widen the
    scale that is meant to catch it. Under four runs there is nothing to compare against.

    [0.96, 0.95, 0.96, 0.40, 0.97] -> [False, False, False, True, False]

    A column whose runs agree exactly has a zero MAD, which would divide by nothing. The
    mean deviation takes over there: it is zero only when every run agrees, and otherwise
    still marks the single run standing away from a set that agrees with itself.
    """
    present = [v for v in values if v is not None]
    if len(present) < 4:
        return [False] * len(values)

    median = statistics.median(present)
    deviations = [abs(v - median) for v in present]
    scale = statistics.median(deviations) * 1.4826
    if scale == 0:
        scale = sum(deviations) / len(deviations)
    if scale == 0:
        return [False] * len(values)
    return [v is not None and abs(v - median) / scale >= _OUTLIER_Z for v in values]


def _links(sub_dir: Path, label: str) -> list[dict[str, str]]:
    return [{"text": text, "href": rel}
            for text, template in _ARTEFACTS
            if (sub_dir / (rel := template.format(label=label))).exists()]


def collect_bad_channels(sub_dir: Path, labels: list[str]) -> dict:
    """Which source-detector pair each run rejected, over all of the subject's runs.

    Read from each run's ``_channel_metrics.csv``, which carries one ``is_bad`` per channel.
    The two wavelengths of a pair are collapsed into the pair: rejecting one rejects the
    optode, and a grid of 44 rows says nothing 22 rows do not.

    Returns ``{"rejected": [{"pair", "bad_in", "n_bad"}], "clean", "n_pairs", "n_clean"}``.
    Only the rejected pairs get a row, ordered by how many runs rejected them and then by
    the montage order the CSV is written in: a grid whose every cell says "kept" is a
    sentence, not a table, and the pairs that survived are named in one. An empty dict when
    no run wrote the file, which is what a tree from before it existed looks like.
    """
    order: list[str] = []
    bad_by_label: dict[str, set[str]] = {}

    for label in labels:
        path = sub_dir / "nirs" / f"{label}_channel_metrics.csv"
        if not path.exists():
            continue
        bad: set[str] = set()
        with path.open(encoding="utf-8", newline="") as fh:
            for entry in csv.DictReader(fh):
                pair = (entry.get("name") or "").rsplit(" ", 1)[0]
                if not pair:
                    continue
                if pair not in order:
                    order.append(pair)
                if (entry.get("is_bad") or "").strip().lower() == "true":
                    bad.add(pair)
        bad_by_label[label] = bad

    if not bad_by_label:
        return {}

    rejected = []
    clean = []
    for index, pair in enumerate(order):
        bad_in = [pair in bad_by_label.get(label, set()) for label in labels]
        if any(bad_in):
            rejected.append({"pair": pair, "bad_in": bad_in,
                             "n_bad": sum(bad_in), "_order": index})
        else:
            clean.append(pair)
    rejected.sort(key=lambda r: (-r["n_bad"], r["_order"]))
    for row in rejected:
        del row["_order"]

    return {
        "rejected": rejected,
        "clean": clean,
        "n_pairs": len(order),
        "n_clean": len(clean),
    }


def _metric_cell(flat: dict, keys: tuple, fmt: str) -> dict:
    """One table cell, taking the first key the record actually carries."""
    for key in keys:
        if flat.get(key) is not None:
            return {"value": fmt.format(flat[key]), "raw": flat[key], "flagged": False}
    return {"value": "n/a", "raw": None, "flagged": False}


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
            "links": _links(sub_dir, label),
            "metrics": [
                _metric_cell(flat, keys, fmt) for _, keys, fmt in _COLUMNS
            ],
        })

    # a run is marked against the subject's other runs, so the comparison can only be made
    # once every row is in hand
    for column in range(len(_COLUMNS)):
        flags = _outlier_flags([row["metrics"][column]["raw"] for row in rows])
        for row, flagged in zip(rows, flags):
            row["metrics"][column]["flagged"] = flagged
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
        channels=collect_bad_channels(sub_dir, [r["label"] for r in rows]),
        columns=[head for head, _, _ in _COLUMNS],
        outlier_z=_OUTLIER_Z,
        run_command=run_command,
        mode=mode or "",
        run_date=date.today().isoformat(),
        versions=collect_software_versions(),
    )
    out_path = sub_dir / f"sub-{subject}_qc.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("sub-%s | run index saved: %s", subject, out_path)
    return out_path
