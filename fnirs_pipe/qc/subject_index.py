"""Subject-level landing page: one row per run, linking to that run's QC report.

Every QC figure is per run, so the subject report is an index rather than a report. What
it adds on top of the links is the three things no single run can show: which run stands
apart from the subject's others on each metric, which channels were rejected in which run
(the set ``--bads-scope subject`` unions), and, for a run driven with ``--by-condition``,
its conditions side by side.

The rows are read back off disk from the records the run already wrote, so this needs
nothing held in memory and can be rebuilt for a past output tree.

The Conditions section covers per-condition *views* only, the ``by_condition`` block a
whole-run pass leaves in its record. A tree cropped per condition first is already listed
above, one run per condition, and belongs there rather than here: a cropped condition is
filtered against its own two edges and lands on its own window grid, so its numbers are not
comparable with a view's and the two must not share a table. See
:mod:`fnirs_pipe.qc.condition_views`.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from fnirs_pipe.qc.boilerplate import collect_software_versions
from fnirs_pipe.qc.report_shell import (
    OUTLIER_Z, footer_vars, guard, outlier_flags, page_vars, render)
from fnirs_pipe.qc.sqm_record import entities_of
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.subject_index")

# Column -> (heading, "section_metric" keys in preference order, format spec). The five that
# say whether a run is usable at a glance; everything else stays in the run's own report.
#
# Long channels first, all channels as the fallback, which is the same preference the run's
# own report and the hyperscanning tables use. Without it this table read `raw_*` while the
# report beside it read `raw_long_*`, so one label named two different numbers. The last two
# `Motion corr.` has no long-channel form: the correction footprint counts what the
# correction touched, over every channel.
_COLUMNS = (
    ("Channels kept", ("raw_long_channel_retention_rate", "raw_channel_retention_rate"), "{:.0%}"),
    ("SCI mean",      ("raw_long_sci_mean", "raw_sci_mean"),                             "{:.2f}"),
    ("GVTD p95",      ("raw_long_gvtd_p95", "raw_gvtd_p95"),                             "{:.2e}"),
    # a fraction of the recording, not a percentage: `motion_corrected_pct` is the mean of
    # a per-sample boolean. Printed with `{:.1f}%` this column said 0.2% where the run's own
    # report, which formats it through the metric registry's "pct", said 17.4%
    ("Motion corr.",  ("motion_motion_corrected_pct",),                                  "{:.1%}"),
    ("HbO-HbR corr",  ("preproc_long_hbo_hbr_corr_mean", "preproc_hbo_hbr_corr_mean"),   "{:+.2f}"),
)


# Other products of a run, as (link text, path relative to sub_dir). Only the ones on disk
# reach the page; the run report links itself and is not repeated here.
_ARTEFACTS = (
    ("MNE",        "{label}_qc_mne.html"),
    ("provenance", "figures/{label}/provenance.png"),
    ("channels",   "nirs/{label}_channel_metrics.csv"),
    ("aux",        "nirs/{label}_desc-aux_timeseries.tsv.gz"),
)


# ---- Conditions ----

# Column -> (heading, key, format spec). The keys are read from a condition's long-channel
# block where it has one, so a row sits under the same heading as the run's `raw_long`
# figures above. `Spike frames` and `Motion corr.` have no per-set block and come off the
# span lists, which are the long set and every channel respectively; each column is one
# measurement down its length either way, which is what the whole-run row is built to keep.
_COND_COLUMNS = (
    ("SCI (10 s)",      "sci_win_mean",            "{:.3f}"),
    ("SNR",             "snr_mean",                "{:.0f}"),
    ("GVTD mean",       "gvtd_mean",               "{:.2e}"),
    ("GVTD above thr.", "gvtd_pct_above_thresh",   "{:.1%}"),
    ("Spike frames",    "spike_pct_frames",        "{:.1%}"),
    ("Motion corr.",    "motion_corrected_pct",    "{:.1%}"),
    ("Retention",       "channel_retention_rate",  "{:.0%}"),
)

# The span list behind each share, and the key it fills. The whole-run row counts these over
# the full recording rather than reading the record's own scalar of the same name: a
# condition's `gvtd_pct_above_thresh` is the corrected file against its own threshold, while
# `raw_long_gvtd_pct_above_thresh` is the uncorrected file against its own, and on
# sub-p1d01 those are 4.0% and 52.4% of one recording. Same rule, same spans, one column.
_WHOLE_RUN_SPANS = (
    ("gvtd_pct_above_thresh", "gvtd_above_spans_s"),
    ("spike_pct_frames",      "spike_spans_s"),
    ("motion_corrected_pct",  "motion_corrected_spans_s"),
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
        flags = outlier_flags([row["metrics"][column]["raw"] for row in rows])
        for row, flagged in zip(rows, flags):
            row["metrics"][column]["flagged"] = flagged
    return rows


def _cond_row(name: str, kind: str, href: "str | None", span: str,
              kept: "tuple[int, int] | None", values: dict) -> dict:
    """One Conditions row, the values formatted by :data:`_COND_COLUMNS`."""
    return {
        "name": name,
        "kind": kind,
        "href": href,
        "span": span,
        "kept": f"{kept[0]}/{kept[1]}" if kept else None,
        "metrics": [
            {"value": fmt.format(values[key]), "raw": values[key], "flagged": False}
            if isinstance(values.get(key), (int, float))
            else {"value": "n/a", "raw": None, "flagged": False}
            for _head, key, fmt in _COND_COLUMNS
        ],
    }


def _whole_run_values(record: dict, duration_s: "float | None") -> dict:
    """The Conditions table's whole-run row, measured the way its condition rows are."""
    from fnirs_pipe.qc.condition_views import span_share

    long_section = record.get("raw_long") or record.get("raw") or {}
    windowed = record.get("windowed") or {}
    values = {key: long_section.get(key)
              for key in ("sci_win_mean", "snr_mean", "channel_retention_rate")}

    per_window = windowed.get("gvtd_per_window")
    values["gvtd_mean"] = statistics.fmean(per_window) if per_window else None
    for key, stored in _WHOLE_RUN_SPANS:
        values[key] = (span_share(windowed.get(stored), 0.0, duration_s)
                       if windowed.get(stored) and duration_s else None)
    return values


def collect_conditions(sub_dir: Path) -> list[dict]:
    """One group per run carrying per-condition views, each a whole-run row and its windows.

    A group is ``{"label", "task", "windows", "by_condition", "record", "rows"}``; ``rows``
    open with the run itself so a condition is read against the whole it was cut from. Runs
    without a ``by_condition`` block contribute nothing, which is what a tree produced
    without ``--by-condition``, or cropped per condition first, looks like.
    """
    nirs_dir = sub_dir / "nirs"
    groups: list[dict] = []
    for sqm_path in sorted(nirs_dir.glob("*_desc-sqm_nirs.json")):
        label = sqm_path.name[: sqm_path.name.index("_desc-sqm")]
        try:
            record = json.loads(sqm_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        by_condition = record.get("by_condition") or {}
        if len(by_condition) < 2:
            continue

        shape = _shape(nirs_dir, label)
        n_channels, duration_s = shape.get("n_channels"), shape.get("duration_s")
        report = sub_dir / f"{label}_qc.html"
        rows = [_cond_row(
            "whole run", "run", report.name if report.exists() else None,
            f"0–{duration_s:.0f} s" if duration_s else "",
            (n_channels - (shape.get("n_bad") or 0), n_channels) if n_channels else None,
            _whole_run_values(record, duration_s),
        )]

        windows: list[tuple[str, float, float]] = []
        for name, block in by_condition.items():
            window = block.get("window_s") or []
            t0, t1 = (float(window[0]), float(window[1])) if len(window) == 2 else (0.0, 0.0)
            windows.append((name, t0, t1))
            page = sub_dir / f"{label}_desc-{name}_qc.html"
            values = {**((block.get("od_by_set") or {}).get("long") or {}),
                      **((block.get("motion_by_set") or {}).get("long") or {})}
            # neither block carries these two, the record splitting only what it measured
            # per channel set; both come off span lists the whole-run row counts as well
            for key in ("spike_pct_frames", "motion_corrected_pct"):
                values[key] = (block.get("scalars") or {}).get(key)
            rows.append(_cond_row(
                name, "view", page.name if page.exists() else None,
                f"{t0:.0f}–{t1:.0f} s",
                (n_channels - len(block.get("bad_channels") or []), n_channels)
                if n_channels else None,
                values,
            ))

        # a condition is marked against the run's other conditions, the whole-run row
        # included: it is the set they are cut from and belongs in the comparison
        for column in range(len(_COND_COLUMNS)):
            flags = outlier_flags([r["metrics"][column]["raw"] for r in rows])
            for row, flagged in zip(rows, flags):
                row["metrics"][column]["flagged"] = flagged

        groups.append({"label": label, "record": record, "rows": rows,
                       "windows": windows, "by_condition": by_condition,
                       "task": entities_of(label).get("task") or label})
    return groups


def write_condition_figures(sub_dir: Path, subject: str, groups: list[dict]) -> dict:
    """Write the Conditions figures and return what the template needs to embed them.

    Three pictures, all of them builders the group report already owns: the profile over
    every run at once, and then per run a channel-by-condition matrix and a timeline. The
    profile takes the runs together because that is the comparison it is for; the other two
    are of one run's channels and one run's clock and cannot be pooled.
    """
    from fnirs_pipe.qc.figure_io import _save_figure_html
    from fnirs_pipe.qc.figures.group_figures import (
        build_channel_condition_matrix, build_condition_panels, build_condition_timeline,
    )
    from fnirs_pipe.qc.group_writer import _sqm_row

    fig_dir = sub_dir / "figures" / f"sub-{subject}"
    out: dict = {"profile": None, "per_run": []}

    def _save(stem: str, name: str, fig) -> "dict | None":
        if fig is None:
            return None
        fname = f"{stem}_desc-{name}_nirs.html"
        height = _save_figure_html(fig, fig_dir / fname)
        return {"src": f"figures/sub-{subject}/{fname}", "h": height,
                "w": getattr(fig.layout, "width", None)}

    panels = build_condition_panels(
        [{"bids_name": g["label"], "by_condition": g["by_condition"]} for g in groups])
    out["profile"] = _save(f"sub-{subject}", "condprofile", panels)

    for group in groups:
        label = group["label"]
        out["per_run"].append({
            "label": label,
            "task": group["task"],
            "channels": _save(label, "condchannels",
                              build_channel_condition_matrix(group["by_condition"])),
            "timeline": _save(label, "condtimeline",
                              build_condition_timeline(_sqm_row(label, group["record"]),
                                                       group["windows"])),
        })
    return out


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

    errors: list[str] = []
    condition_groups = collect_conditions(sub_dir)
    figures: dict = {"profile": None, "per_run": []}
    if condition_groups:
        with guard("Condition figures", errors, f"sub-{subject}"):
            figures = write_condition_figures(sub_dir, subject, condition_groups)

    html = render(
        "subject_index.html.j2",
        # the subject names the page, the way a run names its own; that this is QC is what
        # the reader opened
        **page_vars(
            title=f"sub-{subject}",
            heading=f"sub-{subject}",
        ),
        # the errors block only when a section actually failed: this page has never carried
        # one, and an empty "no errors" panel is chrome it does not need
        **footer_vars(versions=collect_software_versions(),
                      errors=errors or None, scope=f"sub-{subject}"),
        subject=subject,
        rows=rows,
        channels=collect_bad_channels(sub_dir, [r["label"] for r in rows]),
        columns=[head for head, _, _ in _COLUMNS],
        condition_groups=condition_groups,
        condition_columns=[head for head, _, _ in _COND_COLUMNS],
        condition_figures=figures,
        outlier_z=OUTLIER_Z,
        run_command=run_command,
        mode=mode or "",
    )
    out_path = sub_dir / f"sub-{subject}_qc.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("sub-%s | run index saved: %s", subject, out_path)
    return out_path
