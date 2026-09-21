"""Generate hyperscanning group-level raw QC HTML report."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import mne
import numpy as np

from fnirs_pipe.io.derivatives import (
    group_data_dir, group_report_dir, subject_report_dir,
)
from fnirs_pipe.pipeline.hyper.hyper_post import HyperPostResult
from fnirs_pipe.pipeline.hyper import (
    GroupEntry, alignment_params, unfiltered_stage_note,
)
from fnirs_pipe.qc.common.channel_table import (
    channel_columns, channel_rows, format_rows, pair_rows,
)
from fnirs_pipe.qc.boilerplate import collect_software_versions
from fnirs_pipe.qc.boilerplate.vocabulary import (
    MISSING_VALUE, format_metric, is_key_metric, metric_class, metric_label,
    metric_summary,
)
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.qc.common.figure_io import (
    _fig_href,
    _pair_fname,
    pair_slug,
    _save_figure_html,
    save_png,
)
from fnirs_pipe.qc.figures.hyper.hyper_figures import _cond_colors
from fnirs_pipe.qc.hyper.hyper_raw_writer import _process_hyper_raw_group
from fnirs_pipe.qc.common.report_shell import (
    footer_vars,
    guard,
    note,
    page_vars,
    render,
)
from fnirs_pipe.qc.common.windows import crop_provenance, markers_on_data_axis
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.hyper_report")


# ---- Per-subject quality metrics ----
#
# Which scalars the table lists, and in what order. Keys only: the label, the format, the
# better direction and the colour all come from the metric registry, the same one the
# individual reports read, so a metric cannot print to three decimals in a subject report
# and four here. Absent keys render as a dash, so one list serves the raw path (intensity
# metrics only) and the post path (which adds motion and haemoglobin).
_SUBJECT_METRICS = [
    "sci_win_mean",
    "sci_mean",
    # the only one of these that screens; the rest are measured and reported
    "good_frac_mean",
    "psp_mean",
    "cv_mean",
    "snr_mean",
    "gvtd_filt_p95",
    "motion_corrected_pct",
    "spike_pct_frames",
    "hbo_hbr_corr_mean",
    "channel_retention_rate",
]

# What one WTC result yields, keyed to the type of each. Declared once because two places
# read it: `_figure_set` fills a fresh one, and a page renderer falls back to an empty value
# for a window whose figures failed. A key added to only one of the two would leave a panel
# on a condition page showing the whole run's picture with nothing saying so.
#
# Types rather than values, and `_empty_figures` calls them. Holding the empty containers
# themselves would share one dict between every call, so each figure set would write into
# its predecessor's and every page would end up pointing at the last window's figures.
_FIGURE_SET: dict = {"per_channel": dict, "per_roi": dict}


def _empty_figures() -> dict:
    """A figure set with every key present and its own empty value."""
    return {key: factory() for key, factory in _FIGURE_SET.items()}

_CHROMA_LABEL = {"hbo": "HbO", "hbr": "HbR"}


def _metric_class(key: str, value: float, sci_threshold: float) -> str:
    """The registry's verdict, except for SCI, which is judged against the run's own line.

    The exception is the same one the per-channel tables make: this run screened at
    ``--sci-threshold``, so colouring its SCI against the registry's cutoff would show a
    verdict the run did not reach. Both estimates take it, the windowed one included, or the
    only coloured SCI on the page is the whole-run one nobody reads. Everything else is the
    registry's, and a metric with no published cutoff there prints uncoloured on purpose.
    """
    if key in ("sci_mean", "sci_win_mean"):
        return "qm-ok" if value >= sci_threshold else "qm-bad"
    return metric_class(key, value)


def subject_metric_rows(
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    sci_threshold: float,
) -> list[dict]:
    """One entry per metric the members carry a value for, with a cell per member.

    ::

      -> [{"key": "sci_win_mean", "label": "SCI (10 s windows)", "summary": "...",
           "key_metric": True, "cells": [{"text": "0.91", "cls": "qm-ok"}, ...]}]

    Metric-major, which is not how the page prints it: :func:`subject_metric_tables` turns
    it a quarter turn so a metric is a column, the arrangement the subject report's own
    metrics section uses. Kept this way round here because the registry is read per metric.

    No direction arrow. The subject report does not carry one either: which end is better is
    the registry's and reaches the reader through the header's hover text, so a table cannot
    say one thing and a tooltip another.
    """
    rows = []
    for key in _SUBJECT_METRICS:
        cells = []
        for sid in subject_ids:
            value = (sqm_data.get(sid) or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append({"text": format_metric(key, value),
                              "cls": _metric_class(key, value, sci_threshold)})
            else:
                cells.append({"text": MISSING_VALUE, "cls": ""})
        if any(c["text"] != MISSING_VALUE for c in cells):
            rows.append({"key": key,
                         "label": metric_label(key),
                         "summary": metric_summary(key),
                         "key_metric": is_key_metric(key),
                         "cells": cells})
    return rows


# how far two members' copies of one trigger may sit apart and still be the same condition
_TRIGGER_JITTER_SAMPLES = 2.0

# The channel sets a quality table is printed over, and what each is headed on the page.
# Same order and same names the subject report's own metrics section uses.
_CHANNEL_SETS = (("all", "All"), ("long", "Long"), ("short", "Short"))


def subject_metric_tables(
    by_set: "dict[str, dict[str, dict]]",
    subject_ids: list[str],
    sci_threshold: float,
) -> list[dict]:
    """One quality table per channel set.

    ::

      -> [{"set": "Long",
           "columns": [{"key", "label", "summary", "key_metric"}, ...],
           "rows": [{"member": "sub-01", "cells": [{"text", "cls"}, ...]}, ...]}]

    ``by_set`` is ``{set_name: {subject_id: scalars}}``.

    Three tables rather than three columns of one, because a set is a separate measurement
    and not a grouping of the same one: GVTD is an RMS across the channels of its set, and
    a long and a short retention rate are fractions of different montages. This is the
    arrangement the subject report's own metrics section makes, so a reader moving between
    a subject page and a dyad page reads one shape.

    A montage with no short channel measures one set, and ``all`` and ``long`` are then the
    same numbers printed twice; it gets a single unheaded table instead. That is also what
    a caller with no split at all passes, under ``all``.
    """
    def _table(heading: str, data: dict) -> "dict | None":
        metrics = subject_metric_rows(data, subject_ids, sci_threshold)
        if not metrics:
            return None
        # a quarter turn: metric-major in, column-major out. One row per member is what the
        # subject report's metrics section does with its channel sets, and the two pages are
        # read against each other
        return {
            "set": heading,
            "columns": [{k: m[k] for k in ("key", "label", "summary", "key_metric")}
                        for m in metrics],
            "rows": [{"member": sid, "cells": [m["cells"][i] for m in metrics]}
                     for i, sid in enumerate(subject_ids)],
        }

    short = by_set.get("short") or {}
    if not any(short.values()):
        one = _table("", by_set.get("all") or by_set.get("long") or {})
        return [one] if one else []
    return [t for t in (_table(heading, by_set.get(key) or {})
                        for key, heading in _CHANNEL_SETS) if t]


def _record_window_matches(
    window_s, t0: float, t1: float, offset: float, tol: float, sid: str, label: str,
) -> bool:
    """Whether a record's stored window is the one hyper resolved, on one clock.

    ::

      record [3602.4, 3902.4], hyper [3580.0, 3880.0], offset 22.4  ->  True

    A record's ``window_s`` is on that member's own recording; a hyper window is on the
    clock ``align_recordings`` cropped every member onto, and the two differ by that
    member's crop offset. Matching on the label alone would pair ``talk#2`` with a
    different occurrence whenever the crop dropped an earlier one, so the bounds are
    checked rather than assumed.

    The start has to agree: it is an annotation onset and nothing downstream moves it. The
    end is allowed to run past hyper's, because ``trim_to_shortest`` clips every member to
    the shortest of them and a final condition therefore ends early on the aligned clock.
    """
    if not window_s or len(window_s) != 2:
        return False
    r0, r1 = float(window_s[0]) - offset, float(window_s[1]) - offset
    if abs(r0 - t0) > tol or r1 < t1 - tol:
        logger.warning(
            "%s condition %s: the record's window [%.3f, %.3f] is not the one the dyad "
            "resolved, [%.3f, %.3f], with a crop offset of %.3f s. No column for it.",
            sid, label, r0, r1, t0, t1, offset)
        return False
    if r1 > t1 + tol:
        logger.info("%s condition %s: the aligned recording ends %.1f s into it, so these "
                    "numbers cover more of the block than the coherence does",
                    sid, label, t1 - t0)
    return True


def condition_subject_metrics(
    subject_sqm: dict,
    subject_ids: "list[str]",
    windows: "list[tuple[str, float, float]]",
    sci_threshold: float,
    offsets: "dict[str, float] | None" = None,
    sfreq: "float | None" = None,
) -> dict:
    """``{condition: tables}`` for the per-subject quality table, one entry per window.

    Each value is what :func:`subject_metric_tables` returns, one table per channel set.

    Every number is read out of each member's ``by_condition`` record section, which
    :func:`~fnirs_pipe.qc.subject.sqm_record.condition_sections` wrote once after that member's
    pipeline finished. Nothing is measured here and nothing is sliced a second time, so a
    channel's SCI under one condition cannot differ between a subject page and a dyad page.

    **It reports, it does not re-decide.** The coherence on a condition's page was computed
    on the channel set the whole recording was screened into, because the window is read out
    of a transform of the whole recording and a per-condition channel set would need a
    transform of its own. Screening each condition separately would also make a contrast
    between two conditions a contrast between two montages. So these rows say how the
    channels held up over this stretch, and the set they were drawn from is the run's.

    A member whose record predates the section contributes nothing and its column reads as
    absent, which is the honest answer: the values cannot be recovered from the whole-run
    scalars.
    """
    from fnirs_pipe.qc.common.record_views import condition_set_view

    if not windows:
        return {}

    offsets = offsets or {}
    # the dyad resolves its windows from one member, the records carry each member's own
    tol = _TRIGGER_JITTER_SAMPLES / float(sfreq) if sfreq else 0.0

    per_subject: dict = {}
    for sid in subject_ids:
        by_condition = (subject_sqm.get(sid) or {}).get("by_condition") or {}
        if not by_condition:
            logger.info("%s: no by_condition section in the quality record, so the "
                        "per-condition quality table has no column for it", sid)
            continue
        # `windows` drives the loop rather than the record's own set, and the two floors
        # differ on purpose: the record keeps anything holding two screening windows (20 s),
        # hyper anything holding one cycle of the slowest frequency analysed (100 s at
        # 0.01 Hz). The record's set is therefore a superset, and reading it straight
        # through would put a condition too short to have a coherence onto a WTC page.
        for label, t0, t1 in windows:
            entry = by_condition.get(label)
            if entry is None:
                continue
            if not _record_window_matches(entry.get("window_s"), t0, t1,
                                          float(offsets.get(sid, 0.0)), tol, sid, label):
                continue
            for set_name, _ in _CHANNEL_SETS:
                view = condition_set_view(entry, set_name)
                if view:
                    per_subject.setdefault(label, {}).setdefault(set_name, {})[sid] = view

    return {label: subject_metric_tables(by_set, subject_ids, sci_threshold)
            for label, by_set in per_subject.items()}



# Oscillations of the slowest analysed frequency a window needs before its coherence is
# worth reading. Reported, never enforced.
MIN_BAND_CYCLES = 4.0


def _band_cycles(window: "tuple[float, float] | None", band_fmin: float) -> "float | None":
    """How many cycles of the slowest analysed frequency a window holds.

    ::

      a 300 s condition with band_fmin 0.06 Hz -> 18.0
      the same condition with band_fmin 0.01   ->  3.0, below MIN_BAND_CYCLES

    A coherence is a statement about the phase relationship at a frequency, and a window
    holding one cycle of it has seen that relationship once. This is the count, not the cone:
    it stays the same however clean the edges are, and it is the number that moves when
    ``--wtc-band-fmin`` is lowered without lengthening the blocks.
    """
    if not window or not band_fmin:
        return None
    return (float(window[1]) - float(window[0])) * float(band_fmin)


def _member_nirs_dir(output_dir: Path, entry: GroupEntry) -> Path:
    """A member's ``nirs/``, laid out the way :func:`group_data_dir` lays out a group's.

    An entry naming no session still finds a session folder, so a tree with one unnamed
    session yields a Methods paragraph rather than the "no sidecars found" note.
    """
    from fnirs_pipe.io.derivatives import subject_nirs_dirs

    found = subject_nirs_dirs(output_dir, entry.subject_id, entry.session)
    if found:
        return found[0]
    folder = subject_report_dir(output_dir, entry.subject_id)
    if entry.session:
        folder = folder / f"ses-{entry.session}"
    return folder / "nirs"


def group_methods(
    output_dir: Path,
    group: list[GroupEntry],
    group_nirs: Path,
    own_steps: list[tuple[str, dict]],
    versions: dict[str, str],
    notes: list,
    scope: str,
) -> dict[str, str]:
    """Methods prose for a dyad: a member's preprocessing, then the group's own steps.

    Three sources in the order a reader needs them: what was done to each recording (read
    from a member's sidecars), what was done to bring them onto one clock (``own_steps``,
    which leaves no file to scan), then what was measured across the two (read from the
    group's sidecars).

    The paragraph describes **one** preprocessing pipeline, because that is what a
    manuscript can use. When the members were not processed the same way, that becomes a
    note rather than a sentence naming both, since a Methods section hedging every
    parameter is worse than one that says which subject it describes.
    """
    from fnirs_pipe.qc.boilerplate import generate_methods_text
    from fnirs_pipe.qc.boilerplate.vocabulary import steps_from_sidecars

    per_member = {e.subject_id: steps_from_sidecars(_member_nirs_dir(output_dir, e))
                  for e in group}
    chains = list(per_member.values())
    if chains and any(chain != chains[0] for chain in chains[1:]):
        note(notes, scope,
             "the members of this group were not preprocessed identically, so the Methods "
             f"paragraph describes sub-{next(iter(per_member))} only")
    if not any(chains):
        note(notes, scope,
             "no preprocessing sidecars found for the members, so the Methods paragraph "
             "covers the group-level steps only")

    steps = list(chains[0]) if chains else []
    steps += own_steps
    steps += steps_from_sidecars(group_nirs)
    return generate_methods_text(versions=versions, steps=steps)


# The columns a dyad's channel table prints, dropped rather than listed, so a column added
# to CHANNEL_COLUMNS reaches this page by default. Same drops the raw viewer makes: a
# decision is per pair, and this page groups by member rather than by separation.
_CH_COLUMNS = channel_columns(("corr", "separation"))


def decision_rows(sqm_data: dict, subject_ids: list[str],
                  sci_threshold: float) -> list[dict]:
    """The dyad's channel table: one entry per pair, carrying every member's cells.

    ::

      -> [{"pair": "S1_D1", "separation": "long",
           "by_sub": {"sub-01": {"sci": "0.956", "sci_cls": "", ...}, "sub-02": {...}}}]

    Grouped by channel rather than by member, because that is the comparison the page is
    for: a pair's two members sit side by side in one row instead of twenty-two rows apart.
    Every string and every cell class is built here by the functions the subject report and
    the raw viewer use, so the dyad's cells and the member's own report cannot disagree; the
    page's script only lays them out.

    A member with no per-channel section contributes no cells rather than a row of dashes,
    and the pair order is the first member that has one.
    """
    per_sub: dict[str, dict[str, dict]] = {}
    order: list[tuple[str, str]] = []
    for sid in subject_ids:
        member = sqm_data.get(sid) or {}
        record = {"per_channel": member.get("per_channel") or {}}
        # acquisition order, per wavelength: it is what sets the row order, and the pairing
        # below folds the two wavelengths of a pair into the one row a decision applies to
        sci_scores = (member.get("per_channel_all") or {}).get("sci_per_channel") or {}
        try:
            rows = pair_rows(channel_rows(record, sci_scores,
                                          member.get("bad_channels") or []))
            cutoffs = member.get("screen_cutoffs") or {}
            formatted = format_rows(rows, sci_threshold, name_key="pair",
                                    psp_threshold=cutoffs.get("psp"))
        except Exception:
            logger.warning("%s: channel table could not be built", sid, exc_info=True)
            formatted = []
        per_sub[sid] = {r["name"]: r for r in formatted}
        if not order:
            order = [(r["name"], r.get("separation") or "") for r in formatted]

    return [{"pair": pair, "separation": sep,
             "by_sub": {sid: per_sub[sid][pair] for sid in subject_ids
                        if pair in per_sub[sid]}}
            for pair, sep in order]


def build_hyper_report(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    sqm_data: dict[str, dict],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    output_dir: Path,
    raw_raws: dict[str, mne.io.Raw] | None = None,
    intensity_raws: dict[str, mne.io.Raw] | None = None,
    after_raws: dict[str, mne.io.Raw] | None = None,
    session: str | None = None,
    sci_threshold: float = SCI_PASS,
    cardiac_l_freq: float | None = None,
    cardiac_h_freq: float | None = None,
    coherence_fmin: float = 0.01,
    coherence_fmax: float = 0.10,
    sep_bands=None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    notes: list[str] = []

    meta = _process_hyper_raw_group(
        group_id=group_id, task=task, group=group,
        sqm_data=sqm_data, aligned_raws=aligned_raws, offsets=offsets,
        output_dir=output_dir,
        raw_raws=raw_raws, intensity_raws=intensity_raws, after_raws=after_raws,
        session=session, sci_threshold=sci_threshold,
        cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq,
        coherence_fmin=coherence_fmin, coherence_fmax=coherence_fmax,
        sep_bands=sep_bands, errors=errors, notes=notes,
    )

    from fnirs_pipe.qc.boilerplate.vocabulary import template_slots
    versions = collect_software_versions()
    # only the alignment is passed in: it leaves no file, so no sidecar describes it.
    # The coherence sentence comes off the table the writer just wrote.
    own_steps = [
        ("hyper_alignment", template_slots(
            "hyper_alignment", {"n_subjects": len(meta["subject_ids"])})),
    ]
    methods = group_methods(output_dir, group, meta["sqm_dir"], own_steps,
                            versions, notes, meta["label"])

    name_parts = [f"group-{group_id}"]
    if session:
        name_parts.append(f"ses-{session}")
    name_parts.append(f"task-{task}")
    output_path = (group_report_dir(output_dir, group_id)
                   / ("_".join(name_parts) + "_desc-hyperraw_nirs.html"))

    # the post report if `fnirs-hyper run` has written one; a raw-only tree has none, and
    # the index is built from the same coherence tables, so neither link is offered there
    post = output_path.with_name(
        output_path.name.replace("_desc-hyperraw_", "_desc-hyperpost_"))
    post_href = post.name if post.exists() else None

    html = render(
        "hyper_report.html.j2",
        **page_vars(
            title=f"{meta['label']} raw",
            # the run names the page, as it does on a subject report; that this is QC
            # is what the reader opened, and raw against post is in the subtitle
            heading=meta["label"],
            nav_meta=[("group", group_id), ("task", task),
                      ("subjects", ", ".join(meta["subject_ids"]))],
            nav_note=(f"SCI thr: {sci_threshold:.2f} • "
                      f"Coh: {coherence_fmin:.3f}–{coherence_fmax:.3f} Hz"),
        ),
        **footer_vars(
            scope=meta["label"], errors=errors, notes=notes,
            nirs_dir=meta["sqm_dir"],
            methods=methods, versions=versions,
        ),
        group_id=group_id,
        task=task,
        subject_ids=meta["subject_ids"],
        sci_threshold=sci_threshold,
        coherence_fmin=coherence_fmin,
        coherence_fmax=coherence_fmax,
        alignment_json=json.dumps(meta["alignment"]),
        run_command=" ".join(sys.argv),
        # the summary states the pair in one line; the table below it is per member
        align_duration_s=next((r["duration_s"] for r in meta["alignment"]
                               if r["duration_s"] is not None), None),
        align_max_offset_s=max((abs(r["offset_s"]) for r in meta["alignment"]),
                               default=0.0),
        ch_pairs_json=json.dumps(meta["ch_pairs"]),
        sqm_json=json.dumps(meta["sqm"], default=str),
        figure_paths=meta["figure_paths"],
        ch_detail_template_json=json.dumps(meta["figure_paths"].get("ch_detail_template")),
        ch_columns=_CH_COLUMNS,
        decision_rows_json=json.dumps(
            decision_rows(sqm_data, meta["subject_ids"], sci_threshold), default=str),
        blocks_json=json.dumps(meta.get("conditions") or {}),
        member_info_json=json.dumps(meta.get("member_info") or []),
        # one unheaded table: this page's scalars come from the raw pass, which measures
        # every channel and does not split by separation
        subject_metrics_rows=subject_metric_tables(
            {"all": sqm_data}, meta["subject_ids"], sci_threshold),
        # this page's own name, which is what the rating server files a verdict under
        page_stem=output_path.stem,
        post_href=post_href,
        index_href=f"group-{group_id}_index.html" if post_href else "",
    )
    output_path.write_text(html, encoding="utf-8")
    logger.info("Hyper raw report saved: %s", output_path)
    return output_path


def _slice_pair(df, pair: "tuple[str, str] | None"):
    """The rows of a band frame belonging to one pairing.

    The frames are long over every pairing in the group. Everything downstream keys on
    the site pair alone, so handing it the whole frame lets a group of three overwrite
    one pairing's cell with another's; slicing here means no consumer has to know the
    column exists. A frame without the columns is from a tree written before they were
    added and is passed through.
    """
    if df is None or pair is None or not {"sub1", "sub2"}.issubset(
            getattr(df, "columns", [])):
        return df
    return df[(df["sub1"] == pair[0]) & (df["sub2"] == pair[1])]


def _number_table(bands: dict, isc: "dict | None", axis: list[str], kind: str,
                  chroma: tuple, diagonal_only: bool = False) -> dict:
    """Every number behind one scope's panels, one row per pairing.

    ::

      {"hbo": band frame, "hbr": ...} + {"hbo": (matrix, names), ...}
        -> {"kind": "channel", "valid": "87% in COI",
            "rows": [{"a": "S1_D1", "b": "S1_D2",
                      "cells": {"HbO WTC": "0.241", "HbO ISC": "+0.067"}}]}

    The panels above show these as colour and the TSVs hold them to full precision; this
    is the same numbers on the page, so that reading one off a cell does not mean opening
    a file. Every value is looked up by label pair, never by position, which is the rule
    the matrices follow and for the same reason: two members can differ in what they
    rejected.

    Cells are keyed by column name rather than by position, so a scope that filled a
    column no other scope did still lands under the right header once `_merge_scopes` puts
    them side by side.

    ``valid`` is the share of band cells that survived the cone of influence, and it is one
    number for the whole scope rather than a column: the cone depends on the window length
    and the band, so every pairing in a scope has the same share. A window whose coherence
    rests on a third of its cells is not the same measurement as one that kept all of them,
    which is worth saying once at the top rather than repeating down every row. A rejected
    pairing contributes nothing to it, its band mean being NaN.

    Rows are every pairing that carries at least one value, so an uncrossed run shows the
    coherence on the diagonal and the ISC everywhere, which is what those two actually
    computed. ``isc`` is the matrices for this scope at this table's own level, channel or
    ROI, and None where the scope produced none.

    ``diagonal_only`` drops the crossed pairings, for the table whose subject is the
    homologous mean alone. The correlation matrix is a full ROI x ROI whatever the
    coherence beside it covers, so without this that table grew a row per crossed region
    carrying no coherence and repeating the correlation the crossed table above it prints.
    """
    cells: dict = {}
    fracs: set = set()

    def _put(pair: tuple, column: str, text: str) -> None:
        cells.setdefault(pair, {})[column] = text

    for ch_type in chroma:
        name = _CHROMA_LABEL[ch_type]
        df = bands.get(ch_type)
        if df is None or "label" not in getattr(df, "columns", []):
            continue
        for row in df.itertuples():
            # an uncrossed run has no `label2`: every row of it is a site against the
            # other member's copy of the same site, which is this table's diagonal
            pair = (row.label, getattr(row, "label2", row.label))
            # a rejected channel keeps its row and leaves the value empty, which is a dash
            # here as everywhere else on the page rather than the string "nan"
            if np.isfinite(row.coherence):
                _put(pair, f"{name} WTC", f"{row.coherence:.3f}")
            frac = getattr(row, "n_valid_frac", None)
            if frac is not None and np.isfinite(frac):
                fracs.add(round(float(frac), 4))

    for ch_type in (chroma if isc else ()):
        mat, names = isc.get(ch_type) or (None, None)
        if mat is None or not names:
            continue
        mat = np.asarray(mat, dtype=float)
        index = {name: i for i, name in enumerate(names)}
        for a in axis:
            for b in axis:
                if a in index and b in index and np.isfinite(mat[index[a], index[b]]):
                    _put((a, b), f"{_CHROMA_LABEL[ch_type]} ISC",
                         f"{mat[index[a], index[b]]:+.3f}")

    # the columns this table actually filled, in a fixed order rather than in the order
    # the first pairing happened to fill them
    order = ([f"{_CHROMA_LABEL[c]} WTC" for c in chroma]
             + [f"{_CHROMA_LABEL[c]} ISC" for c in (chroma if isc else ())])
    rows = [{"a": a, "b": b, "cells": cells[(a, b)]}
            for a in axis for b in axis
            if (a, b) in cells and not (diagonal_only and a != b)]
    # off the rows that survived, so a column only the dropped pairings filled goes with them
    used = {col for row in rows for col in row["cells"]}
    lo, hi = (min(fracs), max(fracs)) if fracs else (None, None)
    valid = "" if lo is None else (
        f"{100 * lo:.0f}% in COI" if lo == hi
        else f"{100 * lo:.0f}–{100 * hi:.0f}% in COI")
    return ({"kind": kind, "columns": [c for c in order if c in used],
             "rows": rows, "valid": valid} if rows else {})


def _merge_scopes(kind: str, axis: list[str], per_scope: list) -> dict:
    """The whole run and every condition on one row per pairing, side by side.

    ::

      [("Whole run", table), ("rest", table)]
        -> {"scopes": [{"label": "Whole run", "valid": "87% in COI"}, ...],
            "rows": [{"a": "S1_D1", "b": "S1_D2",
                      "cells": {"Whole run": {"HbO WTC": "0.241"}, "rest": {...}}}]}

    Stacked instead, the channel pairings of a 14-channel crossed dyad are 196 rows per
    condition and the whole run's copy of a pairing is hundreds of rows away from the
    condition's. Side by side, the comparison a block design is run for is one row.

    ``columns`` is the union over the scopes, which is what lets one header stand over all
    of them; a scope that filled fewer leaves its cells empty rather than shifting the rest.
    Rows keep ``axis`` order and a pairing appears once however many scopes carry it.
    """
    if not per_scope:
        return {}
    cells: dict = {}
    for heading, table in per_scope:
        for row in table["rows"]:
            cells.setdefault((row["a"], row["b"]), {})[heading] = row["cells"]
    rows = [{"a": a, "b": b, "cells": cells[(a, b)]}
            for a in axis for b in axis if (a, b) in cells]
    if not rows:
        return {}
    return {
        "kind": kind,
        "columns": list(dict.fromkeys(c for _, t in per_scope for c in t["columns"])),
        "scopes": [{"label": h, "valid": t["valid"]} for h, t in per_scope],
        "rows": rows,
    }


def _isc_arc_rule(isc_threshold: "float | None", isc_phase_null: int) -> str:
    """One sentence naming which of the three rules drew the connectogram's chords.

    The figure's own subtitle says the same thing; this is the page's parameter table, which
    a reader reaches without opening a panel.
    """
    if isc_threshold is not None:
        return f"|r| &ge; {isc_threshold:.2f}"
    if isc_phase_null:
        return f"above each pairing's own null, {isc_phase_null} surrogates"
    return "the strongest 10%, a display cut rather than a test"


def build_hyper_post_report(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    output_dir: Path,
    roi_map: dict[str, list[str]] | None = None,
    bad_channels: dict[str, list[str]] | None = None,
    subject_sqm: dict[str, dict] | None = None,
    wtc_fmin: float = 0.004,
    wtc_fmax: float = 0.20,
    wtc_band_fmin: float | None = None,
    wtc_band_fmax: float | None = None,
    wtc_significance: bool = False,
    wtc_seed: int | None = None,
    wtc_mc_count: int = 300,
    wtc_channel_cross: bool = False,
    wtc_by_condition: bool = False,
    wtc_cond_pad_s: "float | None" = None,
    wtc_limit_scales: bool = True,
    wtc_save_maps: bool = False,
    wtc_mask_coi: bool = True,
    wtc_roi_min_channels: int = 2,
    wtc_arrow_min: "float | None" = None,
    wtc_chroma: "tuple[str, ...] | list[str]" = ("hbo", "hbr"),
    isc_threshold: "float | None" = None,
    isc_whiten: int = 0,
    isc_max_lag_s: float = 0.0,
    isc_band: "tuple[float | None, float | None] | None" = None,
    isc_phase_null: int = 0,
    sci_threshold: float = SCI_PASS,
    sep_bands=None,
    cond_windows: "list[tuple[str, float, float]] | None" = None,
    analysis_window: "tuple[float, float] | None" = None,
    no_report: bool = False,
    result: "HyperPostResult | None" = None,
) -> "Path | None":
    """Build hyperscanning post-QC report.

    Sections:
      1. Per-channel WTC  — Morlet wavelet coherence, one heatmap per channel
      2. Per-ROI WTC      — the member channels' maps averaged cell by cell (when roi_map given)
      3. Cross matrices   — band-mean coherence per channel pair and per ROI pair, HbO
                            beside HbR on one scale (when wtc_channel_cross)
      4. ISC              — inter-brain Pearson r, a ROI × ROI heatmap (when roi_map
                            given) and then per chromophore a channel heatmap beside its
                            connectogram, whose arcs are chosen by `_arc_rule`

    Each WTC map is also collapsed to one number per channel over
    [wtc_band_fmin, wtc_band_fmax] and written as a TSV under the group's nirs/, so a
    group analysis reads the same values the figures were drawn from.

    ``wtc_channel_cross`` crosses every long channel with every other, 14 channels giving 196
    rows in ``hyper-wtc.tsv`` instead of 14. The extra pairs reach the TSV and the crossed
    matrix, while the map selector keeps the homologous ones: 196 options is not a list
    anybody reads through, and drawing a full frequency × time map for each of them per
    chromophore is most of what the figures cost. Crossing is also what produces the
    ROI × ROI matrix, since the ROI numbers are grouped from the channel ones.

    ``wtc_by_condition`` repeats the whole coherence analysis inside each task annotation's
    own window, on top of the whole-run pass, which stays as it was. The band means of every
    window land in one ``hyper-wtcbycond.tsv`` with a ``condition`` column, and each window
    gets its own figures. See :func:`condition_windows` for how a window is decided, and note
    that the runtime is roughly doubled: the windows together are about one more pass over
    the recording.

    ``cond_windows`` supplies those windows instead of resolving them here. The caller passes
    the same list to the phase-scrambled null, and the two tables can only be subtracted row by
    row if they describe the same windows. Left at None the windows are resolved here, which
    is what a caller that writes no null wants.

    ``analysis_window`` is ``--tstart``/``--tend``: the stretch the whole-run pass reports
    on. It is taken out of the transform of the whole recording, the way a condition window
    is, so a run restricted to a window carries the recording's cone of influence rather than
    two edges of its own. The recordings themselves are never cut.

    ``wtc_cond_pad_s`` transforms each condition on its own instead of reading it out of the
    whole-run transform, over a cut padded by that many seconds on each side. Left at None,
    which is the default, conditions are windowed out of the whole-run transform. See
    :func:`_transform_condition`: with adequate padding the two give the same number, so this
    exists for a caller who wants per-condition transforms, and 0 reproduces the unpadded cut
    other pipelines take. Costs one transform per condition per chromophore.

    ``wtc_mask_coi`` restricts each band mean to the cone of influence. On by default; the
    share inside the cone is reported either way as ``n_valid_frac``.

    ``wtc_chroma`` is the chromophores to run, ``("hbo",)``, ``("hbr",)`` or both. Both is
    the default and costs exactly twice as much, since the two are the same computation run
    twice: a member's HbO pairs only with the other member's HbO, and the two are never
    mixed and never averaged. The reason to have both is a consistency check rather than two
    results, HbO carrying the larger amplitude and HbR the less scalp contamination, so a
    coupling in HbO with nothing in HbR is a caution flag. Every band-mean table gains a
    ``chromophore`` column rather than splitting per chromophore, the tables being
    long-format. The figures are keyed by chromophore instead and the page carries all of
    them, switched together by one control, which is the comparison the second chromophore
    exists for; ``wtc_chroma[0]`` is only what the page opens on.

    ``wtc_save_maps`` writes the full time-frequency maps beside the tables as ``.npz``, one
    per chromophore, so a different band can be averaged later without a second wavelet
    transform. See :mod:`fnirs_pipe.pipeline.hyper.wtc_store`. ``wtc_limit_scales`` computes only the scales inside
    ``[wtc_fmin, wtc_fmax]`` plus margin, which is most of the runtime and, given that the
    scales land on pycwt's own grid and the margin exceeds the scale-smoothing window,
    reproduces the unrestricted coherences bit for bit.

    ``result`` is an already-computed :class:`~fnirs_pipe.pipeline.hyper.hyper_post.HyperPostResult`.
    Passed one, this draws it and runs no transform, which is how a page is rebuilt after a
    figure or a caption changes without paying for the wavelet pass again. Left at None the
    analysis is run here from the ``wtc_*`` arguments, which is what the CLI does. The
    arguments that only describe the analysis are then ignored, since the result already
    carries what they resolved to.

    ``wtc_arrow_min`` is the coherence a cell has to reach before its phase arrow is drawn
    when no Monte Carlo level was computed. Display only: no table or figure value changes
    with it. ``None`` takes :data:`~fnirs_pipe.qc.figures.hyper.hyper_post_figures.ARROW_MIN_COHERENCE`.
    """
    from fnirs_pipe.pipeline.hyper.hyper_post import HyperPostConfig, run_hyper_post
    from fnirs_pipe.pipeline.hyper import WTCResult, roi_maps_from_channels
    from fnirs_pipe.qc.figures.hyper.hyper_post_figures import (
        ARROW_MIN_COHERENCE,
        build_isc_panel,
        build_isc_roi_matrix,
        build_wtc_channel,
        build_wtc_cross_matrix,
        build_wtc_map_interactive,
        wtc_condition_views,
    )

    arrow_min = ARROW_MIN_COHERENCE if wtc_arrow_min is None else float(wtc_arrow_min)

    errors: list[str] = []
    notes: list[str] = []
    # A page's own failures, keyed (pairing, condition label) with None for the run. `errors`
    # above stays the run-wide list and reaches every page: an analysis that failed is missing
    # from all of them. A figure that failed is missing from one, and only that one says so.
    page_errors: dict = defaultdict(list)
    scope = f"group-{group_id}_task-{task}"

    subject_ids  = [e.subject_id for e in group]
    ref_raw      = aligned_raws.get(subject_ids[0]) if subject_ids else None

    # Every inter-brain figure on this report is of two members. A group of three has three
    # such pairings and the transform carries all of them, so each gets its own page rather
    # than one page standing for the group: the numbers differ per pairing and a single page
    # would have to pick one and name it after the group.
    pairings = list(combinations(subject_ids, 2))

    def _pair_slug(pair: "tuple[str, str] | None") -> str:
        return pair_slug(pair, len(pairings))

    # ---- what put the members on one clock ----
    # Read once and written onto every table this report produces. An inter-brain number
    # assumes a shared time axis, and the stamp is the only thing on disk that says whether
    # one was established: `--no-align` and an alignment whose trigger sits at t=0 both leave
    # every offset at zero, and the numbers cannot be told apart afterwards.
    align_info = alignment_params(aligned_raws)
    if align_info.get("aligned") is False:
        note(notes, scope,
             "these recordings were trimmed to a common length but never aligned on a "
             "shared trigger, so every number below assumes they already shared a clock")

    # ---- is this a segment rather than a recording? ----
    # A cut carries two edges of its own, and everything this report computes from a wavelet
    # transform loses a share of its band at them that grows as the cut shortens. Treating a
    # segment and a whole recording alike is the one way into inflated numbers a reader cannot
    # see, so this still computes but says which it got.
    crop_info = crop_provenance(ref_raw) if ref_raw else None
    if crop_info:
        span = crop_info["window"]
        where = (f"{span[0]:.1f}-{span[1]:.1f} s of the source recording"
                 if isinstance(span, list) and len(span) == 2 and not isinstance(span[0], list)
                 else f"{crop_info['n_windows']} separate windows of the source recording")
        if crop_info["margin_s"] > 0:
            note(notes, scope,
                 f"Input is a cut ({where}), made with a {crop_info['margin_s']:.1f} s "
                 f"margin on each side. --wtc-by-condition windows each block out of the "
                 f"transform, so the margin is what absorbs the cone and is not averaged.")
        else:
            note(notes, scope,
                 f"Input is a cut ({where}), made with no margin. A wavelet transform of a "
                 f"segment has two edges of its own, so this run's band means are inflated "
                 f"by an amount that grows as the segment shortens; n_valid_frac reports the "
                 f"share of band cells that survived. Re-cut with "
                 f"`fnirs-prep crop --margin auto --band-fmin <f>` to avoid it.")
        logger.warning("cropped input: %s, margin %.1f s", where, crop_info["margin_s"])
    markers_list = markers_on_data_axis(ref_raw) if ref_raw else []
    all_descs    = list(dict.fromkeys(m["description"] for m in markers_list))
    cond_colors_ = _cond_colors(all_descs)

    alignment_rows = [
        {
            "subject_id": sid,
            "offset_s":   round(offsets.get(sid, 0.0), 3),
            "duration_s": round(aligned_raws[sid].times[-1], 1)
                          if sid in aligned_raws else None,
        }
        for sid in subject_ids
    ]

    # One subdirectory per group and task, the way a per-run subject report gets one: a
    # group with five cropped tasks writes five sets of these and they would otherwise
    # overwrite each other under one figures/.
    figures_dir = group_report_dir(output_dir, group_id) / "figures" / scope

    def _fig(b64: "str | None", name: str) -> "str | None":
        """One figure onto disk, returning the URL the page links it by, or None.

        Every figure in this report goes out as a file. Embedding them instead is what took
        this page to 174 MB on a 14-channel dyad: a coherence map is 51 x 3962 cells, and
        the page carried one of them per channel per chromophore.
        """
        return save_png(b64, figures_dir, name)

    def _fig_html(fig, name: str, views: "dict | None" = None) -> "dict | None":
        """One Plotly figure onto disk as its own page, with the height its iframe needs.

        The PNG twin is :func:`_fig`. Its URL is under the same ``wtc`` key a PNG entry
        uses, so one shape indexes both panels and a reader of the page's tables does not
        have to know which kind a pairing turned out to be.

        ``views`` is how one file serves the run and each condition off a URL fragment.
        """
        if fig is None:
            return None
        height = _save_figure_html(fig, figures_dir / name, views=views)
        return {"wtc": _fig_href(figures_dir, name), "h": height}


    def _maps(dest: dict, result, pair_key, pair_label: str, axis: list[str],
              stem: str, ch_type: str, suffix: str, what: str, page,
              interactive: bool = False, view_spans=None) -> None:
        """Fill ``dest`` with one coherence map per pairing of ``axis`` against itself.

        ::

          axis ["S1_D1", "S1_D2"], crossed
            -> dest["S1_D1"]["S1_D2"] = {"wtc": "figures/.../wtc_hbo_S1D1_x_S1D2.png"}

        Both map panels are built here, the channels with the channel axis and the ROIs with
        the ROI labels, so the two cannot drift in how they key or name their files. The
        nesting is ``[label of the first member][label of the second]`` and is written even
        for an uncrossed run, which fills only the diagonal: the page then reads one shape
        and shows one selector instead of two.

        A pairing the result has no entry for, or whose builder failed, lands as ``None``
        and the page hides its image. That is the rejected channel's blank row arriving on
        the figure side of the same rule the tables follow.

        ``interactive`` writes each map as its own Plotly page instead of a PNG, and the
        entry then carries the iframe's height beside its URL. The ROI panel takes it and
        the channel panel does not, on volume alone: both cost about 3 MB a map, and a
        14-channel crossed dyad has 2352 channel maps against 192 ROI ones.

        ``view_spans`` puts every condition's window into the file as well, so the run's
        page and each condition's are one file addressed by URL fragment. Interactive only:
        a PNG has no view to open on.
        """
        for label1 in axis:
            row: dict = {}
            for label2 in (axis if wtc_channel_cross else [label1]):
                entry = None
                if result and pair_key:
                    key = (label1, label2) if wtc_channel_cross else label1
                    data = result.pairs.get(pair_key, {}).get(key)
                    site = label1 if label1 == label2 else f"{label1} × {label2}"
                    ext = "html" if interactive else "png"
                    fname = (f"{stem}_{_pair_fname(label1)}{suffix}.{ext}"
                             if label1 == label2 else
                             f"{stem}_{_pair_fname(label1)}_x_{_pair_fname(label2)}"
                             f"{suffix}.{ext}")
                    build = (build_wtc_map_interactive if interactive
                             else build_wtc_channel)
                    with guard(f"wtc map {site} ({what}, {ch_type}) figure",
                               page_errors[page], scope):
                        drawn = build(data, result.freqs, result.times,
                                      pair_label, markers_list, cond_colors_, site,
                                      arrow_min=arrow_min)
                        views = (wtc_condition_views(drawn, view_spans, data, result.freqs,
                                                     result.times, arrow_min)
                                 if view_spans and drawn is not None else None)
                        entry = (_fig_html(drawn, fname, views=views)
                                 if interactive else {"wtc": _fig(drawn, fname)})
                # always a dict, even where nothing was drawn: the page indexes every
                # pairing of the axis and a missing one has to answer with an empty URL
                row[label2] = entry or {"wtc": None}
            dest[label1] = row

    def _figure_set(result, chan_band_df, ch_type: str, page, suffix: str = "",
                    roi_view_of: "dict | None" = None,
                    pair: "tuple[str, str] | None" = None) -> dict:
        """Every figure one WTC result yields: the maps and the three matrices.

        Called once with the whole-run result and again with each condition window's, so a
        condition page carries the panels the run's own page carries instead of the two
        pictures a window used to get. Nothing here decides per panel whether a condition
        has it; the only difference between the two calls is what result comes in.

        ``suffix`` names the files, ``""`` for the run and ``"_<slug>"`` for a window, the
        same rule a per-condition subject page names its panels by. That is what lets both
        sets sit in one ``figures/`` directory without the window overwriting the run.

        Returns ``{"per_channel", "per_roi"}``. The two cross matrices are not
        here: they carry both chromophores on one pair of axes, so they are built once per
        scope by ``_matrix_set`` rather than once per chromophore.
        The two map sets are nested ``{label_sub1: {label_sub2: {"wtc": url}}}`` whether or
        not the run crossed, an uncrossed one holding only the diagonal, so the page reads
        one shape and the pair of selectors above each panel is the only difference.
        ``roi_view_of`` hands a window the run's own ROI map set, and the window then points
        at those files with its slug on the end instead of drawing its own. See
        :func:`~fnirs_pipe.qc.figures.hyper.hyper_post_figures.wtc_condition_views` for when that
        is the same figure and when it is not.

        ``pair`` is which two members these maps are of. A group of three holds three
        pairings and the result carries all of them, so the one drawn has to be named rather
        than taken off the front: the maps are labelled with it, and its slug separates their
        files from the next pairing's.
        """
        out: dict = _empty_figures()
        what = f"condition {suffix.lstrip('_')}" if suffix else "whole run"

        pair_key = pair if pair is not None else (
            next(iter(result.pairs)) if result and result.pairs else None)
        pair_label = f"{pair_key[0]} × {pair_key[1]}" if pair_key else ""

        _maps(out["per_channel"], result, pair_key, pair_label, chan_axis,
              f"wtc_{ch_type}{_pair_slug(pair_key)}", ch_type, suffix, what, page)

        if not roi_map:
            return out

        # the maps grouped the same way, so the picture and the table are one average.
        # A window pointing at the run's files needs no maps of its own, so it does not
        # average for them either
        roi_wtc: WTCResult | None = None
        if result is not None and roi_view_of is None:
            with guard(f"ROI WTC maps from channels ({what}, {ch_type})",
                       page_errors[page], scope):
                roi_wtc = roi_maps_from_channels(result, roi_map)
        roi_pair_key = pair_key if (roi_wtc and pair_key in roi_wtc.pairs) else (
            next(iter(roi_wtc.pairs)) if roi_wtc and roi_wtc.pairs else None)

        if roi_view_of is not None:
            out["per_roi"] = {
                label1: {label2: ({**entry, "wtc": entry["wtc"] + "#" + suffix.lstrip("_")}
                                  if entry.get("wtc") else entry)
                         for label2, entry in row.items()}
                for label1, row in roi_view_of.items()
            }
        else:
            _maps(out["per_roi"], roi_wtc, roi_pair_key, pair_label, roi_labels,
                  f"wtcroi_{ch_type}{_pair_slug(pair_key)}", ch_type, suffix, what, page,
                  interactive=True, view_spans=roi_view_spans)

        return out

    def _figures_for(ch_type: str, computed: dict, pair: "tuple[str, str]") -> dict:
        """Every figure one pairing gets out of one chromophore's transforms.

        ::

          -> {"run_figs": {...}, "cond_figs": [{...}, ...]}   # positional over the windows

        The transforms come in rather than being computed: see :func:`_wtc_pass`. A window
        whose own pass failed contributes an empty set and keeps its place in the list.
        """
        run_figs = _figure_set(computed["result"], computed["chan"], ch_type,
                               (pair, None), pair=pair)
        cond_figs: list[dict] = []
        for i, (label, _, _) in enumerate(cond_windows):
            cond_wtc = computed["cond_wtc"][i] if i < len(computed["cond_wtc"]) else None
            cond_chan = (computed["cond_bands"][i] if i < len(computed["cond_bands"])
                         else {}).get("chan")
            if cond_wtc is None or cond_chan is None:
                cond_figs.append({})
                continue
            # the same panels the run's own page gets, over this window. The ROI maps are
            # the run's files at this window's fragment, where that route is on
            cond_figs.append(_figure_set(
                cond_wtc, cond_chan, ch_type, (pair, label), f"_{_pair_fname(label)}",
                roi_view_of=(run_figs["per_roi"] if roi_view_spans else None),
                pair=pair))
        return {"run_figs": run_figs, "cond_figs": cond_figs}

    def _matrix_set(bands: dict, suffix: str, what: str, page,
                    pair: "tuple[str, str] | None" = None) -> dict:
        """The two cross matrices for one scope, both chromophores on one pair of axes.

        ::

          {"hbo": {"chan": df, "roichan": df}, "hbr": {...}}
            -> {"chan_matrix": {...}, "roi_matrix": {...}}

        One figure per matrix rather than one per chromophore, which is the one thing on
        this page that is not built per chromophore. HbO and HbR are run as a consistency
        check on each other, and side by side on one colour scale is that check; stacked as
        two figures with a colorbar each, they were two results a reader had to hold in their
        head. The maps above stay per chromophore because a map is a picture of one pairing
        and there is no comparison to draw inside it.

        ``suffix`` names the files, ``""`` for the run and ``"_<slug>"`` for a window, the
        same rule the figure sets follow. No chromophore in the name any more, so a re-run
        leaves the old per-chromophore files behind; they are not linked from anywhere.
        """
        out: dict = {"chan_matrix": {}, "roi_matrix": {}}
        if not wtc_channel_cross:
            return out

        def _named(key: str) -> dict:
            return {_CHROMA_LABEL[c]: (bands.get(c) or {}).get(key) for c in chroma}

        def _crossed(df) -> bool:
            return df is not None and "label2" in getattr(df, "columns", [])

        pair_ids = list(pair) if pair else subject_ids
        chan_dfs = {k: _slice_pair(v, pair) for k, v in _named("chan").items()}
        if any(_crossed(df) for df in chan_dfs.values()):
            with guard(f"WTC channel cross matrix ({what})", page_errors[page], scope):
                # the montage, not the labels a table happens to carry: a dyad that lost a
                # channel still gets a matrix of the same shape as one that did not
                labels = chan_axis or sorted({lab for df in chan_dfs.values()
                                              if _crossed(df)
                                              for lab in (*df["label"], *df["label2"])})
                out["chan_matrix"] = _fig_html(build_wtc_cross_matrix(
                    chan_dfs, labels, pair_ids, band_fmin, band_fmax, kind="channel"),
                    f"wtc_chanmatrix{_pair_slug(pair)}{suffix}.html") or {}

        roi_dfs = {k: _slice_pair(v, pair) for k, v in _named("roichan").items()}
        if roi_map and any(_crossed(df) for df in roi_dfs.values()):
            with guard(f"wtc-roi-matrix ({what}) figure", page_errors[page], scope):
                out["roi_matrix"] = _fig_html(build_wtc_cross_matrix(
                    roi_dfs, roi_labels, pair_ids, band_fmin, band_fmax, "ROI"),
                    f"wtc_roimatrix{_pair_slug(pair)}{suffix}.html") or {}
        return out

    # ---- the analysis, which is not this module's ----
    # Everything below draws what this returns.
    if result is None:
        result = run_hyper_post(
            group_id, task, aligned_raws, output_dir,
            HyperPostConfig(
                wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax,
                wtc_band_fmin=wtc_band_fmin, wtc_band_fmax=wtc_band_fmax,
                wtc_significance=wtc_significance, wtc_seed=wtc_seed,
                wtc_mc_count=wtc_mc_count, wtc_channel_cross=wtc_channel_cross,
                wtc_by_condition=wtc_by_condition, wtc_cond_pad_s=wtc_cond_pad_s,
                wtc_limit_scales=wtc_limit_scales, wtc_save_maps=wtc_save_maps,
                wtc_mask_coi=wtc_mask_coi, wtc_roi_min_channels=wtc_roi_min_channels,
                wtc_chroma=wtc_chroma, isc_whiten=isc_whiten,
                isc_max_lag_s=isc_max_lag_s, isc_phase_null=isc_phase_null,
                isc_band=isc_band,
                roi_map=roi_map, sep_bands=sep_bands,
                analysis_window=analysis_window,
            ),
            subject_ids=subject_ids, pairings=pairings, align_info=align_info,
            cond_windows=cond_windows, errors=errors, notes=notes, scope=scope,
        )
    # Everything above is the analysis and has already written its tables; everything below
    # draws them. The figures are most of this step's output on disk, and a study that reads
    # the tables never opens them.
    if no_report:
        logger.info("group-%s | --no-report: tables written, figures skipped", group_id)
        return None
    chroma       = result.chroma
    band_fmin    = result.band_fmin
    band_fmax    = result.band_fmax
    cond_pad_s   = result.cond_pad_s
    cond_windows = result.cond_windows
    chan_axis    = result.chan_axis
    roi_labels   = result.roi_labels
    roi_rows     = result.roi_rows
    passes       = result.passes

    # One ROI map file per pairing, carrying the run and a view per window, rather than one
    # file per window. Only where a window really is a slice of the run's own transform:
    # --wtc-cond-transform gives each window a figure of its own and each then writes its
    # own file. See `wtc_condition_views`.
    roi_view_spans = (cond_windows if (cond_windows and cond_pad_s is None) else None)

    # ---- the ISC panels, drawn from the matrices the run computed ----
    # {label or None: {chromophore: href}}, one entry per page below, and the same keys over
    # the matrices those figures were drawn from.
    def _isc_panel_of(pair, label, ch_type) -> dict:
        isc_mat, isc_ch_names = result.isc.get(pair, {}).get(label, {}).get(
            ch_type, (None, None))
        if isc_mat is None:
            return {}
        arc_level = result.isc_levels.get(pair, {}).get(label, {}).get(ch_type)
        what = f"condition {label}" if label else "whole run"
        suffix = f"_{_pair_fname(label)}" if label else ""
        panel: dict = {}
        with guard(f"ISC panel ({what}, {ch_type})", page_errors[(pair, label)], scope):
            panel = _fig_html(build_isc_panel(
                isc_mat, isc_ch_names, list(pair),
                ch_type=ch_type, isc_threshold=isc_threshold, arc_level=arc_level,
            ), f"isc_{ch_type}{_pair_slug(pair)}{suffix}.html") or {}
        return panel

    def _isc_roi_matrix_of(pair, label) -> dict:
        """The ROI ISC of both chromophores as one figure, the way the WTC matrices are.

        One figure rather than one per chromophore, for the reason `_matrix_set` gives: HbO
        and HbR are a consistency check on each other and side by side on one scale is that
        check. Nothing is drawn where the run had no ROI map, since `isc_roi` is then empty.
        """
        mats = {_CHROMA_LABEL[c]: result.isc_roi.get(pair, {}).get(label, {}).get(
                                      c, (None, None))
                for c in chroma}
        if all(mat is None for mat, _ in mats.values()):
            return {}
        what = f"condition {label}" if label else "whole run"
        suffix = f"_{_pair_fname(label)}" if label else ""
        out: dict = {}
        with guard(f"ISC ROI matrix ({what})", page_errors[(pair, label)], scope):
            out = _fig_html(build_isc_roi_matrix(mats, list(pair)),
                            f"isc_roimatrix{_pair_slug(pair)}{suffix}.html") or {}
        return out

    isc_panels: dict = {}
    isc_roi_matrices: dict = {}
    isc_values: dict = {}
    isc_roi_values: dict = {}
    for pr in pairings:
        labels = [None] + [label for label, _, _ in cond_windows]
        isc_panels[pr] = {k: {c: _isc_panel_of(pr, k, c) for c in ("hbo", "hbr")}
                          for k in labels}
        isc_roi_matrices[pr] = {k: _isc_roi_matrix_of(pr, k) for k in labels}
        isc_values[pr] = {k: {c: result.isc.get(pr, {}).get(k, {}).get(c, (None, None))
                              for c in ("hbo", "hbr")}
                          for k in labels}
        isc_roi_values[pr] = {k: {c: result.isc_roi.get(pr, {}).get(k, {}).get(
                                      c, (None, None))
                                  for c in ("hbo", "hbr")}
                              for k in labels}

    # ISC has no frequency axis, so an unfiltered stage reaches the number directly; WTC
    # does not care. Said on the page as well as in the log, since the two are read by
    # different people
    isc_unfiltered_note = unfiltered_stage_note(aligned_raws)
    if isc_unfiltered_note:
        logger.warning("ISC: %s", isc_unfiltered_note)

    # the quality table: the run's over the whole recording, and each window's read out of
    # that member's own record. The offsets are what puts the two clocks together
    run_metric_rows = subject_metric_tables(
        {set_name: {sid: (subject_sqm or {}).get(sid, {}).get("by_set", {}).get(set_name)
                         or {}
                    for sid in subject_ids}
         for set_name, _ in _CHANNEL_SETS},
        subject_ids, sci_threshold)
    cond_metric_rows: dict = {}
    with guard("Per-condition quality table", errors, scope):
        cond_metric_rows = condition_subject_metrics(
            subject_sqm or {}, subject_ids, cond_windows, sci_threshold,
            offsets=offsets,
            sfreq=float(ref_raw.info["sfreq"]) if ref_raw is not None else None)

    bad_pairs_all: set[str] = set()
    if bad_channels:
        for chs in bad_channels.values():
            bad_pairs_all |= {c.rsplit(" ", 1)[0] for c in chs}

    # desc-hyperpost, matching the raw report's desc-hyperraw: the two are one pair of
    # pages and were named by two conventions, one BIDS-shaped and one not.
    #
    # A condition keeps the run's task- entity and takes a desc- of its own,
    # `..._task-full_desc-baseline_hyperpost_nirs.html`, which is the rule the subject
    # report follows. Putting the label in `task-` instead would name a condition page the
    # same as the run page of a tree where that condition was cropped to its own task, and
    # the two are not the same number: one carries the whole recording's cone of influence
    # and the other two edges of its own.
    def _page_path(label: "str | None", pair: "tuple[str, str] | None" = None) -> Path:
        desc = "hyperpost" if label is None else f"{_pair_fname(label)}_hyperpost"
        return (group_report_dir(output_dir, group_id)
                / f"group-{group_id}_task-{task}_desc-{desc}"
                  f"{_pair_slug(pair)}_nirs.html")

    # every page carries the whole strip, so any one of them reaches the others in a click
    nav_pages = [(None, "Whole run")] + [(label, label) for label, _, _ in cond_windows]

    # Rendered here rather than by the caller: every sidecar the scan reads was written by
    # the passes above, so this is the first moment the graph is complete. Same stem
    # `fnirs-qc provenance` uses, so re-running that refreshes the image this report links.
    provenance_path = None
    with guard("Provenance diagram", errors, scope):
        from fnirs_pipe.qc.figures.common.provenance_figure import write_provenance

        for written in write_provenance(
            group_data_dir(output_dir, group_id),
            group_report_dir(output_dir, group_id) / "figures",
            stem="provenance", title=f"group-{group_id}_task-{task}",
        ):
            if written.suffix == ".png":
                provenance_path = f"figures/{written.name}"

    from fnirs_pipe.qc.boilerplate.vocabulary import template_slots
    versions = collect_software_versions()
    own_steps = [("hyper_alignment", template_slots(
        "hyper_alignment", {"n_subjects": len(subject_ids)}))]
    methods = group_methods(output_dir, group, group_data_dir(output_dir, group_id),
                            own_steps, versions, notes, scope)

    def _render_page(figs: dict, matrices: dict, number_scopes: list, label: "str | None",
                     window: "tuple[float, float] | None",
                     pair: "tuple[str, str] | None" = None) -> Path:
        """One page: the whole run's when ``label`` is None, else that condition's.

        Both go through this, so a panel cannot exist on the run's page and be missing from
        a condition's by anyone forgetting to add it. ``figs`` is
        ``{chromophore: figure set}`` and ``matrices`` the scope's two cross matrices, which
        carry every chromophore in one figure; between them they are the only thing that
        differs between the two kinds of page.

        One panel is the run's alone and is left out rather than repeated: the per-subject
        quality table is measured over the whole recording, and printing it under a
        condition's heading would read as that condition's numbers. It is a click away on
        the run's own page, and the page says so.

        ``number_scopes`` is ``[(heading, band frames, ISC label), ...]``: the run's page
        carries the whole run and then every condition, a condition's page carries itself
        alone. The pictures above are of one scope and the numbers below are not, because a
        block design is read by comparing conditions and every panel repeated per condition
        would be a page nobody scrolls.
        """
        # Every figure's URL keyed by chromophore, which is what the page's chromophore
        # switch reads. The panels are one set of DOM nodes filled from these, not one set
        # per chromophore, so a reader is always looking at one chromophore across the
        # whole page rather than at an HbO channel panel above an HbR ROI panel. What
        # travels is a path under figures/, so the switch changes an img src and the
        # browser fetches one file.
        def _by_chroma(key: str) -> dict:
            return {c: (figs.get(c) or {}).get(key) or _FIGURE_SET[key]() for c in chroma}

        per_channel = _by_chroma("per_channel")
        per_roi     = _by_chroma("per_roi")
        roi_matrix  = matrices.get("roi_matrix") or {}
        chan_matrix = matrices.get("chan_matrix") or {}

        pair_ids = list(pair) if pair else subject_ids
        # One table per kind of pairing rather than one grid holding all three, so a table
        # carries only the columns it filled and the page has one level of nesting instead
        # of a scope inside a kind inside a grid.
        number_tables = []
        for kind, key, axis, values in (
            ("Channel pairs", "chan", chan_axis, isc_values),
            ("ROI pairs", "roichan", roi_labels, isc_roi_values),
            # the homologous ROI mean, which is the reported number and the only ROI
            # value the null can rank; the table above is every pairing in the region
            ("ROI homologous pairs", "roihom", roi_labels, isc_roi_values),
        ):
            per_scope = []
            for heading, band_frames, isc_label in number_scopes:
                bands = {c: (band_frames.get(c) or {}) for c in chroma}
                table = _number_table(
                    {c: _slice_pair(bands[c].get(key), pair) for c in chroma},
                    (values.get(pair) or {}).get(isc_label) or {}, axis, kind, chroma,
                    diagonal_only=key == "roihom")
                if table:
                    per_scope.append((heading, table))
            merged = _merge_scopes(kind, axis, per_scope)
            if merged:
                number_tables.append(merged)

        out_path = _page_path(label, pair)
        heading = f"group-{group_id}_task-{task}"
        nav_meta = [("group", group_id), ("task", task),
                    ("subjects", ", ".join(pair_ids))]
        if label is not None:
            nav_meta.insert(2, ("condition", label))

        html = render(
            "hyper_post_report.html.j2",
            **page_vars(
                title=f"{heading} post" + (f" / {label}" if label else ""),
                heading=heading + (f" — {label}" if label else ""),
                nav_meta=nav_meta,
                nav_note=(f"WTC: {wtc_fmin:.3f}–{wtc_fmax:.3f} Hz · "
                          f"{'+'.join(_CHROMA_LABEL[c] for c in chroma)}"),
            ),
            **footer_vars(
                # the run-wide list plus this page's own, so a window that lost a panel says
                # so and its neighbours do not
                scope=scope, errors=errors + page_errors[(pair, label)], notes=notes,
                nirs_dir=group_data_dir(output_dir, group_id),
                provenance_path=provenance_path,
                methods=methods, versions=versions,
            ),
            group_id=group_id,
            task=task,
            subject_ids=pair_ids,
            index_href=f"group-{group_id}_index.html",
            wtc_fmin=wtc_fmin,
            wtc_fmax=wtc_fmax,
            wtc_band_fmin=band_fmin,
            wtc_band_fmax=band_fmax,
            arrow_min=arrow_min,
            mask_coi=wtc_mask_coi,
            sci_threshold=sci_threshold,
            run_command=" ".join(sys.argv),
            # the summary states the pair in one line; the table below it is per member
            align_duration_s=next((r["duration_s"] for r in alignment_rows
                                   if r["duration_s"] is not None), None),
            align_max_offset_s=max((abs(r["offset_s"]) for r in alignment_rows),
                                   default=0.0),
            wtc_chroma_labels=[_CHROMA_LABEL[c] for c in chroma],
            wtc_chroma_json=json.dumps(list(chroma)),
            isc_arc_rule=_isc_arc_rule(isc_threshold, isc_phase_null),
            alignment_json=json.dumps(alignment_rows),
            per_channel_post_json=json.dumps(per_channel),
            # the long axis, not `ch_pairs_post`: the selector has to name the set the
            # matrix beside it is drawn on, and the short channels have no coherence
            ch_pairs_post_json=json.dumps(chan_axis),
            bad_pairs_json=json.dumps(sorted(bad_pairs_all)),
            roi_rows=roi_rows,
            roi_labels_json=json.dumps(roi_labels),
            per_roi_post_json=json.dumps(per_roi),
            wtc_roi_matrix=roi_matrix,
            wtc_chan_matrix=chan_matrix,
            number_tables=number_tables,
            # a second selector on each map panel, which an uncrossed run has no pairings
            # for: it holds the diagonal alone
            chan_crossed=wtc_channel_cross,
            roi_crossed=wtc_channel_cross,
            # ---- what tells the two kinds of page apart ----
            condition_label=label,
            condition_window=(f"{window[0]:.1f}–{window[1]:.1f} s on the aligned clock"
                              if window else ""),
            analysis_window=(f"{analysis_window[0]:.1f}–{analysis_window[1]:.1f} s"
                             if analysis_window else ""),
            window_cycles=_band_cycles(window or analysis_window, wtc_band_fmin),
            min_band_cycles=MIN_BAND_CYCLES,
            run_href=_page_path(None, pair).name,
            # this page's own name, which is what the rating server files a verdict under
            page_stem=out_path.stem,
            nav_links=[{"label": text, "href": _page_path(lab, pair).name,
                        "current": lab == label} for lab, text in nav_pages]
                      + ([] if len(pairings) < 2 else
                         [{"label": " × ".join(pr), "href": _page_path(label, pr).name,
                           "current": pr == pair} for pr in pairings]),
            isc_unfiltered_note=isc_unfiltered_note,
            isc_roi_matrix=(isc_roi_matrices.get(pair) or {}).get(label) or {},
            isc_panel_hbo=(isc_panels.get(pair) or {}).get(label, {}).get("hbo") or {},
            isc_panel_hbr=(isc_panels.get(pair) or {}).get(label, {}).get("hbr") or {},
            subject_metrics_rows=(run_metric_rows if label is None
                                  else cond_metric_rows.get(label, [])),
        )
        out_path.write_text(html, encoding="utf-8")
        return out_path

    # One set of pages per pairing, and inside each the conditions first and the run last,
    # so the run's page carries the complete error and note lists: a guard that failed while a
    # window was being drawn belongs on both. The path returned is the first pairing's run
    # page, which for a dyad is the only one there has ever been.
    output_path = None
    for pr in pairings:
        pair_figs = {c: _figures_for(c, passes[c], pr) for c in chroma}
        run_matrices = _matrix_set(
            {c: {"chan": passes[c]["chan"], "roichan": passes[c]["roichan"]}
             for c in chroma}, "", "whole run", (pr, None), pr)

        def _cond_bands(i: int) -> dict:
            return {c: (passes[c]["cond_bands"][i]
                        if i < len(passes[c]["cond_bands"]) else {}) for c in chroma}

        run_bands = {c: {"chan": passes[c]["chan"], "roichan": passes[c]["roichan"],
                         "roihom": passes[c].get("roihom")}
                     for c in chroma}

        for i, (label, tstart, tstop) in enumerate(cond_windows):
            figs = {c: (pair_figs[c]["cond_figs"][i]
                        if i < len(pair_figs[c]["cond_figs"]) else {}) for c in chroma}
            bands = _cond_bands(i)
            written = _render_page(
                figs,
                _matrix_set(bands, f"_{_pair_fname(label)}", f"condition {label}",
                            (pr, label), pr),
                [(label, bands, label)], label, (tstart, tstop), pr)
            logger.info("group-%s | %s condition %s -> %s",
                        group_id, " × ".join(pr), label, written.name)

        # the run's page prints every condition under the whole run, which is the comparison
        # a block design is run for and the one thing no condition page can show
        written = _render_page(
            {c: pair_figs[c]["run_figs"] for c in chroma}, run_matrices,
            [("Whole run", run_bands, None)]
            + [(label, _cond_bands(i), label)
               for i, (label, _, _) in enumerate(cond_windows)],
            None, None, pr)
        logger.info("Hyper post report saved: %s", written)
        output_path = output_path or written
    return output_path
