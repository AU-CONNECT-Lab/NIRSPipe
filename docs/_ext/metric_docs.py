"""The QC page's metric entries, generated from the vocabulary the reports print."""

import importlib
import inspect
import re
from functools import cache
from pathlib import Path

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

from cli_parsers import _rst
from nirspipe.qc.boilerplate import vocabulary
from nirspipe.qc.boilerplate.vocabulary import (
    METRIC_DISPLAY,
    METRIC_SUMMARY,
    metric_direction,
    metric_format,
    metric_label,
)

logger = logging.getLogger(__name__)

# one heading of the page each; within a group, the order of METRIC_SUMMARY
GROUPS = {
    "coupling": [
        "sci_mean", "sci_win_mean", "channel_retention_rate", "psp_mean", "good_frac_mean",
        "cp_mean", "cv_mean", "snr_mean", "snr_pass_rate", "n_flat_channels", "mean_amp_mean",
        "n_saturated_channels", "saturated_pct", "ch_dist_mean", "ch_dist_min", "ch_dist_max",
    ],
    "motion": [
        "gvtd_mean", "gvtd_p95", "gvtd_filt_mean", "gvtd_filt_p95", "gvtd_vstd_mean",
        "gvtd_vstd_p95", "gvtd_thresh", "gvtd_thresh_applied", "gvtd_num_above_thresh",
        "gvtd_pct_above_thresh", "gvtd_censor_pct", "gvtd_censor_retained_s",
        "spike_count", "spike_pct", "spike_num_frames", "spike_pct_frames",
    ],
    "correction": [
        "motion_corrected_frac_mean", "motion_corrected_num", "motion_corrected_pct",
        "motion_corrected_n_segments",
    ],
    "imu": [
        "gyro_speed_mean", "gyro_speed_p95", "accel_jerk_mean", "accel_jerk_p95",
        "gyro_speed_gvtd_rho", "accel_jerk_gvtd_rho",
    ],
    "processed": [
        "hbo_hbr_corr_mean", "cnr_hbo_mean", "cnr_hbr_mean", "cnr_n_epochs", "gcor_hbo",
        "gcor_hbr", "lowfreq_drift_amplitude_hbo", "lowfreq_drift_amplitude_hbr",
        "cardiac_band_power_hbo", "cardiac_band_power_hbr", "cardiac_band_frac_hbo",
        "cardiac_band_frac_hbr", "resp_band_power_hbo", "resp_band_power_hbr",
        "resp_band_frac_hbo", "resp_band_frac_hbr", "pct_data_retained",
    ],
    "whitening": ["whitening_lowband_hbo", "whitening_lowband_hbr"],
}

# the function under nirspipe.qc.metrics whose docstring holds the metric's formula or citation
DEFINED_IN = {
    "sci_mean": "coupling.compute_sci_scores",
    "cp_mean": "coupling._cardiac_power_metrics",
    "cv_mean": "coupling.channel_cv",
    "snr_mean": "coupling.channel_snr",
    "gvtd_mean": "gvtd.gvtd_timetrace",
    "gvtd_thresh": "gvtd.gvtd_threshold",
    "hbo_hbr_corr_mean": "haemo.haemo_quality_metrics",
    "cnr_hbo_mean": "haemo._cnr_metrics",
    "gcor_hbo": "haemo._gcor",
}

PER_CHANNEL = {
    "sci_mean": "sci_per_channel",
    "sci_win_mean": "sci_win_per_channel",
    "psp_mean": "psp_per_channel",
    "cp_mean": "cp_per_channel",
    "cv_mean": "cv_per_channel",
    "snr_mean": "snr_per_channel",
    "saturated_pct": "saturated_per_channel",
    "hbo_hbr_corr_mean": "hbo_hbr_corr_per_channel",
    "motion_corrected_frac_mean": "motion_corrected_frac_per_channel",
}

_SECTIONS = {"Parameters", "Returns", "Raises", "Notes", "References", "Examples", "See Also"}


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip(), maxsplit=1)[0]


@cache
def definition(path):
    """(formula lines, citation keys) from a metric function's docstring."""
    module, name = path.rsplit(".", 1)
    obj = getattr(importlib.import_module(f"nirspipe.qc.metrics.{module}"), name)
    lines = (inspect.getdoc(obj) or "").splitlines()

    # citations come from the prose before the first formula or numpydoc section
    i = 0
    while i < len(lines) and not lines[i].startswith(".. math::") and lines[i].strip() not in _SECTIONS:
        i += 1
    cites = [key.strip() for found in re.findall(r":footcite:`([^`]+)`", "\n".join(lines[:i]))
             for key in found.split(",")]

    formula = []
    while i < len(lines) and lines[i].startswith(".. math::"):
        formula.append(lines[i])
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i][0].isspace()):
            formula.append(lines[i])
            i += 1
    while formula and not formula[-1].strip():
        formula.pop()
    # a paragraph that continues the formula's sentence ("where N is ...") keeps its first sentence
    paragraph = []
    while formula and i < len(lines) and lines[i].strip():
        paragraph.append(lines[i].strip())
        i += 1
    follow = " ".join(paragraph)
    if follow[:1].islower():
        formula += ["", first_sentence(follow)]
    return formula, cites


def _number(key, value):
    return f"{value * 100:g} %" if metric_format(key) == "pct" else f"{value:g}"


def colouring(key):
    spec = METRIC_DISPLAY.get(key)
    cut = spec[2] if spec else None
    if not cut:
        return ""
    if metric_direction(key) == "higher":
        return f"The report colours it green at {_number(key, cut[0])} or above and red below {_number(key, cut[1])}."
    return f"The report colours it green below {_number(key, cut[0])} and red at {_number(key, cut[1])} or above."


def entry(key, bib_keys):
    """One definition-list item of reST: label and keys, definition, formula, better end."""
    label = metric_label(key)
    codes = ", ".join(f"``{k}``" for k in (key, PER_CHANNEL.get(key)) if k)
    term = codes if label == key else f"**{_rst(label)}** ({codes})"
    # the hover text: its first sentence defines, the rest says which end is better
    summary = METRIC_SUMMARY[key]
    sentence = first_sentence(summary)
    rest = summary[len(sentence):].strip()
    body = [_rst(sentence)]
    if key in DEFINED_IN:
        formula, cites = definition(DEFINED_IN[key])
        missing = [c for c in cites if c not in bib_keys]
        if missing:
            logger.warning(f"metric_docs: {key} cites {', '.join(missing)}, which references.bib lacks; left out")
        kept = [c for c in cites if c in bib_keys]
        if kept:
            body[0] += f" :footcite:`{','.join(kept)}`"
        if formula:
            body += ["", *formula]
    if tail := " ".join(filter(None, (_rst(rest), colouring(key)))):
        body += ["", tail]
    return [term, *(f"   {line}" if line.strip() else "" for line in body), ""]


def group_lines(group, bib_keys):
    return [line for key in GROUPS[group] for line in entry(key, bib_keys)]


_BIB_KEYS = set()


def bib_keys_of(paths):
    return {key for path in paths
            for key in re.findall(r"^@\w+\{([^,\s]+),", Path(path).read_text(encoding="utf-8"), re.M)}


class MetricReference(SphinxDirective):
    required_arguments = 1

    def run(self):
        group = self.arguments[0]
        if group not in GROUPS:
            raise self.error(f"unknown metric group {group!r}; one of {', '.join(GROUPS)}")
        self.env.note_dependency(vocabulary.__file__)
        holder = nodes.container()
        self.state.nested_parse(StringList(group_lines(group, _BIB_KEYS), source="metric_docs"),
                                self.content_offset, holder)
        return holder.children


def _on_builder_inited(app):
    # bibtex resolves its files against the conf directory, which a directive cannot reach
    _BIB_KEYS.update(bib_keys_of(Path(app.confdir) / p for p in app.config.bibtex_bibfiles))
    # a metric added to the vocabulary without a group would be missing from the page unnoticed
    grouped = [key for keys in GROUPS.values() for key in keys]
    missing = [key for key in METRIC_SUMMARY if key not in grouped]
    unknown = [key for key in grouped if key not in METRIC_SUMMARY]
    if missing:
        logger.warning(f"metric_docs: no group for {', '.join(missing)}")
    if unknown:
        logger.warning(f"metric_docs: not in METRIC_SUMMARY: {', '.join(unknown)}")


def setup(app):
    app.add_directive("metric-reference", MetricReference)
    app.connect("builder-inited", _on_builder_inited)
    return {"parallel_read_safe": True}
