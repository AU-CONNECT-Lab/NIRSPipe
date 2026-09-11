"""The map between the metric keys the pipeline computes and the prose describing them.

`test_step_vocabulary.py` holds the same contract one level up, for steps. This one is for
the numbers those steps produce, and it exists because the two halves live apart: the keys
are written in `qc/metrics/`, the sentences in `vocabulary.py`, and the report template
names keys by hand. Nothing but this file notices when they drift.

The drift is silent in both directions. A metric with no entry renders as a bare number
with no way to act on it, and a template asking for a key that was renamed renders an
empty tooltip. Neither raises.
"""

import ast
import re
from pathlib import Path

import pytest

import fnirs_pipe.qc as qc_pkg
from fnirs_pipe.qc.boilerplate.vocabulary import (
    KEY_METRICS,
    METRIC_DISPLAY,
    METRIC_SUMMARY,
    format_metric,
    is_key_metric,
    metric_class,
    metric_summary,
)
from fnirs_pipe.qc.boilerplate.vocabulary import higher_is_better, metric_direction
from fnirs_pipe.qc.channel_table import OD_SPLIT_COLUMNS, _COLUMN_METRIC
from fnirs_pipe.qc.figures.sci_psp_panel import _TRIAL_METRICS
from fnirs_pipe.qc.prep_raw_report import _VIEW_SCALAR_KEYS

_QC = Path(qc_pkg.__file__).parent
_SUBJECT_TEMPLATE = _QC / "templates" / "subject_report.html.j2"

def _metric_modules() -> list[Path]:
    return sorted((_QC / "metrics").glob("*.py"))


# Per-channel dicts are deliberately undescribed: each is the same quantity as its scalar
# sibling, one value per channel. So is the wildcard family, which has no fixed members.
_NOT_A_SCALAR = ("per_channel", "temporal_derivative_variance")


def _declared_metric_keys() -> set[str]:
    """Every key the `@_safe_metrics` decorators declare, read from the source.

    Read statically rather than by calling the functions: the decorator only exposes its
    keys through the wrapper's behaviour, and running the metrics needs real data.

    Every module in `qc/metrics/`, not one file: a family that moved to a new module would
    otherwise drop out of the check silently, which is the failure this test exists to make
    loud. `test_the_scan_reaches_every_metric_module` guards the glob.
    """
    keys: set[str] = set()
    for path in _metric_modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "_safe_metrics":
                    keys.update(e.value for e in ast.walk(dec.args[1])
                                if isinstance(e, ast.Constant) and isinstance(e.value, str))
    return keys


def _scalar_metric_keys() -> set[str]:
    return {k for k in _declared_metric_keys()
            if not k.endswith("*") and not any(s in k for s in _NOT_A_SCALAR)}


def _template_metric_keys() -> set[str]:
    """The keys the report's metrics panel asks the registry about.

    Three call shapes, because the panel prints three kinds of row: `qm_li('x', …)` for a
    list row, `ql('x', …)` for a bare label, and `('x', 'Label')` for a column in the
    channel-set table. All three end in a registry lookup, so all three are drift risks.
    """
    text = _SUBJECT_TEMPLATE.read_text(encoding="utf-8")
    return (set(re.findall(r"\bqm_li\('([^']+)'", text))
            | set(re.findall(r"\bql\('([^']+)'", text))
            | set(re.findall(r"\('([a-z0-9_]+)',\s*'[A-Z]", text)))


def _view_metric_keys() -> set[str]:
    """The keys the raw viewer and the GUI ask about, which they take from Python.

    Those two render in JavaScript and in Dash components, so they cannot name a key in a
    template: they receive rows already built by the registry. The lists naming which rows
    are still hand-written, and still drift.

    The per-channel table is in here through `_COLUMN_METRIC`, which is where a column gets
    its format: a column is not a scalar key, so it reaches the registry only through the
    scalar it is the same quantity as.
    """
    return (set(_VIEW_SCALAR_KEYS) | {key for key, _ in OD_SPLIT_COLUMNS}
            | set(_COLUMN_METRIC.values()))


# ---- the map ----

def test_every_metric_the_pipeline_computes_is_described():
    undescribed = sorted(_scalar_metric_keys() - set(METRIC_SUMMARY))
    assert not undescribed, (
        f"these metrics reach the report with no explanation: {undescribed}. "
        "Add a line to METRIC_SUMMARY saying what the number is and which way is good."
    )


def test_the_report_never_asks_for_a_metric_that_has_no_entry():
    missing = sorted(_template_metric_keys() - set(METRIC_SUMMARY))
    assert not missing, (
        f"subject_report.html.j2 renders an empty tooltip for {missing}. "
        "Either the key was renamed or its METRIC_SUMMARY line was never written."
    )


def test_no_view_asks_for_a_metric_that_has_no_entry():
    """The same check for the two views that build their rows in Python.

    The raw viewer and the GUI print whatever the registry hands them, so a key with no
    entry there is a row with no label, no format and no verdict.
    """
    missing = sorted(_view_metric_keys() - set(METRIC_SUMMARY))
    assert not missing, f"the raw views name {missing}, which the registry does not describe"


def test_every_key_a_view_prints_has_a_format():
    """METRIC_SUMMARY says what a number means; METRIC_DISPLAY says how to print it.

    A key in the first and not the second renders at a default three decimals with no
    units and no colour, which for a percentage or an exponent is unreadable rather than
    merely plain.
    """
    asked = _template_metric_keys() | _view_metric_keys()
    unformatted = sorted(asked - set(METRIC_DISPLAY))
    assert not unformatted, (
        f"these print with the fallback format and no threshold: {unformatted}. "
        "Add a METRIC_DISPLAY row giving the label, the format and the cutoffs."
    )


def test_the_panel_asks_about_something():
    # guards the regexes above: a template rewrite that drops the call shapes would
    # silently make the previous tests vacuous
    assert len(_template_metric_keys()) > 15
    assert len(_view_metric_keys()) > 5


def test_the_decisive_metrics_are_described_and_real():
    assert KEY_METRICS <= set(METRIC_SUMMARY)
    # they must also be things the pipeline computes, or the report marks a key that
    # never appears
    assert KEY_METRICS <= _scalar_metric_keys() | {
        "sci_mean", "channel_retention_rate", "hbo_hbr_corr_mean",  # undecorated functions
        "good_frac_mean",  # same: _good_frac_metrics is handed its scores, not decorated
    }


@pytest.mark.parametrize("metric", sorted(METRIC_SUMMARY))
def test_a_description_is_a_sentence(metric):
    text = METRIC_SUMMARY[metric]
    assert text.strip() == text
    assert text.endswith("."), f"{metric}: not a sentence"
    # "The same for HbR." and friends are deliberate: the chromophore pairs would
    # otherwise repeat six sentences verbatim. They borrow their substance from a sibling,
    # which test_a_cross_reference_has_something_to_refer_to checks is still there.
    if not text.startswith("The same"):
        assert len(text) > 25, f"{metric}: too short to say which way is good"


def test_a_cross_reference_has_something_to_refer_to():
    """An `_hbr` entry saying "the same" is orphaned if its `_hbo` twin is renamed away."""
    for metric, text in METRIC_SUMMARY.items():
        if text.startswith("The same for HbR"):
            twin = metric.replace("_hbr", "_hbo")
            assert twin != metric and twin in METRIC_SUMMARY, (
                f"{metric} defers to {twin}, which is not in the registry")


def test_the_chromophore_pairs_are_complete():
    # a metric described for one chromophore and not the other is an omission, not a choice
    for metric in METRIC_SUMMARY:
        if metric.endswith("_hbo"):
            assert metric[:-4] + "_hbr" in METRIC_SUMMARY, f"{metric} has no HbR entry"
        if metric.endswith("_hbr"):
            assert metric[:-4] + "_hbo" in METRIC_SUMMARY, f"{metric} has no HbO entry"


def test_an_unknown_metric_reads_as_empty_rather_than_raising():
    # the template calls these for every row, so a lookup miss has to degrade to a plain
    # number with no tooltip and no verdict, never to an exception mid-render
    assert metric_summary("no_such_metric") == ""
    assert is_key_metric("no_such_metric") is False
    assert metric_class("no_such_metric", 0.5) == ""
    assert format_metric("no_such_metric", 0.5) == "0.500"


def test_a_missing_value_prints_as_a_dash_not_as_a_verdict():
    """A metric the run did not measure must not read as one that scored zero."""
    for metric in ("sci_mean", "pct_data_retained", "spike_count", "gvtd_mean"):
        assert format_metric(metric, None) == "\u2014"
        assert metric_class(metric, None) == ""


@pytest.mark.parametrize("metric", sorted(METRIC_DISPLAY))
def test_a_display_row_is_usable(metric):
    label, fmt, thresholds, direction = METRIC_DISPLAY[metric]
    assert label and label.strip() == label
    assert metric in METRIC_SUMMARY, f"{metric} is printed but not described"
    assert direction in (None, "higher", "lower"), f"{metric}: {direction!r}"
    # The two are independent and most metrics have only a direction. A threshold without
    # one is the combination that cannot mean anything: there is no way to read which side
    # of the cutoff passes.
    if thresholds is not None:
        assert direction is not None, f"{metric}: a cutoff with no direction to read it in"
        ok, warn = thresholds
        # ok is the stricter end, so the three bands come out in the right order whichever
        # way the metric is read
        assert (ok > warn) if direction == "higher" else (ok < warn), (
            f"{metric}: ok={ok} warn={warn} reads backwards for direction {direction!r}")
    # the format has to survive a real number, which is the whole point of storing it
    assert format_metric(metric, 0.5, fmt)


def test_a_described_direction_is_encoded():
    """METRIC_SUMMARY states the direction in prose; METRIC_DISPLAY has to agree with it.

    The prose is what a reader sees in the tooltip and the flag is what colours the cell and
    orients the per-trial heatmap, so the two saying different things is a report that
    contradicts itself. Only the unambiguous phrasings are checked: several entries are
    deliberately non-committal ("no universal good value") and those must have no direction.
    """
    for metric, text in METRIC_SUMMARY.items():
        direction = metric_direction(metric)
        lowered = text.lower()
        if "no universal good value" in lowered or "descriptive" in lowered:
            assert direction is None, f"{metric}: described as descriptive but ranked"
        elif "higher is better" in lowered or "higher is a more" in lowered:
            assert direction == "higher", f"{metric}: prose says higher, flag says {direction}"
        elif "lower is better" in lowered or "lower is cleaner" in lowered:
            assert direction == "lower", f"{metric}: prose says lower, flag says {direction}"


def test_the_per_trial_heatmap_can_orient_every_row_it_draws():
    """Its colour is relative within a row, so a metric with no better end has no worse end.

    `_trial_metric_specs` drops such a metric with a warning rather than drawing it
    upside down, which means a key added to _TRIAL_METRICS and not to the registry silently
    vanishes from the panel. This is what notices.
    """
    missing = [key for key, _ in _TRIAL_METRICS if higher_is_better(key) is None]
    assert not missing, (
        f"the per-trial heatmap would drop {missing}: no direction in METRIC_DISPLAY")
    for key, label in _TRIAL_METRICS:
        assert key in METRIC_DISPLAY, f"{key} has no format, so its hover would be bare"
        assert label and len(label) <= 12, f"{label!r} is too wide for a heatmap row"


def test_gvtd_points_at_the_threshold_that_applies_to_it():
    """The one pairing that was documented wrongly once already.

    `gvtd_thresh` is computed from the band-passed trace and compared against it, so it is
    a cutoff for `gvtd_filt_*` and not for the unfiltered `gvtd_mean` / `gvtd_p95`.
    """
    assert "gvtd_thresh" in METRIC_SUMMARY["gvtd_filt_p95"] or \
           "gvtd_thresh" in METRIC_SUMMARY["gvtd_filt_mean"]
    assert "does not apply" in METRIC_SUMMARY["gvtd_mean"]


def test_the_scan_reaches_every_metric_module():
    """`_declared_metric_keys` globs a directory, so a module the glob stops reaching drops
    its whole family out of every test above without failing any of them.

    One key per declaring module, named by hand. Deriving them from the same glob would
    make this test agree with itself whatever the glob found.
    """
    assert {p.stem for p in _metric_modules()} >= {"coupling", "gvtd", "haemo", "motion"}
    assert {"psp_mean", "gvtd_thresh", "cnr_hbo_mean", "spike_pct"} <= _declared_metric_keys()
