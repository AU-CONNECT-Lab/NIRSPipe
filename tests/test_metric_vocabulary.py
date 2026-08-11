"""The map between the metric keys the pipeline computes and the prose describing them.

`test_step_vocabulary.py` holds the same contract one level up, for steps. This one is for
the numbers those steps produce, and it exists because the two halves live apart: the keys
are written in `quantitative_metrics.py`, the sentences in `vocabulary.py`, and the report
template names keys by hand. Nothing but this file notices when they drift.

The drift is silent in both directions. A metric with no entry renders as a bare number
with no way to act on it, and a template asking for a key that was renamed renders an
empty tooltip. Neither raises.
"""

import ast
import re
from pathlib import Path

import pytest

import fnirs_pipe.qc.quantitative_metrics as qm
from fnirs_pipe.qc.boilerplate.vocabulary import (
    KEY_METRICS,
    METRIC_SUMMARY,
    is_key_metric,
    metric_summary,
)

_SUBJECT_TEMPLATE = Path(qm.__file__).parent / "templates" / "subject_report.html.j2"

# Per-channel dicts are deliberately undescribed: each is the same quantity as its scalar
# sibling, one value per channel. So is the wildcard family, which has no fixed members.
_NOT_A_SCALAR = ("per_channel", "temporal_derivative_variance")


def _declared_metric_keys() -> set[str]:
    """Every key the `@_safe_metrics` decorators declare, read from the source.

    Read statically rather than by calling the functions: the decorator only exposes its
    keys through the wrapper's behaviour, and running the metrics needs real data.
    """
    src = Path(qm.__file__).read_text(encoding="utf-8")
    keys: set[str] = set()
    for node in ast.walk(ast.parse(src)):
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
    """The keys the report's metrics panel asks the registry about, i.e. every `ql('x', …)`."""
    return set(re.findall(r"\bql\('([^']+)'", _SUBJECT_TEMPLATE.read_text(encoding="utf-8")))


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


def test_the_panel_asks_about_something():
    # guards the regex above: a template rewrite that drops `ql(` would silently make
    # the previous test vacuous
    assert len(_template_metric_keys()) > 15


def test_the_decisive_metrics_are_described_and_real():
    assert KEY_METRICS <= set(METRIC_SUMMARY)
    # they must also be things the pipeline computes, or the report marks a key that
    # never appears
    assert KEY_METRICS <= _scalar_metric_keys() | {
        "sci_mean", "channel_retention_rate", "hbo_hbr_corr_mean",  # undecorated functions
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
    # the template calls this for every row, so a lookup miss has to degrade to no tooltip
    assert metric_summary("no_such_metric") == ""
    assert is_key_metric("no_such_metric") is False


def test_gvtd_points_at_the_threshold_that_applies_to_it():
    """The one pairing that was documented wrongly once already.

    `gvtd_thresh` is computed from the band-passed trace and compared against it, so it is
    a cutoff for `gvtd_filt_*` and not for the unfiltered `gvtd_mean` / `gvtd_p95`.
    """
    assert "gvtd_thresh" in METRIC_SUMMARY["gvtd_filt_p95"] or \
           "gvtd_thresh" in METRIC_SUMMARY["gvtd_filt_mean"]
    assert "does not apply" in METRIC_SUMMARY["gvtd_mean"]
