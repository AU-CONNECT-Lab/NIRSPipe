"""Which channels get rejected, and on what grounds.

One table, :data:`CRITERIA`, lists every per-channel number the reports carry. A channel is
rejected if it fails any criterion whose ``screens`` is set, so the screening half of the
table is a union and its entries are independent: dropping a criterion is deleting its line,
adding one is adding a line plus a scorer. Everything that prunes goes through
:func:`screen_channels`, so the prep pipeline, the raw QC report, the per-trial scoring and
the dyad path cannot end up screening on different things.

**SCI and PSP are measured and reported but do not decide.** Their published definition
pairs them inside one short window and counts how many windows a channel passes, which is
what ``good_frac`` does; the two whole-run numbers stay in the table because every report
prints them and because they are the lines ``good_frac`` applies per window.

The scores themselves are measured in :mod:`fnirs_pipe.qc.metrics.coupling`; this module
only decides. Cutoffs live in :mod:`fnirs_pipe.qc.metrics._helpers` beside the other
per-channel lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mne

from fnirs_pipe.qc.metrics._helpers import GOOD_FRAC_PASS, PSP_PASS, SCI_PASS
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.metrics.screening")


@dataclass(frozen=True)
class Criterion:
    """One screening criterion: what it measures, where its line is, and how to score it.

    ``scorer`` takes optical density and the cardiac band and returns ``{channel: score}``.
    ``config_field`` names the PrepConfig / CLI field a run may override ``cutoff`` with, or
    None for a criterion with no flag. Every criterion so far fails low, which is what
    ``fails_below`` records; a criterion that fails high would set it False.

    ``screens`` is whether failing this criterion rejects the channel. False means measured
    and reported only. ``needs_context`` marks a scorer that is handed a fourth argument,
    the run's screening context: the resolved cutoffs under ``cutoffs`` and the stretches of
    the recording that count under ``scope``. That is how a criterion built out of other
    criteria's lines gets them without reaching for a global.
    """

    name: str
    label: str
    cutoff: float
    scorer: Callable[..., dict[str, float]]
    config_field: str | None = None
    fails_below: bool = True
    screens: bool = True
    needs_context: bool = False

    def failures(self, scores: dict[str, float], cutoff: float | None = None) -> list[str]:
        line = self.cutoff if cutoff is None else cutoff
        return [ch for ch, v in scores.items()
                if v is not None and ((v < line) if self.fails_below else (v > line))]


def _sci_scorer(raw_od: mne.io.Raw, cardiac_l_freq: float, cardiac_h_freq: float) -> dict:
    from fnirs_pipe.qc.metrics.coupling import compute_sci_scores
    scores, _ = compute_sci_scores(raw_od, cardiac_l_freq, cardiac_h_freq)
    return scores


def _psp_scorer(raw_od: mne.io.Raw, cardiac_l_freq: float, cardiac_h_freq: float) -> dict:
    from fnirs_pipe.qc.metrics.coupling import compute_psp_scores
    return compute_psp_scores(raw_od, cardiac_l_freq, cardiac_h_freq)


def _good_frac_scorer(raw_od: mne.io.Raw, cardiac_l_freq: float, cardiac_h_freq: float,
                      context: dict) -> dict:
    from fnirs_pipe.qc.metrics.windowed import good_window_fraction
    cutoffs = context.get("cutoffs") or {}
    return good_window_fraction(
        raw_od, cardiac_l_freq, cardiac_h_freq,
        sci_cutoff=cutoffs.get("sci", SCI_PASS),
        psp_cutoff=cutoffs.get("psp", PSP_PASS),
        scope=context.get("scope"))


# The criteria, in the order a report lists them. SCI and PSP both measure optode-scalp
# coupling from the cardiac pulsation and they catch different failures: SCI is high
# whenever the two wavelengths agree, which movement can fake, and PSP is near zero when it
# is faked. That is why the rejection is neither of them on its own but `good_frac`, which
# requires both inside the same window and then counts the windows.
CRITERIA: tuple[Criterion, ...] = (
    Criterion("sci", "SCI", SCI_PASS, _sci_scorer,
              config_field="sci_threshold", screens=False),
    Criterion("psp", "PSP", PSP_PASS, _psp_scorer,
              config_field="psp_threshold", screens=False),
    Criterion("good_frac", "coupled windows", GOOD_FRAC_PASS, _good_frac_scorer,
              config_field="min_good_frac", needs_context=True),
)


def criterion_cutoffs(config: object | None = None) -> dict[str, float]:
    """Each criterion's line for this run: its default, overridden from ``config``.

    criterion_cutoffs(config_with_sci_threshold_0_9) -> {"sci": 0.9, "psp": 0.1}

    A criterion with no ``config_field``, or a config that does not carry it, keeps the
    default. Passing None gives the defaults, which is what a caller with no run config has.
    """
    cutoffs = {}
    for c in CRITERIA:
        value = None
        if config is not None and c.config_field:
            value = getattr(config, c.config_field, None)
        cutoffs[c.name] = c.cutoff if value is None else float(value)
    return cutoffs


def resolve_cutoffs(config: object | None = None, **overrides) -> dict[str, float]:
    """:func:`criterion_cutoffs` with loose per-criterion overrides laid on top.

    resolve_cutoffs(sci=0.9, psp=None) -> {"sci": 0.9, "psp": 0.1}

    For a caller holding the lines as separate floats rather than as a config object, which
    is every CLI command and every report builder. A None override is not a value, so it
    keeps whatever the config or the default gave; an unknown criterion name is a typo the
    screening would silently ignore, so it raises.
    """
    unknown = set(overrides) - {c.name for c in CRITERIA}
    if unknown:
        raise ValueError(f"no such screening criterion: {sorted(unknown)}")
    cutoffs = criterion_cutoffs(config)
    cutoffs.update({k: float(v) for k, v in overrides.items() if v is not None})
    return cutoffs


def screening_scores(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    have: dict[str, dict[str, float]] | None = None,
    cutoffs: dict[str, float] | None = None,
    scope: "list[tuple[str, float, float]] | None" = None,
) -> dict[str, dict[str, float]]:
    """Every criterion's per-channel scores, measured on optical density.

    ``have`` is whatever the caller measured already, keyed by criterion name; those are
    passed through rather than recomputed, which is what keeps a caller that needs SCI for
    its own reasons from paying for it twice.

    ``cutoffs`` is only read by criteria built out of other criteria's lines, and defaults
    to the table's own. Pass the same dict here and to :func:`screen_channels`, or a run
    would count windows against one SCI line and colour its report against another.

    ``scope`` is the stretches of the recording that count, ``[(label, tstart, tstop)]``, and
    None counts all of it. It reaches the window-counting criterion only; nothing else in
    the table has a time axis to restrict.
    """
    scores = dict(have or {})
    context = {"cutoffs": cutoffs or criterion_cutoffs(), "scope": scope}
    for c in CRITERIA:
        if c.name in scores:
            continue
        try:
            args = (raw_od, cardiac_l_freq, cardiac_h_freq)
            scores[c.name] = c.scorer(*args, context) if c.needs_context else c.scorer(*args)
        except Exception as exc:
            # a criterion that cannot be measured must not silently reject every channel
            logger.warning("%s could not be measured (%s); it screens nothing this run",
                           c.label, exc)
            scores[c.name] = {}
    return scores


def screen_channels(
    scores: dict[str, dict[str, float]],
    cutoffs: dict[str, float] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """The rejected channels, and which criteria each one failed.

    screen_channels({"good_frac": {"S1_D1 760": 0.4}})
    -> (["S1_D1 760"], {"S1_D1 760": ["coupled windows"]})

    The union over the criteria that screen: one failed is enough. Criteria with ``screens``
    unset are scored and reported but reject nothing, so a channel with a poor whole-run SCI
    that still passes enough windows is kept. The second half is what lets a report say why
    a channel went rather than only that it did.
    """
    cutoffs = cutoffs or criterion_cutoffs()
    why: dict[str, list[str]] = {}
    for c in CRITERIA:
        if not c.screens:
            continue
        for ch in c.failures(scores.get(c.name) or {}, cutoffs.get(c.name)):
            why.setdefault(ch, []).append(c.label)
    # channel order follows the first criterion that carries scores, i.e. acquisition order
    order = next((s for s in (scores.get(c.name) for c in CRITERIA) if s), {})
    bad = [ch for ch in order if ch in why]
    bad += [ch for ch in why if ch not in order]
    return bad, why
