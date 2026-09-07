"""Which channels get rejected, and on what grounds.

One table, :data:`CRITERIA`, lists every criterion a channel is screened on. A channel is
rejected if it fails **any** of them, so the table is a union and the entries are
independent: dropping a criterion is deleting its line, adding one is adding a line plus a
scorer. Everything that prunes goes through :func:`screen_channels`, so the prep pipeline,
the raw QC report, the per-trial scoring and the dyad path cannot end up screening on
different things.

The scores themselves are measured in :mod:`fnirs_pipe.qc.metrics.coupling`; this module
only decides. Cutoffs live in :mod:`fnirs_pipe.qc.metrics._helpers` beside the other
per-channel lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mne

from fnirs_pipe.qc.metrics._helpers import PSP_PASS, SCI_PASS
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.metrics.screening")


@dataclass(frozen=True)
class Criterion:
    """One screening criterion: what it measures, where its line is, and how to score it.

    ``scorer`` takes optical density and the cardiac band and returns ``{channel: score}``.
    ``config_field`` names the PrepConfig / CLI field a run may override ``cutoff`` with, or
    None for a criterion with no flag. Every criterion so far fails low, which is what
    ``fails_below`` records; a criterion that fails high would set it False.
    """

    name: str
    label: str
    cutoff: float
    scorer: Callable[[mne.io.Raw, float, float], dict[str, float]]
    config_field: str | None = None
    fails_below: bool = True

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


# The criteria, in the order a report lists them. Both measure optode-scalp coupling from
# the cardiac pulsation and they catch different failures: SCI is high whenever the two
# wavelengths agree, which movement can fake, and PSP is near zero when it is faked.
CRITERIA: tuple[Criterion, ...] = (
    Criterion("sci", "SCI", SCI_PASS, _sci_scorer, config_field="sci_threshold"),
    Criterion("psp", "PSP", PSP_PASS, _psp_scorer, config_field="psp_threshold"),
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
) -> dict[str, dict[str, float]]:
    """Every criterion's per-channel scores, measured on optical density.

    ``have`` is whatever the caller measured already, keyed by criterion name; those are
    passed through rather than recomputed, which is what keeps a caller that needs SCI for
    its own reasons from paying for it twice.
    """
    scores = dict(have or {})
    for c in CRITERIA:
        if c.name in scores:
            continue
        try:
            scores[c.name] = c.scorer(raw_od, cardiac_l_freq, cardiac_h_freq)
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

    screen_channels({"sci": {"S1_D1 760": 0.4}, "psp": {"S1_D1 760": 0.5}})
    -> (["S1_D1 760"], {"S1_D1 760": ["SCI"]})

    The union: one failed criterion is enough. The second half is what lets a report say why
    a channel went rather than only that it did.
    """
    cutoffs = cutoffs or criterion_cutoffs()
    why: dict[str, list[str]] = {}
    for c in CRITERIA:
        for ch in c.failures(scores.get(c.name) or {}, cutoffs.get(c.name)):
            why.setdefault(ch, []).append(c.label)
    # channel order follows the first criterion that carries scores, i.e. acquisition order
    order = next((s for s in (scores.get(c.name) for c in CRITERIA) if s), {})
    bad = [ch for ch in order if ch in why]
    bad += [ch for ch in why if ch not in order]
    return bad, why
