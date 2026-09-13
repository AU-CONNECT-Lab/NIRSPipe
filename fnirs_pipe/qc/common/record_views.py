"""One condition's numbers, read back out of the record that holds them.

A condition entry in, a flat dict for one channel set out. The subject pages use one of
these and the dyad pages the other, so neither path owns them.
"""

from __future__ import annotations

def condition_verdict_view(entry: dict) -> "dict[str, float | None]":
    """One condition's scalars over the long channels, the set the run's own rows report.

    ::

        {"scalars": {"sci_win_mean": 0.71, "gvtd_filt_p95": 1.2e-4},
         "od_by_set": {"long": {"sci_win_mean": 0.86}}}
        -> {"sci_win_mean": 0.86, "gvtd_filt_p95": 1.2e-4}

    The arrangement :func:`~fnirs_pipe.qc.subject.sqm_record.raw_verdict_view` makes for a run, and
    for the same reason: a page printing a run's row above a condition's needs both on one
    channel set, or what looks like a comparison between two stretches is a comparison
    between two montages as well.

    ``scalars`` is mixed by construction. Its motion and haemoglobin halves are already the
    long set, matching the run's rows; its optical-density half is a mean over every channel,
    a condition being a column selection out of one matrix that holds them all. Only that
    half moves here, and only where the long set has a value: a montage with no short
    channel has a long row equal to the whole of it and nothing changes, and one whose long
    row is empty keeps the numbers it has rather than gaining a row of dashes.
    """
    out = dict(entry.get("scalars") or {})
    long_od = (entry.get("od_by_set") or {}).get("long") or {}
    out.update({key: value for key, value in long_od.items() if value is not None})
    return out


def condition_set_view(entry: dict, set_name: str) -> "dict[str, float | None]":
    """One condition's scalars over one channel set, the three families joined.

    ::

        condition_set_view(entry, "short")
        -> {"sci_win_mean": 0.55, "gvtd_filt_p95": 3.1e-4, "hbo_hbr_corr_mean": -0.31, ...}

    A condition's record keeps its numbers split three ways by family rather than one flat
    dict per set: ``od_by_set`` from the windowed matrices, ``motion_by_set`` from the GVTD
    series and the flagged spans, ``haemo_by_set`` from the crop of the haemoglobin file.
    A page that prints one channel set wants them back together, and joining them here is
    what keeps a set's row from mixing two sets the way the flat ``scalars`` does.

    An empty dict for a set the record has nothing under, which is a montage with no
    channel of that kind; the caller drops the row rather than printing dashes.
    """
    out: dict = {}
    for family in ("od_by_set", "motion_by_set", "haemo_by_set"):
        out.update((entry.get(family) or {}).get(set_name) or {})
    return {key: value for key, value in out.items() if value is not None}
