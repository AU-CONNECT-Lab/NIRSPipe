"""The re-paired null: every draw has to land on the row the real table sits on.

Each draw is against a different person, and the summary groups by ``sub1``/``sub2``. Left
alone, that makes every draw its own group of one and the table reports ``n_iter`` 1 against
a null it never averaged, which reads exactly like a null that was averaged. The pair key is
rewritten to the real dyad's before anything is collected, and these hold that in place
along with the axis rule that lets a stand-in with a different montage be drawn at all.
"""

import numpy as np
import pytest

from fnirs_pipe.pipeline import synchrony
from fnirs_pipe.pipeline.synchrony import WTCResult, compute_wtc_pair_null

FREQS = np.linspace(0.02, 0.30, 12)
TIMES = np.arange(40.0)
TRUE = ("sub-p1d01", "sub-p2d01")


def _map(value):
    """One flat WTC map; a huge coi keeps every cell inside the cone."""
    return {"wtc": np.full((len(FREQS), len(TIMES)), float(value)),
            "coi": np.full(len(TIMES), 1e6), "sig": None,
            "phase": np.zeros((len(FREQS), len(TIMES)))}


@pytest.fixture
def stub(monkeypatch):
    """Stand in for the transform, so these test the bookkeeping rather than pycwt.

    Each draw returns a map whose value is the draw's index, so a summary that really
    averaged the draws is distinguishable from one that kept only the last.
    """
    seen = {"axes": [], "n": 0}

    def fake_signals(raw, ch_type="hbo", sep_bands=None):
        return {"S1_D1": np.zeros(10), "S1_D2": np.zeros(10)}

    def fake_pairs(raws, signals, *args, axis=None, **kwargs):
        seen["axes"].append(axis)
        value = 0.1 * (seen["n"] + 1)
        seen["n"] += 1
        # keyed by the partner, which is exactly what has to be rewritten
        key = tuple(raws)
        return WTCResult(pairs={key: {label: _map(value) for label in (axis or [])}},
                         freqs=FREQS, times=TIMES)

    monkeypatch.setattr(synchrony, "_long_signals", fake_signals)
    monkeypatch.setattr(synchrony, "_wtc_over_pairs", fake_pairs)
    return seen


class _Seg:
    """Stands in for a cropped Raw: the loop only asks it how long it is."""

    def __init__(self, span=39.0):
        self.times = np.array([0.0, span])


def _draws(partners, labels=("early",), span=39.0):
    """One entry per (partner, condition), which is what drawing per condition yields.

    The fourth item is where the condition sits inside the segment: a real draw pads either
    side, and these stubs carry no pad, so it spans the whole of it.
    """
    return [(p, label, {TRUE[0]: _Seg(span), p: _Seg(span)}, (0.0, span))
            for p in partners for label in labels]


def _run(partners, axis=("S1_D1", "S1_D2"), labels=("early",), **kwargs):
    return compute_wtc_pair_null(_draws(partners, labels), TRUE, list(axis),
                                 0.02, 0.30, **kwargs)


# ---- the pair key ----

def test_every_draw_lands_on_the_real_dyads_row(stub):
    null = _run(["sub-p2d02", "sub-p2d03", "sub-p2d04"])
    _, by_cond = null.summarise()
    assert set(by_cond["sub1"]) == {TRUE[0]}
    assert set(by_cond["sub2"]) == {TRUE[1]}


def test_the_draws_are_averaged_rather_than_kept_apart(stub):
    """Three partners give one row per channel with n_iter 3, not three rows of one."""
    null = _run(["sub-p2d02", "sub-p2d03", "sub-p2d04"])
    _, by_cond = null.summarise()
    assert len(by_cond) == 2                     # one row per channel on the axis
    assert set(by_cond["n_iter"]) == {3}
    # 0.1, 0.2, 0.3 really averaged, not the last draw standing alone
    assert by_cond["null_mean"].tolist() == pytest.approx([0.2, 0.2])


def test_a_partners_own_id_never_reaches_the_table(stub):
    null = _run(["sub-p2d02"])
    _, by_cond = null.summarise()
    assert "sub-p2d02" not in set(by_cond["sub1"]) | set(by_cond["sub2"])


def test_who_each_draw_was_against_is_kept(stub):
    """The pool is finite and named, so the sidecar can say which recordings it used."""
    null = _run(["sub-p2d02", "sub-p2d03"])
    assert null.partners == ["sub-p2d02", "sub-p2d03"]


# ---- the axis ----

def test_the_real_dyads_axis_is_used_for_every_draw(stub):
    """Recomputing it per draw would move the rows a stand-in's montage disagrees on."""
    _run(["sub-p2d02", "sub-p2d03"], axis=("S1_D1", "S1_D2"))
    assert stub["axes"] == [["S1_D1", "S1_D2"], ["S1_D1", "S1_D2"]]


def test_a_label_no_stand_in_carries_still_gets_a_row(stub):
    """A blank row keeps the null subtractable from the real table row by row."""
    null = _run(["sub-p2d02"], axis=("S1_D1", "S1_D2", "S9_D9"))
    _, by_cond = null.summarise()
    assert sorted(by_cond["label"]) == ["S1_D1", "S1_D2", "S9_D9"]


# ---- refusals ----

def test_no_usable_partner_is_an_error_rather_than_an_empty_table():
    with pytest.raises(ValueError, match="no usable partner"):
        compute_wtc_pair_null([], TRUE, ["S1_D1"], 0.02, 0.30)


# ---- the levels ----

def test_a_per_frequency_level_is_drawn_for_the_real_pair(stub):
    null = _run(["sub-p2d02", "sub-p2d03"])
    assert set(null.levels) == {(TRUE[0], TRUE[1], "S1_D1"), (TRUE[0], TRUE[1], "S1_D2")}
    assert all(level.shape == FREQS.shape for level in null.levels.values())


# ---- the conditions ----

def test_each_condition_gets_its_own_draw(stub):
    """Each condition is its own pair of segments, cut at each side's own marker.

    They used to be windowed out of one whole-record draw, which needed both sessions on
    one timetable. They drift, so the window that held one dyad's conversation held part of
    another's game, and the null was of that overlap rather than of the condition.
    """
    null = _run(["sub-p2d02", "sub-p2d03"], labels=("early", "late"))
    _, by_cond = null.summarise()
    assert sorted(set(by_cond["condition"])) == ["early", "late"]
    assert set(by_cond["n_iter"]) == {2}


def test_no_whole_run_draw_is_collected(stub):
    """There is no stretch standing in for the whole session, so that half stays empty."""
    null = _run(["sub-p2d02"], labels=("early", "late"))
    whole, by_cond = null.summarise()
    assert whole is None
    assert by_cond is not None and len(by_cond)


# ---- the draw hook ----
# Making a draw is the expensive half, so any other metric over the same re-paired pool has
# to ride along on this loop rather than run a second one. The ISC null is the first caller.

def test_every_draw_reaches_the_hook(stub):
    seen = []
    _run(["sub-p1d03", "sub-p1d04"],
         on_draw=lambda pid, label, pair, inner: seen.append(pid))
    assert seen == ["sub-p1d03", "sub-p1d04"]


def test_the_hook_gets_the_condition_and_the_pair(stub):
    got = []
    _run(["sub-p1d03"], labels=("early", "late"),
         on_draw=lambda pid, label, pair, inner: got.append((label, sorted(pair))))
    assert got == [("early", sorted([TRUE[0], "sub-p1d03"])),
                   ("late", sorted([TRUE[0], "sub-p1d03"]))]


# ---- a null with no whole-run draw ----

def test_the_roi_summary_survives_a_null_that_has_no_whole_run_draw(stub):
    """The re-paired null is per condition only, so its whole-run list is empty by design."""
    from fnirs_pipe.pipeline.synchrony import roi_mean_of_homologous  # noqa: F401

    null = _run(["sub-p2d02", "sub-p2d03"], labels=("early", "late"))
    whole, by_cond = null.summarise_roi({"front": ["S1_D1", "S1_D2"]}, min_channels=2)
    assert whole is None
    assert by_cond is not None and set(by_cond["condition"]) == {"early", "late"}


# ---- the draws behind each metric ----

def test_the_correlation_keeps_its_draws_too(tmp_path):
    """Both metrics ride the same re-paired draws, so both have to be readable above the cell."""
    import pandas as pd
    from fnirs_pipe.pipeline.pair_null import _write_isc_null

    rows = [{"chromophore": "hbo", "condition": "early", "sub1": TRUE[0], "sub2": TRUE[1],
             "label": "S1_D1", "label2": "S1_D1", "coherence": 0.2 + 0.01 * i,
             "n_valid_frac": 1.0, "draw": f"sub-p2d{i:02d}"} for i in range(3)]
    draws = [pd.DataFrame([r]) for r in rows]
    _write_isc_null([], draws, draws, tmp_path, "group-d01_task-full_hyper", [], {}, [],
                    isc_whiten=32, isc_max_lag_s=2.0, isc_band=(0.06, 0.15))

    out = tmp_path / "group-d01_task-full_hyper-iscbycond-pairnull-draws.tsv"
    assert out.exists() and out.with_suffix(".json").exists()
    assert sorted(pd.read_csv(out, sep="\t").draw) == [f"sub-p2d{i:02d}" for i in range(3)]
