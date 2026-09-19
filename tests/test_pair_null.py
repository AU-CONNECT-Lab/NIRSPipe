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


def _draws(partners):
    return [(p, {TRUE[0]: object(), p: object()}) for p in partners]


def _run(partners, axis=("S1_D1", "S1_D2"), **kwargs):
    return compute_wtc_pair_null(_draws(partners), TRUE, list(axis), 0.02, 0.30, **kwargs)


# ---- the pair key ----

def test_every_draw_lands_on_the_real_dyads_row(stub):
    null = _run(["sub-p2d02", "sub-p2d03", "sub-p2d04"])
    whole, _ = null.summarise()
    assert set(whole["sub1"]) == {TRUE[0]}
    assert set(whole["sub2"]) == {TRUE[1]}


def test_the_draws_are_averaged_rather_than_kept_apart(stub):
    """Three partners give one row per channel with n_iter 3, not three rows of one."""
    null = _run(["sub-p2d02", "sub-p2d03", "sub-p2d04"])
    whole, _ = null.summarise()
    assert len(whole) == 2                       # one row per channel on the axis
    assert set(whole["n_iter"]) == {3}
    # 0.1, 0.2, 0.3 really averaged, not the last draw standing alone
    assert whole["null_mean"].tolist() == pytest.approx([0.2, 0.2])


def test_a_partners_own_id_never_reaches_the_table(stub):
    null = _run(["sub-p2d02"])
    whole, _ = null.summarise()
    assert "sub-p2d02" not in set(whole["sub1"]) | set(whole["sub2"])


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
    whole, _ = null.summarise()
    assert sorted(whole["label"]) == ["S1_D1", "S1_D2", "S9_D9"]


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

def test_each_condition_is_cut_from_the_same_draw(stub):
    null = _run(["sub-p2d02", "sub-p2d03"],
                windows=[("early", 0.0, 19.0), ("late", 20.0, 39.0)])
    _, by_cond = null.summarise()
    assert sorted(set(by_cond["condition"])) == ["early", "late"]
    assert set(by_cond["n_iter"]) == {2}


# ---- the draw hook ----
# Making a draw is the expensive half, so any other metric over the same re-paired pool has
# to ride along on this loop rather than run a second one. The ISC null is the first caller.

def test_every_draw_reaches_the_hook(stub):
    seen = []
    _run(["sub-p1d03", "sub-p1d04"], on_draw=lambda pid, aligned: seen.append(pid))
    assert seen == ["sub-p1d03", "sub-p1d04"]


def test_the_hook_gets_the_aligned_pair_not_just_the_name(stub):
    got = {}
    _run(["sub-p1d03"], on_draw=lambda pid, aligned: got.update(aligned))
    assert set(got) == {TRUE[0], "sub-p1d03"}
