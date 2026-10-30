"""The re-paired null: every draw has to land on the row the real table sits on.

Each draw is against a different person, and the summary groups by ``sub1``/``sub2``. Left
alone, that makes every draw its own group of one and the table reports ``n_iter`` 1 against
a null it never averaged, which reads exactly like a null that was averaged. The pair key is
rewritten to the real dyad's before anything is collected, and these hold that in place
along with the axis rule that lets a stand-in with a different montage be drawn at all.
"""

import numpy as np
import pytest

from nirspipe.exceptions import StageError
from nirspipe.pipeline.hyper import surrogate
from nirspipe.pipeline.hyper.surrogate import compute_wtc_pair_null
from nirspipe.pipeline.hyper.wtc import WTCResult
from tests.hyper._names import name

FREQS = np.linspace(0.02, 0.30, 12)
TIMES = np.arange(40.0)
TRUE = ("sub-01G01", "sub-02G01")


def _map(value):
    """One flat WTC map; a huge coi keeps every cell clear of the cone."""
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

    monkeypatch.setattr(surrogate, "_long_signals", fake_signals)
    monkeypatch.setattr(surrogate, "_wtc_over_pairs", fake_pairs)
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
    null = _run(["sub-02G02", "sub-02G03", "sub-02G04"])
    _, by_cond = null.summarise()
    assert set(by_cond["sub1"]) == {TRUE[0]}
    assert set(by_cond["sub2"]) == {TRUE[1]}


def test_the_draws_are_averaged_rather_than_kept_apart(stub):
    """Three partners give one row per channel with n_iter 3, not three rows of one."""
    null = _run(["sub-02G02", "sub-02G03", "sub-02G04"])
    _, by_cond = null.summarise()
    assert len(by_cond) == 2                     # one row per channel on the axis
    assert set(by_cond["n_iter"]) == {3}
    # 0.1, 0.2, 0.3 really averaged, not the last draw standing alone
    assert by_cond["null_mean"].tolist() == pytest.approx([0.2, 0.2])


def test_a_partners_own_id_never_reaches_the_table(stub):
    null = _run(["sub-02G02"])
    _, by_cond = null.summarise()
    assert "sub-02G02" not in set(by_cond["sub1"]) | set(by_cond["sub2"])


def test_who_each_draw_was_against_is_kept(stub):
    """The pool is finite and named, so the sidecar can say which recordings it used."""
    null = _run(["sub-02G02", "sub-02G03"])
    assert null.partners == ["sub-02G02", "sub-02G03"]


# ---- the axis ----

def test_the_real_dyads_axis_is_used_for_every_draw(stub):
    """Recomputing it per draw would move the rows a stand-in's montage disagrees on."""
    _run(["sub-02G02", "sub-02G03"], axis=("S1_D1", "S1_D2"))
    assert stub["axes"] == [["S1_D1", "S1_D2"], ["S1_D1", "S1_D2"]]


def test_a_label_no_stand_in_carries_still_gets_a_row(stub):
    """A blank row keeps the null subtractable from the real table row by row."""
    null = _run(["sub-02G02"], axis=("S1_D1", "S1_D2", "S9_D9"))
    _, by_cond = null.summarise()
    assert sorted(by_cond["label"]) == ["S1_D1", "S1_D2", "S9_D9"]


def test_a_stand_in_sharing_no_channel_is_a_blank_draw_not_a_crash(monkeypatch):
    """The transform hands back no map and no time axis for such a stand-in, as it does for
    every pairing one side lacks. Its rows stay blank, and the other draws are the null."""
    def fake_pairs(raws, signals, *args, axis=None, **kwargs):
        blank = "sub-02G03" in raws
        return WTCResult(pairs={tuple(raws): {label: None if blank else _map(0.4)
                                              for label in axis}},
                         freqs=np.array([]) if blank else FREQS,
                         times=np.array([]) if blank else TIMES)

    monkeypatch.setattr(surrogate, "_long_signals", lambda raw, *a, **k: {})
    monkeypatch.setattr(surrogate, "_wtc_over_pairs", fake_pairs)
    null = _run(["sub-02G02", "sub-02G03"])
    _, by_cond = null.summarise()
    assert by_cond["n_iter"].tolist() == [1, 1]
    assert by_cond["null_mean"].tolist() == pytest.approx([0.4, 0.4])
    assert null.partners == ["sub-02G02", "sub-02G03"]
    blank_rows = [f for f, d in zip(null.cond_draws, null.cond_draw_ids) if d == "sub-02G03"]
    assert len(blank_rows) == 1 and blank_rows[0]["coherence"].isna().all()


def _no_channel_for(who):
    def signals(raw, *a, **k):
        if raw is who:
            raise StageError("no usable long HBO channel: every one is either short-distance "
                             "or marked bad")
        return {"S1_D1": np.zeros(10)}
    return signals


def test_a_stand_in_with_no_usable_channel_is_a_blank_draw_too(monkeypatch):
    """Its signal map is empty, which the transform turns into no map at all, as above."""
    draws = _draws(["sub-02G02", "sub-02G03"])
    empty_partner = draws[1][2]["sub-02G03"]

    def fake_pairs(raws, signals, *args, axis=None, **kwargs):
        blank = not signals["sub-02G03" if "sub-02G03" in raws else "sub-02G02"]
        return WTCResult(pairs={tuple(raws): {label: None if blank else _map(0.4)
                                              for label in axis}},
                         freqs=np.array([]) if blank else FREQS,
                         times=np.array([]) if blank else TIMES)

    monkeypatch.setattr(surrogate, "_long_signals", _no_channel_for(empty_partner))
    monkeypatch.setattr(surrogate, "_wtc_over_pairs", fake_pairs)
    null = compute_wtc_pair_null(draws, TRUE, ["S1_D1", "S1_D2"], 0.02, 0.30)
    _, by_cond = null.summarise()
    assert by_cond["n_iter"].tolist() == [1, 1]
    assert null.partners == ["sub-02G02", "sub-02G03"]


def test_the_fixed_member_without_a_channel_is_still_refused(monkeypatch):
    draws = _draws(["sub-02G02"])
    monkeypatch.setattr(surrogate, "_long_signals", _no_channel_for(draws[0][2][TRUE[0]]))
    with pytest.raises(StageError, match="no usable long"):
        compute_wtc_pair_null(draws, TRUE, ["S1_D1"], 0.02, 0.30)


# ---- refusals ----

def test_no_usable_partner_is_an_error_rather_than_an_empty_table():
    with pytest.raises(ValueError, match="no usable partner"):
        compute_wtc_pair_null([], TRUE, ["S1_D1"], 0.02, 0.30)


# ---- the levels ----

def test_a_per_frequency_level_is_drawn_for_the_real_pair(stub):
    null = _run(["sub-02G02", "sub-02G03"])
    levels = null.cond_levels["early"]
    assert set(levels) == {(TRUE[0], TRUE[1], "S1_D1"), (TRUE[0], TRUE[1], "S1_D2")}
    assert all(level.shape == FREQS.shape for level in levels.values())


def test_each_condition_gets_its_own_level_and_the_whole_run_none(stub):
    """A re-paired draw is one condition, so pooling them would rank one condition's cells
    against another's, and there is no whole-run draw to count a whole-run level from."""
    null = _run(["sub-02G02", "sub-02G03"], labels=("early", "late"))
    assert null.levels == {}
    early = null.cond_levels["early"][(TRUE[0], TRUE[1], "S1_D1")]
    late = null.cond_levels["late"][(TRUE[0], TRUE[1], "S1_D1")]
    assert not np.allclose(early, late)


# ---- the conditions ----

def test_each_condition_gets_its_own_draw(stub):
    """Each condition is its own pair of segments, cut at each side's own marker.

    Not windowed out of one whole-record draw: that needs both sessions on one timetable,
    and two sessions drift apart, so the null would be of an overlap rather than of the
    condition.
    """
    null = _run(["sub-02G02", "sub-02G03"], labels=("early", "late"))
    _, by_cond = null.summarise()
    assert sorted(set(by_cond["condition"])) == ["early", "late"]
    assert set(by_cond["n_iter"]) == {2}


def test_no_whole_run_draw_is_collected(stub):
    """There is no stretch standing in for the whole session, so that half stays empty."""
    null = _run(["sub-02G02"], labels=("early", "late"))
    whole, by_cond = null.summarise()
    assert whole is None
    assert by_cond is not None and len(by_cond)


# ---- the draw hook ----
# Making a draw is the expensive half, so any other metric over the same re-paired pool has
# to ride along on this loop rather than run a second one. The ISC null is the first caller.

def test_every_draw_reaches_the_hook(stub):
    seen = []
    _run(["sub-01G03", "sub-01G04"],
         on_draw=lambda pid, label, pair, inner, excluded: seen.append(pid))
    assert seen == ["sub-01G03", "sub-01G04"]


def test_the_hook_gets_the_condition_and_the_pair(stub):
    got = []
    _run(["sub-01G03"], labels=("early", "late"),
         on_draw=lambda pid, label, pair, inner, excluded: got.append((label, sorted(pair))))
    assert got == [("early", sorted([TRUE[0], "sub-01G03"])),
                   ("late", sorted([TRUE[0], "sub-01G03"]))]


# ---- a null with no whole-run draw ----

def test_the_roi_summary_survives_a_null_that_has_no_whole_run_draw(stub):
    """The re-paired null is per condition only, so its whole-run list is empty by design."""
    from nirspipe.pipeline.hyper.roi import roi_mean_of_homologous  # noqa: F401  re-exported

    null = _run(["sub-02G02", "sub-02G03"], labels=("early", "late"))
    whole, by_cond = null.summarise_roi({"front": ["S1_D1", "S1_D2"]}, min_channels=2)
    assert whole is None
    assert by_cond is not None and set(by_cond["condition"]) == {"early", "late"}


# ---- a region's draws on the real pairing's channels ----

def _region_null(draw_cells, crossed=False):
    """A re-paired null of one condition over region r = a b c, one frame per draw; None is a rejected channel."""
    import numpy as np
    import pandas as pd
    from nirspipe.pipeline.hyper.surrogate import NullDraws

    def frame(cells):
        return pd.DataFrame([{"condition": "early", "sub1": TRUE[0], "sub2": TRUE[1], "label": c,
                              **({"label2": c} if crossed else {}),
                              "coherence": np.nan if v is None else v, "n_valid_frac": 1.0}
                             for c, v in cells.items()])
    return NullDraws(draws=[], cond_draws=[frame(c) for c in draw_cells],
                     keys=["sub1", "sub2", "label"], levels={}), frame


REGION = {"r": ["a", "b", "c"]}
DRAW_CELLS = [{"a": 0.35, "b": 0.25, "c": None},
              {"a": 0.30, "b": 0.30, "c": 0.30},
              {"a": None, "b": None, "c": 0.10}]


def _region_summary(min_channels=1, crossed=False, matched=True):
    import pandas as pd

    null, frame = _region_null(DRAW_CELLS, crossed=crossed)
    real_channels = frame({"a": 0.40, "b": 0.30, "c": 0.20})
    real_roi = pd.DataFrame([{"condition": "early", "sub1": TRUE[0], "sub2": TRUE[1], "label": "r",
                              **({"label2": "r"} if crossed else {}), "coherence": 0.30}])
    _, by_cond = null.summarise_roi(REGION, real_by_cond=real_roi, min_channels=min_channels,
                                    crossed=crossed,
                                    real_channels=real_channels if matched else None)
    return by_cond.iloc[0]


def test_a_region_draw_is_compared_on_the_channels_it_shares_with_the_real_pairing():
    # real .30 over a b c; the draws come out .25 (a b), .30 (all) and .20 (c alone)
    row = _region_summary()
    assert row.null_mean == pytest.approx(0.25)
    assert row.percentile == pytest.approx(200 / 3)
    assert row.n_iter == 3


def test_each_draw_over_its_own_channels_reads_otherwise():
    """What the matching changes: the third draw's lone low channel no longer stands for the region."""
    row = _region_summary(matched=False)
    assert row.null_mean == pytest.approx(0.70 / 3)
    assert row.percentile == pytest.approx(100 / 3)


def test_the_region_minimum_applies_to_the_shared_channels():
    row = _region_summary(min_channels=2)
    assert row.n_iter == 2
    assert row.null_mean == pytest.approx(0.275)


def test_a_crossed_region_matrix_is_matched_the_same_way():
    row = _region_summary(crossed=True)
    assert row.null_mean == pytest.approx(0.25)
    assert row.n_iter == 3


# ---- the draws behind each metric ----

def test_the_correlation_keeps_its_draws_too(tmp_path):
    """Both metrics ride the same re-paired draws, so both have to be readable above the cell."""
    import pandas as pd
    from nirspipe.pipeline.hyper.pair_null import _write_isc_null

    rows = [{"chromophore": "hbo", "condition": "early", "sub1": TRUE[0], "sub2": TRUE[1],
             "label": "S1_D1", "label2": "S1_D1", "coherence": 0.2 + 0.01 * i,
             "n_valid_frac": 1.0, "draw": f"sub-02G{i:02d}"} for i in range(3)]
    draws = [pd.DataFrame([r]) for r in rows]

    def _path(entities):
        return tmp_path / name("G01", "main", "iscpairs", **entities)

    _write_isc_null(draws, draws, _path, [], {}, [],
                    isc_whiten_s=10.0, isc_max_lag_s=2.0, isc_band=(0.06, 0.15))

    out = tmp_path / name("G01", "main", "iscbycond-pairnull-draws")
    assert out.exists() and out.with_suffix(".json").exists()
    assert sorted(pd.read_csv(out, sep="\t").draw) == [f"sub-02G{i:02d}" for i in range(3)]
