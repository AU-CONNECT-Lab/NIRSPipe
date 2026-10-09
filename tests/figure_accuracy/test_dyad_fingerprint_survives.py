"""The dyad fingerprint reaches every table the group pages draw from, so a figure that loses it is the figure's fault."""

import numpy as np
import pytest

from tests._dyad_fingerprint import own_freq
from tests.figure_accuracy._dyad import BAND, wtc_tables
from tests.figure_accuracy._read import share_at

MARGIN = 0.15                        # a planted cell over the best unplanted one in its scope


def _scopes(groups, c):
    """The table scopes a coupling is present in: the run when it spans it, and its blocks."""
    blocks = [name for name, _, _ in groups.truth.blocks]
    if c.block is None:
        return ["", *blocks]
    return [c.block]


def test_each_member_rejects_its_own_pair_and_nothing_else(groups):
    for m in groups.truth.members:
        bad = {ch.rsplit(" ", 1)[0] for ch in groups.sidecar(m.sid, "sci")["bad_channels"]}
        assert bad == {m.rejected}, m.sid


def test_each_long_pair_is_still_identified_by_its_own_frequency(groups):
    truth = groups.truth
    for group in ("G01", "G02"):
        members = truth.group(group)
        candidates = {(m.subject, k): own_freq(m.subject, k)
                      for m in members for k in range(len(truth.long_pairs))}
        for m in members:
            raw = groups.read(m.sid, "preproc")
            for k, pair in enumerate(truth.long_pairs):
                if pair == m.rejected:
                    continue
                y = raw.get_data(picks=[f"{pair} hbo"])[0]
                scores = {key: share_at(raw.times, y, f) for key, f in candidates.items()}
                assert max(scores, key=scores.get) == (m.subject, k), (m.sid, pair)


def test_the_group_record_carries_each_member_s_planted_offset(groups):
    for group in ("G01", "G02"):
        offsets = groups.record(group)["alignment"]["align_offset_s"]
        for m in groups.truth.group(group):
            assert offsets[m.sid] == pytest.approx(m.offset, abs=0.1), m.sid


@pytest.mark.parametrize("group", ["G01", "G02"])
def test_every_planted_coupling_stands_above_every_other_cell_of_its_pairing(groups, group):
    table = wtc_tables(groups, group)
    planted = {}
    for c in groups.truth.couplings:
        if groups.truth.member(c.a).group != group:
            continue
        for scope in _scopes(groups, c):
            planted.setdefault((f"sub-{c.a}", f"sub-{c.b}", c.chroma, scope), []).append(c)
    assert planted
    for (s1, s2, chroma, scope), cs in planted.items():
        rows = table[(table.sub1 == s1) & (table.sub2 == s2) & (table.chromophore == chroma)
                     & (table.condition == scope)]
        hot = {(c.a_pair, c.b_pair) for c in cs}
        is_hot = [(r.label, r.label2) in hot for r in rows.itertuples()]
        coupled = rows[is_hot].coherence
        others = rows[[not h for h in is_hot]].coherence
        assert len(coupled) == len(hot)
        assert coupled.min() > others.max() + MARGIN, (s1, s2, chroma, scope)


@pytest.mark.parametrize("group", ["G01", "G02"])
def test_each_coupling_s_phase_at_every_scale_is_its_lag(groups, group):
    phase = groups.table(group, "stat-wtcphase_relmat.tsv")
    for c in groups.truth.couplings:
        if groups.truth.member(c.a).group != group or c.block is not None:
            continue
        rows = phase[(phase.sub1 == f"sub-{c.a}") & (phase.sub2 == f"sub-{c.b}")
                     & (phase.chromophore == c.chroma) & (phase.label == c.a_pair)
                     & (phase.label2 == c.b_pair)
                     & phase.freq.between(BAND[0] + 0.01, BAND[1] - 0.01)]
        assert len(rows) > 5
        err = (rows.phase_angle - [c.phase_deg(f) for f in rows.freq] + 180) % 360 - 180
        assert np.abs(err).max() < 25, (c, err.round(1).tolist())
