"""The phase-scrambled nulls of a group of more than two members.

Each pairing keeps its earlier member real and has its later member replaced, and within one
draw a member is replaced once, its surrogate serving every pairing it closes. The tests
build members that are copies of one another, so what a draw is made of shows directly: a
pairing of two real copies has coherence 1, two pairings that share a real copy and a
surrogate are equal draw for draw, and two independent surrogates of one copy are not.
"""

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.pipeline.hyper._helpers import long_axis_over
from nirspipe.pipeline.hyper.isc import compute_isc_pairs
from nirspipe.pipeline.hyper.surrogate import compute_wtc_phase_null

SFREQ = 5.0
N = 1500
BAND = (0.03, 0.10)


def _raw(signals: dict, seed: int = 0) -> mne.io.Raw:
    """A haemoglobin Raw with one HbO and one HbR channel per label in ``signals``."""
    rng = np.random.default_rng(seed)
    names, types, data, montage = [], [], [], {}
    for label, sig in signals.items():
        for chroma in ("hbo", "hbr"):
            names.append(f"{label} {chroma}")
            types.append(chroma)
            data.append((sig if chroma == "hbo" else rng.standard_normal(N)) * 1e-6)
        src, det = label.split("_")
        y = int(src[1:]) * 0.03
        montage[src] = np.array([0.0, y, 0.0])
        montage[det] = np.array([0.03, y, 0.0])
    raw = mne.io.RawArray(np.asarray(data), mne.create_info(names, SFREQ, types),
                          verbose=False)
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=montage, coord_frame="head"),
                    on_missing="ignore", verbose=False)
    return raw


def _signals(labels, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    return {label: rng.standard_normal(N) for label in labels}


def _copies(*ids, labels=("S1_D1", "S2_D2")) -> dict:
    raw = _raw(_signals(labels))
    return {sid: raw.copy() for sid in ids}


def _null(raws, n_iter=4, **kwargs):
    return compute_wtc_phase_null(raws, *BAND, n_iter=n_iter, fmin=0.02, fmax=0.2, seed=5,
                                  **kwargs)


def _draws(null, pairing) -> np.ndarray:
    """The pairing's band-mean coherence, one row per iteration and one column per channel."""
    return np.array([f[(f["sub1"] == pairing[0]) & (f["sub2"] == pairing[1])]
                     .sort_values("label")["coherence"].to_numpy() for f in null.draws])


# ---- the coherence's phase null ----

def test_every_pairing_of_three_copies_is_drawn_against_a_surrogate():
    """Three copies of one recording cohere at 1 in every pairing. A pairing whose two
    sides were both left real would keep that 1 in its null."""
    raws = _copies("a", "b", "c")
    summary, _ = _null(raws).summarise()

    assert sorted(set(zip(summary["sub1"], summary["sub2"]))) == [("a", "b"), ("a", "c"),
                                                                  ("b", "c")]
    assert (summary["n_iter"] == 4).all()
    assert (summary["null_mean"] < 0.8).all()


def test_the_later_member_is_replaced_once_for_every_pairing_it_closes():
    """a and b are copies, so a-c and b-c hold the same real side against c's surrogate
    and are equal draw for draw; a-b drew b's own surrogate and is not."""
    raws = _copies("a", "b")
    raws["c"] = _raw(_signals(("S1_D1", "S2_D2"), seed=2))
    null = _null(raws)

    assert np.allclose(_draws(null, ("a", "c")), _draws(null, ("b", "c")), rtol=0, atol=1e-12)
    assert not np.allclose(_draws(null, ("a", "b")), _draws(null, ("a", "c")))


def test_a_condition_draw_holds_every_pairing_under_one_id():
    """One id per iteration, shared by the pairings, which is what a group mean is
    matched on."""
    null = _null(_copies("a", "b", "c"), windows=[("first", 0.0, 150.0)])

    assert null.cond_draw_ids == [f"iter-{i:04d}" for i in range(4)]
    for frame in null.cond_draws:
        assert set(zip(frame["sub1"], frame["sub2"])) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_a_dyad_is_one_pairing_against_its_second_member():
    raws = _copies("a", "b")
    summary, _ = _null(raws).summarise()

    assert set(zip(summary["sub1"], summary["sub2"])) == {("a", "b")}
    assert (summary["null_mean"] < 0.8).all()


# ---- the correlation's phase null ----

def _isc_draws(raws, ids, seed, draw_order=None) -> pd.DataFrame:
    got = []
    compute_isc_pairs(raws, list(ids), "hbo", n_null=5, seed=seed, draw_order=draw_order,
                      on_draws=got.append)
    return got[0].set_index(["label", "label2", "draw"])["r"].sort_index()


def test_one_stream_is_one_surrogate_whatever_the_partners_montage():
    """a lacks S1_D1, so a-c puts c's S1_D1 last on its axis and b-c first. Drawn in the
    group's order the two pairings scramble c alike; drawn in their own, they do not."""
    full = _signals(("S1_D1", "S2_D2", "S3_D3"))
    raws = {"a": _raw({k: full[k] for k in ("S2_D2", "S3_D3")}), "b": _raw(full),
            "c": _raw(_signals(("S1_D1", "S2_D2", "S3_D3"), seed=2))}
    group = long_axis_over(raws.values(), "hbo")
    shared_cells = lambda draws: draws.loc[["S2_D2", "S3_D3"]]

    in_group_order = [shared_cells(_isc_draws(raws, pair, [7, 2], group))
                      for pair in (("a", "c"), ("b", "c"))]
    in_own_order = [shared_cells(_isc_draws(raws, pair, [7, 2]))
                    for pair in (("a", "c"), ("b", "c"))]

    assert np.allclose(*in_group_order, rtol=0, atol=1e-12)
    assert not np.allclose(*in_own_order)


def _isc_null_rows(tmp_path, raws, seed) -> pd.DataFrame:
    """The per-pairing ISC null columns the group's report writes."""
    from nirspipe.io.groups import GroupEntry
    from nirspipe.qc.hyper.hyper_report import build_hyper_post_report

    build_hyper_post_report(
        group_id="G1", task="tap", group=[GroupEntry("G1", sid, "tap") for sid in raws],
        aligned_raws=raws, offsets={sid: 0.0 for sid in raws}, output_dir=tmp_path,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1],
        wtc_chroma=("hbo",), isc_phase_null=6, wtc_seed=seed, no_report=True)
    table = pd.read_csv(next(tmp_path.rglob("*_task-tap_stat-isc_relmat.tsv")), sep="\t")
    return table.set_index(["sub1", "sub2", "label", "label2"])[
        ["null_abs_mean", "null_abs_sd", "null_abs_p95"]]


@pytest.mark.parametrize("seed", [3, None])
def test_each_replaced_member_draws_its_own_stream_in_every_pairing(tmp_path, seed):
    """Three copies: a-c and b-c rank against one surrogate of c, seeded or not, while b's
    and c's surrogates in a-b and a-c come from streams of their own. On one stream the two
    would be the same surrogate, and a group mean's null twice as wide as it is."""
    rows = _isc_null_rows(tmp_path, _copies("sub-a", "sub-b", "sub-c"), seed)
    pairing = lambda a, b: rows.loc[(a, b)]

    pd.testing.assert_frame_equal(pairing("sub-a", "sub-c"), pairing("sub-b", "sub-c"))
    assert not np.allclose(pairing("sub-a", "sub-b"), pairing("sub-a", "sub-c"))
