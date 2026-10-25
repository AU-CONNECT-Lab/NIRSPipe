"""HbT through the dyad commands: added at load, picked by name, and the sum of the pair everywhere.

The members' HbO carries a rhythm they share and their HbR carries the same rhythm with its
sign flipped, so the rhythm cancels in HbT. A pass that read HbO in place of HbT reports the
rhythm as coherence; one that read the real HbT reports noise. The expected values for HbT
come from the pair summed here with numpy and run through the long-standing HbO route.
"""

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.io.groups import GroupEntry
from nirspipe.pipeline.chromophores import with_hbt
from nirspipe.pipeline.hyper._helpers import _long_by_label
from nirspipe.pipeline.hyper.isc import compute_isc
from nirspipe.pipeline.hyper.wtc import compute_wtc, wtc_band_mean
from nirspipe.qc.metrics._helpers import long_channel_picks

SFREQ, DURATION = 5.0, 400.0
BAND = (0.03, 0.10)
CONDITIONS = ("rest", "talk")


def _member(seed: int, shared: np.ndarray, separations=(30, 30, 30)) -> mne.io.Raw:
    """HbO = shared + 0.3 noise and HbR = 0.9 noise - shared, one pair per separation in mm."""
    rng = np.random.default_rng(seed)
    labels = [f"S{i}_D{i}" for i in range(1, len(separations) + 1)]
    names = [f"{label} {c}" for label in labels for c in ("hbo", "hbr")]
    info = mne.create_info(names, SFREQ, [n.split()[1] for n in names])
    for i, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        loc[3:6] = [i * 0.05, 0.0, 0.0]
        loc[6:9] = [i * 0.05 + separations[i // 2] / 1000, 0.0, 0.0]
        loc[:3] = (loc[3:6] + loc[6:9]) / 2
        ch["loc"] = loc
    data = np.empty((len(names), len(shared)))
    for i, name in enumerate(names):
        noise = rng.standard_normal(len(shared))
        data[i] = 1e-6 * (shared + 0.3 * noise if name.endswith("hbo")
                          else 0.9 * noise - shared)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations([20.0, 210.0], [180.0, 180.0], list(CONDITIONS)))
    return raw


def _shared() -> np.ndarray:
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    return np.sin(2 * np.pi * 0.05 * t)


def _summed_as_hbo(raw: mne.io.Raw) -> mne.io.Raw:
    """A copy whose HbO channels hold HbO + HbR, summed here rather than by the package."""
    out = raw.copy()
    for i, name in enumerate(out.ch_names):
        if name.endswith(" hbo"):
            out._data[i] = (raw.get_data(picks=[name])[0]
                            + raw.get_data(picks=[name.replace(" hbo", " hbr")])[0])
    return out


@pytest.fixture(scope="module")
def dyad():
    return {"sub-01": _member(1, _shared()), "sub-02": _member(2, _shared())}


# ---- the picks ----

def test_hbo_and_hbr_picks_are_mne_s_and_hbt_its_own(dyad):
    """By name, the HbO and HbR picks are what MNE's type picking gave before HbT existed;
    HbT, typed hbo, would be picked as HbO by type."""
    raw = _member(3, _shared(), separations=(30, 30, 8))
    raw.info["bads"] = ["S2_D2 hbr"]
    long_names = {f"S{i}_D{i} {c}" for i in (1, 2) for c in ("hbo", "hbr")}
    both = with_hbt(raw, ["hbt"])
    for c in ("hbo", "hbr"):
        by_type = [raw.ch_names[p] for p in mne.pick_types(raw.info, fnirs=c)
                   if raw.ch_names[p] in long_names]
        assert [both.ch_names[p] for p in long_channel_picks(both, c)] == by_type
    assert [both.ch_names[p] for p in long_channel_picks(both, "hbt")] == ["S1_D1 hbt"]
    assert [both.ch_names[p] for p in long_channel_picks(both, "hbt", exclude=[])] == [
        "S1_D1 hbt", "S2_D2 hbt"]


# ---- the numbers are the pair's sum ----

def _band(raws, ch_type):
    return wtc_band_mean(compute_wtc(raws, fmin=0.02, fmax=0.2, ch_type=ch_type), *BAND)


def test_hbt_coherence_is_that_of_the_summed_pair(dyad):
    with_t = {sid: with_hbt(raw, ["hbt"]) for sid, raw in dyad.items()}
    hbt = _band(with_t, "hbt").set_index("label")["coherence"]
    summed = _band({sid: _summed_as_hbo(raw) for sid, raw in dyad.items()},
                   "hbo").set_index("label")["coherence"]
    pd.testing.assert_series_equal(hbt, summed, check_exact=False, rtol=0, atol=1e-12)
    # the planted rhythm cancels in HbT, so a pass reading HbO would stand out
    assert _band(with_t, "hbo")["coherence"].min() > 0.7
    assert hbt.max() < 0.5


def test_hbt_isc_is_numpy_s_correlation_of_the_summed_pair(dyad):
    with_t = {sid: with_hbt(raw, ["hbt"]) for sid, raw in dyad.items()}
    mat, names = compute_isc(with_t, ["sub-01", "sub-02"], "hbt")
    for i, label in enumerate(names):
        a, b = (_summed_as_hbo(dyad[s]).get_data(picks=[f"{label} hbo"])[0]
                for s in ("sub-01", "sub-02"))
        assert mat[i, i] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-12)


# ---- whitening ----

def test_whitening_takes_hbt_and_leaves_hbo_and_hbr_as_they_were():
    from scipy.signal import lfilter

    from nirspipe.pipeline.hyper.whiten import whiten_raws

    def ar1(seed):
        return lfilter([1.0], [1.0, -0.95], np.random.default_rng(seed).standard_normal(2000))

    raws = {}
    for k, sid in enumerate(("sub-01", "sub-02")):
        raw = _member(10 + k, np.zeros(2000))
        raw._data[:] = np.array([ar1(100 * k + i) for i in range(raw.info["nchan"])]) * 1e-6
        raws[sid] = raw
    plain = whiten_raws(raws, 2.0)
    white = whiten_raws({sid: with_hbt(raw, ["hbt"]) for sid, raw in raws.items()}, 2.0)

    def lag1(x):
        return np.corrcoef(x[20:-1], x[21:])[0, 1]

    for sid, raw in white.items():
        assert np.array_equal(raw.get_data(picks=plain[sid].ch_names), plain[sid].get_data())
        before = with_hbt(raws[sid], ["hbt"]).get_data(picks=["S1_D1 hbt"])[0]
        assert lag1(before) > 0.8
        assert abs(lag1(raw.get_data(picks=["S1_D1 hbt"])[0])) < 0.1


# ---- the load ----

def test_hbt_is_added_after_the_rejections_and_before_the_z_score(dyad, monkeypatch):
    """A pair rejected at one wavelength is rejected in HbT, and the z-score is of the sum:
    the sum of z-scored halves would differ, HbR being larger than HbO here."""
    from nirspipe.cli.hyper import _load_aligned_group
    from nirspipe.pipeline.hyper import group_io, group_quality

    monkeypatch.setattr(group_io, "load_group_haemo", lambda out, entries, desc="preproc": {
        e.subject_id: dyad[e.subject_id].copy() for e in entries})
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {
        "sub-01": {"bad_channels": ["S2_D2 850"]}, "sub-02": {}})
    members = [GroupEntry("G1", s, "tap") for s in ("sub-01", "sub-02")]
    aligned, _, _ = _load_aligned_group(None, members, "tap", "preproc", no_align=True,
                                        normalize=True, bads_scope="run", chroma=("hbo", "hbt"))

    assert {"S2_D2 hbo", "S2_D2 hbr", "S2_D2 hbt"} <= set(aligned["sub-01"].info["bads"])
    assert not {"S2_D2 hbt"} & set(aligned["sub-02"].info["bads"])
    hbt = aligned["sub-01"].get_data(picks=["S1_D1 hbt"])[0]
    summed = _summed_as_hbo(dyad["sub-01"]).get_data(picks=["S1_D1 hbo"])[0]
    assert np.corrcoef(hbt, summed)[0, 1] == pytest.approx(1.0, abs=1e-12)
    assert (hbt.mean(), hbt.std()) == pytest.approx((0.0, 1.0), abs=1e-9)
    halves = aligned["sub-01"].get_data(picks=["S1_D1 hbo", "S1_D1 hbr"]).sum(axis=0)
    assert np.corrcoef(halves, summed)[0, 1] < 0.999


def test_no_hbt_is_loaded_unless_asked(dyad, monkeypatch):
    from nirspipe.cli.hyper import _load_aligned_group
    from nirspipe.pipeline.hyper import group_io, group_quality

    monkeypatch.setattr(group_io, "load_group_haemo", lambda out, entries, desc="preproc": {
        e.subject_id: dyad[e.subject_id].copy() for e in entries})
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    members = [GroupEntry("G1", s, "tap") for s in ("sub-01", "sub-02")]
    aligned, _, _ = _load_aligned_group(None, members, "tap", "preproc", no_align=True,
                                        normalize=False, bads_scope="run", chroma=("hbo", "hbr"))
    assert all(raw.ch_names == dyad[sid].ch_names for sid, raw in aligned.items())


# ---- the re-paired null ----

TARGET = ("sub-01", "sub-02")
DONORS = {"G2": ("sub-03", "sub-04"), "G3": ("sub-05", "sub-06")}


def _draw_null(root, raws, chroma, monkeypatch, with_t: bool):
    """The target's real tables, then its re-paired null over the two donor dyads."""
    from nirspipe.pipeline.hyper import alignment, group_io, group_quality
    from nirspipe.pipeline.hyper.pair_null import run_pair_null
    from nirspipe.qc.hyper.hyper_report import build_hyper_post_report

    def entries(gid, subs):
        return [GroupEntry(gid, s, "tap") for s in subs]

    target = {s: with_hbt(raws[s], ["hbt"] if with_t else []) for s in TARGET}
    build_hyper_post_report(
        group_id="G1", task="tap", group=entries("G1", TARGET), aligned_raws=target,
        offsets={s: 0.0 for s in TARGET}, output_dir=root, wtc_fmin=0.02, wtc_fmax=0.2,
        wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1], wtc_chroma=chroma,
        wtc_by_condition=True, no_report=True, desc="preproc", bads_scope="run")
    # the loaders hand back the files' HbO and HbR only, as on disk
    monkeypatch.setattr(group_io, "load_group_haemo", lambda out, ents, desc="preproc": {
        e.subject_id: raws[e.subject_id].copy() for e in ents})
    monkeypatch.setattr(group_quality, "load_group_sqm", lambda *a, **k: {})
    monkeypatch.setattr(group_quality, "apply_group_bads", lambda *a, **k: None)
    monkeypatch.setattr(alignment, "align_recordings",
                        lambda raws, task: (raws, {s: 0.0 for s in raws}))
    groups = {("G1", "tap", None): entries("G1", TARGET),
              **{(gid, "tap", None): entries(gid, subs) for gid, subs in DONORS.items()}}
    run_pair_null("G1", "tap", entries("G1", TARGET), groups, root, root, chroma=chroma)


def _draws(root, statistic, chroma):
    path = next(root.rglob(f"*null-pair_stat-{statistic}_desc-draws_relmat.tsv"))
    table = pd.read_csv(path, sep="\t")
    keys = [k for k in ("sub1", "sub2", "condition", "label", "label2", "draw")
            if k in table.columns]
    return table[table["chromophore"] == chroma].sort_values(keys).set_index(keys)


@pytest.mark.parametrize("statistic", ["wtc", "isc"])
def test_the_re_paired_hbt_null_is_the_hbo_null_of_the_summed_pairs(
        tmp_path, monkeypatch, statistic):
    """Each stand-in, loaded with HbO and HbR only, gets its HbT added before it is drawn."""
    raws = {f"sub-{k:02d}": _member(k, _shared() if k <= 2 else
                                     np.random.default_rng(50 + k).standard_normal(2000))
            for k in range(1, 7)}
    _draw_null(tmp_path / "hbt", raws, ("hbt",), monkeypatch, with_t=True)
    _draw_null(tmp_path / "sum", {s: _summed_as_hbo(r) for s, r in raws.items()}, ("hbo",),
               monkeypatch, with_t=False)

    hbt = _draws(tmp_path / "hbt", statistic, "hbt")["coherence"]
    summed = _draws(tmp_path / "sum", statistic, "hbo")["coherence"]
    assert len(hbt) and hbt.notna().any()
    pd.testing.assert_series_equal(hbt, summed, check_exact=False, rtol=0, atol=1e-12)


# ---- the cohort test ----

def test_the_cohort_test_reads_hbt():
    from nirspipe.cli.hyper import _parsers

    args = _parsers()["nirspipe-hyper-groupnull"].parse_args(
        ["out", "group", "--task", "tap", "--chroma", "hbo", "hbt"])
    assert args.chroma == ["hbo", "hbt"]


# ---- the command, end to end ----

@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One synthetic dyad through `nirspipe`, then `nirspipe-hyper` with and without HbT."""
    from nirspipe.cli import hyper as hyper_cli
    from nirspipe.cli import run as run_cli
    from tests._fingerprint import CLI_ARGS
    from tests._synth import make_hyper_dataset

    root = tmp_path_factory.mktemp("hbt_hyper")
    bids, pairs = make_hyper_dataset(root, tasks=("hold",))
    deriv = root / "deriv"
    run_cli.main([str(bids), str(deriv), "participant", *CLI_ARGS, "--skip-bids-validation"])

    def run(name, *chroma):
        out = root / name
        hyper_cli.main([str(deriv), str(out), "group", "--pairs-csv", str(pairs),
                        "--wtc-fmin", "0.02", "--band-fmin", "0.03", "--band-fmax", "0.12",
                        "--no-channel-cross", "--wtc-chroma", *chroma])
        return out

    return {"hbo": run("hbo", "hbo"), "hbt": run("hbt", "hbt", "hbo")}


def _whole_run(out):
    path = next(p for p in out.rglob("group-*_stat-wtc_relmat.tsv")
                if "_cond-" not in p.name and "_seg-" not in p.name)
    return pd.read_csv(path, sep="\t"), path


def test_the_command_writes_hbt_beside_an_unchanged_hbo(runs):
    with_t, path = _whole_run(runs["hbt"])
    alone, _ = _whole_run(runs["hbo"])
    assert list(with_t["chromophore"].unique()) == ["hbo", "hbt"]
    # a pair rejected in either member is blank in both chromophores, and only such a pair
    blank = {c: set(part.loc[part["coherence"].isna(), "label"])
             for c, part in with_t.groupby("chromophore")}
    assert blank["hbt"] == blank["hbo"] and len(blank["hbt"]) < (with_t["chromophore"] == "hbt").sum()
    pd.testing.assert_frame_equal(
        with_t[with_t["chromophore"] == "hbo"].reset_index(drop=True), alone)
    import json
    assert json.loads(path.with_suffix(".json").read_text())["parameters"]["chroma"] == [
        "hbo", "hbt"]
    assert any(runs["hbt"].rglob("*_chromo-hbt_stat-isc_relmat.tsv"))


def test_the_dyad_page_carries_hbt(runs):
    page = next(p for p in runs["hbt"].rglob("group-*.html")
                if "index" not in p.name and "_cond-" not in p.name).read_text(encoding="utf-8")
    assert '<span class="chroma-name">HbT</span>' in page
    assert 'id="ISCHbt"' in page and 'id="ISCHbo"' in page
    assert 'id="ISCHbr"' not in page
