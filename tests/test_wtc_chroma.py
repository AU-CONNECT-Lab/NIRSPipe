"""WTC on both chromophores: two parallel passes, one table, and no mixing.

HbO and HbR are the same computation run twice. The properties worth pinning are all about
them staying apart: a member's HbO pairs only with the other member's HbO, the two are
never averaged, and a row on disk says which one it is. Averaging them would cancel, since
HbO and HbR anti-correlate by construction, so a bug here produces a plausible-looking
number about nothing.

The decisive test is the first one. It gives the HbO channels a rhythm the two members
share and the HbR channels independent noise, so a pass that reads the wrong chromophore
cannot come out looking right: it would report the noise as coherence, or the rhythm as
noise. Everything after it is bookkeeping.

Signals are built here rather than taken from `_synth`, whose Beer-Lambert output derives
both chromophores from one optical density and so cannot carry two different couplings.
"""

import json

import mne
import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.pipeline.synchrony import (
    _long_by_label,
    compute_pairwise_coherence,
    compute_wtc,
    wtc_band_mean,
)

SFREQ = 5.0
DURATION = 400.0
LABELS = ["S1_D1", "S2_D2", "S3_D3"]
BAND = (0.03, 0.10)
SHARED_FREQ = 0.05        # inside BAND, so the band mean sees it


def _dyad_raw(seed: int, shared: np.ndarray) -> mne.io.Raw:
    """A haemoglobin Raw whose HbO channels carry `shared` and whose HbR channels do not.

    Every channel is 30 mm, so nothing is dropped as short-distance and the chromophore is
    the only thing that separates the two halves of the montage.
    """
    rng = np.random.default_rng(seed)
    names = [f"{label} {c}" for label in LABELS for c in ("hbo", "hbr")]
    types = [c for _ in LABELS for c in ("hbo", "hbr")]
    info = mne.create_info(names, SFREQ, types)
    for i, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        loc[3:6] = [i * 0.05, 0.0, 0.0]              # source
        loc[6:9] = [i * 0.05 + 0.03, 0.0, 0.0]       # detector, 30 mm away
        loc[:3] = (loc[3:6] + loc[6:9]) / 2
        ch["loc"] = loc

    n = len(shared)
    data = np.empty((len(names), n))
    for i, ch_type in enumerate(types):
        noise = rng.standard_normal(n)
        data[i] = 1e-6 * (shared + 0.2 * noise if ch_type == "hbo" else noise)
    return mne.io.RawArray(data, info, verbose="ERROR")


@pytest.fixture(scope="module")
def dyad():
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    shared = np.sin(2 * np.pi * SHARED_FREQ * t)
    return {"sub-01": _dyad_raw(1, shared), "sub-02": _dyad_raw(2, shared)}


def _band_mean(raws, ch_type):
    result = compute_wtc(raws, fmin=0.02, fmax=0.2, ch_type=ch_type)
    return wtc_band_mean(result, *BAND)


# ---- the chromophore decides what is read ----

def test_the_pass_reads_the_chromophore_it_was_asked_for(dyad):
    """The HbO channels share a rhythm and the HbR channels do not, so a pass that read the
    wrong half would report the noise as coherence."""
    hbo = _band_mean(dyad, "hbo")["coherence"].mean()
    hbr = _band_mean(dyad, "hbr")["coherence"].mean()
    assert hbo > 0.7
    assert hbr < hbo


def test_the_picks_never_mix_the_two_chromophores(dyad):
    """A member's HbO pairs only with the other member's HbO. Position would not enforce
    that: HbO and HbR sit interleaved on the montage."""
    for ch_type in ("hbo", "hbr"):
        picks = _long_by_label(dyad["sub-01"], ch_type)
        assert set(picks) == set(LABELS)
        names = [dyad["sub-01"].ch_names[p] for p in picks.values()]
        assert all(name.endswith(f" {ch_type}") for name in names)


def test_the_two_passes_are_different_numbers(dyad):
    """Not a tautology: an implementation that ignored ch_type would pass every test above
    that only looks at one chromophore."""
    hbo = _band_mean(dyad, "hbo")
    hbr = _band_mean(dyad, "hbr")
    assert list(hbo["label"]) == list(hbr["label"])
    assert not np.allclose(hbo["coherence"], hbr["coherence"])


def test_the_error_names_the_chromophore_that_has_no_channel(dyad):
    raw = dyad["sub-01"].copy()
    raw.info["bads"] = [c for c in raw.ch_names if c.endswith(" hbr")]
    with pytest.raises(ValueError, match="no usable long HBR channel"):
        _long_by_label(raw, "hbr")


def test_the_screening_coherence_stays_hbo():
    """`fnirs-qc hyper-raw`'s coherence is a screening number rather than a result, and was
    deliberately left out of scope, so it takes no chromophore at all."""
    import inspect
    assert "ch_type" not in inspect.signature(compute_pairwise_coherence).parameters


# ---- what lands on disk ----

@pytest.fixture(scope="module")
def report(dyad, tmp_path_factory):
    """One report over both chromophores, with the maps archived and the channels crossed."""
    from fnirs_pipe.pipeline.hyperscanning import GroupEntry
    from fnirs_pipe.qc.hyper_report import build_hyper_post_report

    out = tmp_path_factory.mktemp("chroma")
    build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", "sub-01", "tap"), GroupEntry("G1", "sub-02", "tap")],
        aligned_raws=dyad, offsets={"sub-01": 0.0, "sub-02": 0.0}, output_dir=out,
        roi_map={"L": ["S1_D1", "S2_D2"], "R": ["S3_D3"]}, wtc_roi_min_channels=1,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1],
        wtc_channel_cross=True, wtc_save_maps=True, wtc_chroma=("hbo", "hbr"),
    )
    return out / "group-G1" / "nirs"


@pytest.mark.parametrize("kind", ["wtc", "wtc-roichan"])
def test_one_table_holds_both_chromophores(report, kind):
    """Not one file each: the band-mean tables are long-format, so a column keeps the
    filenames stable and a group analysis needs one more grouping key rather than a second
    read. ISC splits per chromophore instead, being a matrix."""
    df = pd.read_csv(report / f"group-G1_task-tap_hyper-{kind}.tsv", sep="\t")
    counts = df["chromophore"].value_counts().to_dict()
    assert set(counts) == {"hbo", "hbr"}
    assert counts["hbo"] == counts["hbr"]
    assert df["chromophore"].notna().all()


def test_the_chromophore_column_comes_first(report):
    """It is the coarsest grouping key in the table, and it survives the ROI aggregation
    only by being re-inserted after it."""
    df = pd.read_csv(report / "group-G1_task-tap_hyper-wtc.tsv", sep="\t")
    assert df.columns[0] == "chromophore"


def test_the_sidecar_records_which_chromophores_ran(report):
    params = json.loads((report / "group-G1_task-tap_hyper-wtc.json").read_text())
    assert params["parameters"]["chroma"] == ["hbo", "hbr"]


def test_the_maps_are_archived_one_file_per_chromophore(report):
    """The table holds both, so the archive cannot be named after it: two would collide."""
    archives = sorted(p.name for p in report.glob("*_hyper-wtc-hb*.npz"))
    assert archives == ["group-G1_task-tap_hyper-wtc-hbo.npz",
                        "group-G1_task-tap_hyper-wtc-hbr.npz"]


def test_a_reband_puts_the_chromophore_column_back(report):
    """The archive is per chromophore and says so in its name, so a re-banded table has the
    same shape as the one the run wrote."""
    from fnirs_pipe.qc.wtc_store import reband_tree

    written = reband_tree(report.parent.parent, 0.04, 0.09)
    seen = {}
    for path in written:
        df = pd.read_csv(path, sep="\t")
        seen[path.name] = df["chromophore"].unique().tolist()
    assert sorted(v for vals in seen.values() for v in vals) == ["hbo", "hbr"]


def test_the_isc_tables_stay_one_file_each(report):
    """ISC is a channel-by-channel matrix, and two cannot share a file the way two long
    tables can, so it keeps the per-chromophore filenames it has always had."""
    for ch_type in ("hbo", "hbr"):
        assert (report / f"group-G1_task-tap_hyper-isc-{ch_type}.tsv").exists()


# ---- one chromophore only ----

def test_asking_for_one_chromophore_writes_only_that_one(dyad, tmp_path):
    from fnirs_pipe.pipeline.hyperscanning import GroupEntry
    from fnirs_pipe.qc.hyper_report import build_hyper_post_report

    build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", "sub-01", "tap"), GroupEntry("G1", "sub-02", "tap")],
        aligned_raws=dyad, offsets={"sub-01": 0.0, "sub-02": 0.0}, output_dir=tmp_path,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1],
        wtc_chroma=("hbr",),
    )
    df = pd.read_csv(tmp_path / "group-G1" / "nirs" / "group-G1_task-tap_hyper-wtc.tsv",
                     sep="\t")
    assert df["chromophore"].unique().tolist() == ["hbr"]


def test_the_figures_say_which_chromophore_they_are(dyad, tmp_path):
    """This round draws one chromophore's panels, so the page has to name it rather than
    leave a reader to assume HbO."""
    from fnirs_pipe.pipeline.hyperscanning import GroupEntry
    from fnirs_pipe.qc.hyper_report import build_hyper_post_report

    path = build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", "sub-01", "tap"), GroupEntry("G1", "sub-02", "tap")],
        aligned_raws=dyad, offsets={"sub-01": 0.0, "sub-02": 0.0}, output_dir=tmp_path,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1],
        wtc_chroma=("hbr",),
    )
    html = path.read_text(encoding="utf-8")
    assert "Wavelet Transform Coherence (per channel) &mdash; HbR" in html
    assert "Wavelet Transform Coherence (per channel) &mdash; HbO" not in html


@pytest.mark.parametrize("bad", [(), ("hbt",), ("hbo", "total")])
def test_an_unknown_chromophore_is_refused(dyad, tmp_path, bad):
    from fnirs_pipe.pipeline.hyperscanning import GroupEntry
    from fnirs_pipe.qc.hyper_report import build_hyper_post_report

    with pytest.raises(ValueError, match="wtc_chroma"):
        build_hyper_post_report(
            group_id="G1", task="tap",
            group=[GroupEntry("G1", "sub-01", "tap"), GroupEntry("G1", "sub-02", "tap")],
            aligned_raws=dyad, offsets={}, output_dir=tmp_path, wtc_chroma=bad,
        )


# ---- the null follows ----

def test_the_null_covers_both_chromophores_in_one_table(dyad, tmp_path):
    """A null on one chromophore says nothing about a coupling in the other, so the real
    table's other half would have nothing to be tested against."""
    from fnirs_pipe.qc.wtc_null import write_wtc_null

    path = write_wtc_null(
        "G1", "tap", dyad, tmp_path, n_iter=1, wtc_fmin=0.02, wtc_fmax=0.2,
        band_fmin=BAND[0], band_fmax=BAND[1], seed=3, chroma=("hbo", "hbr"))
    df = pd.read_csv(path, sep="\t")
    counts = df["chromophore"].value_counts().to_dict()
    assert set(counts) == {"hbo", "hbr"}
    assert counts["hbo"] == counts["hbr"] == len(LABELS)
    params = json.loads(path.with_suffix(".json").read_text())["parameters"]
    assert params["chroma"] == ["hbo", "hbr"]


def test_the_null_refuses_an_unknown_chromophore(dyad, tmp_path):
    from fnirs_pipe.qc.wtc_null import write_wtc_null

    with pytest.raises(ValueError, match="chroma"):
        write_wtc_null("G1", "tap", dyad, tmp_path, n_iter=1, chroma=("hbt",))


# ---- the CLI surface ----

def test_the_flag_defaults_to_both():
    """Taken knowing it doubles the runtime of every existing command: reporting HbO alone
    is the field's habit rather than a defended choice, and the references do not back it."""
    from fnirs_pipe.cli.hyper import _build_parser

    args = _build_parser().parse_args(["run", "out", "--pairs-csv", "pairs.csv"])
    assert args.wtc_chroma == "both"


@pytest.mark.parametrize("value", ["hbo", "hbr", "both"])
def test_the_flag_takes_the_three_settings(value):
    from fnirs_pipe.cli.hyper import _build_parser

    args = _build_parser().parse_args(
        ["run", "out", "--pairs-csv", "pairs.csv", "--wtc-chroma", value])
    assert args.wtc_chroma == value


def test_the_flag_refuses_anything_else():
    from fnirs_pipe.cli.hyper import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["run", "out", "--pairs-csv", "pairs.csv", "--wtc-chroma", "hbt"])
