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


# ---- the traps the structure exists to avoid ----

def test_tagging_before_the_aggregation_would_lose_the_tag():
    """Why `_tag` runs after every aggregation and not before. `roi_mean_of_channels`
    groups on the columns it knows and drops the rest, so a chromophore column added
    upstream of it vanishes without an error."""
    from fnirs_pipe.pipeline.synchrony import roi_mean_of_channels

    tagged = pd.DataFrame({
        "chromophore": ["hbo"] * 2,
        "sub1": ["a", "a"], "sub2": ["b", "b"],
        "label": ["S1_D1", "S2_D2"],
        "coherence": [0.5, 0.6], "coherence_z": [0.55, 0.69], "n_valid_frac": [1.0, 1.0],
    })
    out = roi_mean_of_channels(tagged, {"L": ["S1_D1", "S2_D2"]}, min_channels=1)
    assert "chromophore" not in out.columns


# ---- per condition ----

@pytest.fixture(scope="module")
def by_condition(dyad, tmp_path_factory):
    """The same report with two task windows, so the per-condition branch runs.

    A window shorter than one cycle of `--wtc-fmin` is skipped, so at 0.02 Hz these have to
    be at least 50 s.
    """
    from fnirs_pipe.pipeline.hyperscanning import GroupEntry
    from fnirs_pipe.qc.hyper_report import build_hyper_post_report

    marked = {sid: raw.copy() for sid, raw in dyad.items()}
    for raw in marked.values():
        raw.set_annotations(mne.Annotations(onset=[10.0, 200.0], duration=[100.0, 100.0],
                                            description=["chat", "quiet"]))
    out = tmp_path_factory.mktemp("bycond")
    build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", "sub-01", "tap"), GroupEntry("G1", "sub-02", "tap")],
        aligned_raws=marked, offsets={"sub-01": 0.0, "sub-02": 0.0}, output_dir=out,
        roi_map={"L": ["S1_D1", "S2_D2"], "R": ["S3_D3"]}, wtc_roi_min_channels=1,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1],
        wtc_by_condition=True, wtc_chroma=("hbo", "hbr"),
    )
    return out / "group-G1" / "nirs"


@pytest.mark.parametrize("kind", ["wtcbycond", "wtcbycond-roichan"])
def test_the_per_condition_tables_carry_both_chromophores(by_condition, kind):
    """Its own code path, and the one where the tag is added to a frame that already has a
    `condition` column: two inserts at position 0, so the order matters."""
    df = pd.read_csv(by_condition / f"group-G1_task-tap_hyper-{kind}.tsv", sep="\t")
    assert list(df.columns[:2]) == ["chromophore", "condition"]
    counts = df.groupby(["chromophore", "condition"]).size().unstack()
    assert sorted(counts.index) == ["hbo", "hbr"]
    assert sorted(counts.columns) == ["chat", "quiet"]
    assert (counts.loc["hbo"] == counts.loc["hbr"]).all()


def test_the_whole_run_pass_still_stands_beside_the_conditions(by_condition):
    """`--wtc-by-condition` adds windows rather than replacing the whole-run analysis."""
    whole = pd.read_csv(by_condition / "group-G1_task-tap_hyper-wtc.tsv", sep="\t")
    assert "condition" not in whole.columns
    assert set(whole["chromophore"]) == {"hbo", "hbr"}


def test_the_condition_windows_reach_the_sidecar(by_condition):
    """The one thing a reader cannot reconstruct from the table."""
    params = json.loads(
        (by_condition / "group-G1_task-tap_hyper-wtcbycond.json").read_text())["parameters"]
    assert sorted(params["condition_windows_s"]) == ["chat", "quiet"]
    assert params["chroma"] == ["hbo", "hbr"]


# ---- the null cannot drift from the run ----

def test_the_null_and_the_report_default_to_the_same_chromophores():
    """`cmd_run` hands one tuple to both, so a drift would have to come from the defaults.
    Chroma is inherited unlike crossing: a null missing a chromophore leaves that half of
    the real table with nothing to be tested against."""
    import inspect

    from fnirs_pipe.qc.hyper_report import build_hyper_post_report
    from fnirs_pipe.qc.wtc_null import write_wtc_null

    report = inspect.signature(build_hyper_post_report).parameters["wtc_chroma"].default
    null = inspect.signature(write_wtc_null).parameters["chroma"].default
    assert tuple(report) == tuple(null) == ("hbo", "hbr")


# ---- the chromophore switch ----

def _js_var(html: str, name: str):
    """The JSON literal assigned to one of the page's `var _X = ...;` lines."""
    start = html.index(f"var {name}")
    line = html[start:html.index("\n", start)]
    return json.loads(line.split("=", 1)[1].rsplit(";", 1)[0].strip())


def _page(dyad, where, chroma, **kwargs):
    from fnirs_pipe.pipeline.hyperscanning import GroupEntry
    from fnirs_pipe.qc.hyper_report import build_hyper_post_report

    path = build_hyper_post_report(
        group_id="G1", task="tap",
        group=[GroupEntry("G1", "sub-01", "tap"), GroupEntry("G1", "sub-02", "tap")],
        aligned_raws=dyad, offsets={"sub-01": 0.0, "sub-02": 0.0}, output_dir=where,
        wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=BAND[0], wtc_band_fmax=BAND[1],
        wtc_chroma=chroma, **kwargs,
    )
    return path.read_text(encoding="utf-8")


def test_each_stacked_figure_is_labelled_with_its_chromophore(dyad, tmp_path):
    """The label moved from the panel title to the figure, because a panel now holds more
    than one. A screenshot of a single image still says which chromophore it is, and a
    one-chromophore run names no other."""
    html = _page(dyad, tmp_path, ("hbr",))
    assert '<span class="chroma-name">HbR</span>' in html
    assert "HbO" not in html


def test_the_page_carries_every_chromophore_that_ran(dyad, tmp_path):
    """The switch has to have something to switch to, so the figure URLs are keyed by
    chromophore. What the page carries is a path per figure, not the figure, so a second
    chromophore costs two lines of JSON here and a second set of files under figures/."""
    html = _page(dyad, tmp_path, ("hbo", "hbr"))
    per_ch = _js_var(html, "_PER_CH")
    assert sorted(per_ch) == ["hbo", "hbr"]
    assert per_ch["hbo"] != per_ch["hbr"]


def test_one_chromophore_embeds_only_that_one(dyad, tmp_path):
    """`--wtc-chroma hbo` names no second chromophore anywhere on the page, so nothing
    points at a set of maps that was never drawn."""
    html = _page(dyad, tmp_path, ("hbo",))
    assert sorted(_js_var(html, "_PER_CH")) == ["hbo"]
    assert _js_var(html, "_CHROMA") == ["hbo"]


def test_every_switched_figure_is_keyed_by_chromophore(dyad, tmp_path):
    """One missing key and `_setChroma` would silently leave a panel showing the previous
    chromophore's picture, which is the worst failure this page could have."""
    html = _page(dyad, tmp_path, ("hbo", "hbr"),
                 roi_map={"L": ["S1_D1", "S2_D2"], "R": ["S3_D3"]},
                 wtc_roi_min_channels=1, wtc_channel_cross=True)
    for name in ("_PER_CH", "_PER_ROI", "_ROI_MATRIX", "_CHAN_MATRIX"):
        assert sorted(_js_var(html, name)) == ["hbo", "hbr"], name


def test_the_channel_keys_the_selector_uses_exist_for_both_chromophores(dyad, tmp_path):
    """`_pick` indexes `_PER_CH[_chroma][a][b]` off `_CH_PAIRS`, so a label missing from one
    chromophore's map would blank the plot on switching rather than error."""
    html = _page(dyad, tmp_path, ("hbo", "hbr"))
    pairs = _js_var(html, "_CH_PAIRS")
    per_ch = _js_var(html, "_PER_CH")
    assert pairs
    for ch_type in ("hbo", "hbr"):
        assert set(per_ch[ch_type]) >= set(pairs), ch_type


def test_the_roi_keys_the_selector_uses_exist_for_both_chromophores(dyad, tmp_path):
    html = _page(dyad, tmp_path, ("hbo", "hbr"),
                 roi_map={"L": ["S1_D1", "S2_D2"], "R": ["S3_D3"]},
                 wtc_roi_min_channels=1)
    labels = _js_var(html, "_ROI_LABELS")
    per_roi = _js_var(html, "_PER_ROI")
    assert labels
    for ch_type in ("hbo", "hbr"):
        assert set(per_roi[ch_type]) >= set(labels), ch_type


def test_an_uncrossed_run_fills_the_diagonal_and_shows_one_selector(dyad, tmp_path):
    """The map tables are nested both ways whether or not the run crossed, so the page reads
    one shape. Without crossing there is nothing off the diagonal to pick, and a second
    selector that could only blank the panel is worse than no second selector."""
    html = _page(dyad, tmp_path, ("hbo",),
                 roi_map={"L": ["S1_D1", "S2_D2"], "R": ["S3_D3"]},
                 wtc_roi_min_channels=1)
    per_ch = _js_var(html, "_PER_CH")["hbo"]
    for label, row in per_ch.items():
        assert list(row) == [label], label
    assert 'id="ch-select-post-2"' not in html
    assert 'id="roi-select-post-2"' not in html


def test_a_crossed_run_reaches_every_pairing_from_two_selectors(dyad, tmp_path):
    """The point of the pair: an off-diagonal pairing is what says whether two sites couple
    at a different time or frequency from the homologous one, and it used to be readable
    only as a thumbnail in a grid of every ROI pair."""
    html = _page(dyad, tmp_path, ("hbo",), wtc_channel_cross=True,
                 roi_map={"L": ["S1_D1", "S2_D2"], "R": ["S3_D3"]},
                 wtc_roi_min_channels=1)
    labels = _js_var(html, "_CH_PAIRS")
    per_ch = _js_var(html, "_PER_CH")["hbo"]
    assert len(labels) > 1
    for a in labels:
        assert sorted(per_ch[a]) == sorted(labels), a
    rois = _js_var(html, "_ROI_LABELS")
    per_roi = _js_var(html, "_PER_ROI")["hbo"]
    for a in rois:
        assert sorted(per_roi[a]) == sorted(rois), a
    assert 'id="ch-select-post-2"' in html
    assert 'id="roi-select-post-2"' in html


def test_a_condition_boundary_is_drawn_on_the_axis_the_window_was_cut_on(dyad, tmp_path):
    """An aligned recording keeps its crop offset in `first_time` while everything computed
    from it starts at zero, so the two have to be read through one function. They were not,
    and every boundary line on every coherence map came out late by that offset."""
    from fnirs_pipe.qc.hyper_report import condition_windows, markers_on_data_axis

    raw = dyad["sub-01"].copy()
    raw.set_annotations(mne.Annotations([20.0, 210.0], [180.0, 180.0], ["rest", "talk"]))
    cropped = raw.copy().crop(tmin=20.0)
    assert cropped.first_time == 20.0
    onsets = [m["onset"] for m in markers_on_data_axis(cropped)]
    assert onsets == [0.0, 190.0]
    assert [w[1] for w in condition_windows(cropped, min_duration=50.0)] == onsets


def test_both_chromophores_are_on_the_page_at_once(dyad, tmp_path):
    """There is no chromophore control any more: every panel stacks one labelled image per
    chromophore, in `_CHROMA` order, which is the order the template laid them out in. A
    reader can compare HbO against HbR without operating anything, and nothing is hidden."""
    html = _page(dyad, tmp_path, ("hbo", "hbr"))
    assert _js_var(html, "_CHROMA") == ["hbo", "hbr"]
    assert 'id="chroma-switch"' not in html
    for base in ("wtc-chan-img", "wtc-roi-img", "wtc-chan-matrix-img"):
        assert f'id="{base}-0"' in html, base
        assert f'id="{base}-1"' in html, base
    assert html.count('class="chroma-name"') >= 2


def test_a_single_chromophore_run_stacks_only_that_one(dyad, tmp_path):
    """The stack follows what ran, so a one-chromophore run has one image per panel and no
    empty slot claiming a chromophore that was never computed."""
    html = _page(dyad, tmp_path, ("hbo",))
    assert 'id="wtc-chan-img-0"' in html
    assert 'id="wtc-chan-img-1"' not in html


def test_the_condition_images_line_up_by_window_across_chromophores(dyad, tmp_path_factory):
    """`_drawImages` pairs `_COND_IMGS[chroma][i]` with the card `cond-matrix-<i>`, so the
    lists have to be positional and the same length, one entry per window, even when a
    guard failed for one chromophore and left that entry empty."""
    marked = {sid: raw.copy() for sid, raw in dyad.items()}
    for raw in marked.values():
        raw.set_annotations(mne.Annotations(onset=[10.0, 200.0], duration=[100.0, 100.0],
                                            description=["chat", "quiet"]))
    html = _page(marked, tmp_path_factory.mktemp("condswitch"), ("hbo", "hbr"),
                 roi_map={"L": ["S1_D1", "S2_D2"], "R": ["S3_D3"]},
                 wtc_roi_min_channels=1, wtc_by_condition=True)
    cond = _js_var(html, "_COND_IMGS")
    assert len(cond["hbo"]) == len(cond["hbr"]) == 2
    assert 'id="cond-grid-0"' in html and 'id="cond-grid-1"' in html
    for entries in cond.values():
        for imgs in entries:
            assert set(imgs) == {"matrix", "grid"}
