"""Which channels the inter-brain metrics read, and what a WTC map collapses to.

Three contracts live here. Short channels are not brains: they sample scalp haemodynamics,
so two people sitting in one room share that signal by construction and any inter-brain
metric computed on them reports shared physiology as coupling. And the band mean written
to TSV has to agree with the map the report draws, cone of influence included, or the
figures and the group statistics end up answering different questions.

And the third: ISC and WTC are one code path up to the statistic, so what one of them
refuses the other refuses. The tests at the end of this file hold the parts of that which
have come apart before.

The band-mean tests build a WTCResult by hand rather than running pycwt: the quantity under
test is the collapse, and a hand-built map is the only way to know what the right answer is.
"""

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.exceptions import MissingDerivativesError, StageError
from fnirs_pipe.io.derivatives import find_preproc_snirf
from fnirs_pipe.io.snirf import long_channel_picks, write_snirf
from fnirs_pipe.pipeline.hyper import GroupEntry, load_group_haemo
from fnirs_pipe.pipeline.hyper._helpers import _shared_sfreq
from fnirs_pipe.pipeline.hyper.wtc import WTCResult, compute_wtc, wtc_band_mean
from tests._synth import SHORT_DISTANCE, synth_raw

SHORT_PAIR = "S5_D5"      # the one pair _channel_layout places below 1 cm


def _haemo(subject: str, task: str = "hold", bad_pair: int | None = None) -> mne.io.Raw:
    raw = synth_raw(subject, task, duration=40.0, motion_onset=None, bad_pair=bad_pair)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    return mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)


def _labels(raw: mne.io.Raw, ch_type: str = "hbo") -> set[str]:
    return {raw.ch_names[p].rsplit(" ", 1)[0] for p in long_channel_picks(raw, ch_type)}


# ---- channel scoping ----

def test_the_short_pair_is_dropped_from_the_picks():
    raw = _haemo("11")
    assert SHORT_DISTANCE < 0.01                                   # the premise of the layout
    assert _labels(raw) == {"S1_D1", "S2_D2", "S3_D3", "S4_D4"}
    assert _labels(raw, "hbr") == _labels(raw, "hbo")


def test_bad_channels_stay_excluded_alongside_the_short_ones():
    raw = _haemo("11")
    raw.info["bads"] = [c for c in raw.ch_names if c.startswith("S2_D2")]
    assert _labels(raw) == {"S1_D1", "S3_D3", "S4_D4"}


def test_a_montage_with_no_short_channel_loses_nothing():
    raw = synth_raw("11", "hold", duration=40.0, motion_onset=None,
                    bad_pair=None, short_channels=False)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    haemo = mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)
    assert len(long_channel_picks(haemo, "hbo")) == len(mne.pick_types(haemo.info, fnirs="hbo"))


def test_wtc_never_pairs_the_short_channel():
    raws = {"sub-11": _haemo("11"), "sub-12": _haemo("12")}
    result = compute_wtc(raws, fmin=0.02, fmax=0.5)
    labels = set(next(iter(result.pairs.values())))
    assert SHORT_PAIR not in labels
    assert labels == {"S1_D1", "S2_D2", "S3_D3", "S4_D4"}


def test_isc_reads_the_same_channels_as_wtc():
    from fnirs_pipe.pipeline.hyper.isc import compute_isc

    subject_ids = ["sub-11", "sub-12"]
    raws = dict(zip(subject_ids, (_haemo("11"), _haemo("12"))))
    _, ch_names = compute_isc(raws, subject_ids, "hbo")
    assert SHORT_PAIR not in ch_names


# ---- channel matching across a pair ----

def _tagged(subject: str, drop: str | None = None) -> mne.io.Raw:
    """One haemo recording whose HbO channels each carry their own frequency.

    Whole numbers of cycles over the record, so two different channels correlate at 0 and
    the same channel at 1. That is what makes a mismatched pairing visible at all: with
    ordinary data every pair correlates a little and no assertion can tell them apart.
    """
    raw = _haemo(subject)
    for k, p in enumerate(long_channel_picks(raw, "hbo")):
        raw._data[p] = np.sin(2 * np.pi * (0.1 * (k + 1)) * raw.times)
    if drop:
        raw.info["bads"] = [c for c in raw.ch_names if c.startswith(drop)]
    return raw


def test_isc_matches_channels_by_label_not_position():
    from fnirs_pipe.pipeline.hyper.isc import compute_isc

    # sub-B rejected S2_D2, so its remaining channels sit one position earlier than sub-A's
    subject_ids = ["sub-A", "sub-B"]
    raws = {"sub-A": _tagged("11"), "sub-B": _tagged("12", drop="S2_D2")}
    isc_mat, ch_names = compute_isc(raws, subject_ids, "hbo")

    assert ch_names == ["S1_D1", "S2_D2", "S3_D3", "S4_D4"]
    diag = np.diag(isc_mat)
    assert np.isnan(diag[1])                                   # sub-B has no S2_D2 to pair
    assert diag[[0, 2, 3]] == pytest.approx(1.0, abs=1e-6)     # the rest still line up


def test_the_axis_is_the_union_of_the_two_montages():
    """One member's montage is not the axis: each side can carry a label the other lost."""
    from fnirs_pipe.pipeline.hyper._helpers import long_axis_over

    a, b = _tagged("11"), _tagged("12")
    a.drop_channels([c for c in a.ch_names if c.startswith("S4_D4")])
    b.drop_channels([c for c in b.ch_names if c.startswith("S1_D1")])

    assert long_axis_over([a, b]) == ["S1_D1", "S2_D2", "S3_D3", "S4_D4"]


def test_isc_keeps_a_channel_only_one_member_has():
    from fnirs_pipe.pipeline.hyper.isc import compute_isc

    subject_ids = ["sub-A", "sub-B"]
    a, b = _tagged("11"), _tagged("12")
    b.drop_channels([c for c in b.ch_names if c.startswith("S1_D1")])
    isc_mat, ch_names = compute_isc({"sub-A": a, "sub-B": b}, subject_ids, "hbo")

    assert "S1_D1" in ch_names                                 # sub-B lost it, the axis did not
    col = ch_names.index("S1_D1")
    assert np.isnan(isc_mat[:, col]).all()                     # nothing of sub-B's to pair
    assert not np.isnan(isc_mat[col, ch_names.index("S2_D2")])  # sub-A's own row still runs


def test_the_blanked_column_is_the_one_that_was_named():
    from fnirs_pipe.pipeline.hyper.isc import compute_isc

    # rejections reach ISC on info["bads"], the way load_group_haemo leaves them and the
    # way the WTC path reads them
    subject_ids = ["sub-A", "sub-B"]
    raws = {"sub-A": _tagged("11"), "sub-B": _tagged("12", drop="S3_D3")}
    isc_mat, ch_names = compute_isc(raws, subject_ids, "hbo")
    assert np.isnan(isc_mat[:, ch_names.index("S3_D3")]).all()
    assert not np.isnan(isc_mat[:, ch_names.index("S2_D2")]).any()


# ---- what the pair refuses, and what it flags ----

def test_isc_refuses_two_sampling_rates():
    """WTC raises on this; ISC used to pair sample i with sample i and answer anyway."""
    from fnirs_pipe.pipeline.hyper.isc import compute_isc

    a, b = _tagged("11"), _tagged("12")
    b.resample(b.info["sfreq"] / 2, verbose="error")
    with pytest.raises(ValueError, match="differ in sampling rate"):
        compute_isc({"sub-A": a, "sub-B": b}, ["sub-A", "sub-B"], "hbo")


def test_an_unrecorded_bandpass_is_flagged():
    """--desc defaults to preproc, which is Beer-Lambert output and carries its drift."""
    from fnirs_pipe.pipeline.hyper import unfiltered_stage_note
    from fnirs_pipe.utils.lineage import stamp

    raws = {"sub-A": _haemo("11"), "sub-B": _haemo("12")}
    for raw in raws.values():
        stamp(raw, stage="preproc", step="load")

    note = unfiltered_stage_note(raws)
    assert note is not None
    assert "sub-A" in note and "sub-B" in note and "preproc" in note


def test_a_recorded_bandpass_is_not_flagged():
    from fnirs_pipe.pipeline.hyper import unfiltered_stage_note
    from fnirs_pipe.utils.lineage import stamp

    raws = {"sub-A": _haemo("11"), "sub-B": _haemo("12")}
    for raw in raws.values():
        stamp(raw, stage="errts", step="load", high_pass=0.01, low_pass=0.5)

    assert unfiltered_stage_note(raws) is None


def test_one_filtered_member_is_still_flagged():
    """A dyad half filtered is worse than one not filtered at all, not better."""
    from fnirs_pipe.pipeline.hyper import unfiltered_stage_note
    from fnirs_pipe.utils.lineage import stamp

    raws = {"sub-A": _haemo("11"), "sub-B": _haemo("12")}
    stamp(raws["sub-A"], stage="errts", step="load", high_pass=0.01, low_pass=0.5)
    stamp(raws["sub-B"], stage="preproc", step="load")

    note = unfiltered_stage_note(raws)
    assert note is not None and "sub-B" in note and "sub-A" not in note


def test_coherence_matches_channels_by_label():
    from fnirs_pipe.pipeline.hyper.coherence import compute_pairwise_coherence

    raws = {"sub-A": _tagged("11"), "sub-B": _tagged("12", drop="S2_D2")}
    df = compute_pairwise_coherence(raws, fmin=0.05, fmax=0.15).set_index("ch_name")

    assert list(df.index) == ["S1_D1", "S2_D2", "S3_D3", "S4_D4"]
    assert np.isnan(df.loc["S2_D2", "coherence"])
    assert df.loc[["S1_D1", "S3_D3", "S4_D4"], "coherence"].to_numpy() == pytest.approx(1.0)


def test_screening_coherence_drops_a_pair_one_member_lacks():
    # the windowed coherence this replaced kept a blank row per window; the screening pass
    # drops the pair instead, because a channel one member does not have is not a channel
    # the dyad can be screened on and a NaN row would be averaged into the window's mean
    from fnirs_pipe.pipeline.hyper.coherence import screening_coherence

    raws = {"sub-A": _tagged("11"), "sub-B": _tagged("12", drop="S2_D2")}
    df = screening_coherence(raws, 0.05, 0.15, n_iter=3, seed=0)

    assert set(df["ch_name"]) == {"S1_D1", "S3_D3", "S4_D4"}
    assert list(df["window"].unique()) == ["whole run"]
    # every row of a window carries that window's own rank, so the page can read it off any
    assert df["window_percentile"].nunique() == 1


def test_a_sampling_rate_mismatch_is_refused():
    # alignment equalises duration, not rate, and every metric here takes the rate off one
    # participant, so a mismatch would mislabel the other's frequency axis
    fast, slow = _haemo("11"), _haemo("12")
    slow.resample(slow.info["sfreq"] / 2, verbose="error")
    with pytest.raises(ValueError, match="differ in sampling rate"):
        _shared_sfreq({"sub-11": fast, "sub-12": slow})


# ---- band mean ----

FREQS = np.array([0.05, 0.10, 0.20])
TIMES = np.arange(4.0)
WTC   = np.array([[1.0, 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, 1.0]])
# a period in seconds: 5 s admits 0.2 Hz and above, 20 s admits the whole axis
COI = np.array([5.0, 20.0, 20.0, 5.0])


def _result(wtc=WTC, coi=COI) -> WTCResult:
    return WTCResult(pairs={("sub-A", "sub-B"): {"S1_D1": {"wtc": wtc, "coi": coi}}},
                     freqs=FREQS, times=TIMES)


def test_cells_outside_the_cone_of_influence_do_not_enter_the_mean():
    # the four corner cells hold the 1.0s; masking them leaves 8 cells holding two of them.
    # Both calls name the mask: masking is the default, so the unmasked side is the one
    # that has to ask, and a test naming neither would get the masked number twice
    masked   = wtc_band_mean(_result(), 0.04, 0.25, mask_coi=True).iloc[0]
    unmasked = wtc_band_mean(_result(), 0.04, 0.25, mask_coi=False).iloc[0]
    assert masked["coherence"] == pytest.approx(2 / 8)
    assert unmasked["coherence"] == pytest.approx(6 / 12)
    # n_valid_frac is the share inside the cone, measured whether or not the mask is
    # applied, so it does not tell you which of the two numbers above you are holding
    assert masked["n_valid_frac"] == pytest.approx(8 / 12)
    assert unmasked["n_valid_frac"] == pytest.approx(8 / 12)


def test_the_band_selects_rows_before_the_mask_applies():
    row = wtc_band_mean(_result(), 0.15, 0.25, mask_coi=True).iloc[0]  # the 0.2 Hz row alone
    assert row["coherence"] == pytest.approx(0.5)
    assert row["n_valid_frac"] == pytest.approx(1.0)


def test_a_band_outside_the_computed_axis_is_an_error():
    # silently returning NaN here would look like a dead channel rather than a bad flag
    with pytest.raises(ValueError, match="no WTC frequency bin"):
        wtc_band_mean(_result(), 0.5, 0.8)


def test_a_label_that_failed_keeps_its_row():
    """Every measured column is NaN, `n_valid_frac` included. A 0 there would be averaged
    as a real share by `roi_mean_of_channels`, pulling an ROI's reported cone share down by
    however many of its channels were rejected."""
    result = WTCResult(pairs={("sub-A", "sub-B"): {"S1_D1": None}}, freqs=FREQS, times=TIMES)
    row = wtc_band_mean(result, 0.04, 0.25).iloc[0]
    assert np.isnan(row["coherence"])
    assert np.isnan(row["n_valid_frac"])


def test_every_pair_and_label_gets_one_row():
    pairs = {("sub-A", "sub-B"): {"S1_D1": {"wtc": WTC, "coi": COI},
                                  "S2_D2": {"wtc": WTC, "coi": COI}},
             ("sub-A", "sub-C"): {"S1_D1": {"wtc": WTC, "coi": COI},
                                  "S2_D2": None}}
    df = wtc_band_mean(WTCResult(pairs=pairs, freqs=FREQS, times=TIMES), 0.04, 0.25)
    assert list(df.columns) == ["sub1", "sub2", "label",
                                "coherence", "coherence_z", "n_valid_frac",
                                "phase_angle", "phase_sd", "phase_n"]
    assert len(df) == 4


# ---- stage selection ----

def _write_stage(root: Path, subject: str, task: str, desc: str, raw: mne.io.Raw) -> Path:
    path = root / subject / "nirs" / f"{subject}_task-{task}_desc-{desc}_nirs.snirf"
    write_snirf(raw, path)
    return path


def test_the_desc_picks_the_stage(tmp_path):
    nirs = tmp_path / "sub-01" / "nirs"
    nirs.mkdir(parents=True)
    for desc in ("preproc", "errts"):
        (nirs / f"sub-01_task-hold_desc-{desc}_nirs.snirf").touch()

    assert find_preproc_snirf(tmp_path, "sub-01", "hold").name.endswith("desc-preproc_nirs.snirf")
    assert find_preproc_snirf(tmp_path, "sub-01", "hold", desc="errts").name.endswith(
        "desc-errts_nirs.snirf")


def test_a_missing_stage_names_the_ones_on_disk(tmp_path):
    nirs = tmp_path / "sub-01" / "nirs"
    nirs.mkdir(parents=True)
    (nirs / "sub-01_task-hold_desc-preproc_nirs.snirf").touch()

    with pytest.raises(MissingDerivativesError, match="Available desc: preproc"):
        find_preproc_snirf(tmp_path, "sub-01", "hold", desc="errts")


def test_an_optical_density_stage_is_refused(tmp_path):
    # desc-od, desc-sci and desc-motcorrected all live before Beer-Lambert; coherence
    # computed on them is not coherence between concentrations
    raw = synth_raw("01", "hold", duration=40.0, motion_onset=None, bad_pair=None)
    od  = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    _write_stage(tmp_path, "sub-01", "hold", "od", od)

    group = [GroupEntry(group_id="1", subject_id="sub-01", task="hold")]
    with pytest.raises(StageError, match="optical density"):
        load_group_haemo(tmp_path, group, desc="od")


def test_a_haemoglobin_stage_loads(tmp_path):
    _write_stage(tmp_path, "sub-01", "hold", "errts", _haemo("01"))

    group = [GroupEntry(group_id="1", subject_id="sub-01", task="hold")]
    loaded = load_group_haemo(tmp_path, group, desc="errts")
    assert set(loaded) == {"sub-01"}
    assert "hbo" in set(loaded["sub-01"].get_channel_types())


# ---- what leaves the report ----

def test_the_isc_matrix_is_written_beside_the_panel(tmp_path):
    """The panel is a picture; without this file the correlations cannot reach a group analysis.

    Same contract as the band mean above: what the figure shows and what a stats script reads
    have to be one set of numbers.
    """
    from fnirs_pipe.pipeline.hyper.hyper_post import write_isc_matrix

    names = ["S1_D1 hbo", "S2_D2 hbo"]
    mat = np.array([[0.9, np.nan], [0.4, 0.8]])
    path = tmp_path / "group-G1_task-hold_hyper-isc-hbo.tsv"
    write_isc_matrix(path, mat, names, "hbo", ["/in/sub-01.snirf"], ["sub-01", "sub-02"])

    written = pd.read_csv(path, sep="\t", index_col="channel")
    assert list(written.index) == list(written.columns) == names
    assert written.iloc[1, 0] == pytest.approx(0.4)      # row = sub1, column = sub2
    assert np.isnan(written.iloc[0, 1])                  # a rejected channel stays blank

    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["step"] == "hyper_isc"
    assert sidecar["parameters"]["chromophore"] == "hbo"


def test_a_failed_isc_write_costs_the_file_and_not_the_report(tmp_path):
    # the report is still readable without the TSV, so the writer swallows its own failure
    from fnirs_pipe.pipeline.hyper.hyper_post import write_isc_matrix

    path = tmp_path / "isc.tsv"
    write_isc_matrix(path, np.eye(3), ["a", "b"], "hbo", [], ["sub-01"])   # shapes disagree
    assert not path.exists()


# ---- the ROI mean of the correlations ----
#
# It sits beside the ROI mean of the coherences and is built the same way, off the channel
# values; what differs is the average, since a correlation is signed.

ISC_ROIS = {"L": ["S1_D1", "S2_D2"], "R": ["S3_D3", "S4_D4"]}
ISC_LABELS = ["S1_D1", "S2_D2", "S3_D3", "S4_D4"]


def _isc_roi(mat, **kwargs):
    from fnirs_pipe.pipeline.hyper.isc import roi_mean_of_isc

    return roi_mean_of_isc(np.asarray(mat, dtype=float), ISC_LABELS, ISC_ROIS, **kwargs)


def test_the_roi_correlation_is_the_fisher_z_mean_of_its_channels():
    """Averaging r directly pulls a spread block toward zero; that bias is the whole point
    of the transform, so the test is a block whose two means differ."""
    mat = np.zeros((4, 4))
    mat[np.ix_([0, 1], [0, 1])] = [[0.2, 0.4], [0.6, 0.95]]
    out, labels = _isc_roi(mat, min_channels=1)

    z = np.arctanh(np.clip([0.2, 0.4, 0.6, 0.95], -0.999999, 0.999999))
    assert labels == ["L", "R"]
    assert out[0, 0] == pytest.approx(float(np.tanh(z.mean())))
    assert out[0, 0] > np.mean([0.2, 0.4, 0.6, 0.95])


def test_the_roi_correlation_keeps_the_two_members_on_their_own_axis():
    """Cell (i, j) is sub1's region i against sub2's region j, which is what the channel
    matrix means and the only reason an off-diagonal cell is readable at all."""
    mat = np.zeros((4, 4))
    mat[np.ix_([0, 1], [2, 3])] = 0.5      # sub1's L against sub2's R
    out, labels = _isc_roi(mat, min_channels=1)

    assert out[labels.index("L"), labels.index("R")] == pytest.approx(0.5)
    assert out[labels.index("R"), labels.index("L")] == pytest.approx(0.0)


def test_a_rejected_channel_thins_a_roi_cell_rather_than_voiding_it():
    """A rejected channel arrives as NaN. Those are dropped, so the region is still reported
    off what survived, which is how the ROI coherence treats the same gap."""
    mat = np.full((4, 4), 0.5)
    mat[0, :] = np.nan
    out, _ = _isc_roi(mat, min_channels=2)
    assert np.isfinite(out).all()
    assert out[0, 0] == pytest.approx(0.5)


def test_a_roi_cell_under_the_minimum_is_left_blank():
    """The rule the ROI coherence uses, so one surviving optode never stands for a region."""
    mat = np.full((4, 4), 0.5)
    mat[0, :] = mat[1, :] = np.nan        # region L keeps nothing on sub1's side
    out, labels = _isc_roi(mat, min_channels=2)
    assert np.isnan(out[labels.index("L")]).all()
    assert np.isfinite(out[labels.index("R")]).all()


def test_no_roi_map_is_no_matrix_rather_than_an_empty_one():
    assert _isc_roi(np.zeros((4, 4)))[0] is not None
    from fnirs_pipe.pipeline.hyper.isc import roi_mean_of_isc

    assert roi_mean_of_isc(np.zeros((4, 4)), ISC_LABELS, {}) == (None, None)
    assert roi_mean_of_isc(None, ISC_LABELS, ISC_ROIS) == (None, None)


def test_the_roi_matrix_is_written_beside_the_channel_one(tmp_path):
    """A group analysis over regions reads this file; without it the ROI correlations only
    ever exist inside one page's HTML."""
    from fnirs_pipe.pipeline.hyper.hyper_post import write_isc_matrix

    path = tmp_path / "group-G1_task-hold_hyper-isc-roichan-hbo.tsv"
    write_isc_matrix(path, np.array([[0.4, 0.1], [0.2, 0.3]]), ["L", "R"], "hbo",
                     [], ["sub-01", "sub-02"],
                     step="hyper_isc_roichan", index_label="roi")

    written = pd.read_csv(path, sep="	", index_col="roi")
    assert list(written.index) == ["L", "R"]
    assert written.iloc[1, 0] == pytest.approx(0.2)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["step"] == "hyper_isc_roichan"


# ---- the quality record reaching the Raw ----

def test_a_rejected_pair_is_marked_at_both_chromophores():
    from fnirs_pipe.pipeline.hyper import apply_group_bads

    raw = _haemo("11")
    raws = {"sub-A": raw}
    # the record names channels by wavelength; the haemo Raw names them by chromophore
    apply_group_bads(raws, {"sub-A": {"bad_channels": ["S2_D2 760", "S2_D2 850"]}})

    assert set(raw.info["bads"]) == {"S2_D2 hbo", "S2_D2 hbr"}
    assert _labels(raw) == {"S1_D1", "S3_D3", "S4_D4"}


def test_a_rejected_pair_never_reaches_the_coherence():
    """desc-errts carries no bads, so nothing but this marking keeps a rejected pair out.

    It keeps its label and loses its map, rather than leaving the axis: `long_axis_over`
    holds every label the montage has so two dyads that lost different channels still stack,
    and a reader can tell a blank cell from a channel that was never there.
    """
    from fnirs_pipe.pipeline.hyper import apply_group_bads

    raws = {"sub-11": _haemo("11"), "sub-12": _haemo("12")}
    assert all(not r.info["bads"] for r in raws.values())          # the state on disk

    apply_group_bads(raws, {"sub-11": {"bad_channels": ["S2_D2 760"]}})
    maps = next(iter(compute_wtc(raws, fmin=0.02, fmax=0.5).pairs.values()))

    assert maps["S2_D2"] is None, "a rejected pair must carry no map"
    assert set(maps) == {"S1_D1", "S2_D2", "S3_D3", "S4_D4"}
    # the other half, without which blanking everything would pass the line above
    assert all(maps[label] is not None for label in ("S1_D1", "S3_D3", "S4_D4"))


def test_a_subject_with_nothing_rejected_keeps_every_channel():
    from fnirs_pipe.pipeline.hyper import apply_group_bads

    raw = _haemo("11")
    apply_group_bads({"sub-A": raw}, {"sub-A": {"bad_channels": []}})
    assert raw.info["bads"] == []
    assert _labels(raw) == {"S1_D1", "S2_D2", "S3_D3", "S4_D4"}


# ---- session trees ----
# `derivatives_path` writes a session to its own folder, and this used to build
# `sub-01/nirs` by hand: a session tree raised "Derivatives directory not found" and the
# dyad analysis stopped at its first member. The `ses-` entity also sorts before `task-` in
# a BIDS filename, so the old `{subject}_task-{task}_*` glob missed those names as well.

def test_a_named_session_is_found_in_its_own_folder(tmp_path):
    nirs = tmp_path / "sub-01" / "ses-a" / "nirs"
    nirs.mkdir(parents=True)
    (nirs / "sub-01_ses-a_task-hold_desc-preproc_nirs.snirf").touch()

    found = find_preproc_snirf(tmp_path, "sub-01", "hold", session="a")
    assert found.name == "sub-01_ses-a_task-hold_desc-preproc_nirs.snirf"


def test_one_session_is_found_without_being_named(tmp_path):
    """The group CSV names a session only when it has to. One session never has to."""
    nirs = tmp_path / "sub-01" / "ses-a" / "nirs"
    nirs.mkdir(parents=True)
    (nirs / "sub-01_ses-a_task-hold_desc-preproc_nirs.snirf").touch()

    assert find_preproc_snirf(tmp_path, "sub-01", "hold").name.startswith("sub-01_ses-a_")


def test_two_unnamed_sessions_are_refused_by_name(tmp_path):
    """Ambiguity is the caller's to resolve, and the message has to say what to resolve it
    with. Picking the first in sorted order is what `select_one_run` exists to prevent."""
    for session in ("a", "b"):
        nirs = tmp_path / "sub-01" / f"ses-{session}" / "nirs"
        nirs.mkdir(parents=True)
        (nirs / f"sub-01_ses-{session}_task-hold_desc-preproc_nirs.snirf").touch()

    with pytest.raises(MissingDerivativesError, match="ses: a, b"):
        find_preproc_snirf(tmp_path, "sub-01", "hold")

    assert find_preproc_snirf(tmp_path, "sub-01", "hold", session="b").name.startswith(
        "sub-01_ses-b_")


def test_a_subject_with_no_output_at_all_still_says_so(tmp_path):
    with pytest.raises(MissingDerivativesError, match="Derivatives directory not found"):
        find_preproc_snirf(tmp_path, "sub-99", "hold")
