"""Which channels the inter-brain metrics read, and what a WTC map collapses to.

Two contracts live here. Short channels are not brains: they sample scalp haemodynamics,
so two people sitting in one room share that signal by construction and any inter-brain
metric computed on them reports shared physiology as coupling. And the band mean written
to TSV has to agree with the map the report draws, cone of influence included, or the
figures and the group statistics end up answering different questions.

The band-mean tests build a WTCResult by hand rather than running pycwt: the quantity under
test is the collapse, and a hand-built map is the only way to know what the right answer is.
"""

from pathlib import Path

import mne
import numpy as np
import pytest

from fnirs_pipe.exceptions import MissingDerivativesError, StageError
from fnirs_pipe.io.derivatives import find_preproc_snirf
from fnirs_pipe.io.snirf import long_channel_picks, write_snirf
from fnirs_pipe.pipeline.hyperscanning import GroupEntry, load_group_haemo
from fnirs_pipe.pipeline.synchrony import WTCResult, _shared_sfreq, compute_wtc, wtc_band_mean
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
    raw = _haemo("10031")
    assert SHORT_DISTANCE < 0.01                                   # the premise of the layout
    assert _labels(raw) == {"S1_D1", "S2_D2", "S3_D3", "S4_D4"}
    assert _labels(raw, "hbr") == _labels(raw, "hbo")


def test_bad_channels_stay_excluded_alongside_the_short_ones():
    raw = _haemo("10031")
    raw.info["bads"] = [c for c in raw.ch_names if c.startswith("S2_D2")]
    assert _labels(raw) == {"S1_D1", "S3_D3", "S4_D4"}


def test_a_montage_with_no_short_channel_loses_nothing():
    raw = synth_raw("10031", "hold", duration=40.0, motion_onset=None,
                    bad_pair=None, short_channels=False)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    haemo = mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)
    assert len(long_channel_picks(haemo, "hbo")) == len(mne.pick_types(haemo.info, fnirs="hbo"))


def test_wtc_never_pairs_the_short_channel():
    raws = {"sub-10031": _haemo("10031"), "sub-10032": _haemo("10032")}
    result = compute_wtc(raws, fmin=0.02, fmax=0.5)
    labels = set(next(iter(result.pairs.values())))
    assert SHORT_PAIR not in labels
    assert labels == {"S1_D1", "S2_D2", "S3_D3", "S4_D4"}


def test_isc_reads_the_same_channels_as_wtc():
    from fnirs_pipe.qc.figures.hyper_post_figures import compute_isc

    subject_ids = ["sub-10031", "sub-10032"]
    raws = dict(zip(subject_ids, (_haemo("10031"), _haemo("10032"))))
    _, ch_names = compute_isc(raws, subject_ids, "hbo")
    assert SHORT_PAIR not in ch_names


def test_a_sampling_rate_mismatch_is_refused():
    # alignment equalises duration, not rate, and every metric here takes the rate off one
    # participant, so a mismatch would mislabel the other's frequency axis
    fast, slow = _haemo("10031"), _haemo("10032")
    slow.resample(slow.info["sfreq"] / 2, verbose="error")
    with pytest.raises(ValueError, match="differ in sampling rate"):
        _shared_sfreq({"sub-10031": fast, "sub-10032": slow})


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
    # the four corner cells hold the 1.0s; masking them leaves 8 cells holding two of them
    masked   = wtc_band_mean(_result(), 0.04, 0.25).iloc[0]
    unmasked = wtc_band_mean(_result(), 0.04, 0.25, mask_coi=False).iloc[0]
    assert masked["coherence"] == pytest.approx(2 / 8)
    assert unmasked["coherence"] == pytest.approx(6 / 12)
    assert masked["n_valid_frac"] == pytest.approx(8 / 12)
    assert unmasked["n_valid_frac"] == pytest.approx(1.0)


def test_the_band_selects_rows_before_the_mask_applies():
    row = wtc_band_mean(_result(), 0.15, 0.25).iloc[0]      # the 0.2 Hz row alone
    assert row["coherence"] == pytest.approx(0.5)
    assert row["n_valid_frac"] == pytest.approx(1.0)


def test_a_band_outside_the_computed_axis_is_an_error():
    # silently returning NaN here would look like a dead channel rather than a bad flag
    with pytest.raises(ValueError, match="no WTC frequency bin"):
        wtc_band_mean(_result(), 0.5, 0.8)


def test_a_label_that_failed_keeps_its_row():
    result = WTCResult(pairs={("sub-A", "sub-B"): {"S1_D1": None}}, freqs=FREQS, times=TIMES)
    row = wtc_band_mean(result, 0.04, 0.25).iloc[0]
    assert np.isnan(row["coherence"])
    assert row["n_valid_frac"] == 0.0


def test_every_pair_and_label_gets_one_row():
    pairs = {("sub-A", "sub-B"): {"S1_D1": {"wtc": WTC, "coi": COI},
                                  "S2_D2": {"wtc": WTC, "coi": COI}},
             ("sub-A", "sub-C"): {"S1_D1": {"wtc": WTC, "coi": COI},
                                  "S2_D2": None}}
    df = wtc_band_mean(WTCResult(pairs=pairs, freqs=FREQS, times=TIMES), 0.04, 0.25)
    assert list(df.columns) == ["sub1", "sub2", "label", "coherence", "n_valid_frac"]
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

    with pytest.raises(MissingDerivativesError, match="Available: preproc"):
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
