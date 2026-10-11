"""Targeted PCA motion correction: detection, the PCA and its stitching, the passes, the wiring.

The steps follow the published procedure exactly, quirks included, so several tests pin a
quirk on purpose: half a sample rounds away from zero, a segment is stitched one sample
early, a single component carrying the whole share removes nothing, and later passes remove
the first pass's count rather than re-reading the share.
"""

import json

import mne
import numpy as np

from nirspipe.pipeline import motion
from nirspipe.pipeline.motion import (
    TPCA_MAX_ITER,
    TPCA_N_SV,
    _tpca_motion_correct,
    _tpca_motion_mask,
    _tpca_pca,
    correct_motion,
)
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep
from nirspipe.utils.lineage import lineage_of, stage_of

from tests._synth import synth_raw

SFREQ = 10.0
N_CH = 6


def _quiet(n_t: int = 1000, seed: int = 0) -> np.ndarray:
    """Time x channels white noise far below any detection threshold."""
    return 0.01 * np.random.default_rng(seed).normal(size=(n_t, N_CH))


def _raw(d: np.ndarray, bads: tuple[int, ...] = ()) -> mne.io.Raw:
    names = [f"S{i // 2 + 1}_D1 {760 if i % 2 == 0 else 850}" for i in range(d.shape[1])]
    raw = mne.io.RawArray(d.T.copy(), mne.create_info(names, SFREQ, "fnirs_od"), verbose="error")
    raw.info["bads"] = [names[i] for i in bads]
    return raw


ARTEFACT = slice(400, 420)


def _with_shared_artefact(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A 2 s bump shared by every channel with its own gain, plus channel noise while it lasts.

    The noise keeps the bump's component under TPCA_N_SV of the variance; alone, the bump
    would carry all of it and nothing would be removed.
    """
    rng = np.random.default_rng(seed + 1)
    d = _quiet(seed=seed)
    gain = rng.uniform(0.5, 1.5, N_CH)
    bump = np.sin(np.pi * np.arange(20) / 20) ** 2
    d[ARTEFACT] += np.outer(bump, gain) + 0.15 * rng.normal(size=(20, N_CH))
    return d, gain


# ---- Detection ----

def test_a_step_flags_the_published_window():
    # step between samples 99 and 100; 5-sample window, 10 samples either side: rows 95..99
    # see it, widened to 85..109, and a row marks the sample after it
    d = _quiet()
    d[100:, 0] += 1.0
    mask = _tpca_motion_mask(d, SFREQ, np.ones(N_CH, bool))
    assert (np.flatnonzero(mask) == np.arange(86, 111)).all()


def test_half_a_sample_rounds_away_from_zero():
    # at 5 Hz the 0.5 s window is 2.5 samples: 3, not the 2 Python's round gives
    d = _quiet()
    d[100:, 0] += 1.0
    mask = _tpca_motion_mask(d, 5.0, np.ones(N_CH, bool))
    assert np.flatnonzero(mask)[0] == 100 - 3 - 5 + 1


def test_a_bad_channel_cannot_trigger_detection():
    d = _quiet()
    d[100:, 0] += 1.0
    good = np.ones(N_CH, bool)
    good[0] = False
    assert not _tpca_motion_mask(d, SFREQ, good).any()


# ---- The PCA and its stitching ----

def test_the_published_segment_bookkeeping():
    # removing the only component leaves the segment flat, placed one sample early, and the
    # last motion sample keeps its raw step
    d = np.arange(7.0)[:, None]
    motion_mask = np.array([False, False, True, True, True, False, False])
    out, n_removed = _tpca_pca(d, motion_mask, np.array([True]), 1)
    assert n_removed == 1
    assert (out[:, 0] == [0, 1, 1, 1, 2, 3, 4]).all()


def _motion_with_shares(shares: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Motion samples whose principal components carry exactly these variance shares."""
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(100, len(shares))))
    d = np.zeros((200, len(shares)))
    d[50:150] = q * np.sqrt(shares)
    mask = np.zeros(200, bool)
    mask[50:150] = True
    return d, mask


def test_a_variance_share_becomes_a_component_count():
    d, mask = _motion_with_shares([0.90, 0.08, 0.02])      # cumulative 0.90, 0.98, 1.00
    assert _tpca_pca(d, mask, np.ones(3, bool), TPCA_N_SV)[1] == 1


def test_a_single_component_over_the_share_removes_nothing():
    d, mask = _motion_with_shares([0.98, 0.015, 0.005])
    assert _tpca_pca(d, mask, np.ones(3, bool), TPCA_N_SV)[1] == 0


def test_the_shared_pattern_of_an_artefact_is_removed():
    d, gain = _with_shared_artefact()
    clean = _quiet()
    out, info = _tpca_motion_correct(_raw(d))

    def shared_swing(x):
        res = x[ARTEFACT] - clean[ARTEFACT]
        return np.ptp((res - res.mean(axis=0)) @ gain / (gain @ gain))

    assert info["components_removed"] == [1]
    assert shared_swing(out.get_data().T) < 0.1 * shared_swing(d)


def test_bad_channels_come_back_unchanged():
    d, _ = _with_shared_artefact()
    out, info = _tpca_motion_correct(_raw(d, bads=(0, 1)))
    assert info["components_removed"]                       # something was corrected
    assert (out.get_data()[:2] == d.T[:2]).all()


def test_quiet_data_passes_through_untouched():
    d = _quiet()
    out, info = _tpca_motion_correct(_raw(d))
    assert (out.get_data() == d.T).all()
    assert info == {"components_removed": [], "motion_share_before": 0.0,
                    "motion_share_after": 0.0}


# ---- The passes ----

def _spy_passes(monkeypatch, n_flagged_passes: int) -> list:
    flagged = np.zeros(1000, bool)
    flagged[400:430] = True
    masks = iter([flagged] * n_flagged_passes + [np.zeros(1000, bool)] * 10)
    monkeypatch.setattr(motion, "_tpca_motion_mask", lambda d, sfreq, good: next(masks))
    calls = []
    real = motion._tpca_pca

    def spy(d, mask, good, n_sv):
        out = real(d, mask, good, n_sv)
        calls.append((n_sv, out[1]))
        return out

    monkeypatch.setattr(motion, "_tpca_pca", spy)
    return calls


def test_later_passes_remove_the_first_passes_count(monkeypatch):
    calls = _spy_passes(monkeypatch, 3)
    _tpca_motion_correct(_raw(_with_shared_artefact()[0]))
    first_count = calls[0][1]
    assert [n_sv for n_sv, _ in calls] == [TPCA_N_SV, first_count, first_count]


def test_passes_stop_at_the_maximum(monkeypatch):
    calls = _spy_passes(monkeypatch, 100)
    _, info = _tpca_motion_correct(_raw(_with_shared_artefact()[0]))
    assert len(calls) == TPCA_MAX_ITER == len(info["components_removed"])


# ---- Wiring ----

def test_correct_motion_stamps_tpca_and_what_its_passes_did():
    out = correct_motion(_raw(_with_shared_artefact()[0]), method="tpca")
    params = lineage_of(out).params
    assert stage_of(out) == "motcorrected" and params["method"] == "tpca"
    assert set(params["tpca"]) == {"components_removed", "motion_share_before",
                                   "motion_share_after"}


def test_prep_writes_the_passes_into_the_motcorrected_sidecar(tmp_path):
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.8, motion_correction="tpca",
                        cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)
    run_prep(synth_raw("01", "rest"), config, tmp_path, source_entities={"task": "rest"})
    sidecar = json.loads(next(tmp_path.rglob("*desc-motcorrected_nirs.json")).read_text())
    assert sidecar["parameters"]["motion_correction"] == "tpca"
    assert "components_removed" in sidecar["tpca"]
