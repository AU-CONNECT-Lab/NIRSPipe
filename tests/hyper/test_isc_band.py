"""ISC reads whatever band the preprocessing left, unless it is given one.

Without a band the correlation is dominated by whatever is slowest or loudest in the
residual, which on a stage with no low-pass is the cardiac component. That also makes it
incomparable with the WTC band mean, which the two being averages of the same complex
coherency is the whole reason for wanting.
"""

import logging

import mne
import numpy as np
import pytest

from fnirs_pipe.cli.hyper import _warn_band_mismatch
from fnirs_pipe.pipeline.hyper.isc import _band_limit, compute_isc

SFREQ = 10.0
N = 4000
BAND = (0.06, 0.15)


def _pair(shared_hz: float, seed: int = 0):
    """Two members sharing one oscillation and nothing else, as aligned hbo Raws."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SFREQ
    shared = np.sin(2 * np.pi * shared_hz * t)
    raws = {}
    for i, sid in enumerate(("sub-01", "sub-02")):
        data = np.vstack([shared + 0.3 * rng.standard_normal(N) for _ in range(2)]) * 1e-6
        info = mne.create_info(["S1_D1 hbo", "S2_D2 hbo"], SFREQ, ["hbo", "hbo"])
        for ch, (src, det) in zip(info["chs"], ((0, 0), (1, 1))):
            ch["loc"][3:6] = [src * 0.03, 0.0, 0.0]
            ch["loc"][6:9] = [src * 0.03 + 0.03, 0.0, 0.0]
        raws[sid] = mne.io.RawArray(data, info, verbose="error")
    return raws


def test_band_limit_leaves_all_nan_rows_alone():
    data = np.vstack([np.random.default_rng(0).standard_normal(N), np.full(N, np.nan)])
    out = _band_limit(data, SFREQ, BAND)
    assert np.isnan(out[1]).all()
    assert np.isfinite(out[0]).all()


def test_band_limit_removes_what_is_outside_the_band():
    t = np.arange(N) / SFREQ
    data = np.sin(2 * np.pi * 1.0 * t)[None, :]          # cardiac-ish, far above the band
    out = _band_limit(data, SFREQ, BAND)
    # ignore the filter's own settling at both ends
    assert np.abs(out[0, 500:-500]).max() < 0.05 * np.abs(data).max()


def test_band_limit_keeps_what_is_inside():
    t = np.arange(N) / SFREQ
    data = np.sin(2 * np.pi * 0.1 * t)[None, :]
    out = _band_limit(data, SFREQ, BAND)
    kept = np.abs(out[0, 500:-500]).max() / np.abs(data).max()
    assert kept > 0.9, kept


def test_isc_without_a_band_sees_coupling_the_wtc_band_never_would():
    """The point of the flag: a 1 Hz shared component is coupling ISC reports and WTC cannot."""
    raws = _pair(shared_hz=1.0)
    wide, _ = compute_isc(raws, ["sub-01", "sub-02"], "hbo")
    narrow, _ = compute_isc(raws, ["sub-01", "sub-02"], "hbo", band=BAND)
    assert np.nanmean(np.diag(wide)) > 0.8
    assert np.nanmean(np.diag(narrow)) < 0.5


def test_a_band_does_not_destroy_in_band_coupling():
    raws = _pair(shared_hz=0.1)
    narrow, _ = compute_isc(raws, ["sub-01", "sub-02"], "hbo", band=BAND)
    assert np.nanmean(np.diag(narrow)) > 0.8


# ---- the two bands are checked against each other, never inherited ----
# A flag that silently moved a second metric could not be read off the command line it was
# absent from, so ISC takes only what it was given and a mismatch is reported instead.

def _warnings(caplog, isc_band, wtc=(0.06, 0.15)):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="fnirs_pipe.cli.hyper"):
        _warn_band_mismatch(isc_band, *wtc)
    return [r.getMessage() for r in caplog.records]


def test_an_unbanded_isc_beside_a_banded_coherence_warns(caplog):
    assert any("whole passband" in m for m in _warnings(caplog, None))


def test_two_different_bands_warn(caplog):
    assert any("not comparable" in m for m in _warnings(caplog, (0.02, 0.10)))


def test_the_same_band_is_quiet(caplog):
    assert _warnings(caplog, (0.06, 0.15)) == []


def test_no_coherence_band_means_nothing_to_compare(caplog):
    assert _warnings(caplog, None, wtc=(None, None)) == []


@pytest.mark.parametrize("isc_band, warned", [
    (None, True),
    ((None, 0.1), True),     # a low-pass alone leaves the drift in
    ((0.01, None), False),   # a low edge on the correlation keeps it out
    ((0.01, 0.1), False),
])
def test_the_unfiltered_note_stands_down_when_the_correlation_is_high_passed(isc_band, warned):
    from fnirs_pipe.pipeline.hyper.group_io import unfiltered_stage_note

    raws = _pair(0.05)
    assert (unfiltered_stage_note(raws, isc_band) is not None) is warned
