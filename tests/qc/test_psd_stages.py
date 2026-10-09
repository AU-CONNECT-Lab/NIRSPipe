"""Which rows the PSD panel draws, and which row the filter response may be drawn over."""

import numpy as np
import mne

from nirspipe.qc.figures.subject.psd_plot import psd_figure


def _haemo(sfreq: float = 5.0, dur: float = 200.0, n_pairs: int = 3) -> mne.io.Raw:
    """A minimal HbO/HbR Raw: the figure only reads channel names, rate and samples."""
    n = int(sfreq * dur)
    rng = np.random.default_rng(0)
    names, types = [], []
    for i in range(n_pairs):
        names += [f"S1_D{i + 1} hbo", f"S1_D{i + 1} hbr"]
        types += ["hbo", "hbr"]
    info = mne.create_info(names, sfreq, types)
    return mne.io.RawArray(rng.standard_normal((len(names), n)) * 1e-6, info, verbose=False)


def _titles(fig, n_rows: int) -> list[str]:
    """The subplot titles only. The band shading adds annotations of its own after these."""
    return [a.text for a in fig.layout.annotations[:n_rows]]


def _has_response(fig) -> bool:
    return any(t.name == "Filter response" for t in fig.data)


def test_no_bandpass_draws_no_response_curve():
    """The regression-only run: no cutoffs, so no curve and no row named for a bandpass."""
    raw = _haemo()
    fig = psd_figure(raw, l_freq=None, h_freq=None,
                     stages=[("desc-errts", _haemo())])
    titles = _titles(fig, 2)
    assert titles == ["desc-preproc", "desc-errts"]
    assert not _has_response(fig)


def test_response_curve_follows_the_filtered_row():
    """With desc-resampled and desc-errts stacked after it, the curve stays on desc-filtered."""
    raw = _haemo()
    fig = psd_figure(
        raw, l_freq=0.01, h_freq=0.2,
        stages=[("desc-filtered", _haemo()), ("desc-resampled", _haemo()),
                ("desc-errts", _haemo())],
    )
    titles = _titles(fig, 4)
    assert titles[1].startswith("After bandpass")
    assert titles[2:] == ["desc-resampled", "desc-errts"]
    assert _has_response(fig)
    response = next(t for t in fig.data if t.name == "Filter response")
    # plotly names the second row's y axis "y2"; the curve must sit there, not on a later row
    assert response.yaxis == "y2"


def test_errts_first_is_not_renamed_after_bandpass():
    """A resample-only run puts desc-resampled second; it is not the filter's output."""
    raw = _haemo()
    fig = psd_figure(raw, l_freq=None, h_freq=None,
                     stages=[("desc-resampled", _haemo()), ("desc-errts", _haemo())])
    assert _titles(fig, 3)[1] == "desc-resampled"
    assert not _has_response(fig)
