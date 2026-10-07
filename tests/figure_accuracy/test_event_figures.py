"""The event figures: the events where they happened, and the response on the pairs that responded."""

import re

import mne
import numpy as np
import pytest

from tests._fingerprint import EVENT_DURATION, EVENT_ONSETS, RESPONSE_AMP
from tests.figure_accuracy._payload import _DECODER, _decode, one_figure, plotly_figures
from tests.figure_accuracy._read import traces, xy

TMIN, TMAX = -5.0, 25.0


@pytest.fixture(scope="module")
def evoked(denoise_run):
    """The response the event figures should draw: trials of desc-filtered, baseline before onset."""
    raw = denoise_run.read("filtered")
    events, event_id = mne.events_from_annotations(raw, verbose="error")
    epochs = mne.Epochs(raw, events, event_id, tmin=TMIN, tmax=TMAX, baseline=(TMIN, 0),
                        preload=True, reject_by_annotation=False, verbose="error")
    return epochs


def _kept_long(run, chroma):
    return [f"{p.name} {chroma}" for p in run.truth.long_pairs if not p.bad]


def test_the_timeline_bars_are_the_events(denoise_run):
    fig = one_figure(denoise_run.figure("trigger"))
    for bar in fig["data"]:
        starts = np.asarray(bar.get("base"), float) if bar.get("base") is not None else None
        x = np.asarray(bar["x"], float)
        # a horizontal bar is drawn from base to base + x
        if starts is not None and len(starts) == len(EVENT_ONSETS):
            np.testing.assert_allclose(starts, EVENT_ONSETS, atol=1e-6)
            np.testing.assert_allclose(x, EVENT_DURATION, atol=1e-6)
        else:
            np.testing.assert_allclose(x, EVENT_ONSETS, atol=1e-6)


@pytest.mark.parametrize("label, chroma", [("HBO long", "hbo"), ("HBR long", "hbr")])
def test_the_grand_mean_is_the_trial_average_over_the_kept_long_channels(denoise_run, evoked, label, chroma):
    x, y = xy(traces(one_figure(denoise_run.figure("epochmean")), label)[0])
    expected = evoked.average(picks=_kept_long(denoise_run, chroma)).data.mean(axis=0) * 1e6
    np.testing.assert_allclose(x, evoked.times, atol=1e-6)
    np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-4)


def test_the_grand_mean_carries_the_responders_share_of_the_response(denoise_run):
    _, y = xy(traces(one_figure(denoise_run.figure("epochmean")), "HBO long")[0])
    kept = [p for p in denoise_run.truth.long_pairs if not p.bad]
    share = sum(p.responds for p in kept) / len(kept)
    assert y.max() == pytest.approx(share * RESPONSE_AMP * 1e6, rel=0.25)


def _frames(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Plotly\.addFrames\(\s*'[^']+'\s*,\s*", text)
    frames, _ = _DECODER.raw_decode(text, match.end())
    return {float(f["name"]): f for f in _decode(frames)}


def test_the_evoked_map_colours_each_pair_by_its_own_response(denoise_run, evoked):
    path = denoise_run.figure("evokedtopo", suffix="nirsmap")
    fig = one_figure(path)
    frames = _frames(path)
    average = evoked.average(picks="all")
    long_hbo = next(i for i, t in enumerate(fig["data"]) if "text" in t and (t.get("xaxis") or "x") == "x")
    names = list(fig["data"][long_hbo]["text"])
    for second in (6.0, 10.0, 15.0):
        frame = frames[second]
        colours = np.asarray(frame["data"][frame["traces"].index(long_hbo)]["marker"]["color"], float)
        sample = np.argmin(np.abs(average.times - second))
        for pair in denoise_run.truth.long_pairs:
            if pair.bad:
                continue
            drawn = colours[names.index(pair.name)]
            expected = average.copy().pick([f"{pair.name} hbo"]).data[0, sample] * 1e6
            assert drawn == pytest.approx(expected, abs=0.01), (pair.name, second)
            if second == 10.0:
                assert (drawn > 0.4) if pair.responds else (abs(drawn) < 0.15), pair.name


def test_each_trial_image_is_its_channel_s_trials(denoise_run, evoked):
    for pair in denoise_run.truth.long_pairs:
        if pair.bad:
            continue
        fig = plotly_figures(denoise_run.figure("trialimage", chan=f"{pair.name.replace('_', '')}hbo"))[0]
        heat = next(t for t in fig["data"] if t["type"] == "heatmap")
        trials = evoked.get_data(picks=[f"{pair.name} hbo"])[:, 0, :] * 1e6
        np.testing.assert_allclose(np.asarray(heat["z"], float), trials, rtol=1e-4, atol=1e-3)
        peak = trials.mean(axis=0).max()
        assert (peak > 0.6) if pair.responds else (peak < 0.2), pair.name
