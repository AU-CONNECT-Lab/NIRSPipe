"""The fingerprint reaches the end of the pipeline intact, so a figure that loses it is the figure's fault."""

import json

import numpy as np
import pytest

from tests._fingerprint import HBR_FREQ, RESPONSE_AMP, SYSTEMIC_FREQ
from tests.figure_accuracy._read import evoked_peak, share_at, which_pair


def _stage(run, desc):
    raw = run.read(desc)
    return raw.times, dict(zip(raw.ch_names, raw.get_data()))


@pytest.fixture(scope="module")
def errts(denoise_run):
    return _stage(denoise_run, "errts")


@pytest.fixture(scope="module")
def preproc(denoise_run):
    return _stage(denoise_run, "preproc")


def test_screening_rejects_the_uncoupled_pair_and_nothing_else(denoise_run):
    sidecar = json.loads(denoise_run.stage("sci").with_suffix(".json").read_text())
    bad = {ch.rsplit(" ", 1)[0] for ch in sidecar["bad_channels"]}
    assert bad == {p.name for p in denoise_run.truth.pairs if p.bad}


def test_each_long_pair_is_still_identified_by_its_own_frequency(denoise_run, errts):
    times, data = errts
    for pair in denoise_run.truth.long_pairs:
        if not pair.bad:
            assert which_pair(times, data[f"{pair.name} hbo"], denoise_run.truth) == pair.name


def test_only_hbr_carries_the_hbr_mark(denoise_run, errts):
    times, data = errts
    for pair in denoise_run.truth.long_pairs:
        if not pair.bad:
            assert share_at(times, data[f"{pair.name} hbr"], HBR_FREQ) > 0.03
            assert share_at(times, data[f"{pair.name} hbo"], HBR_FREQ) < 0.01


def test_the_systemic_oscillation_is_there_before_regression_and_gone_after(
        denoise_run, preproc, errts):
    for pair in denoise_run.truth.long_pairs:
        if not pair.bad:
            assert share_at(preproc[0], preproc[1][f"{pair.name} hbo"], SYSTEMIC_FREQ) > 0.1
            assert share_at(errts[0], errts[1][f"{pair.name} hbo"], SYSTEMIC_FREQ) < 0.02


def test_only_the_responders_answer_the_events(denoise_run, errts):
    times, data = errts
    for pair in denoise_run.truth.pairs:
        if pair.bad:
            continue
        peak = evoked_peak(times, data[f"{pair.name} hbo"])
        if pair.responds:
            assert peak > 0.7 * RESPONSE_AMP
        else:
            assert peak < 0.15 * RESPONSE_AMP


def test_hbo_hbr_correlation_matches_the_design_after_regression(denoise_run, errts):
    times, data = errts
    middle = (times > 100) & (times < 300)
    for pair in denoise_run.truth.long_pairs:
        # a motion artefact the correction leaves behind is shared by both chromophores
        if pair.bad or pair.name in denoise_run.truth.moved:
            continue
        clean = denoise_run.truth.haemo_no_systemic
        expected = np.corrcoef(clean[f"{pair.name} hbo"][middle], clean[f"{pair.name} hbr"][middle])[0, 1]
        measured = np.corrcoef(data[f"{pair.name} hbo"][middle], data[f"{pair.name} hbr"][middle])[0, 1]
        assert measured == pytest.approx(expected, abs=0.15)
