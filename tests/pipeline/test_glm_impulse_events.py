"""A zero-duration task event is refused: its beta would scale with the sampling rate."""
import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.pipeline.glm import refuse_impulse_events, run_glm_pipeline


def _events(*durations):
    return pd.DataFrame({"trial_type": ["tap", "rest"][:len(durations)],
                         "onset": [10.0, 40.0][:len(durations)], "duration": list(durations)})


def test_a_zero_duration_names_its_condition():
    with pytest.raises(ValueError, match="1 event.* of rest last 0 s"):
        refuse_impulse_events(_events(5.0, 0.0))


def test_any_positive_duration_passes():
    refuse_impulse_events(_events(0.1, 20.0))


def test_the_glm_stops_before_fitting():
    raw = mne.io.RawArray(np.zeros((2, 600)),
                          mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], 10.0, ["hbo", "hbr"]),
                          verbose=False)
    with pytest.raises(ValueError, match="scales with the sampling rate"):
        run_glm_pipeline(raw, stim_dur=None, hrf_model="spm", noise_model="ols",
                         drift_model="cosine", high_pass=0.01, drift_order=None,
                         fir_delays=None, events=_events(0.0))
