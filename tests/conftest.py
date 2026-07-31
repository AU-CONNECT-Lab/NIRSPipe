"""Shared fixtures.

Only things used by more than one test file belong here. Single-file helpers stay
in that file, so the input a test runs on is readable without jumping.
"""

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end test that needs real data")


@pytest.fixture
def make_raw():
    """Build an fNIRS Raw. Channels are named so MNE and compute_alff both accept them."""
    def _make(data=None, sfreq=10.0, n_ch=4, dur=60.0, chromo="hbo"):
        import mne

        if data is None:
            data = np.random.default_rng(0).normal(size=(n_ch, int(sfreq * dur)))
        data = np.atleast_2d(data)
        names = [f"S{i}_D{i} {chromo}" for i in range(data.shape[0])]
        info = mne.create_info(names, sfreq, [chromo] * data.shape[0])
        return mne.io.RawArray(data, info, verbose="error")

    return _make


@pytest.fixture
def fake_raw(make_raw):
    return make_raw()


# Session-scoped: writing the SNIRF files costs a few seconds, and nothing that
# consumes these datasets modifies them.

@pytest.fixture(scope="session")
def mini_bids(tmp_path_factory):
    """Two subjects, task-tapping (with events) and task-rest (without)."""
    from tests._synth import make_bids_dataset

    return make_bids_dataset(tmp_path_factory.mktemp("synth"))


@pytest.fixture(scope="session")
def mini_hyper_bids(tmp_path_factory):
    """One dyad, task-hold (shared triggers) and task-rest (none). Returns (bids, pairs_csv)."""
    from tests._synth import make_hyper_dataset

    return make_hyper_dataset(tmp_path_factory.mktemp("synth_hyper"))
