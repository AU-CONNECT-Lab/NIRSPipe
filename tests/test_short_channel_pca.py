"""`--short-channel pca`: every short channel as its own column, orthogonalised.

The published comparison ranks the nonselective strategies above picking one short channel
per long one, and puts the pooled decomposition above the mean. What it does not do, and
what neither AnalyzIR nor the paper's own fork of it does, is drop components: the basis is
kept whole. That matters for reading the result, because a full-rank orthonormal basis spans
exactly what the raw short channels span, so the fit is the one entering them all would give
and the decomposition is only there to keep the columns from being collinear. The first test
here is that equivalence, since it is what the methods text has to be honest about.

Everything the mean strategy refuses, this refuses the same way: a montage with no short
channel at all, and a chromophore whose short channels were all rejected.
"""

import numpy as np
import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.glm import _short_channel_basis, _short_channel_regressors


def _block(n_ch=6, n_times=600, rank=None, seed=0):
    """Short-channel data, (n_channels, n_times), sharing a systemic component."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n_times)
    rows = [common * (0.5 + 0.1 * i) + 0.3 * rng.standard_normal(n_times) for i in range(n_ch)]
    if rank is not None:                       # duplicate rows to make it rank deficient
        rows = [rows[i % rank] for i in range(n_ch)]
    return np.array(rows)


def _fit(y, block):
    """Residual of an intercept plus `block` as columns."""
    X = np.column_stack([np.ones(len(y)), block.T])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


# ---- the equivalence the methods text rests on ----

def test_the_basis_spans_what_the_short_channels_span():
    """Keeping every component leaves the residual a raw-channel fit would leave."""
    data = _block()
    y = 0.8 * data[0] + 0.2 * np.random.default_rng(1).standard_normal(data.shape[1])

    raw_resid = _fit(y, data)
    basis = np.array(list(_short_channel_basis(data).values()))
    assert np.abs(_fit(y, basis) - raw_resid).max() < 1e-9


def test_dropping_components_would_change_the_fit():
    """The other half of the claim: keeping all of them is a choice, not a no-op."""
    data = _block()
    y = 0.8 * data[0] + 0.2 * np.random.default_rng(1).standard_normal(data.shape[1])

    basis = np.array(list(_short_channel_basis(data).values()))
    truncated = _fit(y, basis[:2])
    assert np.abs(truncated - _fit(y, data)).max() > 1e-3


# ---- the decomposition itself ----

def test_one_component_per_short_channel():
    assert len(_short_channel_basis(_block(n_ch=6))) == 6


def test_the_components_are_orthogonal():
    basis = np.array(list(_short_channel_basis(_block()).values()))
    gram = basis @ basis.T
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.abs(off_diagonal).max() < 1e-8 * np.abs(np.diag(gram)).max()


def test_each_component_is_scaled_to_unit_variance():
    """So the columns sit beside the drift basis on one scale, as AnalyzIR also does."""
    basis = np.array(list(_short_channel_basis(_block()).values()))
    assert np.allclose(basis.std(axis=1), 1.0)


def test_a_rank_deficient_block_loses_only_the_empty_directions():
    """Two short channels carrying the same signal are one direction, not two."""
    assert len(_short_channel_basis(_block(n_ch=6, rank=3))) == 3


def test_the_column_names_sort_in_order():
    names = list(_short_channel_basis(_block(n_ch=12)))
    assert names == sorted(names)
    assert names[0] == "short_ch_pca01"


# ---- the strategy argument ----

def test_an_unknown_strategy_is_refused_by_name(fake_raw):
    with pytest.raises(ValueError, match="'mean' or 'pca'"):
        _short_channel_regressors(fake_raw, "nearest")


def test_a_config_file_saying_true_still_means_mean(monkeypatch, fake_raw):
    """`short_channel = true` in a TOML predates there being a strategy to name."""
    captured = {}

    def fake_short(raw, sep_bands=None):
        captured["called"] = True
        return ([], [])

    monkeypatch.setattr("fnirs_pipe.qc.metrics._helpers.long_short_channels", fake_short)
    with pytest.raises(StageError):          # no short channels, which is the next check
        _short_channel_regressors(fake_raw, True)
    assert captured["called"]
