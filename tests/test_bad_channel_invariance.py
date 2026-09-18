"""A good channel's metric must not depend on which *other* channels are bad.

`test_bad_channel_exclusion.py` pins three named functions against a leak. This is the
general statement behind them, and it is the shape of the bug that prompted them:
`_spectral_metrics` indexed a PSD array with indices taken from `raw.info`, but
`Spectrum.get_data()` drops bad channels, so the array was shorter than the index space. It
raised only once enough channels were bad, and `@_safe_metrics` turned the exception into
eight silently missing metrics. Clean data never showed it.

So there are two assertions here and the second matters as much as the first:

1. per-channel values for a channel good in both runs are identical, and
2. no key drops out or turns `None` under heavy exclusion.

`@_safe_metrics` is convenient in production and dangerous in a test, because a function
that has started raising looks like a function that returned nothing. Asserting on key
presence is what separates the two.
"""

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from fnirs_pipe.qc.metrics import (
    compute_prep_haemo_sqm,
    compute_raw_sqm,
    compute_sci_scores,
)
from fnirs_pipe.utils.lineage import stamp
from ._synth import synth_raw

CARDIAC = (0.7, 1.5)
RESP = (0.15, 0.4)
N_PAIRS = 8


@pytest.fixture(scope="module")
def intensity():
    return synth_raw("01", "rest", duration=120.0, n_long_pairs=N_PAIRS,
                     bad_pair=None, motion_onset=None)


@pytest.fixture(scope="module")
def haemo(intensity):
    od = mne.preprocessing.nirs.optical_density(intensity.copy(), verbose="error")
    return stamp(mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0),
                 stage="preproc", step="beer_lambert")


def _haemo_record(haemo, bads):
    marked = haemo.copy()
    marked.info["bads"] = list(bads)
    return compute_prep_haemo_sqm(marked, *CARDIAC, *RESP)


def _raw_record(intensity, sci, bads):
    marked = intensity.copy()
    marked.info["bads"] = list(bads)
    return compute_raw_sqm(marked, sci, list(bads), *CARDIAC)


def _per_channel(record):
    return {k: v for k, v in record.items() if isinstance(v, dict) and v}


# ---- the invariant ----

def test_a_surviving_channel_keeps_its_value_whoever_else_is_bad(haemo):
    names = haemo.ch_names
    first = _haemo_record(haemo, names[0:4])
    second = _haemo_record(haemo, names[4:8])

    compared = 0
    for key, values in _per_channel(first).items():
        for channel in set(values) & set(second[key]):
            assert_allclose(values[channel], second[key][channel], rtol=1e-12,
                            err_msg=f"{key} moved for {channel}")
            compared += 1
    assert compared, "no per-channel metric survived both exclusions, so nothing was tested"


def test_the_per_channel_dicts_hold_the_good_channels_and_only_those(haemo):
    # these dicts are keyed by source-detector pair ("S1_D1"), not by channel name
    # ("S1_D1 hbo"), so comparing the two key spaces directly would intersect at nothing
    # and assert nothing
    names = haemo.ch_names
    bads = names[0:4]
    excluded = {name.split()[0] for name in bads}
    record = _haemo_record(haemo, bads)

    checked = 0
    for key, values in _per_channel(record).items():
        assert not set(values) & excluded, f"{key} kept a bad pair"
        checked += 1
    assert checked, "no per-channel dict was inspected"


# ---- the regression guard: heavy exclusion must not silence a metric ----

def test_no_metric_goes_missing_when_most_channels_are_bad(haemo):
    # the original defect needed 16 of 56 bad before it raised, so a two-channel case
    # would not have caught it. Leave two pairs standing and nothing more
    names = haemo.ch_names
    clean = _haemo_record(haemo, [])
    heavy = _haemo_record(haemo, names[:-4])

    assert set(heavy) == set(clean), "a key vanished under heavy exclusion"
    silenced = [k for k, v in heavy.items() if v is None and clean[k] is not None]
    assert not silenced, f"@_safe_metrics swallowed an exception into: {silenced}"


def test_the_spectral_metrics_specifically_survive_it(haemo):
    # named explicitly because these eight are the ones that went missing, and a future
    # refactor that reintroduces info-derived indexing would land here first
    names = haemo.ch_names
    heavy = _haemo_record(haemo, names[:-4])
    for key in ("cardiac_band_power_hbo", "cardiac_band_power_hbr",
                "cardiac_band_frac_hbo", "cardiac_band_frac_hbr",
                "resp_band_power_hbo", "resp_band_power_hbr",
                "resp_band_frac_hbo", "resp_band_frac_hbr"):
        assert heavy[key] is not None, key


# ---- positive control ----

def test_corrupting_one_good_channel_moves_that_channel_and_no_other(haemo):
    # without this, a function that had stopped reading channel data at all would pass
    # every assertion above. Rescaling is not enough to show it: GCOR normalises each
    # channel to unit length and hbo_hbr_corr is a correlation, so both are scale-blind by
    # construction. Replacing the data is what a corrupted channel actually looks like
    target = "S5_D5"
    baseline = _haemo_record(haemo, [])

    corrupted = haemo.copy()
    index = corrupted.ch_names.index(f"{target} hbo")
    corrupted._data[index] = np.random.default_rng(0).normal(scale=1e-6, size=corrupted.n_times)
    moved = compute_prep_haemo_sqm(corrupted, *CARDIAC, *RESP)

    before = baseline["hbo_hbr_corr_per_channel"]
    after = moved["hbo_hbr_corr_per_channel"]
    assert not np.isclose(before[target], after[target]), "the corruption was not seen at all"
    for pair in set(before) - {target}:
        assert_allclose(before[pair], after[pair], rtol=1e-12,
                        err_msg=f"{pair} moved because a different channel was corrupted")


# ---- raw SQM is deliberately computed before exclusion ----

def test_raw_metrics_ignore_the_bad_list_because_they_are_what_decides_it(intensity):
    # raw SQM is the evidence a channel is judged on, so it covers every channel whatever
    # info["bads"] says. Pinning this stops a well-meaning "exclude bads here too" change
    # from quietly emptying the record the exclusion is derived from
    sci, _ = compute_sci_scores(intensity, *CARDIAC)
    names = intensity.ch_names
    first = _raw_record(intensity, sci, names[0:2])
    second = _raw_record(intensity, sci, names[4:6])

    for key, values in _per_channel(first).items():
        assert set(values) == set(names), f"{key} dropped a channel"
        for channel in values:
            assert_allclose(values[channel], second[key][channel], rtol=1e-12,
                            err_msg=f"{key} moved for {channel}")


def test_only_the_retention_accounting_reads_the_bad_list(intensity):
    sci, _ = compute_sci_scores(intensity, *CARDIAC)
    names = intensity.ch_names
    two_bad = _raw_record(intensity, sci, names[0:2])
    four_bad = _raw_record(intensity, sci, names[0:4])

    assert two_bad["channel_retention_rate"] > four_bad["channel_retention_rate"]
    moved = [k for k, v in two_bad.items()
             if isinstance(v, float) and not np.isclose(v, four_bad[k])]
    assert moved == ["channel_retention_rate"], f"unexpectedly sensitive to bads: {moved}"
