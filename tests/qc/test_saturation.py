"""A channel held at the detector's ceiling is counted, per wavelength and per channel set."""

import numpy as np
import pytest

from nirspipe.interface.callbacks.data_prep_callbacks import _CH_COLUMNS as GUI_COLUMNS
from nirspipe.qc.boilerplate.vocabulary import format_metric
from nirspipe.qc.common.channel_table import channel_rows, format_rows, pair_rows
from nirspipe.qc.metrics import compute_sci_scores, saturated_share
from nirspipe.qc.subject.condition_views import UNSLICEABLE
from nirspipe.qc.subject.sqm_record import raw_sections
from tests._synth import synth_raw

CARDIAC = (0.7, 1.5)
SHORT_850 = "S5_D5 850"
LONG_760 = "S2_D2 760"


def test_a_held_ceiling_counts_and_a_lone_peak_or_a_dead_channel_does_not():
    data = np.array([[0.2] * 70 + [1.0] * 30,
                     [0.2] * 49 + [1.0] + [0.2] * 50,
                     [0.0] * 100])
    assert saturated_share(data, 1.0).tolist() == [0.3, 0.0, 0.0]


@pytest.fixture(scope="module")
def clipped():
    """One short channel's 850 nm pinned for a quarter of the run; one long channel held
    for two samples at a peak above every other long sample but below that ceiling."""
    raw = synth_raw("01", "rest", duration=120.0, n_long_pairs=4, bad_pair=None,
                    motion_onset=None)
    data = raw._data
    ceiling = data.max() + 1.0
    short = raw.ch_names.index(SHORT_850)
    data[short, :raw.n_times // 4] = ceiling
    long_ = raw.ch_names.index(LONG_760)
    long_rows = [i for i, ch in enumerate(raw.ch_names) if not ch.startswith("S5_D5")]
    data[long_, 100:102] = data[long_rows].max() + 0.5
    sci, _ = compute_sci_scores(raw, *CARDIAC)
    sections, per_channel = raw_sections(raw, sci, [], *CARDIAC)
    return raw, sci, sections, per_channel


def test_each_set_counts_against_the_whole_recording_s_ceiling(clipped):
    _, _, sections, per_channel = clipped
    assert sections["raw"]["n_saturated_channels"] == 1
    assert sections["raw_short"]["n_saturated_channels"] == 1
    # the long set's own maximum is the held peak; against it this set would count one
    assert sections["raw_long"]["n_saturated_channels"] == 0
    assert per_channel["raw_short"]["saturated_per_channel"][SHORT_850] == pytest.approx(0.25)
    assert per_channel["raw_long"]["saturated_per_channel"][LONG_760] == 0.0


def test_the_per_channel_table_prints_both_wavelengths(clipped):
    _, sci, _, per_channel = clipped
    rows = pair_rows(channel_rows({"per_channel": per_channel}, sci, []), by_wavelength=True)
    shown = {r["name"]: r["saturated"] for r in format_rows(rows, 0.8, name_key="pair")}
    assert shown["S5_D5"] == (f"{format_metric('saturated_pct', 0.0)} / "
                              f"{format_metric('saturated_pct', 0.25)}")


def test_a_condition_page_prints_a_dash_and_the_gui_table_stays_as_it_was():
    assert "saturated_per_channel" in UNSLICEABLE
    assert "saturated" not in {key for key, _ in GUI_COLUMNS}
