"""A pair rejected by screening says which criterion rejected it, as a wavelength row does."""

from fnirs_pipe.qc.common.channel_table import format_rows, pair_rows


def _row(name, good_frac, is_bad):
    return {"name": name, "sci": 0.95, "psp": 0.3, "snr": 50.0, "cv": 0.02, "spike": 0,
            "corr": None, "good_frac": good_frac, "is_bad": is_bad, "separation": "long"}


def test_a_screened_out_pair_names_the_coupled_window_criterion():
    rows = [_row("S1_D1 760", 0.40, True), _row("S1_D1 850", 0.40, True),
            _row("S2_D2 760", 0.95, False), _row("S2_D2 850", 0.95, False)]
    cells = {c["name"]: c for c in format_rows(pair_rows(rows), 0.8, name_key="pair",
                                                psp_threshold=0.1)}
    assert "coupled windows" in cells["S1_D1"]["status"]
    assert "coupled windows" not in cells["S2_D2"]["status"]


def test_the_lower_share_of_the_two_wavelengths_decides():
    rows = [_row("S1_D1 760", 0.90, True), _row("S1_D1 850", 0.40, True)]
    assert pair_rows(rows)[0]["good_frac"] == 0.40


def test_a_rejected_channel_does_not_move_the_grand_mean():
    import numpy as np

    from fnirs_pipe.pipeline.prep_pipeline import intensity_to_od, od_to_haemo
    from fnirs_pipe.qc.figures.subject.raw_figures import build_epoch_preview_figure
    from tests._synth import synth_raw

    haemo = od_to_haemo(intensity_to_od(synth_raw("01", "tapping")), [6.0])
    bad = next(n for n in haemo.ch_names if n.endswith("hbo"))
    haemo.info["bads"] = [bad, bad.replace("hbo", "hbr")]
    spiked = haemo.copy()
    # a ramp, since the baseline correction would take a constant offset straight back out
    spiked._data[haemo.ch_names.index(bad)] += np.linspace(0.0, 1e-3, haemo.n_times)

    before = build_epoch_preview_figure(haemo)
    after = build_epoch_preview_figure(spiked)
    assert before is not None and len(before.data) == len(after.data)
    for a, b in zip(before.data, after.data):
        if getattr(a, "y", None) is not None:
            np.testing.assert_allclose(np.asarray(a.y, float), np.asarray(b.y, float))
