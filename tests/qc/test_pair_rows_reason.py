"""A pair rejected by screening says which criterion rejected it, as a wavelength row does."""

from nirspipe.qc.common.channel_table import channel_rows, format_rows, pair_rows


def _row(name, good_frac, is_bad):
    return {"name": name, "sci": 0.95, "psp": 0.3, "snr": 50.0, "cv": 0.02, "spike": 0,
            "corr": None, "good_frac": good_frac, "is_bad": is_bad, "separation": "long"}


def test_a_screened_out_pair_names_the_coupled_window_criterion():
    rows = [_row("S1_D1 760", 0.40, True), _row("S1_D1 850", 0.40, True),
            _row("S2_D2 760", 0.95, False), _row("S2_D2 850", 0.95, False)]
    cells = {c["name"]: c for c in format_rows(pair_rows(rows), 0.8, name_key="pair",
                                                psp_threshold=0.1)}
    # the usual reason is stated once in each table's note, so Status prints it bare
    assert cells["S1_D1"]["status"] == "BAD"
    assert "coupled windows" in cells["S1_D1"]["reason"]
    assert cells["S2_D2"]["status"] == "OK" and not cells["S2_D2"]["reason"]


def test_the_lower_share_of_the_two_wavelengths_decides():
    rows = [_row("S1_D1 760", 0.90, True), _row("S1_D1 850", 0.40, True)]
    assert pair_rows(rows)[0]["good_frac"] == 0.40


def test_a_pair_row_can_keep_both_wavelengths_of_a_per_wavelength_column():
    rows = [_row("S1_D1 850", 0.9, False) | {"snr": 13.7}, _row("S1_D1 760", 0.9, False) | {"snr": 23.7}]
    row = pair_rows(rows, by_wavelength=True)[0]
    assert row["snr"] == (23.7, 13.7), "760 first, whatever order the rows came in"
    assert pair_rows(rows)[0]["snr"] == 13.7, "the default keeps the first value"
    cell = format_rows([row], 0.8, name_key="pair")[0]
    assert cell["snr"].count(" / ") == 1
    empty = format_rows([row | {"spike": (None, None)}], 0.8, name_key="pair")[0]
    alone = format_rows([row | {"spike": None}], 0.8, name_key="pair")[0]
    assert empty["spike"] == alone["spike"], "nothing on either wavelength prints one dash"


def test_a_rejected_channel_does_not_move_the_grand_mean():
    import numpy as np

    from nirspipe.pipeline.prep_pipeline import intensity_to_od, od_to_haemo
    from nirspipe.qc.figures.subject.raw_figures import build_epoch_preview_figure
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


def test_the_correlation_prints_its_denoised_side_when_the_record_has_one():
    record = {"per_channel": {"raw": {"sci_per_channel": {"S1_D1 760": 0.9, "S1_D1 850": 0.9}},
                              "preproc": {"hbo_hbr_corr_per_channel": {"S1_D1": 0.30}},
                              "errts": {"hbo_hbr_corr_per_channel": {"S1_D1": -0.10}}}}
    rows = channel_rows(record, {"S1_D1 760": 0.9, "S1_D1 850": 0.9}, [])
    assert {(r["corr"], r["corr_post"]) for r in rows} == {(0.30, -0.10)}
    cell = format_rows(pair_rows(rows), 0.8, name_key="pair")[0]
    assert cell["corr"].count("→") == 1 and cell["corr_denoised"]
    assert cell["corr_cls"] == "neg", "coloured by the denoised side"

    del record["per_channel"]["errts"]
    plain = format_rows(pair_rows(channel_rows(record, {"S1_D1 760": 0.9}, [])), 0.8,
                        name_key="pair")[0]
    assert "→" not in plain["corr"] and "corr_denoised" not in plain
