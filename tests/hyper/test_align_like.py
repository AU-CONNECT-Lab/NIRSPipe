import mne
import numpy as np

from nirspipe.pipeline.hyper.alignment import align_like

# a rate whose sample period is not exact in binary, so tmin + duration overshoots the end
SFREQ = 5.0863


def _raw(n=12000):
    return mne.io.RawArray(np.zeros((1, n)), mne.create_info(["a"], SFREQ, "misc"),
                           verbose="ERROR")


def test_window_ending_at_the_last_sample_survives_float_round_off():
    raw = _raw()
    overshoots = 0
    for k in range(1, 200):
        ref = raw.copy().crop(tmin=k / SFREQ)
        tmin = float(ref.first_time) - float(raw.first_time)
        overshoots += tmin + float(ref.times[-1]) > float(raw.times[-1])
        out = align_like({"sub-01": raw}, {"sub-01": ref})
        assert out["sub-01"].n_times == ref.n_times
    assert overshoots, "no offset reproduced the round-off; the test is not exercising it"


def test_window_past_the_end_is_still_dropped():
    raw = _raw()
    longer = _raw(13000)
    assert align_like({"sub-01": raw}, {"sub-01": longer.copy().crop(tmin=1.0)}) == {}


def test_a_window_cut_after_alignment_moves_the_offset_with_it():
    # the dyad raw report cuts the aligned recordings to --tstart/--tend; the offsets it
    # converts each member's own clock with have to include that cut
    from nirspipe.pipeline.hyper.alignment import aligned_offsets, crop_aligned_window

    raws = {"sub-01": _raw(), "sub-02": _raw()}
    aligned = {"sub-01": raws["sub-01"].copy().crop(tmin=10.0),
               "sub-02": raws["sub-02"].copy().crop(tmin=30.0)}
    cut = crop_aligned_window(aligned, 60.0, None)

    offsets = aligned_offsets(raws, cut)
    assert abs(offsets["sub-01"] - 70.0) < 1 / SFREQ
    assert abs(offsets["sub-02"] - 90.0) < 1 / SFREQ
