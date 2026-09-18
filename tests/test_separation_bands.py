"""The long/short separation rule: one definition, configurable, and stamped.

Four things are worth pinning here, and the first three have each failed on their own.

The *rule* itself: two bands that do not meet, so a channel between them belongs to
neither. Channel screening cannot stand in for the lower edge, because a 12 mm channel
scores well for being a strong clean *scalp* signal, so a montage carrying one has to be
told apart from a montage that does not.

The *threading*: one run resolves the bands once and every path reads that value. Half a
threading is worse than a constant, since the reports would then describe a different
montage than the regression used. The consistency test at the end is what enforces that.

The *stamp*: the record carries the bands it was split with, so a reader of an old record
can tell which separations produced its numbers rather than assuming today's defaults.

The *read-back*: `fnirs-hyper run` takes the bands off the members' records rather than
off its own flags, so the dyad metrics cannot be split one way while the member reports
were split another. Two members prepped differently are refused rather than reconciled.

Distances are built by hand rather than taken from `_synth`, whose montage is 8 mm and
30 mm only and so touches neither edge. An fNIRS channel's `loc` is
`[midpoint(3), source(3), detector(3), wavelength(1)]`: setting `loc[:3]` and `loc[3:6]`
instead of `loc[3:6]` and `loc[6:9]` makes every separation read as zero.
"""

import logging

import mne
import numpy as np
import pytest

from fnirs_pipe.cli._shared import separation_bands_from_args
from fnirs_pipe.io.snirf import has_short_channels, long_channel_picks
from fnirs_pipe.qc.metrics._helpers import (
    _ORPHANS_WARNED,
    LONG_MAX_DIST,
    LONG_MIN_DIST,
    SHORT_MAX_DIST,
    bands_from_record,
    bands_to_record,
    long_short_channels,
    separation_bands,
    validate_bands,
)

SFREQ = 10.0


@pytest.fixture(autouse=True)
def _forget_warned_montages():
    """`_ORPHANS_WARNED` is module state, so a test that warns would silence the next
    one for the same montage and make these depend on execution order."""
    _ORPHANS_WARNED.clear()
    yield
    _ORPHANS_WARNED.clear()


def _montage(distances_mm, ch_type="fnirs_cw_amplitude", positioned=True):
    """A Raw whose channels sit at the given separations, one channel each.

    ``[8, 30]`` -> channels "S1_D1" at 8 mm and "S2_D2" at 30 mm.

    ``positioned=False`` leaves every loc at zero, which is what a recording with no
    registered optodes looks like and what mne_nirs used to read as "every channel short".
    """
    names, locs = [], []
    for i, mm in enumerate(distances_mm):
        src = np.array([i * 0.1, 0.0, 0.0])
        det = src + np.array([mm / 1e3, 0.0, 0.0])
        loc = np.zeros(12)
        if positioned:
            loc[0:3] = (src + det) / 2
            loc[3:6] = src
            loc[6:9] = det
        loc[9] = 760.0 if ch_type == "fnirs_cw_amplitude" else 0.0
        names.append(f"S{i + 1}_D{i + 1}" + (" 760" if ch_type == "fnirs_cw_amplitude" else " hbo"))
        locs.append(loc)

    info = mne.create_info(names, SFREQ, ch_types=[ch_type] * len(names))
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc
    return mne.io.RawArray(np.zeros((len(names), int(SFREQ * 20))), info, verbose=False)


def _labels(raw, sep_bands=None):
    """``{channel: "long" | "short" | "neither"}`` for a montage."""
    long, short = long_short_channels(raw, sep_bands)
    return {ch: ("long" if ch in long else "short" if ch in short else "neither")
            for ch in raw.ch_names}


# ---- The rule ----

def test_the_two_bands_do_not_meet():
    """The gap is the whole point: a channel in it is measured by nothing, on purpose."""
    assert SHORT_MAX_DIST < LONG_MIN_DIST


def test_a_channel_between_the_bands_belongs_to_neither():
    """Screening cannot catch this one: at 12 mm the SCI is excellent, being clean scalp."""
    raw = _montage([8, 12, 30])
    assert list(_labels(raw).values()) == ["short", "neither", "long"]


def test_the_long_band_has_no_upper_bound_by_default():
    """An upper bound is a geometric proxy for SNR, which SCI and PSP measure directly, so
    a far channel is screened on its own numbers rather than excluded by its distance."""
    assert LONG_MAX_DIST is None
    assert _labels(_montage([30, 58, 64]))["S3_D3 760"] == "long"


def test_an_upper_bound_puts_the_far_channels_back_in_neither():
    labels = _labels(_montage([30, 58, 64]), (0.01, 0.015, 0.045))
    assert list(labels.values()) == ["long", "neither", "neither"]


def test_the_bands_move_the_split():
    raw = _montage([8, 13, 30])
    assert _labels(raw)["S2_D2 760"] == "neither"
    # closing the gap makes the 13 mm channel short, which is a decision, not a default
    assert _labels(raw, (0.014, 0.015, None))["S2_D2 760"] == "short"
    # raising the long edge drops a 30 mm channel out instead
    assert _labels(raw, (0.01, 0.035, None))["S3_D3 760"] == "neither"


def test_a_montage_with_no_positions_splits_into_nothing():
    """Every distance reads as zero there. mne_nirs' `get_short_channels` counts that as
    short, which is how `--short-channel mean` came to build its regressors out of the
    whole montage; the `0 < d` guard here is what refuses it."""
    long, short = long_short_channels(_montage([8, 30], positioned=False))
    assert (long, short) == ([], [])


def test_the_short_edge_is_inclusive():
    """mne_nirs compares with strict `<`, so a montage specified at exactly 10.0 mm had
    no short channels by its reckoning and some by ours. One rule now, and it includes
    the edge."""
    assert _labels(_montage([SHORT_MAX_DIST * 1e3]))["S1_D1 760"] == "short"


def test_bad_channels_stay_in_both_lists():
    """Separation is the only thing being asked; a metric that excluded bads here would
    average over channels selected for being good."""
    raw = _montage([8, 30])
    raw.info["bads"] = list(raw.ch_names)
    long, short = long_short_channels(raw)
    assert len(long) == 1 and len(short) == 1


# ---- Resolution and validation ----

def test_a_config_overrides_only_what_it_carries():
    cfg = type("C", (), {"short_max_dist": None, "long_min_dist": 0.02,
                         "long_max_dist": 0.045})
    assert separation_bands(cfg) == (SHORT_MAX_DIST, 0.02, 0.045)


def test_no_config_gives_the_defaults():
    assert separation_bands() == separation_bands(None) == (SHORT_MAX_DIST, LONG_MIN_DIST, None)


@pytest.mark.parametrize("bad, because", [
    ((0.02, 0.015, None), "overlap"),
    ((0.01, 0.015, 0.012), "no channel is long"),
    ((0.0, 0.015, None), "positive"),
    ((0.01, -0.015, None), "positive"),
])
def test_bands_that_do_not_describe_two_ranges_are_refused(bad, because):
    """The three are independent flags, so nothing stops a caller moving one and leaving
    the others; overlapping bands would put a channel in both lists at once."""
    with pytest.raises(ValueError, match=because):
        validate_bands(bad)


def test_the_defaults_are_valid():
    assert validate_bands(separation_bands()) == separation_bands()


# ---- The CLI surface ----

def test_the_flags_arrive_in_metres_from_millimetres():
    """A montage is described in mm and MNE reports metres, so the conversion is here
    rather than in every reader."""
    assert separation_bands_from_args(
        {"short_max_dist": 12, "long_min_dist": 20, "long_max_dist": 55}
    ) == {"short_max_dist": 0.012, "long_min_dist": 0.020, "long_max_dist": 0.055}


def test_an_omitted_flag_stays_none_so_the_default_survives():
    """None is not a value here: it means "keep whatever the package default is", which
    is what lets `--long-max-dist` be omitted to mean no upper bound at all."""
    fields = separation_bands_from_args(
        {"short_max_dist": None, "long_min_dist": None, "long_max_dist": None})
    assert fields == {"short_max_dist": None, "long_min_dist": None, "long_max_dist": None}
    assert separation_bands(type("C", (), fields)) == separation_bands()


def test_the_flags_are_validated_together_rather_than_one_at_a_time():
    """`--short-max-dist 30` alone is invalid against the *default* long edge, which a
    per-flag check would never see."""
    with pytest.raises(ValueError, match="overlap"):
        separation_bands_from_args(
            {"short_max_dist": 30, "long_min_dist": None, "long_max_dist": None})


def test_every_cli_that_splits_channels_offers_the_flags():
    """Four commands split a montage; a fifth that grew the split later must not quietly
    keep the constants."""
    from fnirs_pipe.cli.hyper import _build_parser as hyper_parser
    from fnirs_pipe.cli.qc import _build_parser as qc_parser
    from fnirs_pipe.cli.run import _build_parser as run_parser

    wanted = {"--short-max-dist", "--long-min-dist", "--long-max-dist"}

    def _flags(parser, subcommand=None):
        target = parser
        if subcommand:
            actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
            target = next(a.choices[subcommand] for a in actions if subcommand in a.choices)
        return {f for a in target._actions for f in a.option_strings}

    assert wanted <= _flags(run_parser())
    assert wanted <= _flags(qc_parser(), "prep-raw")
    assert wanted <= _flags(qc_parser(), "hyper-raw")
    assert wanted <= _flags(hyper_parser(), "run")


def test_no_cli_offers_a_gvtd_channel_set_any_more():
    """`--gvtd-channels all` used to widen the GVTD scalars to every channel, which judged
    a run on channels `long_channel_picks` and the GLM never touch. The bands decide the
    set now, and both values were in the record either way, so the flag is gone from all
    three parsers rather than deprecated in one."""
    from fnirs_pipe.cli.qc import _build_parser as qc_parser
    from fnirs_pipe.cli.run import _build_parser as run_parser

    def _flags(parser, subcommand=None):
        target = parser
        if subcommand:
            actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
            target = next(a.choices[subcommand] for a in actions if subcommand in a.choices)
        return {f for a in target._actions for f in a.option_strings}

    for parser, sub in ((run_parser(), None), (qc_parser(), "prep-raw"),
                        (qc_parser(), "hyper-raw")):
        assert "--gvtd-channels" not in _flags(parser, sub)


def test_the_old_channel_set_argument_cannot_be_passed_by_position():
    """The second positional is `sep_bands` now. A caller left over from the flag would
    hand it "long", which has to fail loudly rather than be read as a set of bands."""
    from fnirs_pipe.qc.metrics import gvtd_channel_picks

    with pytest.raises(ValueError):
        gvtd_channel_picks(_montage([8, 30]), "long")


def test_the_analysis_page_emits_the_flags():
    """The GUI's command builder is a hand-written copy of the CLI surface; this is the
    same contract `test_gui_cli_surface` holds for the postprocessing flags."""
    from fnirs_pipe.cli.run import _build_parser
    from fnirs_pipe.interface.callbacks.analysis_callbacks import _build_cli_args

    base = dict(bids_dir="/b", output_dir="/o", subjects=["001"], dpf=6.0, sci_thresh=0.8,
                cardiac_l=0.7, cardiac_h=2.0, resp_l=0.1, resp_h=0.5, post_mode="none")
    argv = _build_cli_args({**base, "short_max_dist": 12, "long_max_dist": 55})
    assert {"--short-max-dist", "--long-max-dist"} <= set(argv)
    _build_parser().parse_args(argv[1:])          # raises SystemExit on an unknown flag

    # and nothing at all when the boxes are empty, so the defaults are not overwritten
    assert not [a for a in _build_cli_args(base) if "dist" in a]


# ---- The stamp ----

def test_the_record_round_trips_the_bands():
    for sep_bands in [(0.01, 0.015, None), (0.012, 0.02, 0.055)]:
        assert bands_from_record(bands_to_record(sep_bands)) == sep_bands


def test_the_stamp_is_in_millimetres():
    """The record is read by people, and every other separation in the report is in mm."""
    assert bands_to_record((0.01, 0.015, 0.045)) == {
        "sep_short_max_mm": 10.0, "sep_long_min_mm": 15.0, "sep_long_max_mm": 45.0}


def test_a_record_with_nothing_stamped_falls_back_to_the_defaults():
    """Every record written before prep started stamping the bands. The alternative,
    refusing to read it, would make an existing tree unreportable."""
    assert bands_from_record({"n_long_channels": 28}) == separation_bands()


def test_an_upper_bound_switched_off_is_not_the_same_as_one_never_stamped():
    """Both read back as None, so presence is what tells them apart. Getting this wrong
    would silently re-apply a 45 mm bound to a run that had asked for none."""
    off = {"sep_short_max_mm": 10.0, "sep_long_min_mm": 15.0, "sep_long_max_mm": None}
    assert bands_from_record(off)[2] is None
    never = {"sep_short_max_mm": 10.0, "sep_long_min_mm": 15.0}
    assert bands_from_record(never)[2] is LONG_MAX_DIST


def test_the_writer_and_the_reader_share_one_set_of_keys():
    from fnirs_pipe.qc.metrics._helpers import BANDS_RECORD_KEYS
    assert set(bands_to_record(separation_bands())) == set(BANDS_RECORD_KEYS)


# ---- Read back from the members' records ----
# `fnirs-hyper run` works on derivatives prep already split and stamped, so being told the
# bands again is an invitation to type a number that does not match the one on disk.

def _dyad():
    from fnirs_pipe.pipeline.hyperscanning import GroupEntry
    return [GroupEntry("G1", "sub-01", "tap"), GroupEntry("G1", "sub-02", "tap")]


def _sqm(*per_member):
    """{subject_id: record scalars} from one Bands per member; None means nothing stamped."""
    return {f"sub-0{i}": ({} if bands is None else bands_to_record(bands))
            for i, bands in enumerate(per_member, start=1)}


def test_agreeing_records_decide_the_bands():
    from fnirs_pipe.pipeline.hyperscanning import resolve_group_bands
    bands = (0.012, 0.02, None)
    assert resolve_group_bands(_dyad(), _sqm(bands, bands)) == bands


def test_members_prepped_with_different_bands_are_refused():
    """Not reconciled: the bands also chose what short-channel regression removed from each
    member upstream, so a band taken from both would describe neither. Nothing is lost by
    refusing, since the homologous channel set already intersects by label."""
    from fnirs_pipe.pipeline.hyperscanning import resolve_group_bands
    with pytest.raises(ValueError, match="different separation bands"):
        resolve_group_bands(_dyad(), _sqm((0.01, 0.015, None), (0.012, 0.02, None)))


def test_the_refusal_names_both_members_and_their_bands():
    """A message saying only "they disagree" leaves the operator to grep two records."""
    from fnirs_pipe.pipeline.hyperscanning import resolve_group_bands
    with pytest.raises(ValueError) as excinfo:
        resolve_group_bands(_dyad(), _sqm((0.01, 0.015, None), (0.012, 0.02, 0.055)))
    message = str(excinfo.value)
    assert "sub-01 task-tap" in message and "sub-02 task-tap" in message
    assert "10 mm" in message and "20-55 mm" in message


def test_an_unstamped_dyad_falls_back_to_the_defaults_with_a_warning(caplog):
    """Every tree prepped before the bands were stamped. Refusing those outright would make
    them unanalysable, so it is a warning; the value is a guess and says so."""
    from fnirs_pipe.pipeline.hyperscanning import resolve_group_bands
    with caplog.at_level(logging.WARNING):
        assert resolve_group_bands(_dyad(), _sqm(None, None)) == separation_bands()
    assert "no separation bands stamped" in caplog.text


def test_an_unstamped_member_is_warned_with_the_bands_it_is_being_given(caplog):
    """The other member's stamp is the best evidence available, so it is what gets applied
    -- but the warning has to name that value rather than the package defaults, or a reader
    is told 10 mm was assumed while 12 mm was used."""
    from fnirs_pipe.pipeline.hyperscanning import resolve_group_bands
    with caplog.at_level(logging.WARNING):
        assert resolve_group_bands(_dyad(), _sqm((0.012, 0.02, None), None)) == (0.012, 0.02, None)
    assert "sub-02 task-tap" in caplog.text
    assert "short <= 12 mm" in caplog.text


def test_a_flag_overrides_the_records_and_says_so(caplog):
    from fnirs_pipe.pipeline.hyperscanning import resolve_group_bands
    bands = (0.012, 0.02, None)
    with caplog.at_level(logging.WARNING):
        resolved = resolve_group_bands(_dyad(), _sqm(bands, bands),
                                       {"short_max_dist": 0.008, "long_min_dist": 0.015})
    assert resolved == (0.008, 0.015, None)
    assert "overriding" in caplog.text


def test_a_flag_left_off_keeps_the_records_value_rather_than_the_package_default():
    """Capping the long band must not silently re-assert 10 / 15 mm on a dyad prepped at
    12 / 20 mm, which is what falling back to `separation_bands()` would do."""
    from fnirs_pipe.pipeline.hyperscanning import resolve_group_bands
    bands = (0.012, 0.02, None)
    resolved = resolve_group_bands(_dyad(), _sqm(bands, bands), {"long_max_dist": 0.055})
    assert resolved == (0.012, 0.02, 0.055)


def test_an_override_of_only_none_values_is_not_an_override(caplog):
    """argparse hands over three Nones when no flag was given, which must not read as a
    request to force the package defaults."""
    from fnirs_pipe.pipeline.hyperscanning import resolve_group_bands
    bands = (0.012, 0.02, None)
    with caplog.at_level(logging.WARNING):
        resolved = resolve_group_bands(_dyad(), _sqm(bands, bands), {
            "short_max_dist": None, "long_min_dist": None, "long_max_dist": None})
    assert resolved == bands
    assert "overriding" not in caplog.text


def test_a_forced_band_is_still_validated():
    """The override merges with the records, so the pair it produces was never validated by
    the CLI: forcing a short edge past the records' long edge has to be caught here."""
    from fnirs_pipe.pipeline.hyperscanning import resolve_group_bands
    bands = (0.012, 0.02, None)
    with pytest.raises(ValueError, match="overlap"):
        resolve_group_bands(_dyad(), _sqm(bands, bands), {"short_max_dist": 0.03})


def test_a_record_stamps_whether_it_carries_bands_at_all():
    from fnirs_pipe.qc.metrics._helpers import record_has_bands
    assert record_has_bands(bands_to_record((0.01, 0.015, None)))
    assert not record_has_bands({"n_long_channels": 28})
    # partial is not a stamp: the writer always writes the three together, and mixing a
    # stamped value with a defaulted one is worse than defaulting all three
    assert not record_has_bands({"sep_short_max_mm": 10.0})


# ---- One run, one split ----

def test_the_analysis_paths_agree_with_the_split():
    """`long_channel_picks` and `has_short_channels` used to reach mne_nirs directly, so
    "long" meant 15-45 mm to the reports and "anything over 10 mm" to the dyad metrics.
    A 58 mm channel was outside the montage in one half of the package and inside it in
    the other."""
    raw = _montage([8, 12, 30, 58], ch_type="hbo")
    for sep_bands in [None, (0.01, 0.015, 0.045), (0.014, 0.015, None)]:
        long, short = long_short_channels(raw, sep_bands)
        picks = long_channel_picks(raw, "hbo", exclude=[], sep_bands=sep_bands)
        assert [raw.ch_names[p] for p in picks] == long
        assert has_short_channels(raw, sep_bands) is bool(short)


def test_a_positionless_montage_builds_no_short_channel_regressors():
    """The bug this closes: every separation reads as zero, mne_nirs called that short,
    and the "systemic" signal regressed out of every channel was the whole montage.

    The refusal replaced an empty return: a skip would leave the methods text naming
    regressors the residual does not carry."""
    from fnirs_pipe.exceptions import StageError
    from fnirs_pipe.pipeline.glm import _short_channel_regressors

    raw = _montage([8, 30], ch_type="hbo", positioned=False)
    with pytest.raises(StageError, match="no channel at or under"):
        _short_channel_regressors(raw, "mean")
    assert has_short_channels(raw) is False


def test_the_refusal_names_the_shortest_channel_there_is():
    """The other arm of that message, and the one a montage sitting just outside the short
    band hits: the number to raise --short-max-dist to has to be in the error rather than
    left for the reader to go measure."""
    from fnirs_pipe.exceptions import StageError
    from fnirs_pipe.pipeline.glm import _short_channel_regressors

    raw = _montage([12.8, 30], ch_type="hbo")
    with pytest.raises(StageError, match="shortest is 12.8 mm"):
        _short_channel_regressors(raw, "mean")


def test_the_gvtd_channel_set_follows_the_bands():
    from fnirs_pipe.qc.metrics import gvtd_channel_picks

    raw = _montage([8, 30, 58])
    assert gvtd_channel_picks(raw) == (["S2_D2 760", "S3_D3 760"], "long")
    # an upper bound takes the 58 mm channel out of the trace as well as out of the table
    assert gvtd_channel_picks(raw, (0.01, 0.015, 0.045)) == (["S2_D2 760"], "long")


def test_the_orphan_warning_repeats_when_the_bands_change(caplog):
    """It is deduplicated per montage because it is called once per figure. Keying on the
    channel names alone would silence it for a second run with different bands, which is
    exactly the run whose split a reader would want explained."""
    raw = _montage([8, 12, 30])
    with caplog.at_level(logging.WARNING):
        long_short_channels(raw)
        first = caplog.text.count("neither separation band")
        long_short_channels(raw)                      # same bands: still one warning
        assert caplog.text.count("neither separation band") == first
        long_short_channels(raw, (0.01, 0.02, None))  # different bands: warns again
        assert caplog.text.count("neither separation band") == first + 1


def test_the_orphans_are_named_with_their_own_separations():
    """The gap is a package default; the separations are this montage's. A note that says
    only "10-15 mm" tells a reader a gap exists, not whether moving a bound by 1 mm or 5
    would take their channels in."""
    from fnirs_pipe.qc.metrics import separation_orphans

    raw = _montage([8, 12.8, 13.8, 30])
    assert separation_orphans(raw) == pytest.approx({"S2_D2 760": 12.8, "S3_D3 760": 13.8})
    # a clean split has none, and neither does a montage with no positions, which the
    # callers word as a missing-registration problem instead
    assert separation_orphans(_montage([8, 30])) == {}
    assert separation_orphans(_montage([8, 30], positioned=False)) == {}
    # the bands move it: at 14 mm those two are short, so nothing is orphaned
    assert separation_orphans(raw, (0.014, 0.015, None)) == {}


def test_the_note_says_where_the_orphans_sit_and_which_bound_would_take_them():
    from fnirs_pipe.qc.common.channel_table import separation_notes

    scalars = {"n_long_channels": 1, "n_short_channels": 1}
    rows = [{"separation": "unclassified"}, {"separation": "unclassified"}]
    plain = separation_notes(scalars, rows)[0]
    assert "10-15 mm" in plain and "Theirs sit at" not in plain
    named = separation_notes(scalars, rows, orphan_mm={"a": 12.8, "b": 13.8})[0]
    assert "12.8 to 13.8 mm" in named
    # ceil the short bound and floor the long one, or a bound is suggested that leaves
    # one of the orphans out: --long-min-dist 13 still excludes a 12.8 mm channel
    assert "--short-max-dist 14" in named and "--long-min-dist 12" in named
    # one orphan, or several at the same separation, reads as a point not a span
    one = separation_notes(scalars, rows, orphan_mm={"a": 12.8})[0]
    assert "at 12.8 mm" in one and "to" not in one.split("Theirs sit")[1][:20]


def test_a_config_toml_can_carry_the_separation_bands():
    """They have to be resolved before prep runs, because prep stamps them into the record
    and only the post config builder is handed the TOML."""
    from fnirs_pipe.cli import _shared

    assert _shared.SEPARATION_BAND_KEYS == ("short_max_dist", "long_min_dist", "long_max_dist")
    args = {"short_max_dist": None, "long_min_dist": 16.0}
    toml = {"short_max_dist": 14.0, "long_min_dist": 25.0, "long_max_dist": 55.0}
    for band in _shared.SEPARATION_BAND_KEYS:
        if args.get(band) is None and toml.get(band) is not None:
            args[band] = toml[band]
    # the CLI value stands, the two absent ones come from the TOML
    assert args == {"short_max_dist": 14.0, "long_min_dist": 16.0, "long_max_dist": 55.0}
    assert _shared.separation_bands_from_args(args) == {
        "short_max_dist": 0.014, "long_min_dist": 0.016, "long_max_dist": 0.055}
