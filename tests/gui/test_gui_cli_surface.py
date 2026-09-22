"""The contract between the GUI's command builder and the CLI it drives.

The analysis page does not run the pipeline; it assembles an argv and hands it to
`fnirs-pipe`. That makes `_build_cli_args` a hand-written copy of the CLI's flag surface,
and a copy falls behind. It had: `--drift-model` was never emitted at all, so the GLM and
Rest entries in the mode picker could only ever produce a command that died in `run_post`,
and `--short-channel` was locked to glm long after rest and denoise honoured it.

Nothing announced any of that, because the two surfaces are only compared by a person
reading both. These tests do the comparing. `NOT_EXPOSED` is the deliberate half of the
contract, in the same spirit as `ALL_STEPS` in `test_step_vocabulary.py`: a flag may be left
out of the GUI, but only on purpose and only in writing.
"""

import os

import dash
import dash_bootstrap_components as dbc
import pytest
from dash import dcc, html

from fnirs_pipe.cli.run import _build_parser
from fnirs_pipe.cli.workflows import _build_post_config
from fnirs_pipe.interface.callbacks.analysis_callbacks import _build_cli_args
from fnirs_pipe.interface.cli_args import (
    _AGGREGATE,
    _HYPER,
    _RAW_QC,
    build_prep_args,
    build_qc_args,
    build_raw_qc_args,
    missing_raw_qc,
)

# Flags the analysis page deliberately does not offer, and why. A flag listed here must still
# exist in the CLI, and must not also be emitted; both are asserted below.
NOT_EXPOSED = {
    "--bad-channels": "a per-subject channel list or a table path, and the page has no file picker",
    "--epoch-single-trial": "only meaningful when no condition repeats",
    "--gvtd-min-epoch-s": "housekeeping beside the threshold, which is the knob that moves the result",
    "--config": "the form is the config surface; a TOML overriding it would make the preview lie",
    "--events-path": "a file path, and mutually exclusive with the Stim Duration field",
    "--contrast-file": "a file path, and the page has no file picker",
    "--fir-delays": "only meaningful with --hrf-model fir",
    "--no-combine-runs": "the negative half of a paired flag; the checkbox emits the positive",
    "--aux-regressors": "withheld pending evaluation; suppressed from --help too",
    "--aux-channels": "only meaningful with --aux-regressors, which is withheld",
}

# every control filled in, so the union over modes is everything the builder can emit
_FULL_OPTS = dict(
    bids_dir="/bids", output_dir="/out", subjects=["001"],
    session_label="ses-1", task_label="tapping",
    dpf=6.0, sci_thresh=0.5, psp_thresh=0.1, motion_correction="tddr",
    min_good_frac=0.75, screen_scope="run", window_length=10.0,
    short_max_dist=10.0, long_min_dist=15.0, long_max_dist=45.0,
    epoch_tmin=-5.0, epoch_tmax=25.0, epoch_chunk=25.0,
    by_condition=True, gvtd_censor="long", gvtd_n_std=10.0,
    cardiac_l=0.7, cardiac_h=1.5, resp_l=0.1, resp_h=0.5,
    high_pass=0.01, low_pass=0.1, filter_method="iir", filter_order=4,
    resample=2.0, n_jobs=1,
    hrf_model="spm", noise_model="ar1", short_channel="mean",
    drift_model="cosine", drift_high_pass=0.01, drift_order=1,
    stim_dur=5.0, roi_mapping="/roi.json", fc=True,
    flags=["dry_run", "skip_bids_validation", "no_report", "combine_runs"],
)

_MODES = ["denoise", "glm", "rest"]

# built by a callback into a container rather than declared in the layout, which is why the
# app is constructed with suppress_callback_exceptions
_DYNAMIC_IDS = {"an-subjects-checklist", "dp-subject-radio"}

# Controls the callbacks may leave alone, and why. The deliberate half of the contract below,
# in the same spirit as NOT_EXPOSED: an id may go unread, but only on purpose and only in
# writing. An id nothing reads and nothing will read belongs deleted, not listed here.
_UNBOUND_BY_DESIGN = {
    "dp-tabs": "a dbc.Tabs container; switching is client-side and reaches no callback",
    "an-noise-model-suggest": "an html.Datalist; the browser reads it through the input's "
                              "list= attribute, so no callback ever names it",
}

# bound by callbacks but declared in app.launch's own layout rather than on a page, so the
# page sweep below cannot see them
_APP_LEVEL_IDS = {"dp-run-store"}

# every page whose controls are held against the callbacks that drive them
_PREFIXES = ("an-", "qc-", "bp-", "dp-", "ha-", "rc-")


def _long_flags(action) -> set[str]:
    """Every long spelling the action answers to, ``--no-`` forms included."""
    return {f for f in action.option_strings if f.startswith("--")}


def _alias_map() -> dict[str, frozenset[str]]:
    """Every long spelling of one option, keyed by each of them.

    ``--coh-fmin`` / ``--fmin`` and ``--participant-label`` / ``--participant_label`` are
    each one option under two names. Without this the sweeps below would read a second
    spelling as a second thing for the page to grow a control for, and would read a
    write-off naming either spelling as covering only that one.
    """
    parsers = [_build_parser(), *_qc_subparsers().values(),
               *_hyper_parsers_by_command().values(), *_prep_subparsers().values()]
    out: dict[str, frozenset[str]] = {}
    for parser in parsers:
        for action in parser._actions:
            longs = frozenset(_long_flags(action))
            for flag in longs:
                out[flag] = longs
    return out


def _with_aliases(flags: set[str]) -> set[str]:
    """*flags* plus every other spelling of the same options."""
    alias = _alias_map()
    return {spelling for f in flags for spelling in alias.get(f, (f,))}


def _group_flags(prefix: str) -> set[str]:
    """Every long flag in the CLI argument groups whose title starts with *prefix*."""
    parser = _build_parser()
    return {
        flag
        for group in parser._action_groups
        if group.title and group.title.startswith(prefix)
        for action in group._group_actions
        for flag in _long_flags(action)
    }


def _post_flags() -> set[str]:
    return _group_flags("postprocessing")


def _cli_flags() -> set[str]:
    """Preprocessing and postprocessing together: both halves drift the same way.

    The preprocessing half was uncovered until `--min-good-frac` and `--screen-scope` had
    been in the CLI for a while with no way to set them here, which meant the page could
    only ever screen at the default share of coupled windows.
    """
    return _group_flags("preprocessing") | _post_flags()


def _emitted(**overrides) -> set[str]:
    opts = {**_FULL_OPTS, **overrides}
    return {a for a in _build_cli_args(opts) if a.startswith("--")}


def _emitted_any_mode() -> set[str]:
    """Everything the builder can emit, over every mode and drift model.

    Some flags are conditional on more than the mode: --drift-order is polynomial-only and
    --drift-high-pass is cosine-only, so a union over modes alone would call them missing.
    """
    flags: set[str] = set()
    for mode in _MODES:
        for drift in ("cosine", "polynomial", "none"):
            flags |= _emitted(post_mode=mode, drift_model=drift)
    return flags


# ---- the flag surface ----

def test_every_flag_is_offered_or_written_off():
    missing = _cli_flags() - _with_aliases(_emitted_any_mode() | set(NOT_EXPOSED))
    assert not missing, (
        f"the CLI grew {sorted(missing)} and the analysis page cannot send them; "
        f"add a control, or add them to NOT_EXPOSED with a reason"
    )


def test_the_written_off_flags_still_exist():
    """Otherwise the list quietly becomes an excuse for flags nobody removed from it."""
    stale = set(NOT_EXPOSED) - _cli_flags()
    assert not stale, f"NOT_EXPOSED names flags the CLI no longer has: {sorted(stale)}"


def test_nothing_is_both_written_off_and_emitted():
    assert not set(NOT_EXPOSED) & _emitted_any_mode()


# ---- the commands the page actually produces ----

@pytest.mark.parametrize("mode", _MODES)
def test_the_generated_command_parses(mode):
    argv = _build_cli_args({**_FULL_OPTS, "post_mode": mode})
    assert argv[0] == "fnirs-pipe"
    _build_parser().parse_args(argv[1:])          # raises SystemExit on an unknown flag


@pytest.mark.parametrize("mode", _MODES)
def test_the_generated_command_satisfies_its_modes_requirements(mode):
    """The regression test for the bug: GLM and Rest used to arrive without a drift model."""
    argv = _build_cli_args({**_FULL_OPTS, "post_mode": mode})
    args = vars(_build_parser().parse_args(argv[1:]))
    config = _build_post_config("001", None, args, {})

    if mode in ("glm", "rest"):
        assert config.drift_model is not None
    if mode == "glm":
        assert config.hrf_model is not None and config.noise_model is not None
    # __post_init__ refuses a cosine drift with no cutoff, which is what the page defaults to
    assert config.drift_high_pass is not None


@pytest.mark.parametrize("mode", _MODES)
def test_short_channel_reaches_every_mode(mode):
    """It was locked behind glm, long after rest and denoise both honoured it."""
    argv = _build_cli_args({**_FULL_OPTS, "post_mode": mode})
    assert "--short-channel" in _emitted(post_mode=mode)
    args = vars(_build_parser().parse_args(argv[1:]))
    assert _build_post_config("001", None, args, {}).short_channel == "mean"


def test_a_none_short_channel_is_left_out_rather_than_sent():
    assert "--short-channel" not in _emitted(post_mode="glm", short_channel="none")


def test_fc_is_not_sent_to_rest_mode():
    """rest writes the connectivity products regardless, so the flag would be noise."""
    assert "--fc" in _emitted(post_mode="denoise")
    assert "--fc" in _emitted(post_mode="glm")
    assert "--fc" not in _emitted(post_mode="rest")


def test_no_postprocessing_flags_without_a_mode():
    assert _emitted(post_mode="none") & _post_flags() == set()


# ---- controls and the callbacks that read them ----

@pytest.fixture(scope="module")
def analysis_page():
    """The built app, the contracted pages' ids, and the ids the callbacks bind to.

    Dash refuses `register_page` before an app exists, so the page modules cannot simply be
    imported; the app has to be constructed the way `interface.app.launch` constructs it.
    """
    import fnirs_pipe.interface.app as app_module

    app = dash.Dash(
        __name__, use_pages=True,
        pages_folder=os.path.join(os.path.dirname(app_module.__file__), "pages"),
        external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True,
    )
    import fnirs_pipe.interface.callbacks.analysis_callbacks
    import fnirs_pipe.interface.callbacks.batch_prep_callbacks
    import fnirs_pipe.interface.callbacks.data_prep_callbacks
    import fnirs_pipe.interface.callbacks.hyper_align_callbacks
    import fnirs_pipe.interface.callbacks.qc_callbacks
    import fnirs_pipe.interface.callbacks.recon_callbacks  # noqa: F401

    app.layout = html.Div([dcc.Store(id="app-bids-dir"), dash.page_container])
    app._setup_server()

    layouts = [p["layout"] for p in dash.page_registry.values()]
    layouts = [lay() if callable(lay) else lay for lay in layouts]

    def _ids(component):
        found = set()
        cid = getattr(component, "id", None)
        if isinstance(cid, str):
            found.add(cid)
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                found |= _ids(child)
        elif children is not None:
            found |= _ids(children)
        return found

    # pattern-matching bindings carry a dict id and match nothing declared in a layout
    bound = set()
    for entry in app.callback_map.values():
        for spec in list(entry["inputs"]) + list(entry.get("state") or []):
            if isinstance(spec["id"], str):
                bound.add(spec["id"])
        outputs = entry["output"]
        for out in (outputs if isinstance(outputs, (list, tuple)) else [outputs]):
            if isinstance(out.component_id, str):
                bound.add(out.component_id)

    on_page = set()
    for lay in layouts:
        on_page |= _ids(lay)

    return on_page, bound


def test_every_control_the_callbacks_bind_to_exists_on_the_page(analysis_page):
    on_page, bound = analysis_page
    dangling = ({i for i in bound if i.startswith(_PREFIXES)}
                - on_page - _DYNAMIC_IDS - _APP_LEVEL_IDS)
    assert not dangling, f"callbacks bind ids the page does not define: {sorted(dangling)}"


def test_no_control_on_the_page_is_decoration(analysis_page):
    """`an-stim-dur` was drawn, never read, and never reached a command for two releases.

    An Output counts: a preview pane is driven by a callback rather than read by one. What
    this catches is a control wired to nothing in either direction.
    """
    on_page, bound = analysis_page
    unread = ({i for i in on_page if i.startswith(_PREFIXES)}
              - bound - set(_UNBOUND_BY_DESIGN))
    assert not unread, f"controls nothing reads: {sorted(unread)}"


def test_the_unbound_controls_still_exist(analysis_page):
    """Otherwise the list quietly becomes an excuse for ids nobody removed from it."""
    on_page, _ = analysis_page
    stale = set(_UNBOUND_BY_DESIGN) - on_page
    assert not stale, f"_UNBOUND_BY_DESIGN names ids no page defines: {sorted(stale)}"


# ---- the same contract for the QC page, which drives two tools ----

# The page builds `fnirs-qc` for the aggregate commands and `fnirs-hyper` for the dyad
# analysis. Both are subparser CLIs, so each command's flag list is read off its own parser.

# fnirs-qc subcommands the QC page does not offer, and why.
QC_COMMANDS_NOT_OFFERED: dict[str, str] = {}

# per-command flags the page leaves out. Most are the negative half of a paired
# BooleanOptionalAction, where the switch emits the positive. --wtc-limit-scales is the
# exception and goes the other way: it defaults to on, and the CLI's own help says the
# restricted and unrestricted coherences match bit for bit, so neither half is a choice
# worth putting on screen. It is an escape hatch for comparing against old output.
QC_NOT_EXPOSED = {
    "run": {"--no-normalize", "--wtc-limit-scales", "--no-wtc-limit-scales",
            # a log-level switch, not a parameter of the analysis
            "--verbose",
            # only tints the per-subject quality table, against a threshold the run was
            # already prepped with. A second control here could be set to a different number
            # than prep used, with nothing saying which one the colours mean
            "--sci-threshold",
            # same reason, one step worse: the bands are stamped in each member's record
            # by prep, so a second control could split the dyad metrics one way while the
            # member reports were split another. The CLI keeps the flags for a tree prepped
            # before they were stamped
            "--short-max-dist", "--long-min-dist", "--long-max-dist",
            # display only: it decides where a phase arrow is drawn on the WTC maps and
            # changes no table or figure value, so the page has nothing to preview for it
            "--wtc-arrow-min",
            # the page exists to show the report; a run started from it that writes no
            # report leaves the user looking at an empty panel with no way to tell the run
            # from a failure. It is for a batch that will be read as tables
            "--no-report",

            # the positive half of a paired flag, and it is the default; the checkbox emits
            # the negative one
            "--wtc-mask-coi",
            # four spellings of one switch. The per-condition pass is the default and the
            # checkbox turns it off with the canonical --no-by-condition; offering the
            # aliases would be three controls doing one thing
            "--by-condition", "--wtc-by-condition", "--no-wtc-by-condition",
            # route P: transform each condition on its own instead of reading it out of the
            # whole-run transform. With the default padding it reproduces the default route
            # to four decimals, and without it the cone eats a share that grows as the
            # condition shortens. It exists to reproduce a published result, which is not a
            # thing to put in front of someone filling in a form
            "--wtc-cond-transform", "--wtc-cond-pad-s"},
    "band":  {"--verbose",
              # as above: the default, and the checkbox emits --no-wtc-mask-coi
              "--wtc-mask-coi"},
    "pair-null": {
        # a log-level switch, not a parameter of the analysis
        "--verbose",
        # a speed setting whose kept scales land on pycwt's own grid, so the coherences
        # match the unrestricted ones bit for bit. Nothing about the result moves with it
        "--wtc-limit-scales", "--no-wtc-limit-scales"},
    "group-null": {"--verbose"},
    "merge": {"--verbose"},
    # every group-* directory by default, which is the whole shape of the aggregate form
    "index": {"--verbose", "--group-id"},
}

_QC_FULL_OPTS = dict(
    derivatives_dir="/deriv", output_dir="/out", pairs_csv="/pairs.csv", group_id="01",
    desc="errts", roi_mapping="/roi.json",
    wtc_fmin=0.004, wtc_fmax=0.2, wtc_band_fmin=0.01, wtc_band_fmax=0.1,
    wtc_mc_count=300, wtc_seed=42, isc_threshold=0.3,
    wtc_phase_null=100, wtc_roi_min_channels=2, wtc_chroma="both", wtc_window_s=30.0,
    isc_whiten=32, isc_max_lag=2.0, isc_phase_null=100, isc_fmin=0.06, isc_fmax=0.15,
    hyper_task="rest",
    hyper_flags=["wtc_significance", "wtc_no_mask_coi", "wtc_channel_cross", "wtc_phase_null_cross",
                 "no_by_condition", "bads_subject", "wtc_save_maps", "no_align", "normalize",
                 "check_only"],
    wtc_pair_pool="position", wtc_pair_max=20, pair_flags=["wtc_pair_cross"],
    gn_task="full", gn_chroma="hbo", gn_null="repaired",
    gn_roi_mapping="/roi.json", gn_resample=20000, gn_seed=7,
    band_fmin=0.05, band_fmax=0.2, band_suffix="band0p05-0p2",
    band_flags=["band_no_mask_coi"],
    tstart=0.0, tend=60.0,
)

_QC_OFFERED = [*_HYPER, *_AGGREGATE]


def _subcommands(build):
    parser = build()
    action = next(a for a in parser._actions if getattr(a, "choices", None))
    return action.choices


def _build_qc_parser():
    from fnirs_pipe.cli.qc import _build_parser as build
    return build()


def _hyper_parsers_by_command():
    """Each dyad command's own parser, keyed the way the page names it."""
    from fnirs_pipe.cli.hyper import _parsers
    from fnirs_pipe.interface.cli_args import _HYPER_PROG

    built = _parsers()
    return {command: built[prog] for command, prog in _HYPER_PROG.items()}


def _qc_subparsers():
    return _subcommands(_build_qc_parser)


def _hyper_subparsers():
    return _hyper_parsers_by_command()


def _qc_flags(command: str) -> set[str]:
    parser = (_hyper_subparsers() if command in _HYPER else _qc_subparsers())[command]
    return {flag for action in parser._actions
            if action.dest not in ("help", "version")
            for flag in _long_flags(action)}


def _qc_emitted(command: str) -> set[str]:
    return {a for a in build_qc_args(command, _QC_FULL_OPTS) if a.startswith("--")}


def test_some_page_offers_every_fnirs_qc_subcommand_or_writes_it_off():
    """The aggregates are on Cohort Reports; the two raw reports are on the QC pages."""
    reachable = set(_QC_OFFERED) | set(_RAW_QC) | set(QC_COMMANDS_NOT_OFFERED)
    missing = set(_qc_subparsers()) - reachable
    assert not missing, (
        f"fnirs-qc grew {sorted(missing)} and no page either offers or declines them"
    )


def test_the_page_offers_every_hyper_subcommand():
    missing = set(_hyper_subparsers()) - set(_QC_OFFERED)
    assert not missing, f"fnirs-hyper grew {sorted(missing)} and the page cannot reach them"


def test_the_declined_subcommands_still_exist():
    stale = set(QC_COMMANDS_NOT_OFFERED) - set(_qc_subparsers())
    assert not stale, f"QC_COMMANDS_NOT_OFFERED names commands fnirs-qc no longer has: {sorted(stale)}"


def test_the_dyad_analysis_left_fnirs_qc():
    """hyper-post, hyper-null, wtc-band and group-hyper-wtc are fnirs-hyper now."""
    gone = {"hyper-post", "hyper-null", "wtc-band", "group-hyper-wtc"} & set(_qc_subparsers())
    assert not gone, f"fnirs-qc still carries {sorted(gone)}"


@pytest.mark.parametrize("command", _QC_OFFERED)
def test_every_flag_of_an_offered_command_is_sendable_or_written_off(command):
    missing = _qc_flags(command) - _with_aliases(
        _qc_emitted(command) | QC_NOT_EXPOSED.get(command, set()))
    assert not missing, (
        f"{command} grew {sorted(missing)} and the page cannot send them; "
        f"add a control, or add them to QC_NOT_EXPOSED with a reason"
    )


@pytest.mark.parametrize("command", sorted(QC_NOT_EXPOSED))
def test_the_written_off_qc_flags_still_exist(command):
    stale = QC_NOT_EXPOSED[command] - _qc_flags(command)
    assert not stale, f"QC_NOT_EXPOSED[{command!r}] names flags that are gone: {sorted(stale)}"


@pytest.mark.parametrize("command", _QC_OFFERED)
def test_the_generated_qc_command_parses(command):
    argv = build_qc_args(command, _QC_FULL_OPTS)
    if command in _HYPER:
        from fnirs_pipe.interface.cli_args import _HYPER_PROG
        assert argv[0] == _HYPER_PROG[command]
        # raises SystemExit on an unknown flag or a missing positional
        _hyper_parsers_by_command()[command].parse_args(argv[1:])
    else:
        assert argv[:2] == ["fnirs-qc", command]
        _build_qc_parser().parse_args(argv[1:])


def test_neither_tool_on_this_page_asks_for_a_bids_directory():
    """Both read derivatives rather than raw BIDS, whichever tree they are pointed at."""
    for command in _QC_OFFERED:
        argv = build_qc_args(command, _QC_FULL_OPTS)
        assert "/bids" not in argv


def test_a_space_separated_box_repeats_its_flag_rather_than_joining():
    """argparse nargs="+" takes repeats; one string with a space in it is one label."""
    argv = build_qc_args("run", dict(_QC_FULL_OPTS, hyper_task="rest tap"))
    args = _hyper_parsers_by_command()["run"].parse_args(argv[1:])
    assert args.task_label == ["rest", "tap"]


def test_the_aggregate_commands_take_only_an_output_directory():
    for command in _AGGREGATE:
        assert build_qc_args(command, _QC_FULL_OPTS) == ["fnirs-qc", command, "/out"]


# ── Batch Prep → fnirs-prep ──────────────────────────────────────────────────

# per-subcommand flags the Batch Prep page leaves out, and why
PREP_NOT_EXPOSED = {
    "crop": {
        # the page's radio is the two modes; a trigger-relative origin needs a name the
        # page cannot offer without reading the recording, which is Data Preparation's job
        "--align", "--trigger-name",
        # a margin is only meaningful against the band a later analysis will average over,
        # and that band is set on the Hyper Analysis page. Setting it here would let the two
        # disagree with nothing saying which one the output was cut for
        "--margin", "--band-fmin",
        # cutting a processed stage rather than a recording. The page takes a BIDS root, so
        # offering this would make the two directory fields mean different things per mode
        "--input-desc",
        # the negative half of a paired flag; the checkbox emits the positive
        "--no-combine",
        "--skip-bids-validation", "--no-skip-bids-validation",
    },
    "align": {"--skip-bids-validation", "--no-skip-bids-validation"},
    "edit-markers": {
        # an edited events TSV applied to every subject, and the page has no file picker.
        # Its three radio options are the edits that need no file
        "--tsv",
        "--skip-bids-validation", "--no-skip-bids-validation",
    },
}

_PREP_FULL_OPTS = dict(
    bids_dir="/bids", deriv_dir="/deriv", subjects=["001", "002"],
    ses="01", task="tapping", run="01", n_jobs=2,
    shift=-2.5, set_duration=30.0, rename=["old:new"],
    crop_tmin=10.0, crop_tmax=600.0,
    segments_path="/deriv/batch-crop-segments.tsv", combine=True,
    group_csv="/groups.csv",
)

# every branch of the page's two radios, as (operation, extra opts)
_PREP_CASES = [
    ("markers", {"marker_op": "shift"}),
    ("markers", {"marker_op": "set_duration"}),
    ("markers", {"marker_op": "rename"}),
    ("crop", {"crop_mode": "single"}),
    ("crop", {"crop_mode": "multi"}),
    ("hyper_align", {}),
]

# the page's operation names, and the fnirs-prep subcommand each one drives
_PREP_SUBCOMMAND = {"markers": "edit-markers", "crop": "crop", "hyper_align": "align"}


def _build_prep_parser():
    from fnirs_pipe.cli.prep import _build_parser as build
    return build()


def _prep_subparsers():
    return _subcommands(_build_prep_parser)


def _prep_flags(subcommand: str) -> set[str]:
    parser = _prep_subparsers()[subcommand]
    if subcommand == "edit-markers":
        parser = _subcommands(lambda: parser)["apply"]
    return {flag for action in parser._actions if action.dest != "help"
            for flag in _long_flags(action)}


def _prep_emitted(subcommand: str) -> set[str]:
    emitted: set[str] = set()
    for operation, extra in _PREP_CASES:
        if _PREP_SUBCOMMAND[operation] != subcommand:
            continue
        argv = build_prep_args(operation, dict(_PREP_FULL_OPTS, **extra))
        emitted |= {a for a in argv if a.startswith("--")}
    return emitted


def test_the_batch_page_reaches_every_prep_subcommand():
    missing = set(_prep_subparsers()) - set(_PREP_SUBCOMMAND.values())
    assert not missing, f"fnirs-prep grew {sorted(missing)} and Batch Prep cannot reach them"


@pytest.mark.parametrize("subcommand", sorted(set(_PREP_SUBCOMMAND.values())))
def test_every_prep_flag_is_sendable_or_written_off(subcommand):
    missing = _prep_flags(subcommand) - _with_aliases(
        _prep_emitted(subcommand) | PREP_NOT_EXPOSED.get(subcommand, set()))
    assert not missing, (
        f"{subcommand} grew {sorted(missing)} and Batch Prep cannot send them; "
        f"add a control, or add them to PREP_NOT_EXPOSED with a reason"
    )


@pytest.mark.parametrize("subcommand", sorted(PREP_NOT_EXPOSED))
def test_the_written_off_prep_flags_still_exist(subcommand):
    stale = PREP_NOT_EXPOSED[subcommand] - _prep_flags(subcommand)
    assert not stale, f"PREP_NOT_EXPOSED[{subcommand!r}] names flags that are gone: {sorted(stale)}"


@pytest.mark.parametrize("operation,extra", _PREP_CASES,
                         ids=[f"{op}-{'-'.join(e.values())}" if e else op
                              for op, e in _PREP_CASES])
def test_the_generated_prep_command_parses(operation, extra):
    argv = build_prep_args(operation, dict(_PREP_FULL_OPTS, **extra))
    assert argv[0] == "fnirs-prep"
    _build_prep_parser().parse_args(argv[1:])   # raises SystemExit on an unknown flag


def test_the_selected_subjects_reach_participant_label():
    argv = build_prep_args("crop", dict(_PREP_FULL_OPTS, crop_mode="single"))
    args = _build_prep_parser().parse_args(argv[1:])
    assert args.participant_label == ["001", "002"]


def test_align_takes_no_subject_selection():
    """Its subjects come from the group CSV, so a selection flag would contradict it."""
    argv = build_prep_args("hyper_align", _PREP_FULL_OPTS)
    assert "--participant-label" not in argv


def test_one_marker_edit_is_sent_at_a_time():
    """The page's radio picks one; sending two would let the CLI apply both."""
    for marker_op, flag in (("shift", "--shift"),
                            ("set_duration", "--set-duration"),
                            ("rename", "--rename")):
        argv = build_prep_args("markers", dict(_PREP_FULL_OPTS, marker_op=marker_op))
        others = {"--shift", "--set-duration", "--rename"} - {flag}
        assert flag in argv
        assert not (others & set(argv)), f"{marker_op} also sent {sorted(others & set(argv))}"


# ── The two QC pages → fnirs-qc prep-raw / hyper-raw ─────────────────────────

# These are the static counterparts of what Data Preparation and Hyper Preparation show
# interactively. Both take a BIDS root, which is what separates them from the aggregate
# commands on the Cohort Reports page.
RAW_QC_NOT_EXPOSED = {
    "prep-raw": {
        # the page screens on SCI alone, and its figures are drawn against that. A second
        # threshold here could be set to a number the viewer never used
        "--psp-threshold", "--min-good-frac", "--screen-scope",
        # the page shows one run at a time, so there is no second condition to split by
        "--by-condition",
        # the raw report is the recording before correction; the page has no control for it
        # because nothing it draws is motion-corrected
        "--motion-correction",
        "--skip-bids-validation", "--no-skip-bids-validation",
    },
    "hyper-raw": {
        "--psp-threshold", "--min-good-frac", "--screen-scope",
        # alignment and normalisation are what the page itself does and shows; the report
        # is of the recordings as they are, so it does not re-decide them
        "--normalize", "--no-normalize", "--no-align",
        # the analysis window belongs to Hyper Analysis, which is where a window changes a
        # result. Here it would only trim what the report draws
        "--tstart", "--tend",
        # the coherence band is Hyper Analysis's to set; this report does not average one
        "--coh-fmin", "--fmin", "--coh-fmax", "--fmax",
        # the page reads a task per row out of the group CSV
        "--task-label",
        "--skip-bids-validation", "--no-skip-bids-validation",
    },
}

_RAW_QC_FULL_OPTS = dict(
    bids_dir="/bids", output_dir="/out", subject="001", ses="01", task="tapping",
    dpf=6.0, cardiac_l=0.7, cardiac_h=1.5, sci_threshold=0.8,
    window_length=10.0, epoch_tmin=-5.0, epoch_tmax=25.0, epoch_qc=True,
    short_max_dist=10.0, long_min_dist=15.0, long_max_dist=45.0,
    pairs_csv="/pairs.csv", group_id="1003",
)


def _raw_qc_flags(command: str) -> set[str]:
    parser = _qc_subparsers()[command]
    return {flag for action in parser._actions if action.dest != "help"
            for flag in _long_flags(action)}


def _raw_qc_emitted(command: str) -> set[str]:
    return {a for a in build_raw_qc_args(command, _RAW_QC_FULL_OPTS) if a.startswith("--")}


@pytest.mark.parametrize("command", _RAW_QC)
def test_the_qc_pages_reach_the_raw_report_commands(command):
    assert command in _qc_subparsers()


@pytest.mark.parametrize("command", _RAW_QC)
def test_every_raw_qc_flag_is_sendable_or_written_off(command):
    missing = _raw_qc_flags(command) - _with_aliases(
        _raw_qc_emitted(command) | RAW_QC_NOT_EXPOSED.get(command, set()))
    assert not missing, (
        f"{command} grew {sorted(missing)} and the QC page cannot send them; "
        f"add a control, or add them to RAW_QC_NOT_EXPOSED with a reason"
    )


@pytest.mark.parametrize("command", _RAW_QC)
def test_the_written_off_raw_qc_flags_still_exist(command):
    stale = RAW_QC_NOT_EXPOSED[command] - _raw_qc_flags(command)
    assert not stale, f"RAW_QC_NOT_EXPOSED[{command!r}] names flags that are gone: {sorted(stale)}"


@pytest.mark.parametrize("command", _RAW_QC)
def test_the_generated_raw_qc_command_parses(command):
    argv = build_raw_qc_args(command, _RAW_QC_FULL_OPTS)
    assert argv[:2] == ["fnirs-qc", command]
    _build_qc_parser().parse_args(argv[1:])   # raises SystemExit on an unknown flag


@pytest.mark.parametrize("command", _RAW_QC)
def test_the_required_raw_qc_arguments_are_all_sent(command):
    """dpf and the cardiac band have no defaults anywhere, by design."""
    emitted = _raw_qc_emitted(command)
    assert {"--dpf", "--cardiac-l-freq", "--cardiac-h-freq"} <= emitted


@pytest.mark.parametrize("command", _RAW_QC)
def test_a_raw_report_refuses_without_the_values_that_have_no_default(command):
    for field in ("dpf", "cardiac_l", "cardiac_h"):
        opts = dict(_RAW_QC_FULL_OPTS, **{field: None})
        assert missing_raw_qc(command, opts), f"{command} accepted a missing {field}"


def test_the_raw_reports_take_a_bids_directory():
    """Unlike everything on the Cohort Reports page, these two read the recordings."""
    for command in _RAW_QC:
        argv = build_raw_qc_args(command, _RAW_QC_FULL_OPTS)
        assert argv[2] == "/bids"
