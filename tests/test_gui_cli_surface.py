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
from fnirs_pipe.interface.callbacks.qc_callbacks import _AGGREGATE, _HYPER, build_qc_args

# Postprocessing flags the analysis page deliberately does not offer, and why. A flag listed
# here must still exist in the CLI, and must not also be emitted; both are asserted below.
NOT_EXPOSED = {
    "--config": "the form is the config surface; a TOML overriding it would make the preview lie",
    "--events-path": "a file path, and mutually exclusive with the Stim Duration field",
    "--contrast-file": "a file path, and the page has no file picker",
    "--fir-delays": "only meaningful with --hrf-model fir",
    "--no-combine-runs": "the negative half of a paired flag; the checkbox emits the positive",
}

# every control filled in, so the union over modes is everything the builder can emit
_FULL_OPTS = dict(
    bids_dir="/bids", output_dir="/out", subjects=["001"],
    session_label="ses-1", task_label="tapping",
    dpf=6.0, sci_thresh=0.5, motion_correction="tddr",
    cardiac_l=0.7, cardiac_h=1.5, resp_l=0.1, resp_h=0.5,
    high_pass=0.01, low_pass=0.1, filter_method="iir", filter_order=4,
    resample=2.0, n_jobs=1,
    hrf_model="spm", noise_model="ar1", short_channel="mean",
    aux=True, aux_channels="AUX1 AUX2",
    drift_model="cosine", drift_high_pass=0.01, drift_order=1,
    stim_dur=5.0, roi_mapping="/roi.json", fc=True,
    flags=["dry_run", "skip_bids_validation", "no_report", "combine_runs"],
)

_MODES = ["denoise", "glm", "rest"]

# built by a callback into an-subjects-container rather than declared in the layout, which
# is why the app is constructed with suppress_callback_exceptions
_DYNAMIC_IDS = {"an-subjects-checklist"}

# pages whose command builder this file holds against the CLI
_PREFIXES = ("an-", "qc-")


def _post_flags() -> set[str]:
    """Every long flag in the CLI's two postprocessing argument groups."""
    parser = _build_parser()
    return {
        flag
        for group in parser._action_groups
        if group.title and group.title.startswith("postprocessing")
        for action in group._group_actions
        for flag in action.option_strings
        if flag.startswith("--")
    }


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

def test_every_postprocessing_flag_is_offered_or_written_off():
    missing = _post_flags() - _emitted_any_mode() - set(NOT_EXPOSED)
    assert not missing, (
        f"the CLI grew {sorted(missing)} and the analysis page cannot send them; "
        f"add a control, or add them to NOT_EXPOSED with a reason"
    )


def test_the_written_off_flags_still_exist():
    """Otherwise the list quietly becomes an excuse for flags nobody removed from it."""
    stale = set(NOT_EXPOSED) - _post_flags()
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
    """It was locked behind glm; rest has always honoured it and denoise does since 0.22.0."""
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
    import fnirs_pipe.interface.callbacks.analysis_callbacks  # noqa: F401

    app.layout = html.Div([dcc.Store(id="app-bids-dir"), dash.page_container])
    app._setup_server()

    layouts = [p["layout"] for p in dash.page_registry.values()
               if p["path"] in ("/analysis", "/qc")]
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

    bound = set()
    for entry in app.callback_map.values():
        for spec in list(entry["inputs"]) + list(entry.get("state") or []):
            bound.add(spec["id"])
        outputs = entry["output"]
        for out in (outputs if isinstance(outputs, (list, tuple)) else [outputs]):
            bound.add(out.component_id)

    on_page = set()
    for lay in layouts:
        on_page |= _ids(lay)

    return on_page, bound


def test_every_control_the_callbacks_bind_to_exists_on_the_page(analysis_page):
    on_page, bound = analysis_page
    dangling = {i for i in bound if i.startswith(_PREFIXES)} - on_page - _DYNAMIC_IDS
    assert not dangling, f"callbacks bind ids the page does not define: {sorted(dangling)}"


def test_no_control_on_the_page_is_decoration(analysis_page):
    """`an-stim-dur` was drawn, never read, and never reached a command for two releases.

    An Output counts: a preview pane is driven by a callback rather than read by one. What
    this catches is a control wired to nothing in either direction.
    """
    on_page, bound = analysis_page
    unread = {i for i in on_page if i.startswith(_PREFIXES)} - bound
    assert not unread, f"controls nothing reads: {sorted(unread)}"


# ---- the same contract for the QC page, which drives two tools ----

# The page builds `fnirs-qc` for the aggregate commands and `fnirs-hyper` for the dyad
# analysis. Both are subparser CLIs, so each command's flag list is read off its own parser.

# fnirs-qc subcommands the QC page does not offer, and why.
QC_COMMANDS_NOT_OFFERED = {
    "prep-raw": "single-subject QC; the Data Prep page's QC tab already does this interactively",
    "hyper-raw": "pre-analysis dyad QC; belongs with Hyper Align, not with the post-analysis page",
}

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
            # the positive half of a paired flag, and it is the default; the checkbox emits
            # the negative one. See memory/project_mask_coi_default_on
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
    "merge": {"--verbose"},
    # every group-* directory by default, which is the whole shape of the aggregate form
    "index": {"--verbose", "--group-id"},
}

_QC_FULL_OPTS = dict(
    output_dir="/out", pairs_csv="/pairs.csv", group_id="01",
    desc="errts", roi_mapping="/roi.json",
    wtc_fmin=0.004, wtc_fmax=0.2, wtc_band_fmin=0.01, wtc_band_fmax=0.1,
    wtc_mc_count=300, wtc_seed=42, isc_threshold=0.3,
    wtc_pseudo=100, wtc_roi_min_channels=2, wtc_chroma="both",
    isc_whiten=32, isc_pseudo=100,
    hyper_task="rest",
    hyper_flags=["wtc_significance", "wtc_no_mask_coi", "wtc_channel_cross", "wtc_pseudo_cross",
                 "no_by_condition", "bads_subject", "wtc_save_maps", "no_align", "normalize",
                 "check_only"],
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


def _build_hyper_parser():
    from fnirs_pipe.cli.hyper import _build_parser as build
    return build()


def _qc_subparsers():
    return _subcommands(_build_qc_parser)


def _hyper_subparsers():
    return _subcommands(_build_hyper_parser)


def _qc_flags(command: str) -> set[str]:
    parser = (_hyper_subparsers() if command in _HYPER else _qc_subparsers())[command]
    return {flag for action in parser._actions for flag in action.option_strings
            if flag.startswith("--") and action.dest != "help"}


def _qc_emitted(command: str) -> set[str]:
    return {a for a in build_qc_args(command, _QC_FULL_OPTS) if a.startswith("--")}


def test_the_qc_page_offers_every_subcommand_or_writes_it_off():
    missing = set(_qc_subparsers()) - set(_QC_OFFERED) - set(QC_COMMANDS_NOT_OFFERED)
    assert not missing, (
        f"fnirs-qc grew {sorted(missing)} and the QC page neither offers nor declines them"
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
    missing = _qc_flags(command) - _qc_emitted(command) - QC_NOT_EXPOSED.get(command, set())
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
        assert argv[:2] == ["fnirs-hyper", command]
        _build_hyper_parser().parse_args(argv[1:])   # raises SystemExit on an unknown flag
    else:
        assert argv[:2] == ["fnirs-qc", command]
        _build_qc_parser().parse_args(argv[1:])


def test_neither_tool_on_this_page_asks_for_a_bids_directory():
    """Both read derivatives only, which is why the page has one directory field."""
    for command in _QC_OFFERED:
        argv = build_qc_args(command, _QC_FULL_OPTS)
        assert "/bids" not in argv


def test_a_space_separated_box_repeats_its_flag_rather_than_joining():
    """argparse nargs="+" takes repeats; one string with a space in it is one label."""
    argv = build_qc_args("run", dict(_QC_FULL_OPTS, hyper_task="rest tap"))
    args = _build_hyper_parser().parse_args(argv[1:])
    assert args.task_label == ["rest", "tap"]


def test_the_aggregate_commands_take_only_an_output_directory():
    for command in _AGGREGATE:
        assert build_qc_args(command, _QC_FULL_OPTS) == ["fnirs-qc", command, "/out"]
