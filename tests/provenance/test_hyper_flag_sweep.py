"""Every setting of the nirspipe-hyper family: change it alone, and the output and the record move together.

The dyad counterpart of test_flag_sweep.py. nirspipe-hyper writes a run record, so a case
checks its [hyper] key there as well as the sidecars of the tables it names; pairnull,
groupnull and band write none, so their tables' sidecars are the record. A setting that
must equal what the real tables recorded is checked by being refused when it does not.
"""

import shutil
from dataclasses import dataclass, field

import pytest

from nirspipe.cli import hyper as hyper_cli
from nirspipe.cli import run as run_cli

from tests._dyad_fingerprint import TASK, make_group_dataset, write_roi_map
from tests._fingerprint import CLI_ARGS
from tests.provenance._sweep import argv, moved, snapshot

# the flags that pick what runs rather than how
SELECTION = {"help", "version", "verbose", "config", "pairs_csv", "group_id",
             "participant_label", "task_label", "check_only", "no_report", "task"}
# where it runs: test_hyper_nprocs.py and test_wtc_gpu.py hold the outputs to the serial CPU run's
EXECUTION = {"nprocs", "device"}


@dataclass(frozen=True)
class Case:
    change: dict
    sidecars: tuple = ()             # stages whose parameters must change
    context: dict = field(default_factory=dict)
    report: bool = False             # the setting acts on the report only
    refused: bool = False            # any value but the recorded one is refused


SIGNIFICANCE = {"--wtc-significance": [], "--wtc-mc-count": ["10"]}
PHASE_NULL = {"--wtc-phase-null": ["5"]}
ROI = {"--roi-mapping": ["{roi}"]}
COND_TRANSFORM = {"--wtc-cond-transform": []}
# the short pair at 8 mm becomes a long one, so a long band edge has a channel to move
SHORT_AS_LONG = {"--short-max-dist": ["5"], "--long-min-dist": ["6"]}
BOTH = ("wtc", "isc")

HYPER = {
    "normalize": Case({"--normalize": []}),
    "no_align": Case({"--no-align": []}, BOTH),
    "tstart": Case({"--tstart": ["20"]}, BOTH),
    "tend": Case({"--tend": ["380"]}, BOTH),
    "wtc_band_fmin": Case({"--wtc-band-fmin": ["0.04"]}, ("wtc",)),
    "wtc_band_fmax": Case({"--wtc-band-fmax": ["0.08"]}, ("wtc",)),
    "wtc_mask_coi": Case({"--no-wtc-mask-coi": []}, ("wtc",)),
    "desc": Case({"--desc": ["filtered"]}, BOTH),
    "roi_mapping": Case(ROI, BOTH),
    "wtc_fmin": Case({"--wtc-fmin": ["0.03"]}, ("wtc",)),
    "wtc_fmax": Case({"--wtc-fmax": ["0.15"]}, ("wtc",)),
    "wtc_significance": Case(SIGNIFICANCE, report=True, context={"--wtc-mc-count": ["10"]}),
    "wtc_mc_count": Case({"--wtc-mc-count": ["20"]}, context=SIGNIFICANCE, report=True),
    "wtc_seed": Case({"--wtc-seed": ["1"]}, ("level",), PHASE_NULL),
    "wtc_chroma": Case({"--wtc-chroma": ["hbo"]}, ("wtc",)),
    "wtc_roi_min_channels": Case({"--wtc-roi-min-channels": ["2"]}, ("wtc",), ROI),
    "wtc_arrow_min": Case({"--wtc-arrow-min": ["0.8"]}, report=True),
    "wtc_channel_cross": Case({"--no-channel-cross": []}, BOTH),
    "wtc_window_s": Case({"--wtc-window-s": ["40"]}, BOTH),
    "wtc_by_condition": Case({"--no-by-condition": []}, BOTH),
    "wtc_cond_transform": Case(COND_TRANSFORM, ("wtc",)),
    "wtc_cond_pad_s": Case({"--wtc-cond-pad-s": ["10"]}, ("wtc",), COND_TRANSFORM),
    "wtc_limit_scales": Case({"--no-wtc-limit-scales": []}),
    "wtc_save_maps": Case({"--wtc-save-maps": []}),
    "wtc_phase_null": Case(PHASE_NULL, ("level",)),
    "wtc_phase_null_cross": Case({"--no-wtc-phase-null-cross": []}, ("draws",), PHASE_NULL),
    "bads_scope": Case({"--bads-scope": ["subject"]}, BOTH),
    "isc_threshold": Case({"--isc-threshold": ["0.3"]}, report=True),
    "band_fmin": Case({"--band-fmin": ["0.04"]}, ("isc",)),
    "band_fmax": Case({"--band-fmax": ["0.08"]}, ("isc",)),
    "isc_fmin": Case({"--isc-fmin": ["0.02"]}, ("isc",)),
    "isc_fmax": Case({"--isc-fmax": ["0.12"]}, ("isc",)),
    "wtc_whiten": Case({"--wtc-whiten": ["2"]}, ("wtc",)),
    "isc_whiten_s": Case({"--isc-whiten-s": ["2"]}, ("isc",)),
    "isc_max_lag": Case({"--isc-max-lag": ["2"]}, ("isc",)),
    "isc_phase_null": Case({"--isc-phase-null": ["5"]}, ("isc",)),
    "phase_null_shared": Case({"--phase-null-shared": []}, ("isc", "level", "draws"),
                              {"--isc-phase-null": ["5"], **PHASE_NULL}),
    "short_max_dist": Case({"--short-max-dist": ["5"]}, BOTH),
    "long_min_dist": Case({"--long-min-dist": ["6"]}, BOTH, {"--short-max-dist": ["5"]}),
    "long_max_dist": Case({"--long-max-dist": ["20"]}, BOTH, SHORT_AS_LONG),
}

PAIRNULL = {
    "desc": Case({"--desc": ["filtered"]}, refused=True),
    "roi_mapping": Case(ROI, BOTH),
    "bads_scope": Case({"--bads-scope": ["subject"]}, refused=True),
    "chroma": Case({"--chroma": ["hbo"]}, BOTH),
    "wtc_pair_pool": Case({"--wtc-pair-pool": ["any"]}, BOTH),
    # two dyads leave one stand-in per member by position; any gives two to cap
    "wtc_pair_max": Case({"--wtc-pair-max": ["1"]}, BOTH, {"--wtc-pair-pool": ["any"]}),
    "wtc_pair_cross": Case({"--no-wtc-pair-cross": []}, BOTH),
    "wtc_roi_min_channels": Case({"--wtc-roi-min-channels": ["2"]}, refused=True),
    "wtc_limit_scales": Case({"--no-wtc-limit-scales": []}),
}

GROUPNULL = {
    "chroma": Case({"--chroma": ["hbo"]}, ("cohort",)),
    "null": Case({"--null": ["phase"]}, ("cohort",)),
    "roi_mapping": Case(ROI, ("cohort",)),
    "n_resample": Case({"--n-resample": ["60"]}, ("cohort",)),
    "seed": Case({"--seed": ["1"]}, ("cohort",)),
    "p_correction": Case({"--p-correction": ["fdr_bh"]}, ("cohort",)),
    "isc_test": Case({"--isc-test": ["magnitude"]}, ("cohort",)),
}

BAND = {
    "wtc_band_fmin": Case({"--wtc-band-fmin": ["0.05"]}, ("wtc",)),
    "wtc_band_fmax": Case({"--wtc-band-fmax": ["0.07"]}, ("wtc",)),
    "wtc_mask_coi": Case({"--no-wtc-mask-coi": []}, ("wtc",)),
    "wtc_suffix": Case({"--wtc-suffix": ["mid"]}, ("wtc",)),
}

COMMANDS = {"nirspipe-hyper": HYPER, "nirspipe-hyper-pairnull": PAIRNULL,
            "nirspipe-hyper-groupnull": GROUPNULL, "nirspipe-hyper-band": BAND,
            "nirspipe-hyper-index": {}, "nirspipe-hyper-merge": {}}

# a setting whose output may stay put, with the reason
ALLOWED_SAME = {
    ("nirspipe-hyper", "bads_scope"): "differs from run only across cropped tasks; one run per member here",
    ("nirspipe-hyper", "wtc_save_maps"): "adds the maps beside the tables; the tables stay as they were",
    ("nirspipe-hyper", "normalize"): "coherence and correlation do not change under a per-channel z-score",
    ("nirspipe-hyper", "wtc_limit_scales"): "a speed option: the scales it skips lie outside the band",
    ("nirspipe-hyper-pairnull", "wtc_limit_scales"): "a speed option, as in nirspipe-hyper",
    ("nirspipe-hyper", "wtc_fmin"): "the band means read inside the band, which both axes cover",
    ("nirspipe-hyper", "short_max_dist"): "a short channel and one in neither band are both left out",
}

BASE = {
    "nirspipe-hyper": {"--pairs-csv": ["{pairs}"], "--group-id": ["G01"], "--wtc-fmin": ["0.02"],
                       "--wtc-band-fmin": ["0.03"], "--wtc-band-fmax": ["0.10"],
                       "--wtc-seed": ["0"]},
    "nirspipe-hyper-pairnull": {"--pairs-csv": ["{pairs}"]},
    "nirspipe-hyper-groupnull": {"--task-label": [TASK], "--n-resample": ["50"], "--seed": ["0"]},
    "nirspipe-hyper-band": {"--wtc-band-fmin": ["0.04"], "--wtc-band-fmax": ["0.08"]},
}


# ---- the inputs: two dyads through denoise, and one full tree the later stages start from ----

@pytest.fixture(scope="module")
def dyads(tmp_path_factory):
    root = tmp_path_factory.mktemp("hyper_flag_sweep")
    bids, _, _ = make_group_dataset(root)
    pairs = root / "dyads.csv"
    pairs.write_text("group_id,subject_id,task\n" + "".join(
        f"{g},sub-{s},{TASK}\n" for g, s in (("G01", "01"), ("G01", "02"), ("G02", "03"), ("G02", "04"))))
    deriv = root / "deriv"
    run_cli.main([str(bids), str(deriv), "participant", *CLI_ARGS, "--mode", "denoise",
                  "--participant-label", "01", "02", "03", "04", "--skip-bids-validation",
                  "--no-report"])
    paths = {"pairs": str(pairs), "roi": str(write_roi_map(root))}
    full = root / "full"
    hyper_cli.main([str(deriv), str(full), "group", "--pairs-csv", str(pairs),
                    "--wtc-fmin", "0.02", "--wtc-band-fmin", "0.03", "--wtc-band-fmax", "0.10",
                    "--wtc-save-maps", "--wtc-phase-null", "5", "--isc-phase-null", "5",
                    "--wtc-seed", "0", "--no-report"])
    with_draws = root / "with_draws"
    shutil.copytree(full, with_draws)
    hyper_cli.main_pair_null([str(deriv), str(with_draws), "group", "--pairs-csv", str(pairs)])
    return {"root": root, "deriv": deriv, "paths": paths, "full": full, "with_draws": with_draws}


def _hyper(dyads, out, settings, report):
    hyper_cli.main([str(dyads["deriv"]), str(out), "group", *settings,
                    *([] if report else ["--no-report"])])
    group = out / "group-G01"
    return snapshot(group, group / "logs" / f"group-G01_task-{TASK}.toml", {"hyper": None})


def _on_a_copy(start: str, call):
    def run(dyads, out, settings, report):
        shutil.copytree(dyads[start], out)
        call(dyads, out, settings)
        return snapshot(out)
    return run


RUNNERS = {
    "nirspipe-hyper": _hyper,
    "nirspipe-hyper-pairnull": _on_a_copy("full", lambda d, out, s: hyper_cli.main_pair_null(
        [str(d["deriv"]), str(out), "group", *s])),
    "nirspipe-hyper-groupnull": _on_a_copy("with_draws", lambda d, out, s: hyper_cli.main_group_null(
        [str(out), "group", *s])),
    "nirspipe-hyper-band": _on_a_copy("full", lambda d, out, s: hyper_cli.main_band(
        [str(out), "group", *s])),
}


@pytest.fixture(scope="module")
def runs(dyads):
    bases: dict = {}

    def run(command: str, name: str, settings: dict, report: bool):
        out = dyads["root"] / f"{command}_{name}"
        return RUNNERS[command](dyads, out, argv(settings, dyads["paths"]), report)

    def pair(command: str, key: str):
        case = COMMANDS[command][key]
        base = {**BASE[command], **case.context}
        tag = repr((command, case.report, sorted(base.items())))
        if tag not in bases:
            bases[tag] = run(command, f"base{len(bases)}", base, case.report)
        return bases[tag], run(command, key, {**base, **case.change}, case.report)

    def refused(command: str, key: str):
        case = COMMANDS[command][key]
        return run(command, key, {**BASE[command], **case.context, **case.change}, False)

    return pair, refused


# ---- the sweep ----

def _settings(parser) -> set[str]:
    return {a.dest for a in parser._actions if a.option_strings} - SELECTION - EXECUTION


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_every_setting_of_the_family_has_a_case(command):
    assert set(COMMANDS[command]) == _settings(hyper_cli._parsers()[command])


@pytest.mark.parametrize("command, key", [(c, k) for c, cases in COMMANDS.items()
                                          for k, case in cases.items() if not case.refused])
def test_a_changed_setting_moves_the_output_and_the_record(runs, command, key):
    pair, _ = runs
    base, changed = pair(command, key)
    if base["record"] or changed["record"]:
        assert base["record"].get(f"hyper.{key}") != changed["record"].get(f"hyper.{key}")
    if (command, key) not in ALLOWED_SAME:
        assert moved(base, changed), "the record changed and no output did"
    for stage in COMMANDS[command][key].sidecars:
        assert base["params"].get(stage) != changed["params"].get(stage), \
            f"the {stage} tables changed nothing in their sidecar parameters"


@pytest.mark.parametrize("command, key", [(c, k) for c, cases in COMMANDS.items()
                                          for k, case in cases.items() if case.refused])
def test_a_setting_the_real_tables_fix_is_refused_when_it_differs(runs, capsys, command, key):
    _, refused = runs
    with pytest.raises(SystemExit):
        refused(command, key)
    assert "disagrees with the real table" in capsys.readouterr().err
