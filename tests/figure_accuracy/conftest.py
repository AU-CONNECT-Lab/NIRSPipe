import pytest

from tests._fingerprint import CLI_ARGS, EVENT_DURATION, make_fingerprint_dataset
from tests.figure_accuracy._run import Run

# prep-raw takes no respiration band
RAW_VIEWER_ARGS = [a for i, a in enumerate(CLI_ARGS)
                   if not (a.startswith("--resp-") or (i and CLI_ARGS[i - 1].startswith("--resp-")))]


def run_capturing(root, args: list, spied: tuple, task: str = "tapping", rest: bool = False,
                  blocks: bool = False, raw_viewer: bool = False, aux: bool = False) -> dict:
    """Run fnirs-pipe, or fnirs-qc prep-raw, on a fresh fingerprint dataset, recording what each
    named figure builder was handed."""
    import fnirs_pipe.qc.figures as figures
    import fnirs_pipe.qc.figures.common.provenance_figure as provenance
    from fnirs_pipe.cli import qc as qc_cli
    from fnirs_pipe.cli import run as run_cli
    from fnirs_pipe.qc.subject import report

    captured: dict = {}
    # the report imports most builders by name and a few at call time from the package; the
    # provenance diagram is drawn inside its own module
    patched = [(module, name, getattr(module, name)) for name in spied
               for module in (report, figures, provenance) if hasattr(module, name)]

    def spy(name, original):
        def wrapper(*a, **kw):
            captured[name] = (a, kw)
            return original(*a, **kw)
        return wrapper

    bids, truth = make_fingerprint_dataset(root, task=task, rest=rest, blocks=blocks, aux=aux)
    for module, name, original in patched:
        setattr(module, name, spy(name, original))
    try:
        if raw_viewer:
            qc_cli.main(["prep-raw", str(bids), str(root / "out"), *RAW_VIEWER_ARGS, *args,
                         "--skip-bids-validation"])
        else:
            run_cli.main([str(bids), str(root / "out"), "participant", *CLI_ARGS, *args,
                          "--skip-bids-validation"])
    finally:
        for module, name, original in patched:
            setattr(module, name, original)
    return {"truth": truth, "captured": captured}


# in the task recording each of A and B holds one responder; in the resting one, A holds the
# network pair in phase and B the one out of phase
ROI_MAP = {"A": ["S1_D1", "S2_D2"], "B": ["S4_D4", "S5_D5"], "C": ["S6_D6"]}


def _roi_file(root) -> str:
    import json

    path = root / "roi_net.json"
    path.write_text(json.dumps(ROI_MAP))
    return str(path)


@pytest.fixture(scope="session")
def denoise_run(tmp_path_factory) -> Run:
    """One fingerprint recording through prep and denoise with short-channel regression."""
    root = tmp_path_factory.mktemp("fingerprint")
    done = run_capturing(root, ["--mode", "denoise", "--short-channel", "mean",
                                "--roi-mapping", _roi_file(root)],
                         ("quality_brain_views", "provenance_figure", "_build_mne_report"))
    return Run(root / "out", done["truth"], captured=done["captured"])


@pytest.fixture(scope="session")
def glm_run(tmp_path_factory) -> Run:
    """The same recording, with an accelerometer axis, through the GLM with aux regressors; its
    design and activation builders' inputs captured."""
    root = tmp_path_factory.mktemp("fingerprint_glm")
    done = run_capturing(root, ["--mode", "glm", "--drift-high-pass", "0.008",
                                "--stim-dur", f"{EVENT_DURATION:g}", "--short-channel", "mean",
                                "--aux-regressors"],
                         ("design_matrix_static_figure", "design_matrix_heatmap",
                          "activation_condition_figures"), aux=True)
    return Run(root / "out", done["truth"], captured=done["captured"])


@pytest.fixture(scope="session")
def rest_run(tmp_path_factory) -> Run:
    """The resting-state fingerprint through rest mode, with an ROI map."""
    root = tmp_path_factory.mktemp("fingerprint_rest")
    done = run_capturing(root, ["--mode", "rest", "--short-channel", "mean",
                                "--roi-mapping", _roi_file(root)], (), task="rest", rest=True)
    return Run(root / "out", done["truth"], task="rest")


@pytest.fixture(scope="session")
def condition_run(tmp_path_factory) -> Run:
    """The two-level design, blocks naming the conditions, through denoise with condition pages."""
    root = tmp_path_factory.mktemp("fingerprint_blocks")
    done = run_capturing(root, ["--mode", "denoise", "--short-channel", "mean", "--by-condition"],
                         (), task="main", blocks=True)
    return Run(root / "out", done["truth"], task="main")


# a prep run with two settings off the defaults that every condition page has to carry
PREP_SHORT_MAX_MM = 5.0
HAND_MARKED = "S2_D2"


@pytest.fixture(scope="session")
def prep_condition_run(tmp_path_factory) -> Run:
    """The two-level design through prep only, with its own separation bands and a pair
    marked bad by hand, with condition pages."""
    root = tmp_path_factory.mktemp("fingerprint_prep_blocks")
    done = run_capturing(root, ["--by-condition", "--short-max-dist", f"{PREP_SHORT_MAX_MM:g}",
                                "--bad-channels", HAND_MARKED], (), task="main", blocks=True)
    return Run(root / "out", done["truth"], task="main")


@pytest.fixture(scope="session")
def raw_condition_run(tmp_path_factory) -> Run:
    """The two-level design through fnirs-qc prep-raw, with condition pages."""
    root = tmp_path_factory.mktemp("fingerprint_raw_blocks")
    done = run_capturing(root, ["--participant-label", "01", "--epoch-qc", "--by-condition"], (),
                         task="main", blocks=True, raw_viewer=True)
    return Run(root / "out", done["truth"], task="main")


# ---- the group commands, on the dyad fingerprint ----

def _flag(name: str) -> list[str]:
    """One flag of CLI_ARGS with its values."""
    i = CLI_ARGS.index(name) + 1
    j = next((k for k in range(i, len(CLI_ARGS)) if CLI_ARGS[k].startswith("--")), len(CLI_ARGS))
    return [name, *CLI_ARGS[i:j]]


# hyper-raw takes the physiology and the SCI line, and none of the pipeline's other settings
DYAD_RAW_ARGS = [a for flag in ("--dpf", "--sci-threshold", "--cardiac-l-freq", "--cardiac-h-freq")
                 for a in _flag(flag)]
DYAD_HYPER_ARGS = ["--wtc-fmin", "0.02", "--wtc-band-fmin", "0.03", "--wtc-band-fmax", "0.10"]


@pytest.fixture(scope="session")
def groups(tmp_path_factory):
    """The dyad and the triad through every command that draws a group or cohort page:
    fnirs-pipe, fnirs-qc cohort, fnirs-qc hyper-raw, fnirs-hyper, its index, and
    fnirs-qc cohort-hyper, with the channel coherence maps' builder inputs captured."""
    from tests._dyad_fingerprint import make_group_dataset, write_roi_map
    from tests.figure_accuracy._dyad import Groups
    from fnirs_pipe.cli import hyper as hyper_cli
    from fnirs_pipe.cli import qc as qc_cli
    from fnirs_pipe.cli import run as run_cli
    from fnirs_pipe.qc.hyper import hyper_report

    root = tmp_path_factory.mktemp("dyad")
    bids, pairs, truth = make_group_dataset(root)
    deriv, hyper = root / "deriv", root / "hyper"
    calls: dict = {}
    spied = [(hyper_report, "build_wtc_channel")]

    def spy(name, original):
        def wrapper(*a, **kw):
            out = original(*a, **kw)
            calls.setdefault(name, []).append((a, kw, out))
            return out
        return wrapper

    originals = [(module, name, getattr(module, name)) for module, name in spied]
    for module, name, original in originals:
        setattr(module, name, spy(name, original))
    try:
        run_cli.main([str(bids), str(deriv), "participant", *CLI_ARGS, "--by-condition",
                      "--skip-bids-validation"])
        qc_cli.main(["cohort", str(deriv)])
        qc_cli.main(["hyper-raw", str(bids), str(hyper), "group", "--pairs-csv", str(pairs),
                     "--derivatives-dir", str(deriv), "--skip-bids-validation", *DYAD_RAW_ARGS])
        hyper_cli.main([str(deriv), str(hyper), "group", "--pairs-csv", str(pairs),
                        "--roi-mapping", str(write_roi_map(root)), *DYAD_HYPER_ARGS])
        hyper_cli.main_index([str(hyper), "group"])
        qc_cli.main(["cohort-hyper", str(hyper)])
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
    return Groups(root, bids, deriv, hyper, truth, calls=calls)


@pytest.fixture(scope="session")
def raw_viewer_run(tmp_path_factory) -> Run:
    """The task recording through fnirs-qc prep-raw, with motion correction and per-trial scoring."""
    root = tmp_path_factory.mktemp("fingerprint_raw")
    done = run_capturing(root, ["--participant-label", "01", "--epoch-qc"], (), raw_viewer=True)
    return Run(root / "out", done["truth"])


@pytest.fixture(scope="session")
def dyad_variants(groups):
    """G01 again through the flags the main run leaves at their defaults, one tree each, off the
    same derivatives; and the cohort-hyper page over a copy of the tree with one dyad's block
    missing from its usable table."""
    import shutil

    import pandas as pd

    from tests._dyad_fingerprint import write_roi_map
    from tests.figure_accuracy._dyad import Groups
    from fnirs_pipe.cli import hyper as hyper_cli
    from fnirs_pipe.cli import qc as qc_cli
    from fnirs_pipe.qc.hyper import hyper_report

    roi = str(write_roi_map(groups.root))
    pairs = str(groups.root / "bids_dyad_pairs.csv")
    out: dict = {}

    def _post(name, *flags):
        calls: dict = {}
        original = hyper_report.build_wtc_channel

        def wrapper(*a, **kw):
            got = original(*a, **kw)
            calls.setdefault("build_wtc_channel", []).append((a, kw, got))
            return got

        hyper_report.build_wtc_channel = wrapper
        tree = groups.root / f"hyper_{name}"
        try:
            hyper_cli.main([str(groups.deriv), str(tree), "group", "--pairs-csv", pairs,
                            "--group-id", "G01", "--roi-mapping", roi, *DYAD_HYPER_ARGS, *flags])
            hyper_cli.main_index([str(tree), "group"])
        finally:
            hyper_report.build_wtc_channel = original
        out[name] = Groups(groups.root, groups.bids, groups.deriv, tree, groups.truth, calls=calls)

    def _raw(name, *flags):
        tree = groups.root / f"hyper_{name}"
        qc_cli.main(["hyper-raw", str(groups.bids), str(tree), "group", "--pairs-csv", pairs,
                     "--group-id", "G01", "--derivatives-dir", str(groups.deriv),
                     "--skip-bids-validation", *DYAD_RAW_ARGS, *flags])
        out[name] = Groups(groups.root, groups.bids, groups.deriv, tree, groups.truth)

    _post("condtransform", "--wtc-cond-transform")
    _post("phasenull", "--wtc-phase-null", "10", "--wtc-seed", "0")
    _post("phasenull_homologous", "--wtc-phase-null", "10", "--wtc-seed", "0",
          "--no-wtc-phase-null-cross")
    _post("uncrossed", "--no-channel-cross")
    _raw("tstart", "--tstart", "20", "--tend", "380")
    _raw("normalize", "--normalize")

    gap = groups.root / "hyper_cohortgap"
    shutil.copytree(groups.hyper, gap, ignore=shutil.ignore_patterns("figures", "*.html"))
    usable = gap / "group-G01" / "nirs" / "group-G01_task-main_desc-usable_qc.tsv"
    table = pd.read_csv(usable, sep="\t")
    table[table["condition"] != "cb"].to_csv(usable, sep="\t", index=False)
    qc_cli.main(["cohort-hyper", str(gap)])
    out["cohortgap"] = Groups(groups.root, groups.bids, groups.deriv, gap, groups.truth)
    return out
