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


@pytest.fixture(scope="session")
def raw_viewer_run(tmp_path_factory) -> Run:
    """The task recording through fnirs-qc prep-raw, with motion correction and per-trial scoring."""
    root = tmp_path_factory.mktemp("fingerprint_raw")
    done = run_capturing(root, ["--participant-label", "01", "--epoch-qc"], (), raw_viewer=True)
    return Run(root / "out", done["truth"])
