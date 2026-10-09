"""A dyad's provenance diagram draws that task's chain only, on its pages and from `nirspipe-qc provenance`."""

import pytest

TASKS = ("hold", "rest")
PHYS = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
        "--sci-threshold", "0.8"]


def _mmd(figures, stem):
    return next(figures.glob(f"{stem}_desc-provenance_*.mmd")).read_text(encoding="utf-8")


def _only_its_own(text, own, others):
    # mermaid ids are the sidecar stems with every non-word character as "_"
    return f"_{own}_" in text and not any(f"_{o}_" in text for o in others)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """One dyad, two tasks, through hyper-raw and nirspipe-hyper; the raw pages' graphs kept."""
    from nirspipe.cli import hyper as hyper_cli
    from nirspipe.cli import qc as qc_cli
    from nirspipe.cli import run as run_cli
    from tests._fingerprint import CLI_ARGS
    from tests._synth import make_hyper_dataset

    root = tmp_path_factory.mktemp("provenance")
    bids, pairs = make_hyper_dataset(root, tasks=TASKS)
    deriv, hyper = root / "deriv", root / "hyper"
    figures = hyper / "group-G01" / "figures"
    run_cli.main([str(bids), str(deriv), "participant", *CLI_ARGS, "--skip-bids-validation"])
    qc_cli.main(["hyper-raw", str(bids), str(hyper), "group", "--pairs-csv", str(pairs),
                 "--skip-bids-validation", "--no-align", *PHYS])
    # the post page writes its graph under the same name, so the raw page's is read first
    raw = {task: _mmd(figures, f"group-G01_task-{task}") for task in TASKS}
    hyper_cli.main([str(deriv), str(hyper), "group", "--pairs-csv", str(pairs),
                    "--no-align", "--wtc-fmin", "0.02"])
    post = {task: _mmd(figures, f"group-G01_task-{task}") for task in TASKS}
    return {"hyper": hyper, "figures": figures, "raw": raw, "post": post}


@pytest.mark.parametrize("page", ["raw", "post"])
@pytest.mark.parametrize("task", TASKS)
def test_a_dyad_page_s_provenance_holds_its_own_task_only(tree, page, task):
    text = tree[page][task]
    assert _only_its_own(text, f"task_{task}", [f"task_{t}" for t in TASKS if t != task]), text


def test_the_provenance_command_draws_each_dyad_task_apart(tree):
    from nirspipe.cli import qc as qc_cli

    for path in tree["figures"].glob("*_desc-provenance_*"):
        path.unlink()
    qc_cli.main(["provenance", str(tree["hyper"])])
    for task in TASKS:
        text = _mmd(tree["figures"], f"group-G01_task-{task}")
        assert _only_its_own(text, f"task_{task}", [f"task_{t}" for t in TASKS if t != task])


def test_the_provenance_command_reaches_a_dyad_recorded_per_session(tmp_path):
    from nirspipe.cli import qc as qc_cli
    from tests._synth import make_hyper_dataset

    bids, pairs = make_hyper_dataset(tmp_path, tasks=("hold",), sessions=("a", "b"))
    hyper = tmp_path / "hyper"
    qc_cli.main(["hyper-raw", str(bids), str(hyper), "group", "--pairs-csv", str(pairs),
                 "--skip-bids-validation", *PHYS])
    figures = hyper / "group-G01" / "figures"
    for path in figures.glob("*_desc-provenance_*"):
        path.unlink()
    qc_cli.main(["provenance", str(hyper)])
    for ses, other in (("a", "b"), ("b", "a")):
        text = _mmd(figures, f"group-G01_ses-{ses}_task-hold")
        assert _only_its_own(text, f"ses_{ses}", [f"ses_{other}"]), text
