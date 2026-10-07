import pytest

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset
from tests.figure_accuracy._run import Run


def run_capturing(root, args: list, spied: tuple) -> dict:
    """Run fnirs-pipe on a fresh fingerprint dataset, recording what each named report builder was handed."""
    from fnirs_pipe.cli.run import main
    from fnirs_pipe.qc.subject import report

    captured: dict = {}
    originals = {name: getattr(report, name) for name in spied}

    def spy(name):
        def wrapper(*a, **kw):
            captured[name] = (a, kw)
            return originals[name](*a, **kw)
        return wrapper

    bids, truth = make_fingerprint_dataset(root)
    for name in spied:
        setattr(report, name, spy(name))
    try:
        main([str(bids), str(root / "out"), "participant", *CLI_ARGS, *args,
              "--skip-bids-validation"])
    finally:
        for name, original in originals.items():
            setattr(report, name, original)
    return {"truth": truth, "captured": captured}


@pytest.fixture(scope="session")
def denoise_run(tmp_path_factory) -> Run:
    """One fingerprint recording through prep and denoise with short-channel regression."""
    root = tmp_path_factory.mktemp("fingerprint")
    done = run_capturing(root, ["--mode", "denoise", "--short-channel", "mean"],
                         ("quality_brain_views",))
    return Run(root / "out", done["truth"], captured=done["captured"])
