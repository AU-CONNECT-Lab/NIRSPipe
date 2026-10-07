import pytest

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset
from tests.figure_accuracy._run import Run


@pytest.fixture(scope="session")
def denoise_run(tmp_path_factory) -> Run:
    """One fingerprint recording through prep and denoise with short-channel regression."""
    from fnirs_pipe.cli.run import main

    root = tmp_path_factory.mktemp("fingerprint")
    bids, truth = make_fingerprint_dataset(root)
    out = root / "out"
    main([str(bids), str(out), "participant", *CLI_ARGS, "--mode", "denoise",
          "--short-channel", "mean", "--skip-bids-validation"])
    return Run(out, truth)
