"""The generated script has to name the noise model the run actually fitted.

Every mode honours --noise-model now, but the generator kept the constant inside its glm
branch and hard-coded "ols" in the denoise and rest blocks, so a denoise run fitted with an
AR model handed the reader a script that would reproduce something else.
"""

import sys

import pytest

from fnirs_pipe.cli.workflows import run_participant_level

_ARGS = dict(
    analysis_level="participant", session_label=None, task_label=["tapping"],
    bids_filter_file=None, work_dir=None, verbose=False, skip_bids_validation=True,
    ignore=None, n_jobs=1, no_report=True,
    dpf=[6.0, 6.0], sci_threshold=0.8, motion_correction="tddr",
    cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5,
    mode="denoise", high_pass=0.01, low_pass=0.5,
    # a drift basis is what makes denoise regress at all, and so what emits the block
    drift_model="cosine", drift_high_pass=0.01,
)


def _script_for(bids, out_dir, **extra) -> str:
    args = {**_ARGS, "bids_dir": bids, "output_dir": out_dir,
            "participant_label": ["01"], "dry_run": True, **extra}
    original = sys.argv
    sys.argv = ["fnirs-pipe", str(bids), str(out_dir), "participant", "--dry-run"]
    try:
        run_participant_level(args)
    finally:
        sys.argv = original
    return (out_dir / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("asked", ["ar5", "ols", "auto"])
def test_denoise_reports_the_model_it_was_given(mini_bids, tmp_path_factory, asked):
    out_dir = tmp_path_factory.mktemp(f"noise_{asked}") / "out"
    text = _script_for(mini_bids, out_dir, noise_model=asked)

    assert f"NOISE_MODEL    = {asked!r}" in text
    assert "noise_model=NOISE_MODEL," in text
    # the defect spelled it straight into the call, so the constant alone does not prove it
    assert 'noise_model="ols"' not in text
