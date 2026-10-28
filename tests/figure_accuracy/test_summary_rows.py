"""The run report's summary names the recording's length and the screening's passing share, as the raw viewer does."""

import re

import pytest

from tests._fingerprint import BLOCKS, CLI_ARGS, DURATION, SFREQ


def _cell(html: str, heading: str) -> str:
    return re.search(rf"<th>{heading}</th><td>([^<]*)</td>", html).group(1)


def test_the_run_page_names_the_recording_s_length_and_passing_share(denoise_run):
    html = denoise_run.report.read_text(encoding="utf-8")
    # the last sample sits one period before the recording's nominal end
    assert _cell(html, "Duration") == f"{DURATION - 1 / SFREQ:.0f} s"
    share = float(CLI_ARGS[CLI_ARGS.index("--min-good-frac") + 1])
    assert _cell(html, "Passing windows") == f"{share * 100:.0f}% of windows"


@pytest.mark.parametrize("name,duration", [(b[0], b[2]) for b in BLOCKS])
def test_a_condition_page_names_its_own_window_as_its_duration(condition_run, name, duration):
    html = (condition_run.report.parent / f"sub-01_task-main_cond-{name}_report.html").read_text(
        encoding="utf-8")
    assert _cell(html, "Duration") == f"{duration:.0f} s"
