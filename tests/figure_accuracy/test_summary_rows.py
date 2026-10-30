"""The run report's summary and Run summary carry what the raw viewer shows of the same recording."""

import json
import re

import pytest

from tests._fingerprint import BLOCKS, CLI_ARGS, DPF, DURATION, SFREQ


def _cell(html: str, heading: str) -> str:
    return re.search(rf"<th>{heading}</th><td>([^<]*)</td>", html).group(1)


def _metric_row(html: str, label: str) -> "list[str] | None":
    for row in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)]
        if cells and cells[0] == label:
            return cells[1:]
    return None


def _condition_page(run, name: str) -> str:
    return (run.report.parent / f"sub-01_task-main_cond-{name}_report.html").read_text(
        encoding="utf-8")


def test_the_run_page_names_the_recording_s_length_and_passing_share(denoise_run):
    html = denoise_run.report.read_text(encoding="utf-8")
    # the last sample sits one period before the recording's nominal end
    assert _cell(html, "Duration") == f"{DURATION - 1 / SFREQ:.0f} s"
    share = float(CLI_ARGS[CLI_ARGS.index("--min-good-frac") + 1])
    assert _cell(html, "Passing windows") == f"{share * 100:.0f}% of windows"


def test_the_run_page_prints_the_dpf_the_run_was_given(denoise_run):
    html = denoise_run.report.read_text(encoding="utf-8")
    assert _cell(html, "DPF") == ", ".join(f"{d:g}" for d in DPF)


@pytest.mark.parametrize("name,duration", [(b[0], b[2]) for b in BLOCKS])
def test_a_condition_page_names_its_own_window_as_its_duration(condition_run, name, duration):
    assert _cell(_condition_page(condition_run, name), "Duration") == f"{duration:.0f} s"


def test_the_run_summary_lists_cardiac_power_and_flat_channels_per_set(denoise_run):
    html = denoise_run.report.read_text(encoding="utf-8")
    record = denoise_run.record()
    cp = _metric_row(html, "Mean CP (exp.)")
    assert cp[:2] == [f"{record[s]['cp_mean']:.3f}" for s in ("raw", "raw_long")]
    # the short set measures neither, so its cell holds no number
    assert not re.search(r"\d", cp[2])
    # the fingerprint montage has no flat channel
    assert _metric_row(html, "Flat channels")[:2] == ["0", "0"]


def test_a_condition_page_leaves_out_what_only_the_whole_run_measures(condition_run):
    html = _condition_page(condition_run, BLOCKS[0][0])
    assert _metric_row(html, "Mean CP (exp.)") is None
    assert _metric_row(html, "Flat channels") is None


def _summary_cell(html: str, heading: str) -> str:
    cell = re.search(rf"<th>{heading}</th>\s*<td>(.*?)</td>", html, flags=re.S).group(1)
    return re.sub(r"<[^>]+>", "", cell).split(",")[0].strip()


def test_the_raw_viewer_prints_the_run_report_s_summary_cells(denoise_run, raw_viewer_run):
    report = denoise_run.report.read_text(encoding="utf-8")
    page = raw_viewer_run.report.with_name(
        raw_viewer_run.report.name.replace("_report.html", "_desc-raw_report.html"))
    html = page.read_text(encoding="utf-8")
    data = json.loads(re.search(r"var _STATIC_DATA = (\[.*?\]);\n", html, re.S).group(1))[0]
    cells = data["summary"]["cells"]
    for key, heading in (("bad", "Bad channels"), ("frac", "Passing windows"),
                         ("heart_rate", "Heart rate"), ("duration", "Duration")):
        assert cells[key] == _summary_cell(report, heading), key

