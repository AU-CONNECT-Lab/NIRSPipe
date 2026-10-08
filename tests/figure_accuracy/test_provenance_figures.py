"""The provenance diagram draws this run's chain, and the MNE report is handed this run's recordings."""

import json
import re

import mne
import numpy as np
import pytest

from fnirs_pipe.io.snirf import read_snirf
from tests._fingerprint import CLI_ARGS, DPF


def _arg(flag):
    return float(CLI_ARGS[CLI_ARGS.index(flag) + 1])


@pytest.fixture(scope="module")
def diagram(denoise_run):
    """The nodes the PNG was drawn from, keyed by file stem."""
    args, _ = denoise_run.captured["provenance_figure"]
    return args[0]


def _key(path):
    return path.name.split(".")[0]


# ---- the boxes ----

def test_every_stage_file_is_a_box_with_its_own_shape(denoise_run, diagram):
    stages = sorted(denoise_run.nirs.glob("*_nirs.snirf"))
    assert len(stages) == 6
    for path in stages:
        node = diagram[_key(path)]
        # the package's reader: the marks live in the sidecar, which SNIRF has no field for
        raw = read_snirf(path)
        n, bad = len(raw.ch_names), len(raw.info["bads"])
        shape = f"{n - bad}/{n} ch" if bad else f"{n} ch"
        expected = f"{shape} · {raw.info['sfreq']:g} Hz · {raw.n_times / raw.info['sfreq']:g} s"
        assert node.state == expected, path.name
        assert not node.missing, path.name


def test_the_rejected_pair_is_counted_out_from_screening_on(denoise_run, diagram):
    rejected = 2 * sum(p.bad for p in denoise_run.truth.pairs)
    n = 2 * len(denoise_run.truth.pairs)
    for desc in ("sci", "motcorrected", "preproc", "filtered", "errts"):
        assert diagram[f"sub-01_task-tapping_desc-{desc}_nirs"].state.startswith(
            f"{n - rejected}/{n} ch"), desc


# ---- the arrows ----

def test_the_chain_is_the_denoise_mode_s_order(diagram):
    chain = ["sub-01_task-tapping_nirs"] + [f"sub-01_task-tapping_desc-{d}_nirs" for d in
                                            ("od", "sci", "motcorrected", "preproc", "filtered",
                                             "errts")]
    for parent, child in zip(chain, chain[1:]):
        assert diagram[child].sources == [parent], child


@pytest.mark.parametrize("desc, detail", [
    ("sci", f"thr {_arg('--sci-threshold'):g}"),
    ("motcorrected", "tddr"),
    ("preproc", "dpf " + ", ".join(f"{d:g}" for d in DPF)),
])
def test_each_arrow_names_the_setting_the_run_was_given(diagram, desc, detail):
    assert diagram[f"sub-01_task-tapping_desc-{desc}_nirs"].detail == detail


def test_the_filter_arrow_names_the_band_the_filtered_stage_was_cut_to(denoise_run, diagram):
    params = json.loads(denoise_run.stage("filtered").with_suffix(".json").read_text())["parameters"]
    assert diagram["sub-01_task-tapping_desc-filtered_nirs"].detail == \
        f"{params['high_pass']:g}-{params['low_pass']:g} Hz"


@pytest.mark.xfail(strict=True, reason="finding 16: the screening arrow names the SCI line only, "
                                       "while the run rejects on the coupled-window share")
def test_the_screening_arrow_names_the_lines_that_reject(diagram):
    detail = diagram["sub-01_task-tapping_desc-sci_nirs"].detail
    for flag in ("--psp-threshold", "--min-good-frac"):
        assert f"{_arg(flag):g}" in detail, flag


@pytest.mark.xfail(strict=True, reason="finding 17: the denoising arrow leaves out the short-channel "
                                       "regression that made the stage")
def test_the_denoising_arrow_names_the_short_channel_regression(diagram):
    assert "short" in diagram["sub-01_task-tapping_desc-errts_nirs"].detail


def test_the_mermaid_source_is_the_graph_the_png_drew(denoise_run, diagram):
    mmd = next(denoise_run.figures.glob("*_desc-provenance_nirs.mmd")).read_text(encoding="utf-8")

    def ident(key):
        return re.sub(r"\W", "_", key)

    drawn = {(ident(src), ident(node.key)) for node in diagram.values() if not node.checkpoint
             for src in node.sources}
    written = set(re.findall(r"^\s+(\w+) --.*?--> (\w+)$", mmd, re.M))
    assert written == drawn
    assert next(denoise_run.figures.glob("*_desc-provenance_nirs.png")).stat().st_size > 0


# ---- the MNE report ----

@pytest.fixture(scope="module")
def mne_inputs(denoise_run):
    args, _ = denoise_run.captured["_build_mne_report"]
    return args[1], args[2]


def test_the_mne_report_is_handed_the_input_intensity(denoise_run, mne_inputs):
    intensity, _ = mne_inputs
    source = next((denoise_run.out.parent / "bids_fingerprint").rglob("*_nirs.snirf"))
    recorded = mne.io.read_raw_snirf(source, verbose="error").load_data()
    assert intensity.ch_names == recorded.ch_names
    np.testing.assert_allclose(intensity.get_data(), recorded.get_data(), rtol=1e-6)


def test_the_mne_report_is_handed_the_preprocessed_haemoglobin_with_the_run_s_marks(
        denoise_run, mne_inputs):
    _, haemo = mne_inputs
    preproc = denoise_run.read("preproc")
    assert haemo.ch_names == preproc.ch_names
    np.testing.assert_allclose(haemo.get_data(), preproc.get_data(), rtol=1e-6, atol=1e-12)
    rejected = {f"{p.name} {c}" for p in denoise_run.truth.pairs if p.bad for c in ("hbo", "hbr")}
    assert set(haemo.info["bads"]) == rejected


def test_the_mne_report_is_written_beside_the_run_page(denoise_run):
    page = denoise_run.out / "sub-01" / "sub-01_task-tapping_desc-mne_report.html"
    assert page.stat().st_size > 0
