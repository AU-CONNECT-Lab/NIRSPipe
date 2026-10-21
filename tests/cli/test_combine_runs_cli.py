"""`nirspipe --combine-runs` on a subject with two runs of a task, through the CLI.

The joined FC is held to a re-derivation from the two runs' own written residuals: each
z-scored per channel over its kept frames with numpy, joined, and correlated over the kept
frames.
"""

import json

import numpy as np
import pandas as pd
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.glm_censored import kept_frames

from tests._fingerprint import CLI_ARGS, fingerprint_raw
from tests._synth import _write_dataset_root, _write_subject
from tests.figure_accuracy._payload import one_figure

ROI = {"A": ["S1_D1", "S2_D2"], "B": ["S4_D4", "S5_D5"]}


def _tree(root, runs, report=False):
    bids = root / "bids"
    _write_dataset_root(bids, ["01"])
    for run, seed in runs:
        raw, _ = fingerprint_raw("01", "tapping", seed=seed, blocks=True)
        _write_subject(bids, "01", f"tapping_run-{run}", raw)
    roi = root / "roi.json"
    roi.write_text(json.dumps(ROI))
    run_cli.main([str(bids), str(root / "out"), "participant", *CLI_ARGS, "--mode", "rest",
                  "--skip-bids-validation", "--combine-runs", "--roi-mapping", str(roi),
                  *([] if report else ["--no-report"])])
    return root / "out" / "sub-01" / "nirs"


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    return _tree(tmp_path_factory.mktemp("combine_two"), (("01", 1), ("02", 2)), report=True)


def _side(path):
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def _runs(nirs):
    return [nirs / f"sub-01_task-tapping_run-{r}_desc-errts_nirs.snirf" for r in ("01", "02")]


def test_the_joined_series_and_its_fc_are_written_without_the_run(two):
    for name in ("desc-errts_nirs.snirf", "chromo-hbo_stat-pearson_relmat.tsv",
                 "chromo-hbr_stat-pearson_relmat.tsv",
                 "chromo-hbo_seg-roi_agg-roi_stat-pearson_relmat.tsv",
                 "chromo-hbo_seg-roi_agg-seed_stat-pearson_relmat.tsv"):
        assert (two / f"sub-01_task-tapping_{name}").exists(), name
    # ALFF stays per run, as the GLM's tables would
    assert not [p for p in two.glob("*stat-alff*") if "_run-" not in p.name]


def test_the_sidecar_names_each_run_its_onset_and_every_runs_bads(two):
    side = _side(two / "sub-01_task-tapping_desc-errts_nirs.snirf")
    runs = _runs(two)
    assert [s.rsplit("/", 1)[-1] for s in side["Sources"]] == [p.name for p in runs]
    first = read_snirf(runs[0])
    assert side["parameters"]["combined_runs"] == [
        {"run": runs[0].name, "onset_s": 0.0},
        {"run": runs[1].name, "onset_s": round(first.n_times / first.info["sfreq"], 6)}]
    assert side["bad_channels"] == sorted({ch for p in runs for ch in _side(p)["bad_channels"]},
                                          key=first.ch_names.index)


def test_the_joined_fc_is_the_runs_z_scored_joined_and_correlated(two):
    parts, keeps = [], []
    for path in _runs(two):
        raw = read_snirf(path)
        keep = kept_frames(raw)
        data = raw.get_data(picks=["S1_D1 hbo", "S2_D2 hbo"])
        kept = data[:, keep]
        parts.append(((data.T - kept.mean(axis=1)) / kept.std(axis=1)).T)
        keeps.append(keep)
    joined, keep = np.hstack(parts), np.concatenate(keeps)
    expected = np.corrcoef(joined[0, keep], joined[1, keep])[0, 1]
    fc = pd.read_csv(two / "sub-01_task-tapping_chromo-hbo_stat-pearson_relmat.tsv", sep="\t",
                     index_col=0)
    assert fc.loc["S1_D1 hbo", "S2_D2 hbo"] == pytest.approx(expected, abs=1e-6)


def test_the_joined_series_gets_no_quality_record_of_its_own(two):
    assert sorted(p.name for p in two.glob("*desc-sqm_qc.json")) == [
        "sub-01_task-tapping_run-01_desc-sqm_qc.json",
        "sub-01_task-tapping_run-02_desc-sqm_qc.json"]


def test_the_subject_index_draws_the_joined_fc(two):
    sub = two.parent
    html = (sub / "sub-01_desc-index_report.html").read_text(encoding="utf-8")
    assert 'id="Combined"' in html and "run-01, run-02" in html
    fig = one_figure(sub / "figures" / "sub-01_task-tapping_desc-joinedfc_nirs.html")
    hbo = fig["data"][0]
    cell = np.asarray(hbo["z"])[list(hbo["y"]).index("S2_D2"), list(hbo["x"]).index("S1_D1")]
    fc = pd.read_csv(two / "sub-01_task-tapping_chromo-hbo_stat-pearson_relmat.tsv", sep="\t",
                     index_col=0)
    assert cell == pytest.approx(fc.loc["S1_D1 hbo", "S2_D2 hbo"], abs=1e-6)
    assert (sub / "figures" / "sub-01_task-tapping_desc-joinedroifc_nirs.html").exists()


def test_one_run_is_not_joined(tmp_path):
    nirs = _tree(tmp_path, (("01", 1),))
    assert not [p for p in nirs.iterdir() if "_run-" not in p.name
                and p.suffix in (".snirf", ".tsv")]
