"""`nirspipe --combine-runs` on a subject with two runs of a task, through the CLI.

The joined FC is held to a re-derivation from the two runs' own written residuals: each
z-scored per channel over its kept frames with numpy, blanked where the run marked the channel
bad, joined, and correlated over the kept frames both channels have. S1_D1 is marked bad in
run-01 only, so it takes part in run-02 alone.
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
    bads = root / "bads.tsv"
    bads.write_text("run\tbad_channels\n01\tS1_D1\n", encoding="utf-8")
    run_cli.main([str(bids), str(root / "out"), "participant", *CLI_ARGS, "--mode", "rest",
                  "--skip-bids-validation", "--combine-runs", "--roi-mapping", str(roi),
                  "--bad-channels", str(bads), *([] if report else ["--no-report"])])
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


def test_the_sidecar_names_each_run_its_onset_and_the_runs_each_channel_took_part_in(two):
    side = _side(two / "sub-01_task-tapping_desc-errts_nirs.snirf")
    runs = _runs(two)
    assert [s.rsplit("/", 1)[-1] for s in side["Sources"]] == [p.name for p in runs]
    first = read_snirf(runs[0])
    assert side["parameters"]["combined_runs"] == [
        {"run": runs[0].name, "onset_s": 0.0},
        {"run": runs[1].name, "onset_s": round(first.n_times / first.info["sfreq"], 6)}]
    bads = [set(_side(p)["bad_channels"]) for p in runs]
    assert "S1_D1 hbo" in bads[0] - bads[1]
    # bad in the joined series only where every run marked it
    assert side["bad_channels"] == [ch for ch in first.ch_names if all(ch in b for b in bads)]
    assert side["parameters"]["channel_runs"] == {
        ch: [f"run-{r}" for r, b in zip(("01", "02"), bads) if ch not in b]
        for ch in first.ch_names}


def test_a_channel_bad_in_one_run_is_nan_there_and_kept_in_the_other(two):
    joined = read_snirf(two / "sub-01_task-tapping_desc-errts_nirs.snirf")
    first = read_snirf(_runs(two)[0])
    row = joined.get_data(picks=["S1_D1 hbo"])[0]
    assert np.isnan(row[:first.n_times]).all() and np.isfinite(row[first.n_times:]).all()


@pytest.mark.parametrize("pair", [("S1_D1 hbo", "S2_D2 hbo"), ("S2_D2 hbo", "S4_D4 hbo")])
def test_the_joined_fc_is_the_runs_z_scored_joined_and_correlated(two, pair):
    parts, keeps = [], []
    for path in _runs(two):
        raw = read_snirf(path)
        keep = kept_frames(raw)
        data = raw.get_data(picks=list(pair))
        kept = data[:, keep]
        part = ((data.T - kept.mean(axis=1)) / kept.std(axis=1)).T
        part[[ch in raw.info["bads"] for ch in pair]] = np.nan
        parts.append(part)
        keeps.append(keep)
    joined = np.hstack(parts)
    both = np.concatenate(keeps) & np.isfinite(joined).all(axis=0)
    expected = np.corrcoef(joined[0, both], joined[1, both])[0, 1]
    fc = pd.read_csv(two / "sub-01_task-tapping_chromo-hbo_stat-pearson_relmat.tsv", sep="\t",
                     index_col=0)
    assert fc.loc[pair] == pytest.approx(expected, abs=1e-6)


def test_the_joined_roi_fc_averages_each_run_over_its_own_good_members(two):
    parts = {"A": [], "B": []}
    keeps = []
    for path in _runs(two):
        raw = read_snirf(path)
        keep = kept_frames(raw)
        keeps.append(keep)
        for roi, bases in ROI.items():
            picks = [f"{b} hbo" for b in bases if f"{b} hbo" not in raw.info["bads"]]
            mean = raw.get_data(picks=picks).mean(axis=0)
            parts[roi].append((mean - mean[keep].mean()) / mean[keep].std())
    keep = np.concatenate(keeps)
    a, b = (np.concatenate(parts[roi])[keep] for roi in ("A", "B"))
    roi_fc = "sub-01_task-tapping_chromo-hbo_seg-roi_agg-roi_stat-pearson_relmat.tsv"
    fc = pd.read_csv(two / roi_fc, sep="\t", index_col=0)
    assert fc.loc["A", "B"] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-6)
    assert _side(two / roi_fc)["parameters"]["roi_channels"]["run-01"]["A"] == ["S2_D2 hbo"]


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


def test_files_told_apart_by_another_entity_are_not_runs_of_one_series(tmp_path):
    bids = tmp_path / "bids"
    _write_dataset_root(bids, ["01"])
    for acq, seed in (("a", 1), ("b", 2)):
        raw, _ = fingerprint_raw("01", "tapping", seed=seed, blocks=True)
        _write_subject(bids, "01", f"tapping_acq-{acq}_run-01", raw)
    run_cli.main([str(bids), str(tmp_path / "out"), "participant", *CLI_ARGS, "--mode", "rest",
                  "--skip-bids-validation", "--combine-runs", "--no-report"])
    nirs = tmp_path / "out" / "sub-01" / "nirs"
    assert not [p for p in nirs.iterdir() if "_run-" not in p.name
                and p.suffix in (".snirf", ".tsv")]
