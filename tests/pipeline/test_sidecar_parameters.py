"""Each stage's sidecar records the settings that stage used, plus the state its file carries."""

import json
import re
import shutil
import subprocess
import sys

import pytest

from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_snirf
from nirspipe.qc.boilerplate.vocabulary import steps_from_sidecars
from nirspipe.qc.common.provenance import scan
from nirspipe.qc.subject.sqm_record import build_sqm_records
from nirspipe.utils.lineage import lineage_of

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

ARGS = (*CLI_ARGS, "--mode", "denoise", "--drift-model", "polynomial", "--drift-order", "1",
        "--high-pass", "0.01", "--low-pass", "0.1", "--resample-sfreq", "5",
        "--participant-label", "01", "--no-report", "--skip-bids-validation")
SCREENING = {"sci_threshold", "psp_threshold", "min_good_frac", "screen_scope",
             "screen_scope_counted", "cardiac_l_freq", "cardiac_h_freq", "resp_l_freq",
             "resp_h_freq", "qc_window_s", "bad_channels", "channels_tsv_bads", "ignore",
             "keep_spans", "keep_spans_from", "bad_spans"}
FILTER_STATE = {"high_pass", "low_pass", "filter_method", "filter_order", "censor_fill"}
SEPARATION = {"sep_short_max_mm", "sep_long_min_mm", "sep_long_max_mm"}
EXPECTED = {
    "od": set(),
    "sci": SCREENING,
    "motcorrected": {"motion_correction"},
    "preproc": {"dpf"},
    "filtered": FILTER_STATE | {"filter_description"},
    "resampled": FILTER_STATE | {"resample_sfreq"},
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("sidecar_parameters")
    bids, _ = make_fingerprint_dataset(root)
    out = root / "out"
    run_cli.main([str(bids), str(out), "participant", *ARGS])
    files = {p.name.split("_desc-")[1].split("_")[0]: p for p in out.rglob("*_desc-*_nirs.snirf")}
    return bids, out, files


def _sidecar(path):
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def _params(files, stage):
    return _sidecar(files[stage])["parameters"]


# ---- what each stage records ----

@pytest.mark.parametrize("stage", sorted(EXPECTED))
def test_each_stage_records_only_its_own_settings(run, stage):
    _, _, files = run
    assert set(_params(files, stage)) == EXPECTED[stage]


def test_the_residual_keeps_the_state_and_adds_its_regression(run):
    _, _, files = run
    params = _params(files, "errts")
    assert FILTER_STATE | {"resample_sfreq"} | SEPARATION | {"noise_model", "drift_model"} <= set(params)


def test_no_sidecar_repeats_a_setting_under_a_second_name(run):
    _, _, files = run
    for stage, path in files.items():
        assert not {"l_freq", "h_freq", "sfreq", "subject", "session"} & set(_sidecar(path)["parameters"]), stage


def test_the_aux_table_records_no_settings(tmp_path):
    bids, _ = make_fingerprint_dataset(tmp_path, aux=True)
    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--participant-label", "01",
                  "--no-report", "--skip-bids-validation"])
    (aux,) = out.rglob("*_desc-aux_timeseries.json")
    assert json.loads(aux.read_text(encoding="utf-8"))["parameters"] == {}


def test_the_values_are_the_settings_the_run_was_given(run):
    _, _, files = run
    sci = _params(files, "sci")
    assert (sci["sci_threshold"], sci["psp_threshold"], sci["min_good_frac"]) == (0.75, 0.12, 0.7)
    assert (sci["cardiac_l_freq"], sci["resp_h_freq"]) == (0.8, 0.45)
    assert _params(files, "motcorrected") == {"motion_correction": "tddr"}
    assert _params(files, "resampled")["resample_sfreq"] == 5.0


# ---- the exported script splits them the same way ----

def test_the_exported_script_writes_the_same_split(run, tmp_path):
    _, out, files = run
    script = (out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({(tmp_path / 'script').as_posix()!r})",
                    script, count=1, flags=re.M)
    (tmp_path / "script.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(tmp_path / "script.py")], check=True, cwd=tmp_path)
    written = {p.name.split("_desc-")[1].split("_")[0]: p
               for p in (tmp_path / "script").rglob("*_desc-*_nirs.snirf")}
    assert sorted(written) == sorted(files)
    for stage, path in written.items():
        assert _sidecar(path)["parameters"] == _params(files, stage), stage


# ---- every reader answers the same off the old layout ----

def _old_layout(out, files, dest):
    """A copy of the tree with the sidecars written the way they were before the split."""
    copy = shutil.copytree(out, dest)
    sci = _params(files, "sci")
    config = {"subject": "01", "session": None, "dpf": _params(files, "preproc")["dpf"],
              "motion_correction": _params(files, "motcorrected")["motion_correction"],
              **{k: v for k, v in sci.items() if k != "screen_scope_counted"}}
    errts = _params(files, "errts")
    added = {
        "od": config, "sci": config, "motcorrected": config, "preproc": config,
        "filtered": {"resample_sfreq": errts["resample_sfreq"],
                     **{k: errts[k] for k in SEPARATION},
                     "l_freq": errts["high_pass"], "h_freq": errts["low_pass"]},
        "resampled": {**{k: errts[k] for k in SEPARATION}, "sfreq": errts["resample_sfreq"]},
    }
    for stage, extra in added.items():
        path = copy / files[stage].relative_to(out)
        side = _sidecar(path)
        side["parameters"] = {**extra, **side["parameters"]}
        path.with_suffix(".json").write_text(json.dumps(side, indent=2), encoding="utf-8")
    return copy


def test_readers_answer_the_same_off_the_old_layout(run, tmp_path):
    bids, out, files = run
    old = _old_layout(out, files, tmp_path / "old")
    new_dir = files["preproc"].parent
    old_dir = old / new_dir.relative_to(out)
    label = files["preproc"].name.split("_desc-")[0]

    assert steps_from_sidecars(new_dir, label) == steps_from_sidecars(old_dir, label)
    new_nodes, old_nodes = scan(new_dir, label), scan(old_dir, label)
    assert sorted(new_nodes) == sorted(old_nodes)
    for key in new_nodes:
        assert new_nodes[key].detail == old_nodes[key].detail, key
    for stage, path in files.items():
        old_path = old / path.relative_to(out)
        assert lineage_of(read_snirf(path)).params == lineage_of(read_snirf(old_path)).params, stage


def test_the_quality_record_is_the_same_off_the_old_layout(run):
    bids, out, files = run
    # beside the original, so the relative link to the raw dataset still resolves
    old = _old_layout(out, files, out.parent / "record_old")
    new_copy = shutil.copytree(out, out.parent / "record_new")
    records = {}
    for name, tree in (("new", new_copy), ("old", old)):
        nirs_dir = tree / files["preproc"].parent.relative_to(out)
        (path,) = build_sqm_records(nirs_dir, bids_root=bids)
        side = json.loads(path.read_text(encoding="utf-8"))
        side.pop("timestamp", None)
        records[name] = side
    assert records["new"] == records["old"]
