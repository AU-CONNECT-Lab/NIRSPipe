"""Every setting a run record holds: change it alone, and the output and the record move together.

The breadth companion to test_wiring.py. Each case runs the CLI twice on the fingerprint
recording, once as the base and once with one flag changed, and compares three things: the
output files with their sidecars' parameters left out, the flag's key in the run record, and
the parameters of the stage sidecars named for the flag. A flag whose output does not move
needs an entry in ALLOWED_SAME saying why.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import mne
import numpy as np
import pytest

from nirspipe.cli import run as run_cli

from tests._fingerprint import (AUX_FS, DURATION, EVENT_DURATION, fingerprint_raw,
                                make_fingerprint_dataset)
from tests._synth import _write_dataset_root, _write_subject
from tests.provenance._sweep import argv, moved, snapshot

BASE = {"--dpf": ["5.8", "6.4"], "--sci-threshold": ["0.75"], "--psp-threshold": ["0.12"],
        "--min-good-frac": ["0.7"], "--cardiac-l-freq": ["0.8"], "--cardiac-h-freq": ["1.9"],
        "--resp-l-freq": ["0.15"], "--resp-h-freq": ["0.45"], "--window-length": ["12"],
        "--motion-correction": ["tddr"]}
# ols keeps the GLM fit to seconds; the noise model case moves it to ar1
MODE = {"denoise": {"--mode": ["denoise"]},
        "glm": {"--mode": ["glm"], "--stim-dur": [f"{EVENT_DURATION:g}"],
                "--drift-high-pass": ["0.008"], "--noise-model": ["ols"]},
        "rest": {"--mode": ["rest"]}}
QUIET = ["--participant-label", "01", "--skip-bids-validation"]
RECORD = {"prep": None, "post": None, "execution": ("ignore", "allow_cropped_input")}


@dataclass(frozen=True)
class Case:
    mode: str
    change: dict
    sidecars: tuple                  # stages whose parameters must change; () for none
    context: dict = field(default_factory=dict)
    dataset: str = "main"
    report: bool = False             # the flag acts on the report only


SPLINE = {"--motion-correction": ["spline"]}
WAVELET = {"--motion-correction": ["wavelet"]}
EPOCHS = {"--epoch-tmin": ["-5"], "--epoch-tmax": ["25"]}
GVTD = {"--gvtd-censor": [], "--gvtd-censor-n-std": ["3"]}
BAD_SPANS = {"--bad-spans": ["{bad_spans}"]}
AUX = {"--aux-regressors": []}
SHORT = {"--short-channel": ["mean"]}

CASES = {
    "dpf": Case("denoise", {"--dpf": ["6.0", "7.0"]}, ("preproc",)),
    "sci_threshold": Case("denoise", {"--sci-threshold": ["0.9"]}, ("sci",)),
    "motion_correction": Case("denoise", WAVELET, ("motcorrected",)),
    "motion_std_thresh": Case("denoise", {"--motion-std-thresh": ["10"]}, ("motcorrected",), SPLINE),
    "wavelet_iqr": Case("denoise", {"--wavelet-iqr": ["0.8"]}, ("motcorrected",), WAVELET),
    "bad_channels": Case("denoise", {"--bad-channels": ["S1_D1"]}, ("sci",)),
    "keep_spans": Case("denoise", {"--keep-spans": ["{keep}"]}, ("sci",)),
    "keep_spans_from": Case("denoise", {"--keep-spans-from": ["ca"]}, ("sci",),
                            {"--keep-spans": ["{keep}"]}),
    "bad_spans": Case("denoise", BAD_SPANS, ("sci",)),
    "cardiac_l_freq": Case("denoise", {"--cardiac-l-freq": ["1.2"]}, ("sci",)),
    "cardiac_h_freq": Case("denoise", {"--cardiac-h-freq": ["1.5"]}, ("sci",)),
    "resp_l_freq": Case("denoise", {"--resp-l-freq": ["0.1"]}, ("sci",)),
    "resp_h_freq": Case("denoise", {"--resp-h-freq": ["0.3"]}, ("sci",)),
    "psp_threshold": Case("denoise", {"--psp-threshold": ["0.3"]}, ("sci",)),
    "min_good_frac": Case("denoise", {"--min-good-frac": ["0.85"]}, ("sci",)),
    "screen_scope": Case("denoise", {"--screen-scope": ["task"]}, ("sci",)),
    # with no channel left short, the censoring falls back to every channel
    "short_max_dist": Case("denoise", {"--short-max-dist": ["5"]}, ("sci",),
                           {"--gvtd-censor": ["short"], "--gvtd-censor-n-std": ["3"]}),
    "long_min_dist": Case("denoise", {"--long-min-dist": ["20"]}, ("errts",), SHORT),
    "long_max_dist": Case("denoise", {"--long-max-dist": ["25"]}, ("errts",), SHORT),
    "window_length": Case("denoise", {"--window-length": ["15"]}, ("sci",)),
    "epoch_tmin": Case("denoise", {"--epoch-tmin": ["-2"]}, (), EPOCHS, report=True),
    "epoch_tmax": Case("denoise", {"--epoch-tmax": ["15"]}, (), EPOCHS, report=True),
    "epoch_chunk_duration": Case("denoise", {"--epoch-chunk-duration": ["40"]}, (), report=True),
    "epoch_single_trial": Case("denoise", {"--epoch-single-trial": []}, (), dataset="single",
                               report=True),
    "by_condition": Case("denoise", {"--by-condition": []}, (), report=True),
    "gvtd_censor": Case("denoise", GVTD, ("sci",), {"--gvtd-censor-n-std": ["3"]}),
    "gvtd_censor_n_std": Case("denoise", {"--gvtd-censor-n-std": ["5"]}, ("sci",), GVTD),
    "gvtd_min_epoch_s": Case("denoise", {"--gvtd-min-epoch-s": ["5"]}, ("sci",), GVTD),
    "mode": Case("denoise", {"--mode": ["rest"]}, ("filtered",)),
    "high_pass": Case("denoise", {"--high-pass": ["0.02"]}, ("filtered",)),
    "low_pass": Case("denoise", {"--low-pass": ["0.3"]}, ("filtered",)),
    "filter_method": Case("denoise", {"--filter-method": ["fir"]}, ("filtered",)),
    "filter_order": Case("denoise", {"--filter-order": ["2"]}, ("filtered",)),
    "export_mean_filled": Case("denoise", {"--export-mean-filled": []}, ("meanfilled",), BAD_SPANS),
    "censor_fill": Case("denoise", {"--censor-fill": ["spline"]}, ("filtered",), BAD_SPANS),
    "resample_sfreq": Case("denoise", {"--resample-sfreq": ["5"]}, ("resampled",)),
    "combine_runs": Case("denoise", {"--combine-runs": []}, (), dataset="two_runs"),
    "stim_dur": Case("glm", {"--stim-dur": ["5"]}, ("errts", "design")),
    "hrf_model": Case("glm", {"--hrf-model": ["glover"]}, ("errts", "design")),
    "noise_model": Case("glm", {"--noise-model": ["ar1"]}, ("errts", "design")),
    "drift_model": Case("glm", {"--drift-model": ["polynomial"]}, ("errts", "design")),
    "drift_high_pass": Case("glm", {"--drift-high-pass": ["0.01"]}, ("errts", "design")),
    "drift_order": Case("glm", {"--drift-order": ["2"]}, ("errts", "design"),
                        {"--drift-model": ["polynomial"]}),
    "fir_delays": Case("glm", {"--fir-delays": ["0,2,4"]}, ("errts", "design"),
                       {"--hrf-model": ["fir"], "--fir-delays": ["0,1,2"]}),
    "short_channel": Case("denoise", SHORT, ("errts",)),
    "aux_regressors": Case("glm", AUX, ("errts", "design")),
    "aux_channels": Case("glm", {"--aux-channels": ["ACC_X"]}, ("errts", "design"), AUX),
    "fc": Case("denoise", {"--fc": []}, ("pearson",)),
    "min_time": Case("rest", {"--min-time": ["1000"]}, ("pearson",)),
    "events_path": Case("glm", {"--stim-dur": None, "--events-path": ["{events}"]},
                        ("errts", "design")),
    "contrast_file": Case("glm", {"--contrast-file": ["{contrast}"]}, ("contrast",)),
    "roi_mapping": Case("rest", {"--roi-mapping": ["{roi}"]}, ("pearson",)),
    "ignore": Case("denoise", {"--ignore": ["channel-status"]}, ("sci",)),
    "allow_cropped_input": Case("denoise", {"--allow-cropped-input": []}, ()),
}

# a flag whose output may stay put, with the reason
ALLOWED_SAME = {
    "allow_cropped_input": "acts only on a crop tree, where the base run is refused outright",
}
# confirmed defects, pinned until they are fixed
KNOWN = {
    "combine_runs": "PostConfig.combine_runs is set from the flag and read by no step",
}


# ---- the inputs ----

def _second_aux_axis(snirf: Path) -> None:
    n = int(DURATION * AUX_FS)
    with h5py.File(snirf, "a") as handle:
        group = handle["nirs"].create_group("aux2")
        group.create_dataset("name", data=b"ACC_Y")
        group.create_dataset("dataTimeSeries", data=np.random.default_rng(8).standard_normal(n))
        group.create_dataset("time", data=np.arange(n) / AUX_FS)
        group.create_dataset("dataUnit", data=b"m/s^2")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("flag_sweep")
    bids, _ = make_fingerprint_dataset(root, blocks=True, aux=True)
    nirs = bids / "sub-01" / "nirs"
    _second_aux_axis(nirs / "sub-01_task-tapping_nirs.snirf")
    # one pair marked bad in channels.tsv, which --ignore channel-status then skips
    names = mne.io.read_raw_snirf(nirs / "sub-01_task-tapping_nirs.snirf", verbose="error").ch_names
    (nirs / "sub-01_task-tapping_channels.tsv").write_text("name\ttype\tstatus\n" + "".join(
        f"{n}\tNIRSCWAMPLITUDE\t{'bad' if n.startswith('S5_D5') else 'good'}\n" for n in names))

    # no condition repeats: the epoch section is skipped unless --epoch-single-trial
    single = root / "bids_single"
    _write_dataset_root(single, ["01"])
    raw, _ = fingerprint_raw("01", "tapping", blocks=True)
    raw.annotations.delete([i for i, d in enumerate(raw.annotations.description) if d not in ("ca", "cb")])
    _write_subject(single, "01", "tapping", raw)

    two = root / "bids_two_runs"
    _write_dataset_root(two, ["01"])
    for run, seed in (("01", 1), ("02", 2)):
        raw, _ = fingerprint_raw("01", "tapping", seed=seed, blocks=True)
        _write_subject(two, "01", f"tapping_run-{run}", raw)

    files = {
        "keep": ("keep.tsv", "task\tonset\tduration\ntask-tapping\t0\t300\n"),
        "bad_spans": ("bad_spans.tsv", "onset\tduration\n120\t15\n"),
        "events": ("events.tsv", "onset\tduration\ttrial_type\n"
                   + "".join(f"{o}\t5\ttrial\n" for o in (40, 100, 160, 220, 280, 340))),
        "contrast": ("contrast.toml", "ca_minus_cb = { ca = 1, cb = -1 }\n"),
        "roi": ("roi.json", json.dumps({"A": ["S1_D1", "S2_D2"], "B": ["S4_D4", "S5_D5"]})),
    }
    paths = {}
    for key, (name, text) in files.items():
        paths[key] = str(root / name)
        (root / name).write_text(text)
    return {"root": root, "datasets": {"main": bids, "single": single, "two_runs": two},
            "paths": paths}


# ---- running and reading a run ----

@pytest.fixture(scope="module")
def runs(inputs):
    """``(case) -> (base snapshot, changed snapshot)``, each base run once per configuration."""
    bases: dict = {}

    def _run(name: str, case: Case, settings: dict) -> dict:
        out = inputs["root"] / name
        report = [] if case.report else ["--no-report"]
        run_cli.main([str(inputs["datasets"][case.dataset]), str(out), "participant",
                      *argv(settings, inputs["paths"]), *QUIET, *report])
        return snapshot(out / "sub-01", out / "sub-01" / "logs" / "sub-01.toml", RECORD)

    def pair(key: str):
        case = CASES[key]
        base = {**BASE, **MODE[case.mode], **case.context}
        tag = json.dumps([case.dataset, case.report, base], sort_keys=True)
        if tag not in bases:
            bases[tag] = _run(f"base_{len(bases)}", case, base)
        return bases[tag], _run(f"case_{key}", case, {**base, **case.change})

    return pair


# ---- the sweep ----

def test_every_setting_of_a_run_record_has_a_case():
    keys = run_cli.config_keys()
    assert set(CASES) == set(keys["prep"]) | set(keys["post"]) | {"ignore", "allow_cropped_input"}


def _section(key: str) -> str:
    return "execution" if key in ("ignore", "allow_cropped_input") else (
        "prep" if key in run_cli.config_keys()["prep"] else "post")


@pytest.mark.parametrize("key", [
    pytest.param(k, marks=pytest.mark.xfail(strict=True, reason=KNOWN[k])) if k in KNOWN else k
    for k in CASES])
def test_a_changed_setting_moves_the_output_and_the_record(runs, key):
    base, changed = runs(key)
    assert base["record"].get(f"{_section(key)}.{key}") != changed["record"].get(f"{_section(key)}.{key}")
    if key not in ALLOWED_SAME:
        assert moved(base, changed), "the record changed and no output did"
    for stage in CASES[key].sidecars:
        assert base["params"].get(stage) != changed["params"].get(stage), \
            f"desc-{stage} changed nothing in its sidecar parameters"
