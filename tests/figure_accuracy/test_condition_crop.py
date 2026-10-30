"""A condition page's haemoglobin panels measure the condition's own window, and nothing of the run past its end."""

from types import SimpleNamespace

import pytest

from nirspipe.qc import metrics
from nirspipe.qc.subject.report import _section_haemo

from tests._fingerprint import CLI_ARGS


def _config():
    def flag(name):
        return float(CLI_ARGS[CLI_ARGS.index(name) + 1])
    return SimpleNamespace(**{key: flag(f"--{key.replace('_', '-')}") for key in (
        "cardiac_l_freq", "cardiac_h_freq", "resp_l_freq", "resp_h_freq")})


def _namer(kind, **_):
    return f"{kind}.html"


def _haemo(run, tmp_path, crop):
    errors: list = []
    out = _section_haemo(run.read("preproc"), _config(), "01", errors, tmp_path, _namer,
                         raw_errts=run.read("errts"), crop=crop, mode="denoise")
    return out, errors


def test_a_window_past_the_recording_draws_nothing_of_the_run(denoise_run, tmp_path):
    end = float(denoise_run.read("preproc").times[-1])
    out, errors = _haemo(denoise_run, tmp_path, (end + 10.0, end + 20.0))
    assert out["hbo_hbr_path"] is None
    assert out["stage_metrics_path"] is None
    assert out["psd_panel_path"] is None
    assert errors == []


def test_a_window_without_a_band_measures_its_stages_on_the_window(denoise_run, tmp_path,
                                                                   monkeypatch):
    lengths = []
    real = metrics.comparable_stage_metrics

    def spy(stages, *args, **kwargs):
        lengths.append([float(raw.times[-1]) for _, raw in stages])
        return real(stages, *args, **kwargs)

    monkeypatch.setattr(metrics, "comparable_stage_metrics", spy)
    _haemo(denoise_run, tmp_path, (20.0, 80.0))
    assert lengths and all(t == pytest.approx(60.0, abs=0.5) for t in lengths[0])
