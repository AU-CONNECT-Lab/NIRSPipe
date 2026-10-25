"""Methods sentences and report numbers for BAD_ spans, and none of them on a run without spans."""

import json
import re

import pytest

from nirspipe.cli import run as run_cli
from nirspipe.qc.boilerplate import generate_methods_text
from nirspipe.qc.boilerplate.vocabulary import span_keys, template_slots

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

SPAN_WORDS = ("BAD_", "kept time", "GVTD")


def _run(tmp_path, *extra, report=False):
    bids, _ = make_fingerprint_dataset(tmp_path, blocks=True)
    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--mode", "denoise",
                  "--high-pass", "0.01", "--low-pass", "0.1", "--short-channel", "mean",
                  "--noise-model", "ar1", "--fc", *extra, "--skip-bids-validation",
                  *([] if report else ["--no-report"])])
    (nirs,) = {p.parent for p in out.rglob("*_desc-sci_nirs.json")}
    label = next(nirs.glob("*_desc-sci_nirs.json")).name.split("_desc-")[0]
    return out, nirs, label


def _methods(nirs, label):
    return generate_methods_text(versions={"nirspipe": "x"}, nirs_dir=nirs,
                                 label=label)["plain"].split("\n\nReferences")[0]


def _keep(tmp_path, until):
    table = tmp_path / "keep.tsv"
    table.write_text(f"onset\tduration\n0\t{until}\n", encoding="utf-8")
    return ["--keep-spans", str(table)]


# ---- pieces ----

def test_the_span_slots_are_empty_without_spans():
    assert span_keys("sci_pruning", {}, []) == []
    assert span_keys("bandpass", {"censor_fill": "linear"}, []) == []
    assert template_slots("sci_marking", {"sci_threshold": 0.8})["spans"] == ""
    assert template_slots("glm", {"noise_model": "ar1"})["censoring"] == ""
    fc = template_slots("fc", {"chromophores": ["hbo"]})
    assert fc["kept"] == "" and fc["min_time"] == ""
    assert template_slots("alff", {"chromophores": ["hbo"]})["kept"] == ""
    assert template_slots("hyper_wtc", {"excluded_spans_s": {"a": [], "b": []}})["excluded"] == ""


def test_the_regression_sentence_names_the_noise_model_only_when_one_was_fitted():
    ar1 = template_slots("glm", {"noise_model": "ar1", "censored_frac": 0.25})["censoring"]
    ols = template_slots("glm", {"noise_model": "ols", "censored_frac": 0.25})["censoring"]
    assert "25% of the recording" in ar1 and "noise model" in ar1
    assert "25% of the recording" in ols and "noise model" not in ols


def test_a_dyad_sentence_says_both_members_spans_are_left_out():
    slots = template_slots("hyper_wtc", {"excluded_spans_s": {"a": [[1.0, 2.0]], "b": []},
                                         "wtc_whiten_s": 10.0, "wtc_whiten_order": 50})
    assert "either member" in slots["excluded"] and "outside BAD_ spans" in slots["whiten"]


# ---- the whole run ----

def test_a_run_without_spans_has_no_span_sentence_and_no_gvtd_parameters(tmp_path):
    _, nirs, label = _run(tmp_path)
    text = _methods(nirs, label)
    assert not any(word in text for word in SPAN_WORDS)
    sci = json.loads(next(nirs.glob("*_desc-sci_nirs.json")).read_text(encoding="utf-8"))
    assert not any(k.startswith("gvtd") for k in sci["parameters"])


def test_a_run_with_spans_describes_each_of_them(tmp_path):
    _, nirs, label = _run(tmp_path, *_keep(tmp_path, 330), "--gvtd-censor", "long",
                          "--censor-fill", "spline", "--min-time", "10")
    text = _methods(nirs, label)
    assert "marked BAD_unselected" in text
    assert "Windows touching a BAD_ span were not counted." in text
    assert re.search(r"plus 10 times its spread below the mode were marked BAD_gvtd, as was "
                     r"every stretch left between them shorter than 30 s", text)
    assert "filled with one cubic spline" in text
    assert re.search(r"Frames inside BAD_ spans \(\d+% of the recording\) were left out of the "
                     r"fit", text)
    assert "between channels over the frames outside BAD_ spans" in text
    assert "Connectivity was computed only when at least 10 s lay outside BAD_ spans." in text
    fc = json.loads(next(nirs.glob("*_stat-pearson_relmat.json")).read_text(encoding="utf-8"))
    assert {s["source"] for s in fc["excluded_spans"]} == {"--keep-spans", "--gvtd-censor"}


@pytest.mark.parametrize("spans", [False, True])
def test_the_report_shows_excluded_time_only_with_spans(tmp_path, spans):
    out, _, _ = _run(tmp_path, *(_keep(tmp_path, 330) if spans else []), "--by-condition",
                     report=True)
    page = next(out.rglob("*_report.html")).parent
    run_page = next(p for p in page.glob("*_report.html") if "_cond-" not in p.name
                    and "desc-" not in p.name).read_text(encoding="utf-8")
    cond_page = next(page.glob("*_cond-cb_report.html")).read_text(encoding="utf-8")
    assert ('id="Excluded"' in run_page) is spans
    assert ("<th>Kept</th>" in cond_page) is spans
    if spans:
        assert "BAD_unselected" in run_page and "--keep-spans" in run_page
        assert re.search(r"<th>Kept</th><td>120\.0 of 160\.0 s outside every BAD_ span", cond_page)


def test_an_output_skipped_under_min_time_is_noted_with_or_without_spans(tmp_path):
    out, _, _ = _run(tmp_path, "--min-time", "100000", report=True)
    run_page = next(p for p in out.rglob("*_report.html") if "_cond-" not in p.name
                    and "desc-" not in p.name).read_text(encoding="utf-8")
    assert 'id="Excluded"' not in run_page
    assert re.search(r"FC was not computed: [\d.]+ s lie outside its BAD_ spans, under "
                     r"--min-time 100000 s", run_page)
