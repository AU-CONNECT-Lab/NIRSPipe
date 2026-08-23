"""The SQM record: does what the run computed survive the trip through disk.

The record is not returned by the pipeline. It is rebuilt afterwards by reading the files
the run left behind, which makes it the most wiring-shaped thing in the package and the
easiest to lose quietly: `compute_run_sections` wraps every section in its own
`except Exception`, and has further fallbacks for the SCI scores, for resolving the BIDS
input, and for reading it. Nine paths in one function, each of which drops data and logs a
warning rather than failing. A record missing half its sections looks exactly like a
record whose run had less to measure.

So the first test simply demands that a normal run produce all six sections. It is worth
more than its length suggests: it converts all nine of those paths into something loud.

The rest enforce the rules the module docstring states, because those are the ones that
rot when a section is added or a metric moves:

  - SCI is the one input the record cannot recover from any output file, so it travels
    run -> sci sidecar -> record, and nothing else would notice if that link broke
  - Beer-Lambert is the dividing line for bad channels: raw* and motion include them,
    preproc and final exclude them. Reading it the other way gives an sci_mean over
    channels chosen for good SCI, which can never fall below the threshold
  - `final` follows resampled > filtered > preproc, so it names the last haemo file
  - one run per BIDS label, never one that keeps the last task
"""

import json

import numpy as np
import pytest

from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep
from fnirs_pipe.qc.quantitative_metrics import long_short_channels
from fnirs_pipe.qc.sqm_record import (
    SECTIONS,
    build_sqm_records,
    compute_run_sections,
    record_path,
    scan_runs,
)

from ._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)


def _run(out_dir, subject="01", task="tapping", post=True):
    """A full run on synthetic data, left on disk the way the pipeline leaves it."""
    bids = out_dir / "bids"
    (bids / f"sub-{subject}" / "nirs").mkdir(parents=True, exist_ok=True)
    source = bids / f"sub-{subject}" / "nirs" / f"sub-{subject}_task-{task}_nirs.snirf"

    from fnirs_pipe.io.snirf import read_snirf, write_snirf
    write_snirf(synth_raw(subject, task), source)

    # read_snirf, not the in-memory object: it is what stamps the input, and Recorder
    # only registers a source path for an object that carries a stamp
    entities = {"task": task}
    prep = run_prep(read_snirf(source), PrepConfig(subject=subject, dpf=[6.0, 6.0],
                                           sci_threshold=0.8, motion_correction="tddr",
                                           **_BANDS),
                    output_dir=out_dir, source_entities=entities, source_path=source)
    if post:
        run_post(prep.raw_haemo.copy(),
                 PostConfig(subject=subject, high_pass=0.01, low_pass=0.5, **_BANDS),
                 output_dir=out_dir, mode="denoise", source_entities=entities)
    return prep, out_dir / f"sub-{subject}" / "nirs"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    prep, nirs_dir = _run(tmp_path_factory.mktemp("sqm_run"))
    stages = scan_runs(nirs_dir)["sub-01_task-tapping"]
    return prep, nirs_dir, stages, compute_run_sections(stages, **_BANDS)


# ---- Nothing went missing on the way through disk ----

def test_a_normal_run_produces_every_section(run):
    """Each section can vanish on its own without raising, so demand all of them."""
    _, _, _, sections = run
    assert set(SECTIONS) <= set(sections), set(SECTIONS) - set(sections)


def test_every_section_carries_metrics(run):
    _, _, _, sections = run
    for name in SECTIONS:
        assert sections[name], f"{name} is empty"


def test_the_per_channel_maps_cover_every_section(run):
    _, _, _, sections = run
    assert set(sections["per_channel"]) >= set(SECTIONS) - {"final"}


# ---- SCI: the one value no output file carries ----

def test_the_sci_scores_the_run_computed_reach_the_record(run):
    """run -> sci sidecar -> record. Nothing else in the record depends on that link."""
    prep, _, _, sections = run
    stored = sections["per_channel"]["raw"]["sci_per_channel"]
    assert set(stored) == set(prep.sci_scores)
    for ch, score in prep.sci_scores.items():
        assert stored[ch] == pytest.approx(score, rel=1e-9)


def test_sci_mean_averages_every_channel_not_only_the_good_ones(run):
    """The circularity the module docstring warns about: excluding bads floors sci_mean."""
    prep, _, _, sections = run
    assert prep.bad_channels                                   # the synthetic bad pair
    over_all = np.mean(list(prep.sci_scores.values()))
    good_only = np.mean([v for ch, v in prep.sci_scores.items()
                         if ch not in set(prep.bad_channels)])
    assert sections["raw"]["sci_mean"] == pytest.approx(over_all, rel=1e-9)
    assert over_all < good_only                                # the two are distinguishable


def test_the_retention_rate_reports_the_rejected_channels(run):
    prep, _, _, sections = run
    expected = (len(prep.sci_scores) - len(prep.bad_channels)) / len(prep.sci_scores)
    assert sections["raw"]["channel_retention_rate"] == pytest.approx(expected)


# ---- Which file each section measured ----

def test_final_names_the_last_haemo_file(run):
    _, _, _, sections = run
    assert sections["final"]["stage"] == "filtered"


def test_final_falls_back_to_preproc_when_nothing_was_filtered(tmp_path_factory):
    _, nirs_dir = _run(tmp_path_factory.mktemp("sqm_prep_only"), post=False)
    sections = compute_run_sections(scan_runs(nirs_dir)["sub-01_task-tapping"], **_BANDS)
    assert sections["final"]["stage"] == "preproc"


def test_the_long_short_split_keeps_bad_channels():
    """It answers a question about separation, so marking a channel bad must not move it.

    pick_types drops bads by default, and the record marks them before asking for the
    split, so a bad channel used to fall out of both lists. Everything computed from them
    then averaged over channels that were kept for being good: raw_long's sci_mean could
    not fall below the threshold, and its channel_retention_rate was always 1.0.
    """
    raw = synth_raw("01", "tapping")
    before = long_short_channels(raw)
    raw.info["bads"] = [raw.ch_names[0]]
    assert long_short_channels(raw) == before


def test_the_long_section_counts_the_channels_it_rejected(run):
    prep, _, _, sections = run
    in_long = set(sections["per_channel"]["raw_long"]["sci_per_channel"])
    rejected = in_long & set(prep.bad_channels)
    assert rejected                                            # the synthetic bad pair is long
    expected = (len(in_long) - len(rejected)) / len(in_long)
    assert sections["raw_long"]["channel_retention_rate"] == pytest.approx(expected)
    assert sections["raw_long"]["channel_retention_rate"] < 1.0


def test_the_long_and_short_sections_split_the_same_recording(run):
    """raw_long and raw_short are one file through two channel sets, so they partition it."""
    _, _, _, sections = run
    every = set(sections["per_channel"]["raw"]["sci_per_channel"])
    longs = set(sections["per_channel"]["raw_long"]["sci_per_channel"])
    shorts = set(sections["per_channel"]["raw_short"]["sci_per_channel"])
    assert longs | shorts == every
    assert not longs & shorts


# ---- Rebuilding from a tree alone ----

def test_a_tree_alone_rebuilds_the_same_numbers(run):
    """build_sqm_records takes no band arguments; it reads them back from the sidecars."""
    _, nirs_dir, _, sections = run
    written = build_sqm_records(nirs_dir)
    assert written == [record_path(nirs_dir, "sub-01_task-tapping")]
    on_disk = json.loads(written[0].read_text(encoding="utf-8"))
    for name in SECTIONS:
        for key, value in sections[name].items():
            if isinstance(value, float):
                assert on_disk[name][key] == pytest.approx(value, rel=1e-9), f"{name}.{key}"
            else:
                assert on_disk[name][key] == value, f"{name}.{key}"


def test_two_tasks_get_two_records_not_one(tmp_path_factory):
    """A subject with several tasks used to collapse into whichever finished last."""
    out = tmp_path_factory.mktemp("sqm_two_tasks")
    _run(out, task="tapping")
    _run(out, task="rest")
    labels = set(scan_runs(out / "sub-01" / "nirs"))
    assert labels == {"sub-01_task-tapping", "sub-01_task-rest"}
    assert len(build_sqm_records(out / "sub-01" / "nirs")) == 2
