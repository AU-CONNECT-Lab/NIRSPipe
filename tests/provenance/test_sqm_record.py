"""The SQM record: does what the run computed survive the trip through disk.

The record is not returned by the pipeline. It is rebuilt afterwards by reading the files
the run left behind, which makes it the most wiring-shaped thing in the package and the
easiest to lose quietly: `compute_run_sections` wraps every section in its own
`except Exception`, and has further fallbacks for the SCI scores, for resolving the BIDS
input, and for reading it. Nine paths in one function, each of which drops data and logs a
warning rather than failing. A record missing half its sections looks exactly like a
record whose run had less to measure.

So the first test simply demands that a run exercising every optional step produce all of
them. It is worth more than its length suggests: it converts all nine of those paths into
something loud.

The rest enforce the rules the module docstring states, because those are the ones that
rot when a section is added or a metric moves:

  - SCI is the one input the record cannot recover from any output file, so it travels
    run -> sci sidecar -> record, and nothing else would notice if that link broke
  - Beer-Lambert is the dividing line for bad channels: raw* and motion include them,
    the haemo sections exclude them. Reading it the other way gives an sci_mean over
    channels chosen for good SCI, which can never fall below the threshold
  - every section is named after the file it measured, so a section exists only when its
    file does, and a run that skipped a step is missing that section rather than holding
    one whose meaning silently moved
  - one run per BIDS label, never one that keeps the last task
"""

import json

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.io.auxiliary import aux_table_path
from nirspipe.pipeline.post_pipeline import PostConfig, run_post
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep
from nirspipe.qc.common.channel_table import CHANNEL_METRICS_SUFFIX
from nirspipe.qc.metrics import long_short_channels
from nirspipe.exceptions import StageError
from nirspipe.qc.subject.group_writer import _scalars, build_group_raw_report
from nirspipe.qc.subject.record_io import read_record, write_record
from nirspipe.qc.subject.sqm_record import (
    SECTIONS,
    build_sqm_records,
    compute_run_sections,
    record_path,
    scan_runs,
    sqm_record_dict,
)

from tests._synth import SFREQ, synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)
# below the synthetic recording's rate, so `resampled` is a different file from `filtered`
_RESAMPLE_SFREQ = SFREQ / 2


# ---- GVTD censoring, deliberately on none of its defaults ----
# PrepConfig's defaults are gvtd_censor_spans' defaults, so a config value that never
# reached the function would still produce a plausible record. These do not match, which
# turns "the kwarg went somewhere else" into a failing assertion.
# gvtd_censor carries the channel set as well as the switch, so this is "long" and
# not True; the assertion below on gvtd_censor_channel_set is what pins that
_CENSOR = dict(gvtd_censor="long", gvtd_censor_n_std=8.0, gvtd_min_epoch_s=20.0)


def _run(out_dir, subject="01", task="tapping", post=True, censor=None, blocks=None):
    """A full run on synthetic data, left on disk the way the pipeline leaves it.

    Post is asked for every optional step it has, so the record comes out with one section
    per haemo stage. A leaner post is what `test_a_skipped_step_leaves_no_section` covers.
    ``blocks`` replaces the events with ``[(onset, duration, label), ...]`` conditions.
    """
    bids = out_dir / "bids"
    (bids / f"sub-{subject}" / "nirs").mkdir(parents=True, exist_ok=True)
    source = bids / f"sub-{subject}" / "nirs" / f"sub-{subject}_task-{task}_nirs.snirf"

    from nirspipe.io.snirf import read_snirf, write_snirf
    raw = synth_raw(subject, task)
    if blocks:
        raw.set_annotations(mne.Annotations(*zip(*blocks)))
    write_snirf(raw, source)

    # read_snirf, not the in-memory object: it is what stamps the input, and Recorder
    # only registers a source path for an object that carries a stamp
    entities = {"task": task}
    prep = run_prep(read_snirf(source), PrepConfig(subject=subject, dpf=[6.0, 6.0],
                                           sci_threshold=0.8, motion_correction="tddr",
                                           **_BANDS, **(censor or {})),
                    output_dir=out_dir, source_entities=entities, source_path=source)
    if post:
        run_post(prep.raw_haemo.copy(),
                 PostConfig(subject=subject, high_pass=0.01, low_pass=0.5,
                            resample_sfreq=_RESAMPLE_SFREQ, short_channel="mean",
                            drift_model="cosine", drift_high_pass=0.01, **_BANDS),
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
    # `windowed` holds per-window series and `excluded` whole-run seconds, not per-channel
    # values, so neither has an entry there
    _, _, _, sections = run
    assert set(sections["per_channel"]) >= set(SECTIONS) - {"windowed", "excluded"}


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

def test_each_haemo_section_measured_its_own_file(run):
    """The sections are distinct files, not three views of one.

    `resampled` is the cheapest proof: it is the only stage whose sampling rate differs, so
    a section wired to the wrong file shows up as a Nyquist that cannot be there.
    """
    _, _, stages, sections = run
    assert {"filtered", "resampled", "errts"} <= set(stages)
    # the bandpass, the resample and the regression each moved the signal somewhere else
    values = [sections[d]["hbo_hbr_corr_mean"] for d in ("preproc", "filtered", "errts")]
    assert len(set(values)) == len(values), values


def test_a_skipped_step_leaves_no_section(tmp_path_factory):
    """A section exists only when its file does. Nothing stands in for a step that was
    skipped."""
    _, nirs_dir = _run(tmp_path_factory.mktemp("sqm_prep_only"), post=False)
    sections = compute_run_sections(scan_runs(nirs_dir)["sub-01_task-tapping"], **_BANDS)
    assert sections["preproc"]
    for desc in ("filtered", "resampled", "errts"):
        assert desc not in sections, desc


def test_the_long_short_split_keeps_bad_channels():
    """It answers a question about separation, so marking a channel bad must not move it.

    pick_types drops bads by default, and the record marks them before asking for the
    split, so a split through pick_types would lose a bad channel from both lists. Everything
    computed from them would then average over channels kept for being good: raw_long's
    sci_mean could not fall below the threshold, and its channel_retention_rate would be 1.0.
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
    on_disk = read_record(written[0])
    for name in SECTIONS:
        for key, value in sections[name].items():
            if isinstance(value, float):
                assert on_disk[name][key] == pytest.approx(value, rel=1e-9), f"{name}.{key}"
            else:
                assert on_disk[name][key] == value, f"{name}.{key}"


def test_the_channel_table_is_written_without_a_report(run):
    """The index and the dyad pages read is_bad from it, so it cannot depend on --no-report."""
    prep, nirs_dir, _, _ = run
    table = nirs_dir / ("sub-01_task-tapping" + CHANNEL_METRICS_SUFFIX)
    table.unlink(missing_ok=True)
    build_sqm_records(nirs_dir)
    rows = pd.read_csv(table, sep="\t")
    assert list(rows["name"]) == list(prep.sci_scores)
    assert set(rows.loc[rows["is_bad"], "name"]) == set(prep.bad_channels)
    # a kept channel has no reason, which BIDS writes as n/a rather than an empty cell
    assert not (pd.read_csv(table, sep="\t", keep_default_na=False) == "").any().any()


def test_two_tasks_get_two_records_not_one(tmp_path_factory):
    """Each task keeps its own record rather than one that keeps whichever finished last."""
    out = tmp_path_factory.mktemp("sqm_two_tasks")
    _run(out, task="tapping")
    _run(out, task="rest")
    labels = set(scan_runs(out / "sub-01" / "nirs"))
    assert labels == {"sub-01_task-tapping", "sub-01_task-rest"}
    assert len(build_sqm_records(out / "sub-01" / "nirs")) == 2


def test_every_cohort_column_is_described(tmp_path_factory):
    """A metric added to the record without a description fails here, not in a reader's R."""
    out = tmp_path_factory.mktemp("cohort")
    _, nirs_dir = _run(out, censor=_CENSOR)
    build_sqm_records(nirs_dir)
    build_group_raw_report(out)

    table = out / "desc-subjects_qc.tsv"
    side = json.loads(table.with_suffix(".json").read_text(encoding="utf-8"))
    columns = pd.read_csv(table, sep="\t").columns
    assert [c for c in columns if "Description" not in side.get(c, {})] == []


# ---- The record on disk: scalars in the JSON, arrays in tables beside it ----

_LABEL = "sub-01_task-tapping"


def _table(path, stat=None, suffix="timeseries", ext=".tsv"):
    stat_part = f"_stat-{stat}" if stat else ""
    return path.parent / f"{_LABEL}{stat_part}_desc-sqm_{suffix}{ext}"


@pytest.fixture(scope="module")
def on_disk(run, tmp_path_factory):
    _, _, _, sections = run
    record = sqm_record_dict(sections, ["bids::sub-01/nirs/sub-01_task-tapping_nirs.snirf"])
    path = tmp_path_factory.mktemp("record_io") / f"{_LABEL}_desc-sqm_qc.json"
    write_record(path, record)
    return record, path


def test_the_tables_fold_back_into_the_record_the_writer_was_handed(on_disk):
    record, path = on_disk
    back = read_record(path)
    back["data"].pop("tables")
    np.testing.assert_equal(back, record)


def test_the_json_keeps_no_array_a_table_holds(on_disk):
    _, path = on_disk
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "per_channel" not in stored
    assert not [k for k in stored["windowed"]
                if k.endswith(("_matrix", "_channels")) or "_per_window" in k]
    expected = [_table(path, stat).name for stat in ("cv", "psp", "sci", "summary")]
    assert stored["data"]["tables"] == sorted([*expected, _table(path, suffix="nirsmap").name])
    for name in stored["data"]["tables"]:
        assert (path.parent / name).with_suffix(".json").exists(), name


def test_a_matrix_table_is_headed_by_the_channels_its_rows_used_to_follow(on_disk):
    """The names make explicit the order readers used to borrow from per_channel."""
    record, path = on_disk
    frame = pd.read_csv(_table(path, "sci"), sep="\t")
    assert list(frame.columns) == list(record["per_channel"]["raw"]["sci_per_channel"])
    assert len(frame) == len(record["windowed"]["sci_times"])


def test_the_window_tables_say_how_long_a_row_is(on_disk):
    record, path = on_disk
    start, stop = record["windowed"]["sci_times"][0]
    for stat in ("sci", "psp", "cv", "summary"):
        side = json.loads(_table(path, stat, ext=".json").read_text(encoding="utf-8"))
        assert side["WindowLength"] == pytest.approx(stop - start), stat
        assert side["SamplingFrequency"] == pytest.approx(1.0 / (stop - start)), stat
        assert side["Sources"] == record["Sources"], stat


def test_every_column_and_level_written_is_described(on_disk):
    _, path = on_disk
    summary = pd.read_csv(_table(path, "summary"), sep="\t")
    side = json.loads(_table(path, "summary", ext=".json").read_text(encoding="utf-8"))
    assert [c for c in summary.columns if "Description" not in side.get(c, {})] == []

    channels = pd.read_csv(_table(path, suffix="nirsmap"), sep="\t")
    side = json.loads(_table(path, suffix="nirsmap", ext=".json").read_text(encoding="utf-8"))
    assert set(side["section"]["Levels"]) == set(channels["section"])
    assert set(side["metric"]["Levels"]) == set(channels["metric"])


def test_imu_columns_carry_their_unit_and_a_missing_window_stays_missing(tmp_path):
    path = tmp_path / "sub-01_task-rest_desc-sqm_qc.json"
    write_record(path, {
        "Sources": [], "data": {}, "imu": {"gyro_speed_unit": "deg/s"},
        "windowed": {"gvtd_window_times_s": [5.0, 15.0],
                     "gyro_speed_per_window": [1.0, float("nan")],
                     "gyro_speed_p95_per_window": [2.0, 3.0]},
    })
    table = tmp_path / "sub-01_task-rest_stat-summary_desc-sqm_timeseries.tsv"
    side = json.loads(table.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["gyro_speed_per_window"]["Units"] == "deg/s"
    assert side["WindowLength"] == 10.0
    assert "n/a" in table.read_text(encoding="utf-8")
    np.testing.assert_equal(read_record(path)["windowed"]["gyro_speed_per_window"],
                            [1.0, float("nan")])


def test_a_record_with_its_arrays_inline_is_refused(tmp_path):
    path = tmp_path / "sub-01_task-rest_desc-sqm_qc.json"
    path.write_text(json.dumps({"step": "sqm", "windowed": {"sci_matrix": [[0.9]]}}))
    with pytest.raises(StageError, match="rerun"):
        read_record(path)


def test_a_listed_table_missing_from_disk_is_an_error(tmp_path):
    path = tmp_path / "sub-01_task-rest_desc-sqm_qc.json"
    write_record(path, {"Sources": [], "data": {},
                        "per_channel": {"raw": {"sci_per_channel": {"S1_D1 760": 0.9}}}})
    (tmp_path / "sub-01_task-rest_desc-sqm_nirsmap.tsv").unlink()
    with pytest.raises(StageError, match="missing"):
        read_record(path)


# ---- GVTD censoring ----
# The step is opt-in and everything it produces is opt-in with it: the annotations, the
# sidecar entry and the record section. So every assertion here has a mirror in
# `test_censoring_off_...`, because a censoring step wired to nothing at all would satisfy
# the off case on its own.

@pytest.fixture(scope="module")
def censored_run(tmp_path_factory):
    prep, nirs_dir = _run(tmp_path_factory.mktemp("sqm_censored"), censor=_CENSOR)
    stages = scan_runs(nirs_dir)["sub-01_task-tapping"]
    return prep, nirs_dir, stages, compute_run_sections(stages, **_BANDS)


def _n_bad_gvtd(path):
    """How many BAD_gvtd annotations the file on disk carries."""
    from nirspipe.io.snirf import read_snirf
    return sum(1 for d in read_snirf(path).annotations.description if d == "BAD_gvtd")


def test_the_censored_run_actually_censored_something(censored_run):
    """Guards every assertion below: they are all vacuous on a run that flagged nothing."""
    prep, _, _, _ = censored_run
    assert prep.censor_spans


def test_the_marks_reach_every_derivative_taken_after_the_censoring(censored_run):
    """desc-sci is written after the annotations are set, so the three files past it
    inherit them through OD -> TDDR -> Beer-Lambert and two snirf round trips."""
    prep, _, stages, _ = censored_run
    for desc in ("sci", "motcorrected", "preproc"):
        assert _n_bad_gvtd(stages[desc]) == len(prep.censor_spans), desc


def test_the_od_file_predates_the_censoring_and_carries_none(censored_run):
    """desc-od is step 1 and the censoring is step 2, so a marked desc-od would mean the
    annotations were attached to the wrong object."""
    _, _, stages, _ = censored_run
    assert _n_bad_gvtd(stages["od"]) == 0


def test_the_censor_section_is_the_sidecar_the_run_wrote(censored_run):
    """Nothing recomputes this one: censoring is a decision the run made, so the section
    is the sidecar read back, and the link is the only thing that can break."""
    _, _, stages, sections = censored_run
    assert "censor" in sections
    sidecar = json.loads(stages["sci"].with_suffix(".json").read_text(encoding="utf-8"))
    assert sections["censor"] == sidecar["gvtd_censor"]


def test_the_config_the_run_was_given_is_what_censored_it(censored_run):
    """The four fields travel config -> gvtd_censor_spans -> sidecar -> record, and the
    record echoes three of them verbatim. This is the assertion that catches a kwarg
    handed to the wrong function, which is silent everywhere else."""
    prep, _, _, sections = censored_run
    censor = sections["censor"]
    assert censor["gvtd_censor_n_std"] == _CENSOR["gvtd_censor_n_std"]
    assert censor["gvtd_censor_min_epoch_s"] == _CENSOR["gvtd_min_epoch_s"]
    # not a config value: the set follows the separation bands, and the record
    # stores which one the picks actually landed on
    assert censor["gvtd_censor_channel_set"] == "long"
    assert censor["gvtd_censor_n_spans"] == len(prep.censor_spans)


def test_censoring_off_leaves_no_marks_and_no_section(run):
    """The default run. A section named after an opt-in step must be absent, not empty,
    and the rest of the record must be untouched by the option not being taken."""
    _, _, stages, sections = run
    for desc in ("od", "sci", "motcorrected", "preproc"):
        assert _n_bad_gvtd(stages[desc]) == 0, desc
    assert "censor" not in sections
    assert set(SECTIONS) <= set(sections), set(SECTIONS) - set(sections)


# ---- The IMU, where the recording carried one ----
# A gyroscope still except for one 10 s turn at 5 deg/s inside the "talk" block, so every
# number the section holds has a closed form: the run's mean is 5 x 10 / 400, its p95 is
# zero, and the turn belongs to one condition and not the other.

_BLOCKS = [(20.0, 150.0, "rest"), (200.0, 150.0, "talk")]
_TURN = (250.0, 260.0)


def _write_aux_table(nirs_dir, stage_path):
    t = np.round(np.arange(0.0, 400.0, 0.01), 2)
    turn = np.where((t >= _TURN[0]) & (t < _TURN[1]), 5.0, 0.0)
    table = pd.DataFrame({"time": t, "GYRO_X_1": turn, "GYRO_Y_1": 0.0, "GYRO_Z_1": 0.0,
                          "ACCEL_X_1": 0.0, "ACCEL_Y_1": 0.0, "ACCEL_Z_1": 9.8})
    path = aux_table_path(stage_path)
    table.to_csv(path, sep="	", index=False, compression="gzip")
    units = {**{f"GYRO_{a}_1": "o/s" for a in "XYZ"}, **{f"ACCEL_{a}_1": "m/s^2" for a in "XYZ"}}
    path.with_name(path.name.removesuffix(".gz")).with_suffix(".json").write_text(
        json.dumps({"Units": units}), encoding="utf-8")


@pytest.fixture(scope="module")
def imu_run(tmp_path_factory):
    _, nirs_dir = _run(tmp_path_factory.mktemp("sqm_imu"), blocks=_BLOCKS)
    stages = scan_runs(nirs_dir)["sub-01_task-tapping"]
    _write_aux_table(nirs_dir, stages["sci"])
    return compute_run_sections(stages, **_BANDS)


def test_no_aux_table_no_imu_section(run):
    _, _, _, sections = run
    assert "imu" not in sections


def test_the_imu_section_is_the_trace_over_the_run(imu_run):
    imu = imu_run["imu"]
    duration = 400.0
    assert np.isclose(imu["gyro_speed_mean"], 5.0 * (_TURN[1] - _TURN[0]) / duration, rtol=0.01)
    assert imu["gyro_speed_p95"] == 0.0
    assert imu["gyro_speed_unit"] == "°/s"
    assert imu["accel_jerk_mean"] == 0.0
    # a sensor that never moved has no rank order to agree with
    assert imu["accel_jerk_gvtd_rho"] is None
    assert -1.0 <= imu["gyro_speed_gvtd_rho"] <= 1.0


def test_each_condition_holds_its_own_stretch_of_the_imu(imu_run):
    by_condition = imu_run["by_condition"]
    assert by_condition["rest"]["scalars"]["gyro_speed_mean"] == 0.0
    talk = by_condition["talk"]["scalars"]["gyro_speed_mean"]
    assert np.isclose(talk, 5.0 * (_TURN[1] - _TURN[0]) / 150.0, rtol=0.01)


def test_the_imu_windows_sit_on_the_gvtd_grid(imu_run):
    windowed = imu_run["windowed"]
    means = np.asarray(windowed["gyro_speed_per_window"])
    centres = np.asarray(windowed["gvtd_window_times_s"])
    assert len(means) == len(centres) == len(windowed["gyro_speed_p95_per_window"])
    turned = centres[means > 0]
    assert turned.size and np.all((turned > _TURN[0] - 10) & (turned < _TURN[1] + 10))


def test_the_group_table_flattens_the_numbers_and_drops_the_units(imu_run):
    flat = _scalars(imu_run)
    assert "imu_gyro_speed_mean" in flat and "imu_gyro_speed_gvtd_rho" in flat
    assert "imu_gyro_speed_unit" not in flat
