"""Both dyad pages list each member's IMU summaries, as the subject pages do."""

import re

import h5py
import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.io.naming import parse_path
from nirspipe.io.groups import GroupEntry
from nirspipe.pipeline.hyper.group_quality import load_group_sqm
from nirspipe.qc.boilerplate.vocabulary import MISSING_VALUE, format_metric
from nirspipe.qc.hyper.hyper_report import imu_member_rows
from tests._raw_qc import raw_qc
from tests._synth import make_hyper_dataset

SIDS = ["sub-11", "sub-12"]
RAW_ARGS = ("--dpf", "6", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
            "--cardiac-h-freq", "1.5", "--skip-bids-validation")


def _section(html: str) -> str:
    return re.search(r'<section id="MemberMetrics".*?</section>', html, re.S).group(0)


# ---- the rows ----

def test_the_rows_sit_under_the_first_set_and_skip_a_sensor_nobody_has():
    rows = imu_member_rows({"sub-11": {"gyro_speed_mean": 1.25, "gyro_speed_p95": 4.0},
                            "sub-12": {}}, SIDS, n_sets=2)
    assert [r["key"] for r in rows] == ["gyro_speed_mean", "gyro_speed_p95"]
    assert [c["text"] for c in rows[0]["cells"]] == [
        format_metric("gyro_speed_mean", 1.25), MISSING_VALUE, "", ""]


def test_no_member_with_an_imu_gives_no_rows():
    assert imu_member_rows({"sub-11": {}, "sub-12": {}}, SIDS, n_sets=1) == []


# ---- the raw pass ----

def _add_gyro(snirf):
    # longer than the recording: the summary reads only the optical span
    n = 50 * 2000
    with h5py.File(snirf, "a") as handle:
        group = handle["nirs"].create_group("aux1")
        group.create_dataset("name", data=b"GYRO_X")
        group.create_dataset("dataTimeSeries",
                             data=np.random.default_rng(3).standard_normal(n))
        group.create_dataset("time", data=np.arange(n) / 50.0)
        group.create_dataset("dataUnit", data=b"deg/s")


@pytest.fixture(scope="module")
def raw_pass(tmp_path_factory):
    root = tmp_path_factory.mktemp("dyad_raw_imu")
    bids, pairs = make_hyper_dataset(root, tasks=("hold",))
    _add_gyro(next((bids / "sub-11" / "nirs").glob("*.snirf")))
    out = root / "out"
    raw_qc(bids, out, pairs, participant=RAW_ARGS)
    return out


def test_hyper_raw_records_and_prints_the_member_with_a_gyroscope(raw_pass):
    (path,) = raw_pass.glob("group-*/nirs/*_desc-subject_qc.tsv")
    table = pd.read_csv(path, sep="\t").set_index("subject_id")
    assert np.isfinite(table.loc["sub-11", "gyro_speed_mean"])
    assert table.loc["sub-11", "gyro_speed_unit"] == "°/s"
    assert pd.isna(table.loc["sub-12", "gyro_speed_mean"])
    section = _section(next(raw_pass.glob("group-*/*_desc-raw_report.html")).read_text(
        encoding="utf-8"))
    assert "Gyroscope speed mean" in section
    assert "Accelerometer jerk" not in section


# ---- the analysis pages ----

def test_the_member_record_s_imu_section_reaches_the_dyad(tmp_path):
    from nirspipe.qc.subject.record_io import write_record
    from nirspipe.qc.subject.sqm_record import RECORD_SUFFIX

    nirs = tmp_path / "sub-11" / "nirs"
    nirs.mkdir(parents=True)
    write_record(nirs / ("sub-11_task-tap" + RECORD_SUFFIX),
                 {"Sources": [], "imu": {"gyro_speed_mean": 1.5, "gyro_speed_unit": "°/s"}})
    sqm = load_group_sqm(tmp_path, [GroupEntry("G1", "sub-11", "tap")])["sub-11"]
    assert sqm["gyro_speed_mean"] == 1.5


SFREQ, DURATION, CROP = 5.0, 400.0, 20.0
LABELS = ["S1_D1", "S2_D2", "S3_D3"]
WINDOWS = {"rest": (20.0, 200.0), "talk": (210.0, 390.0)}


def _raw(seed: int, shared: np.ndarray) -> mne.io.Raw:
    rng = np.random.default_rng(seed)
    names = [f"{label} {c}" for label in LABELS for c in ("hbo", "hbr")]
    info = mne.create_info(names, SFREQ, [c for _ in LABELS for c in ("hbo", "hbr")])
    for i, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        loc[3:6] = [i * 0.05, 0.0, 0.0]
        loc[6:9] = [i * 0.05 + 0.03, 0.0, 0.0]
        loc[:3] = (loc[3:6] + loc[6:9]) / 2
        ch["loc"] = loc
    data = 1e-6 * (shared + 0.2 * rng.standard_normal((len(names), len(shared))))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations([w[0] for w in WINDOWS.values()], [180.0, 180.0],
                                        list(WINDOWS)))
    return raw


def _entry(window, gyro=None):
    return {"window_s": list(window),
            "scalars": {} if gyro is None else {"gyro_speed_mean": gyro},
            "od_by_set": {"all": {"channel_retention_rate": 1.0, "sci_win_mean": 0.9}}}


# sub-11 carries an IMU, sub-12 none; the record's own clock starts CROP before the dyad's
RUN = {"by_set": {"all": {"channel_retention_rate": 1.0, "sci_win_mean": 0.9}}}
SQM = {"sub-11": {**RUN, "gyro_speed_mean": 1.5,
                  "by_condition": {"rest": _entry(WINDOWS["rest"], 0.75),
                                   "talk": _entry(WINDOWS["talk"], 2.25)}},
       "sub-12": {**RUN, "by_condition": {label: _entry(w) for label, w in WINDOWS.items()}}}


@pytest.fixture(scope="module")
def post_pages(tmp_path_factory) -> dict:
    from nirspipe.qc.hyper.hyper_report import build_hyper_post_report

    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    shared = np.sin(2 * np.pi * 0.05 * t)
    raws = {sid: _raw(i, shared).crop(tmin=CROP) for i, sid in enumerate(SIDS)}
    out = tmp_path_factory.mktemp("dyad_post_imu")
    path = build_hyper_post_report(
        group_id="G1", task="tap", group=[GroupEntry("G1", sid, "tap") for sid in SIDS],
        aligned_raws=raws, offsets={sid: CROP for sid in SIDS}, output_dir=out,
        subject_sqm=SQM, wtc_fmin=0.02, wtc_fmax=0.2, wtc_band_fmin=0.03, wtc_band_fmax=0.10,
        wtc_chroma=("hbo",), wtc_by_condition=True,
    )
    return {parse_path(p.name).get("condition") or "": _section(p.read_text(encoding="utf-8"))
            for p in path.parent.glob("*.html") if "index" not in p.name}


@pytest.mark.parametrize("page, value", [("", 1.5), ("rest", 0.75), ("talk", 2.25)])
def test_each_page_prints_the_member_s_value_for_its_own_window(post_pages, page, value):
    section = post_pages[page]
    row = re.search(r"Gyroscope speed mean.*?</tr>", section, re.S).group(0)
    assert format_metric("gyro_speed_mean", value) in row
    assert MISSING_VALUE in row
